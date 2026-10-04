"""Independent SimPy environment wrapped as a co-simulation model.

每个模型拥有*自己*的 :class:`~simpy.core.Environment` 与自己的
事件堆;协调器绝不会把各环境的内部堆合并成共享堆。

跨模型的直接调度由 :class:`GuardedEnvironment` 拦截:
业务代码只能通过 :class:`~simpy.cosim.connection.Connection`
向其他模型发送事件。
"""
import threading

from simpy.core import Environment
from simpy.resources.store import Store

from simpy.cosim.exceptions import CrossModelScheduleError


# The model whose environment the coordinator is currently advancing.
# 协调器当前正在推进的模型;用于检测跨环境直连调度。
_state = threading.local()


def current_model():
    """The model currently being stepped by the coordinator, if any."""
    return getattr(_state, 'model', None)


class _active(object):
    """Context marker: *model* is the only environment allowed to schedule."""

    def __init__(self, model):
        self.model = model
        self.previous = None

    def __enter__(self):
        self.previous = getattr(_state, 'model', None)
        _state.model = self.model
        return self

    def __exit__(self, exc_type, exc, tb):
        _state.model = self.previous


class GuardedEnvironment(Environment):
    """Environment that rejects events scheduled for another model.

    如果业务代码在推进模型 A 时,直接把事件调度进模型 B 的环境堆,
    抛出 :class:`CrossModelScheduleError`。合法的跨模型事件必须走连接。
    """

    def __init__(self, model, initial_time=0):
        super(GuardedEnvironment, self).__init__(initial_time)
        self._cosim_model = model

    def schedule(self, event, priority=None, delay=0):
        active = current_model()
        if active is not None and active is not self._cosim_model:
            raise CrossModelScheduleError(
                'model %r scheduled an event directly into environment '
                'of model %r; cross-model events must be sent through a '
                'Connection'
                % (active.name, self._cosim_model.name))
        if priority is None:
            from simpy.events import NORMAL
            priority = NORMAL
        return super(GuardedEnvironment, self).schedule(
            event, priority, delay)


class Model(object):
    """A federate: one guarded SimPy environment plus an incoming inbox.

    :param name: unique model name.
    :param initial_time: local clock start (normally ``0``).
    """

    def __init__(self, name, initial_time=0):
        self.name = name
        self.env = GuardedEnvironment(self, initial_time)
        # Cross-model messages, once due, are deposited in this FIFO store.
        # 到达时刻的跨模型消息最终进入这个收件箱,由业务进程自行取出。
        self.inbox = Store(self.env)
        self._coordinator = None
        self._incoming = []  # Connections targeting this model.
        self.processes = []  # All processes ever started on this model.

    def process(self, generator):
        """Register a SimPy process in this model's environment."""
        proc = self.env.process(generator)
        self.processes.append(proc)
        return proc

    @property
    def all_processes_terminated(self):
        """True if every started process has ended.

        所有已启动进程均已终止——即使仍有未完成的收件箱 ``get``
        (终止的进程留下的请求不再代表可恢复的活动)。
        """
        return all(not proc.is_alive for proc in self.processes)

    @property
    def now(self):
        """Local simulation time of this model. 模型本地仿真时间。"""
        return self.env.now

    @property
    def is_waiting(self):
        """True if the model only blocks on an empty inbox.

        本地事件堆为空、唯一挂起的是对收件箱的 ``get`` 请求,
        说明它在等待其他模型的消息(可能是死锁/停滞)。
        """
        if self.env.peek() != float('inf'):
            return False
        return bool(self.inbox.get_queue)

    @property
    def is_idle(self):
        """True if nothing remains locally: no events, no live process."""
        return (self.env.peek() == float('inf')
                and self.all_processes_terminated)

    def __repr__(self):
        return 'Model(%r, now=%g)' % (self.name, self.env.now)
