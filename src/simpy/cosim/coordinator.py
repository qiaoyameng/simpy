"""Conservative coordinator for independent SimPy environments.

保守(Chandy-Misra 风格,基于连接 lookahead)协调器。

不变量
------
对每条连接 ``c: source -> target``,最小延迟 ``L = c.min_delay`` 严格为正。
只要源环境尚未越过时刻 ``t``,该连接上任何消息的到达时刻都
``>= t + L``。因此目标环境可以安全地前进到

    safe_horizon(target) = min(source.now + L) over all incoming c

而绝不会在之后收到"过去"的消息。没有入连接的模型上界为 ``+inf``。

协调器在每一轮:

1. 依据各环境*当前*时间和连接声明重算安全上界;
2. 只选择满足 ``内部下一事件时刻 <= 上界`` 或
   ``存在到达时刻 <= 上界的在途消息`` 的模型推进;
3. 先把所有 ``到达时刻 <= t`` 的入消息按公开稳定序投递,
   再前进一个本地事件(绝不跨过上界,也不回退);
4. 没有可推进模型时,区分"全部模型完成"与"协议无法继续推进(停滞)"。

各环境的内部堆从不被合并;协调器通过公开的 ``peek/step`` 驱动。
"""
from heapq import heappop
from itertools import count

from simpy.cosim.model import _active


INF = float('inf')

# Result of a coordinator run.
COMPLETED = 'completed'
STALLED = 'stalled'


class Eligible(object):
    """A model the coordinator is allowed to advance next.

    候选快照:``time`` 是将前进到的时刻,``has_event``/``has_message``
    说明该时刻由内部事件还是到达的跨模型消息触发。自定义选择策略
    接收 :class:`Eligible` 的列表并返回其中一个。
    """

    __slots__ = ('model', 'time', 'has_event', 'has_message')

    def __init__(self, model, time, has_event, has_message):
        self.model = model
        self.time = time
        self.has_event = has_event
        self.has_message = has_message

    def __repr__(self):
        return 'Eligible(%r at %g, event=%s, message=%s)' % (
            self.model.name, self.time, self.has_event,
            self.has_message)


class RunResult(object):
    """Outcome of :meth:`Coordinator.run`."""

    __slots__ = ('status', 'reason', 'iterations', 'times')

    def __init__(self, status, reason, iterations, times):
        self.status = status
        self.reason = reason
        self.iterations = iterations
        self.times = times

    @property
    def completed(self):
        """True iff every model finished with no messages in transit."""
        return self.status == COMPLETED

    def __repr__(self):
        return 'RunResult(status=%r, iterations=%d)' % (
            self.status, self.iterations)


class Coordinator(object):
    """Advances several :class:`~simpy.cosim.model.Model` instances safely.

    :param models: federates taking part in the co-simulation.
    :param connections: declared :class:`Connection` objects. Every
        connection endpoint must be one of *models*; names must be unique.
    :param select: how the model advanced next is chosen:

        - ``'round-robin'`` (default): rotate through eligible models;
        - ``'name'`` / ``'reverse'``: eligible models in name order /
          reverse name order (the selection order changes, the resulting
          business trajectory must not);
        - ``'earliest'``: the eligible model with the smallest event time;
        - a callable ``f(list_of_Eligible) -> Eligible``.
    """

    def __init__(self, models, connections, select='round-robin'):
        self.models = list(models)
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError('model names must be unique')
        self._by_name = {m.name: m for m in self.models}
        self.connections = list(connections)
        if len({c.name for c in self.connections}) != \
                len(self.connections):
            raise ValueError('connection names must be unique')
        for conn in self.connections:
            if (conn.source.name not in self._by_name or
                    conn.target.name not in self._by_name or
                    self._by_name[conn.source.name] is not conn.source or
                    self._by_name[conn.target.name] is not conn.target):
                raise ValueError(
                    'connection %r references an unregistered model'
                    % conn.name)
            conn._coordinator = self
            conn.target._incoming.append(conn)
        for model in self.models:
            model._coordinator = self
        self._select_name = select if isinstance(select, str) else None
        self._select_fn = None if isinstance(select, str) else select
        self._rr = count()
        if isinstance(select, str) and select not in (
                'round-robin', 'name', 'reverse', 'earliest'):
            raise ValueError("unknown selection policy %r" % (select,))

    # -- conservative lower bounds / 保守时间下界 --------------------

    def _channel_bound(self, conn):
        """Time up to which one incoming channel can currently promise.

        - 若源模型仍有存活进程,它可能再发消息,承诺下界为
          ``min(最早在途到达, source.now + min_delay)``;
        - 若源模型所有进程已终止,它永远不会再发,承诺下界就是
          已在途消息的最早确定到达时刻(无在途消息则 ``+inf``)。
        """
        in_transit = conn.earliest
        if conn.source.all_processes_terminated:
            return in_transit
        return min(in_transit, conn.source.now + conn.min_delay)

    def safe_horizon(self, model):
        """Latest time *model* may reach without risk of a late message.

        ``min`` of every incoming channel's promised bound;
        ``+inf`` for models with no incoming connections.
        """
        bound = INF
        for conn in model._incoming:
            bound = min(bound, self._channel_bound(conn))
        return bound

    def _eligible(self):
        out = []
        for model in self.models:
            horizon = self.safe_horizon(model)
            event_t = model.env.peek()
            has_event = event_t != INF and event_t <= horizon
            message_t = INF
            for conn in model._incoming:
                earliest = conn.earliest
                if earliest != INF and earliest <= horizon:
                    message_t = min(message_t, earliest)
            has_message = message_t != INF
            # A concrete trigger is mandatory: a local event or a due
            # message. Advancing an empty clock only to widen other
            # models' bounds is an unbounded "null advance" and must not
            # be mistaken for progress (that situation is a stall).
            # 必须有具体触发;仅为放宽他者下界而空推时钟是无界空推进,
            # 不视为进展(应判停滞)。
            if not has_event and not has_message:
                continue
            t = INF
            if has_event:
                t = min(t, event_t)
            if has_message:
                t = min(t, message_t)
            out.append(Eligible(model, t, event_t <= t,
                                message_t <= t))
        return out

    def _choose(self, eligible):
        if self._select_fn is not None:
            return self._select_fn(list(eligible))
        policy = self._select_name
        if policy == 'round-robin':
            offset = next(self._rr) % len(eligible)
            return eligible[offset]
        if policy == 'name':
            return sorted(eligible, key=lambda e: e.model.name)[0]
        if policy == 'reverse':
            return sorted(eligible, key=lambda e: e.model.name,
                          reverse=True)[0]
        return sorted(eligible,
                      key=lambda e: (e.time, e.model.name))[0]

    # -- message delivery / 同刻消息稳定投递 --------------------------

    def _due_messages(self, model, t):
        due = []
        for conn in model._incoming:
            heap = conn.in_transit
            while heap and heap[0][0] <= t:
                _, _, message = heappop(heap)
                due.append(message)
        # Public, stable same-time order:
        # (arrival, connection name, per-connection sequence).
        # 公开稳定排序:(到达时刻, 连接名, 连接内序号)。
        return sorted(due, key=lambda m: m.order_key)

    def _deliver(self, model, messages):
        with _active(model):
            for message in messages:
                model.inbox.put(message)

    def _advance(self, eligible):
        model = eligible.model
        t = eligible.time
        with _active(model):
            if model.env.peek() == t:
                # A local event at t fires first; messages at t are
                # delivered afterwards, independently of selection order.
                # 本地已排在 t 的事件先执行;t 时刻的消息随后投递,
                # 该先后与协调器选择次序无关。
                model.env.step()
            else:
                # No earlier local event: advance the clock exactly to t
                # via a zero payload timeout, never past t.
                # 本地没有更早事件:用精确到 t 的超时把时钟推进到 t,
                # 绝不越过 t。
                tick = model.env.timeout(t - model.env.now)
                model.env.step()
                assert tick.processed and model.env.now == t
            due = self._due_messages(model, t)
        self._deliver(model, due)

    # -- termination analysis / 停止原因判定 ---------------------------

    def _pending_in_transit(self):
        return [(conn.name, len(conn.in_transit))
                for conn in self.connections if conn.in_transit]

    def _stop_reason(self):
        pending = self._pending_in_transit()
        terminated = all(m.all_processes_terminated for m in self.models)
        if terminated and not pending:
            reason = ('all models completed: every process on every model '
                      'has terminated and no messages remain in transit')
            return COMPLETED, reason
        parts = []
        waiting = [m.name for m in self.models if m.is_waiting]
        if waiting:
            parts.append('models %s block on empty inboxes' % waiting)
        if pending:
            parts.append('messages still in transit %s' % pending)
        reached = {m.name: m.now for m in self.models}
        parts.append('local times %s' % reached)
        detail = '; '.join(parts)
        return STALLED, (
            'protocol cannot make further progress: ' + detail +
            '. The conservative protocol stops here (it never advances '
            'past a message time and never rolls back).')

    # -- main loop / 主循环 -------------------------------------------

    def run(self, on_stall='return', max_iterations=10_000_000):
        """Run the federation until completion or an unresolvable stall.

        :param on_stall: ``'return'`` gives a :class:`RunResult` with
            ``status='stalled'``; ``'raise'`` raises
            :class:`ProtocolStalledError` instead.
        :param max_iterations: safety bound on coordinator decisions.
        """
        if on_stall not in ('return', 'raise'):
            raise ValueError("on_stall must be 'return' or 'raise'")
        iterations = 0
        while True:
            eligible = self._eligible()
            if not eligible:
                status, reason = self._stop_reason()
                result = RunResult(
                    status, reason, iterations,
                    {m.name: m.now for m in self.models})
                if status == STALLED and on_stall == 'raise':
                    from simpy.cosim.exceptions import ProtocolStalledError
                    raise ProtocolStalledError(reason)
                return result
            choice = self._choose(eligible)
            self._advance(choice)
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError(
                    'exceeded max_iterations=%d (possible zero-delay '
                    'livelock)' % max_iterations)

    def _notify_send(self, connection):
        # Heaps are inspected directly each round; this hook exists so
        # connections stay coupled to the coordinator they were declared on.
        if connection not in self.connections:
            raise RuntimeError(
                'connection %r is not registered with this coordinator'
                % connection.name)
