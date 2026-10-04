"""Conservative co-simulation of independent SimPy environments.

把多个各带独立 :class:`~simpy.core.Environment` 的模型放在同一进程里
做保守联合仿真。跨模型事件只能通过声明了严格正最小延迟的连接发送,
由 :class:`Coordinator` 依据连接时间下界决定哪些环境可以安全前进。

公开接口::

    from simpy.cosim import Connection, Coordinator, Model, Message
    from simpy.cosim import COMPLETED, STALLED, Eligible, RunResult
    from simpy.cosim.exceptions import (
        CosimError, ConnectionDeclarationError, MinimumDelayError,
        PastMessageError, CrossModelScheduleError, ProtocolStalledError)

The coordinator never merges per-environment heaps, never advances past a
promised message time, and never rolls a clock back.
协调器不合并各环境内部堆,不越过消息时间,也不做时间回退。
"""
from simpy.cosim.connection import Connection
from simpy.cosim.coordinator import (Coordinator, Eligible, RunResult,
                                     COMPLETED, STALLED)
from simpy.cosim.messages import Message
from simpy.cosim.model import Model
from simpy.cosim import exceptions

__all__ = [
    'Connection', 'Coordinator', 'Model', 'Message',
    'Eligible', 'RunResult', 'COMPLETED', 'STALLED', 'exceptions',
]
