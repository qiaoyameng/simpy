"""Exceptions raised by the conservative co-simulation layer.

保守协同仿真层的异常类型。每一种都对应一类协议违例,
协调器不会通过"先越过消息时间再回退"来修复它们。
"""


class CosimError(Exception):
    """Base class for all co-simulation protocol errors."""


class ConnectionDeclarationError(CosimError):
    """A model-to-model connection was declared with an invalid bound.

    连接声明非法,例如最小传输延迟不是严格正数、源与汇是同一个模型,
    或连接引用了没有向协调器注册的模型。
    """


class MinimumDelayError(CosimError):
    """A message was offered with a delay smaller than the declared bound.

    请求的传输时间早于 发送时刻 + 连接声明的最小延迟。
    """


class PastMessageError(CosimError):
    """A message was offered at a time already left behind by the receiver.

    消息的到达时刻早于接收环境当前时间(给过去发消息)。
    """


class CrossModelScheduleError(CosimError):
    """An event was scheduled directly into another model's environment.

    业务代码绕过连接,直接在另一个模型的 SimPy 环境里调度事件。
    跨模型事件只能通过 :class:`~simpy.cosim.connection.Connection` 发送。
    """


class ProtocolStalledError(CosimError):
    """Optional strict-mode failure when the protocol cannot advance.

    当 ``on_stall='raise'`` 时,协调器确认无法继续推进会抛出此异常;
    默认情况下它返回一个 ``stalled`` 状态的结果对象。
    """
