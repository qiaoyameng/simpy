"""Declared connections with strictly positive minimum transmission delay.

模型间连接是跨模型事件的唯一通道。每条连接在声明时给出一个
*严格为正* 的最小传输延迟 ``min_delay``;任何通过该连接发送的消息,
其到达时刻都不得早于 ``source.now + min_delay``。
"""
from heapq import heappush
from itertools import count

from simpy.cosim.exceptions import (ConnectionDeclarationError,
                                    MinimumDelayError, PastMessageError)
from simpy.cosim.messages import Message


class Connection(object):
    """A directed, delay-bounded channel from ``source`` to ``target``.

    连接声明即协议的时间下界(lookahead):协调器据此保证,
    只要源环境尚未越过时刻 ``t``,目标环境在 ``t + min_delay``
    之前不会收到来自本连接的消息。

    :param name: globally unique name; doubles as the public tie-break
        between different connections for same-time delivery.
    :param min_delay: strictly positive minimum transmission delay.
    """

    def __init__(self, name, source, target, min_delay):
        if not isinstance(min_delay, (int, float)):
            raise ConnectionDeclarationError(
                'min_delay must be a number, got %r' % (min_delay,))
        if not min_delay > 0:
            raise ConnectionDeclarationError(
                'connection %r: min_delay must be strictly positive, '
                'got %r' % (name, min_delay))
        if source is target:
            raise ConnectionDeclarationError(
                'connection %r: source and target must differ' % (name,))
        self.name = name
        self.source = source
        self.target = target
        self.min_delay = min_delay
        self._counter = count(1)
        # Messages in transit, as (arrival, seq, Message).
        # 在途消息:堆按到达时刻组织。
        self.in_transit = []
        self._coordinator = None

    @property
    def earliest(self):
        """Arrival time of the earliest message still in transit.

        仍在途的最早消息到达时刻;无在途消息时为 ``inf``。
        """
        if not self.in_transit:
            return float('inf')
        return self.in_transit[0][0]

    def send(self, payload, delay=None, at=None):
        """Send a cross-model event. Exactly one of ``delay``/``at``.

        通过连接发送跨模型事件,这是跨模型通信的唯一合法入口。

        - ``delay`` 省略时默认使用声明的 ``min_delay``;
        - ``delay`` 不得小于 ``min_delay``(抛 :class:`MinimumDelayError`);
        - 到达时刻不得早于接收环境当前时间(抛 :class:`PastMessageError`);
        - 也可用 ``at`` 直接指定到达时刻,接受同样的下界检查。
        """
        if delay is not None and at is not None:
            raise TypeError("specify either 'delay' or 'at', not both")
        source_env = self.source.env
        now = source_env.now
        if at is None:
            delay = self.min_delay if delay is None else delay
            if not delay >= self.min_delay:
                raise MinimumDelayError(
                    'connection %r: requested delay %g < min_delay %g '
                    '(source time %g)'
                    % (self.name, delay, self.min_delay, now))
            arrival = now + delay
        else:
            arrival = at
            earliest_arrival = now + self.min_delay
            if arrival < earliest_arrival:
                # 对接收方而言这是"过去"时,单独报过去错误;
                # 否则属于违反最小延迟。
                if arrival < self.target.env.now:
                    raise PastMessageError(
                        'connection %r: arrival %g is before receiver '
                        'time %g'
                        % (self.name, arrival, self.target.env.now))
                raise MinimumDelayError(
                    'connection %r: arrival %g violates min_delay %g '
                    '(source time %g, earliest allowed %g)'
                    % (self.name, arrival, self.min_delay, now,
                       earliest_arrival))
        if arrival < self.target.env.now:
            raise PastMessageError(
                'connection %r: arrival %g is before receiver time %g'
                % (self.name, arrival, self.target.env.now))
        message = Message(payload, arrival, next(self._counter),
                          self.source, self.target, self)
        heappush(self.in_transit, (arrival, message.seq, message))
        if self._coordinator is not None:
            self._coordinator._notify_send(self)
        return message

    def __repr__(self):
        return 'Connection(%r: %s->%s, min_delay=%g)' % (
            self.name, self.source.name, self.target.name, self.min_delay)
