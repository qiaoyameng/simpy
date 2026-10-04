"""Messages exchanged between independent simulation environments."""


class Message(object):
    """A single cross-model event in transit on a connection.

    ``payload`` 是业务数据;``arrival`` 是承诺的到达时刻;
    ``seq`` 由连接分配,在同一条连接上严格递增。

    Same-time messages are ordered with the *public, stable* key
    ``(arrival, connection_name, seq)`` (see :meth:`order_key`).
    同刻消息按 (到达时刻, 连接名, 连接内序号) 公开稳定排序。
    """

    __slots__ = ('payload', 'arrival', 'seq', 'sender', 'receiver',
                 'connection')

    def __init__(self, payload, arrival, seq, sender, receiver, connection):
        self.payload = payload
        self.arrival = arrival
        self.seq = seq
        self.sender = sender
        self.receiver = receiver
        self.connection = connection

    @property
    def order_key(self):
        """Stable sort key for same-time delivery.

        先按到达时刻,再按连接名(全局确定),最后按连接内 FIFO 序号。
        """
        return (self.arrival, self.connection.name, self.seq)

    def __lt__(self, other):
        if not isinstance(other, Message):
            return NotImplemented
        return self.order_key < other.order_key

    def __repr__(self):
        return ('Message(arrival=%r, conn=%r, seq=%d, payload=%r)'
                % (self.arrival, self.connection.name, self.seq,
                   self.payload))
