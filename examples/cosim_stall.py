"""Conservative protocol stall (a real lookahead deadlock).

两个模型 X、Y 互相等待对方的第一条消息,且没有任何本地启动事件:
安全上界要求"对方先推进",形成 CMB 死锁。协调器不回退、不越过,
而是报告协议无法继续推进。

Run with::

    python examples/cosim_stall.py
"""
from simpy.cosim import Connection, Coordinator, Model
from simpy.cosim.exceptions import ProtocolStalledError


def main():
    x, y = Model('X'), Model('Y')
    # Both processes wait forever for a first message; nobody sends one.
    def waiter(model):
        yield model.inbox.get()
    x.process(waiter(x))
    y.process(waiter(y))

    conns = [Connection('X->Y', x, y, min_delay=1),
             Connection('Y->X', y, x, min_delay=2)]
    result = Coordinator([x, y], conns).run()
    print('status:', result.status)
    print('reason:', result.reason)
    assert not result.completed

    # Strict mode raises instead.
    x, y = Model('X'), Model('Y')
    x.process(waiter(x))
    y.process(waiter(y))
    conns = [Connection('X->Y', x, y, min_delay=1),
             Connection('Y->X', y, x, min_delay=2)]
    try:
        Coordinator([x, y], conns).run(on_stall='raise')
    except ProtocolStalledError as exc:
        print('strict mode raised ProtocolStalledError:', exc)
    else:
        raise AssertionError('expected ProtocolStalledError')


if __name__ == '__main__':
    main()
