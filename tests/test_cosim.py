"""Tests for the conservative co-simulation layer (``simpy.cosim``).

覆盖:
- 三环联合仿真在不同推进次序下业务轨迹相同;
- 同刻消息的公开稳定排序;
- 跨环境直连、发送过去时间、违反最小延迟三类检查;
- 停止时区分"全部完成"与"协议无法继续推进(停滞)"。
"""
import pytest

from simpy.cosim import (Connection, Coordinator, Model, COMPLETED,
                         STALLED, Eligible)
from simpy.cosim.exceptions import (ConnectionDeclarationError,
                                    CrossModelScheduleError,
                                    MinimumDelayError, PastMessageError,
                                    ProtocolStalledError)


# --------------------------------------------------------------------------
# Ring federation / 带延迟的环
# --------------------------------------------------------------------------

RING_DELAYS = {('A', 'B'): 2, ('B', 'C'): 1, ('C', 'A'): 3}
TOTAL_HOPS = 6
EXPECTED_RING = [
    (0, 'A', 'token-0', 0),
    (2, 'B', 'token-0', 1),
    (3, 'C', 'token-0', 2),
    (6, 'A', 'token-0', 3),
    (8, 'B', 'token-0', 4),
    (9, 'C', 'token-0', 5),
    (12, 'A', 'token-0', 6),
]


def build_ring():
    models = {name: Model(name) for name in ('A', 'B', 'C')}
    connections = []
    by_pair = {}
    for (src, dst), delay in RING_DELAYS.items():
        conn = Connection('%s->%s' % (src, dst), models[src], models[dst],
                          min_delay=delay)
        connections.append(conn)
        by_pair[(src, dst)] = conn

    next_name = {'A': 'B', 'B': 'C', 'C': 'A'}

    def make(model):
        def pem():
            token = ({'id': 'token-0', 'hops': 0}
                     if model.name == 'A'
                     else (yield model.inbox.get()).payload)
            while True:
                if token.get('shutdown'):
                    by_pair[(model.name, next_name[model.name])].send(
                        {'shutdown': True})
                    break
                model.trace.append((model.env.now, token['id'],
                                    token['hops']))
                if token['hops'] >= TOTAL_HOPS:
                    by_pair[(model.name,
                             next_name[model.name])].send(
                        {'shutdown': True})
                    break
                by_pair[(model.name, next_name[model.name])].send(
                    {'id': token['id'],
                     'hops': token['hops'] + 1})
                token = (yield model.inbox.get()).payload
        return pem

    for model in models.values():
        model.trace = []
        model.process(make(model)())
    return models, connections


def ring_trace(models):
    rows = []
    for name in sorted(models):
        for time, token_id, hops in models[name].trace:
            rows.append((time, name, token_id, hops))
    return sorted(rows)


@pytest.mark.parametrize('policy',
                         ['round-robin', 'name', 'reverse', 'earliest'])
def test_ring_trajectory_is_order_independent(policy):
    models, connections = build_ring()
    result = Coordinator(list(models.values()), connections,
                         select=policy).run()
    assert result.status == COMPLETED
    assert ring_trace(models) == EXPECTED_RING
    assert result.times == {'A': 18, 'B': 14, 'C': 15}


def test_ring_trajectory_custom_callable_selector():
    # Always pick the eligible model with the greatest local time.
    models, connections = build_ring()
    result = Coordinator(
        list(models.values()), connections,
        select=lambda eligible: max(eligible,
                                    key=lambda e: e.model.now)).run()
    assert result.status == COMPLETED
    assert ring_trace(models) == EXPECTED_RING


def test_eligible_exposes_decision_inputs():
    models, connections = build_ring()
    seen = []

    def select(eligible):
        seen.append(sorted(e.model.name for e in eligible))
        return min(eligible, key=lambda e: e.model.name)

    result = Coordinator(list(models.values()), connections,
                         select=select).run()
    assert result.completed
    # All three ring members have local startup events at t=0.
    assert seen[0] == ['A', 'B', 'C']


# --------------------------------------------------------------------------
# Same-time stable delivery / 同刻消息公开稳定排序
# --------------------------------------------------------------------------

def test_same_time_messages_use_public_stable_order():
    src = Model('S')
    dst = Model('D')
    conn1 = Connection('a-conn', src, dst, min_delay=1)
    conn2 = Connection('b-conn', src, dst, min_delay=1)
    received = []

    def consumer():
        while True:
            message = yield dst.inbox.get()
            received.append((message.connection.name, message.payload))
            if message.payload == 4:
                break

    def driver():
        yield src.env.timeout(1)
        # Send deliberately out of order; the public key must still order
        # by (arrival, connection name, per-connection sequence).
        conn2.send(2, at=3)
        conn1.send(1, at=3)
        conn1.send(3, at=4)
        conn2.send(4, at=4)

    src.process(driver())
    dst.process(consumer())
    result = Coordinator([src, dst], [conn1, conn2]).run()
    assert result.completed
    assert received == [('a-conn', 1), ('b-conn', 2),
                        ('a-conn', 3), ('b-conn', 4)]


def test_same_connection_same_time_keeps_fifo():
    src, dst = Model('S'), Model('D')
    conn = Connection('c', src, dst, min_delay=1)
    received = []

    def consumer():
        for _ in range(2):
            received.append((yield dst.inbox.get()).payload)

    def driver():
        yield src.env.timeout(1)
        conn.send('first', at=5)
        conn.send('second', at=5)

    src.process(driver())
    dst.process(consumer())
    assert Coordinator([src, dst], [conn]).run().completed
    assert received == ['first', 'second']


# --------------------------------------------------------------------------
# Declaration checks / 连接声明检查
# --------------------------------------------------------------------------

def test_min_delay_must_be_strictly_positive():
    a, b = Model('A'), Model('B')
    pytest.raises(ConnectionDeclarationError, Connection,
                  'c', a, b, 0)
    pytest.raises(ConnectionDeclarationError, Connection,
                  'c', a, b, -1)
    pytest.raises(ConnectionDeclarationError, Connection,
                  'c', a, b, 'soon')


def test_self_connection_rejected():
    a = Model('A')
    pytest.raises(ConnectionDeclarationError, Connection,
                  'c', a, a, 1)


def test_unregistered_endpoint_rejected():
    a, b, rogue = Model('A'), Model('B'), Model('R')
    conn = Connection('c', a, b, 1)
    pytest.raises(ValueError, Coordinator, [a, b],
                  [Connection('x', a, rogue, 1)])
    pytest.raises(ValueError, Coordinator, [a, b],
                  [Connection('y', rogue, b, 1)])
    Coordinator([a, b], [conn])  # well-formed registration is fine


def test_duplicate_names_rejected():
    a, b = Model('A'), Model('B')
    pytest.raises(ValueError, Coordinator, [a, b],
                  [Connection('c', a, b, 1),
                   Connection('c', a, b, 2)])


# --------------------------------------------------------------------------
# Cross-model direct scheduling / 跨环境直连检查
# --------------------------------------------------------------------------

def test_direct_cross_environment_schedule_rejected():
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, 1)

    def cheater():
        # Reach into the other model's heap directly instead of using the
        # declared connection. 直接向 B 的环境堆里塞事件。
        b.env.timeout(0)
        yield a.env.timeout(0)

    a.process(cheater())
    with pytest.raises(CrossModelScheduleError):
        Coordinator([a, b], [conn]).run()


def test_messages_only_travel_via_connections():
    # Sanity: the legitimate path works and the other model sees payloads.
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, 1)
    got = []

    def sender():
        yield a.env.timeout(2)
        conn.send('hello')

    def receiver():
        got.append((yield b.inbox.get()).payload)

    a.process(sender())
    b.process(receiver())
    result = Coordinator([a, b], [conn]).run()
    assert result.completed
    assert got == ['hello']


# --------------------------------------------------------------------------
# Minimum-delay and past-send checks / 最小延迟与过去时间检查
# --------------------------------------------------------------------------

def test_delay_below_minimum_rejected():
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, min_delay=5)

    def sender():
        yield a.env.timeout(10)
        conn.send('x', delay=3)  # 3 < 5

    a.process(sender())
    with pytest.raises(MinimumDelayError):
        Coordinator([a, b], [conn]).run()


def test_explicit_arrival_violating_min_delay_rejected():
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, min_delay=5)

    def sender():
        yield a.env.timeout(10)
        conn.send('x', at=12)  # < 10 + 5, but not in B's past

    a.process(sender())
    with pytest.raises(MinimumDelayError):
        Coordinator([a, b], [conn]).run()


def test_send_into_receivers_past_rejected():
    # The coordinator never advances a receiver past promised arrivals, so
    # a correct federation cannot hit this; the check is the explicit
    # defence-in-depth guard of the Connection contract.
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, min_delay=1)
    # Advance the receiver's clock beyond the offered arrival time.
    def advance_b():
        yield b.env.timeout(10)
    b.process(advance_b())
    b.env.run()
    assert b.now == 10
    with pytest.raises(PastMessageError):
        conn.send('late', at=5)  # 5 is in B's past
    # 'delay' that would land in the past is rejected the same way.
    with pytest.raises(PastMessageError):
        conn.send('late', delay=1)


def test_default_delay_equals_minimum_is_allowed():
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, min_delay=4)
    seen = []

    def sender():
        yield a.env.timeout(2)
        conn.send('z')  # arrives exactly at 6 == 2 + 4

    def receiver():
        seen.append((yield b.inbox.get()).payload)
        seen[-1] = (b.env.now, seen[-1])

    a.process(sender())
    b.process(receiver())
    assert Coordinator([a, b], [conn]).run().completed
    assert seen == [(6, 'z')]


# --------------------------------------------------------------------------
# Termination: completion vs stall / 停止原因
# --------------------------------------------------------------------------

def test_stall_when_each_side_waits_for_the_other():
    x, y = Model('X'), Model('Y')

    def waiter(model):
        yield model.inbox.get()

    x.process(waiter(x))
    y.process(waiter(y))
    conns = [Connection('x2y', x, y, 1), Connection('y2x', y, x, 2)]
    result = Coordinator([x, y], conns).run()
    assert result.status == STALLED
    assert not result.completed
    assert 'cannot make further progress' in result.reason
    assert 'X' in result.reason and 'Y' in result.reason


def test_stall_strict_mode_raises():
    x, y = Model('X'), Model('Y')

    def waiter(model):
        yield model.inbox.get()

    x.process(waiter(x))
    y.process(waiter(y))
    conns = [Connection('x2y', x, y, 1), Connection('y2x', y, x, 2)]
    with pytest.raises(ProtocolStalledError):
        Coordinator([x, y], conns).run(on_stall='raise')


def test_completion_when_all_processes_end():
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, 1)

    def sender():
        yield a.env.timeout(3)
        conn.send('done')

    def receiver():
        yield b.inbox.get()

    a.process(sender())
    b.process(receiver())
    result = Coordinator([a, b], [conn]).run()
    assert result.status == COMPLETED
    assert 'all models completed' in result.reason


def test_positive_lookahead_bound_drives_safe_advance():
    # Receiver may safely consume a message whose arrival is promised by
    # source.now + min_delay, without ever being rolled back.
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, min_delay=2)
    log = []

    def sender():
        yield a.env.timeout(5)
        conn.send('m')

    def receiver():
        yield b.env.timeout(6)  # local event safe while 6 <= 0+inf (no
        log.append(b.env.now)   # incoming needed here), then await mail
        log.append((yield b.inbox.get()).payload)

    a.process(sender())
    b.process(receiver())
    result = Coordinator([a, b], [conn]).run()
    assert result.completed
    assert log == [6, 'm']


def test_unknown_policy_rejected():
    a, b = Model('A'), Model('B')
    conn = Connection('c', a, b, 1)
    with pytest.raises(ValueError):
        Coordinator([a, b], [conn], select='bogus')


# --------------------------------------------------------------------------
# No shared heap and no rollback / 不合并堆、不回退
# --------------------------------------------------------------------------

def test_environment_heaps_are_never_merged_or_rolled_back():
    models, connections = build_ring()
    heaps = {name: m.env._queue for name, m in models.items()}
    # Distinct heap identities throughout, and monotonically non-decreasing
    # local clocks. 各环境拥有独立堆,且本地时间单调不回退。
    assert len({id(h) for h in heaps.values()}) == 3
    last = {name: 0 for name in models}

    def select_and_watch(eligible):
        choice = min(eligible, key=lambda e: e.model.name)
        return choice

    coordinator = Coordinator(list(models.values()), connections,
                              select=select_and_watch)
    # Wrap advance to assert monotonicity after every single decision.
    original = coordinator._advance

    def watched(elig):
        original(elig)
        model = elig.model
        assert model.now >= last[model.name]
        last[model.name] = model.now
        assert model.env._queue is heaps[model.name]
    coordinator._advance = watched
    result = coordinator.run()
    assert result.completed
    assert {id(m.env._queue) for m in models.values()} == \
        {id(h) for h in heaps.values()}
