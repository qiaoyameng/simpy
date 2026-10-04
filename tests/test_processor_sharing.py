"""
Tests for the weighted processor-sharing resource.

All expected values are computed by hand; the tests only advance the
simulation clock (no wall-clock sleeps).

"""
import pytest

import simpy


def test_three_load_phases(env, log):
    """Three phases of load with hand-computable completion times.

    Capacity is 1. At t=0 tasks A and B (work=6, weight=1 each) share the
    capacity equally (rate 1/2 each). At t=4 task C (work=6, weight=1)
    joins; A and B have 4 units of work left each.

    - Phase [0, 4):  A, B served at 1/2 -> 2 units each.
    - Phase [4, 16): A, B, C served at 1/3 -> A and B finish at t=16,
      C receives 4 units.
    - Phase [16, 18): C alone at rate 1 -> finishes at t=18.

    A and B complete at the same instant and must be settled together.

    """
    def pem(env, name, work, resource, log, delay=0):
        yield env.timeout(delay)
        req = resource.request(work=work, weight=1)
        yield req
        log.append((name, env.now))

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    env.process(pem(env, 'a', 6, resource, log))
    env.process(pem(env, 'b', 6, resource, log))
    env.process(pem(env, 'c', 6, resource, log, delay=4))
    env.run()

    assert log == [('a', 16), ('b', 16), ('c', 18)]
    assert resource.count == 0


def test_simultaneous_completion_settled_together(env, log):
    """Identical requests started together finish at the same instant."""
    def pem(env, name, resource, log):
        yield resource.request(work=2, weight=1)
        log.append((name, env.now))

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    for name in 'abcd':
        env.process(pem(env, name, resource, log))
    env.run()

    # Each request is served at rate 1/4 and needs 2 units -> t=8.
    assert log == [('a', 8), ('b', 8), ('c', 8), ('d', 8)]


def test_weighted_sharing(env, log):
    """Capacity is split proportionally to the weights.

    Capacity 1, A (work=1, weight=1) and B (work=3, weight=3) start at
    t=0. A gets rate 1/4, B gets 3/4, so both finish at t=4.

    """
    def pem(env, name, work, weight, resource, log):
        yield resource.request(work=work, weight=weight)
        log.append((name, env.now))

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    env.process(pem(env, 'a', 1, 1, resource, log))
    env.process(pem(env, 'b', 3, 3, resource, log))
    env.run()

    assert log == [('a', 4), ('b', 4)]


def test_cancel_redistributes_capacity(env, log):
    """A cancelled request frees its share for the survivors.

    Capacity 2. A (work=8, weight=1) and B (work=100, weight=3) start at
    t=0, so A is served at rate 0.5 and B at rate 1.5. At t=6 A cancels
    after receiving 3 units of service (5 remaining). B then gets the full
    capacity of 2 and finishes its remaining 91 units at t=6 + 45.5 = 51.5.

    """
    def pem_a(env, resource, log):
        req = resource.request(work=8, weight=1)
        yield env.timeout(6)
        req.cancel()
        log.append(('a-cancelled', env.now, req.served, req.remaining))

    def pem_b(env, resource, log):
        req = resource.request(work=100, weight=3)
        yield req
        log.append(('b', env.now, req.served, req.remaining))

    resource = simpy.ProcessorSharingResource(env, capacity=2)
    env.process(pem_a(env, resource, log))
    env.process(pem_b(env, resource, log))
    env.run()

    assert log == [
        ('a-cancelled', 6, 3, 5),
        ('b', 51.5, 100, 0),
    ]
    assert resource.count == 0


def test_cancel_settles_service_of_survivors(env):
    """Cancelling first accounts for the service of the surviving tasks."""
    resource = simpy.ProcessorSharingResource(env, capacity=1)
    req_a = resource.request(work=10, weight=1)
    req_b = resource.request(work=10, weight=1)

    env.run(until=4)
    # Both served at rate 1/2 for 4 time units. Cancelling A must first
    # settle B's account: B has received 2 units.
    req_a.cancel()
    assert req_b.served == 2
    assert req_b.remaining == 8

    # B now gets the full capacity and finishes at t=4 + 8 = 12.
    env.run(until=req_b)
    assert env.now == 12


def test_stale_completion_event_does_not_trigger(env, log):
    """A completion event scheduled before a state change must not fire.

    A (work=4) and B (work=4) start at t=0 with capacity 1; a completion
    event for both is scheduled at t=8. At t=2, C (work=4) joins, pushing
    the completions of A and B to t=11. The stale event at t=8 must not
    trigger any request.

    """
    def pem(env, name, resource, log, delay=0):
        yield env.timeout(delay)
        req = resource.request(work=4, weight=1)
        yield req
        log.append((name, env.now))

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    env.process(pem(env, 'a', resource, log))
    env.process(pem(env, 'b', resource, log))
    env.process(pem(env, 'c', resource, log, delay=2))

    env.run(until=9)
    # The stale completion event fired at t=8; nothing may have completed.
    assert log == []
    assert resource.count == 3

    env.run()
    # [0, 2): a, b at rate 1/2 -> 1 unit each, 3 left. [2, 11): all three
    # at rate 1/3 -> a and b finish at t=11, c has 4 - 3 = 1 unit left.
    # c alone at rate 1 finishes at t=12.
    assert log == [('a', 11), ('b', 11), ('c', 12)]


def test_capacity_change_pause_and_resume(env, log):
    """Capacity changes at scheduled times, including a pause at zero.

    Capacity 1. A (work=6, weight=1) and B (work=6, weight=2) start at
    t=0 with rates 1/3 and 2/3. At t=6 the capacity drops to 0 (A has 4
    units left, B has 2). At t=10 the capacity is restored at 3, so A gets
    rate 1 and B gets rate 2. B finishes at t=11; A then has 3 units left
    and the full capacity of 3, finishing at t=12.

    """
    def pem(env, name, work, weight, resource, log):
        yield resource.request(work=work, weight=weight)
        log.append((name, env.now))

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    env.process(pem(env, 'a', 6, 1, resource, log))
    env.process(pem(env, 'b', 6, 2, resource, log))
    resource.set_capacity(0, at=6)
    resource.set_capacity(3, at=10)

    env.run(until=9)
    # The resource is paused; nothing completes while the capacity is 0.
    assert log == []
    assert resource.capacity == 0

    env.run()
    assert log == [('b', 11), ('a', 12)]


def test_set_capacity_now(env, log):
    """Capacity changes apply immediately when no time is given."""
    def pem(env, resource, log):
        yield resource.request(work=4, weight=1)
        log.append(env.now)

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    env.process(pem(env, resource, log))

    env.run(until=2)
    resource.set_capacity(3)
    assert resource.capacity == 3

    env.run()
    # 2 units served at rate 1, the remaining 2 at rate 3: t=2 + 2/3.
    assert log == [pytest.approx(2 + 2.0 / 3.0)]


def test_zero_work_request_completes_immediately(env, log):
    def pem(env, resource, log):
        yield resource.request(work=0, weight=1)
        log.append(env.now)

    resource = simpy.ProcessorSharingResource(env, capacity=1)
    env.process(pem(env, resource, log))
    env.run()

    assert log == [0]
    assert resource.count == 0


def test_invalid_arguments(env):
    resource = simpy.ProcessorSharingResource(env, capacity=1)
    pytest.raises(ValueError, simpy.ProcessorSharingResource, env, -1)
    pytest.raises(ValueError, resource.request, work=-1, weight=1)
    pytest.raises(ValueError, resource.request, work=1, weight=0)
    pytest.raises(ValueError, resource.request, work=1, weight=-2)
    pytest.raises(ValueError, resource.set_capacity, -1)
    pytest.raises(ValueError, resource.set_capacity, 2, at=-1)
