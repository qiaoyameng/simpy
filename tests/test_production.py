import pytest

import simpy
from simpy.production import (
    OwnershipLedger,
    Part,
    WorkstationFailure,
    WorkstationInterrupted,
    ZeroCapacityHandoff,
)


def test_zero_capacity_handoff_is_rendezvous():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    handoff = ZeroCapacityHandoff(env, ledger)
    part = ledger.register(Part('a'), 'sender')
    log = []

    def sender():
        log.append((env.now, 'offered', ledger.owner_of(part)))
        yield handoff.put(part, 'sender')
        log.append((env.now, 'accepted', ledger.owner_of(part)))

    def receiver():
        yield env.timeout(3)
        received = yield handoff.get('receiver')
        log.append((env.now, 'received', received.ident))

    env.process(sender())
    env.process(receiver())
    env.run()

    assert log == [
        (0, 'offered', 'sender'),
        (3, 'accepted', 'receiver'),
        (3, 'received', 'a'),
    ]
    assert handoff.capacity == 0
    assert ledger.owner_of(part) == 'receiver'


def test_workstation_blocks_until_downstream_receives():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    inbound = ZeroCapacityHandoff(env, ledger)
    outbound = ZeroCapacityHandoff(env, ledger)
    station = WorkstationWithState(env, ledger, inbound, outbound)

    def offer_registered():
        yield inbound.put(station.part, 'source')

    env.process(offer_registered())
    env.run(until=5)

    assert station.station.state == 'handing_off'
    assert station.station.processing_time == 2
    assert station.station.elapsed_in_state() == 3
    assert station.station.blocking_time == 0
    assert ledger.owner_of(station.part) == 'M'


def test_pending_put_cancellation_keeps_part_with_holder():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    handoff = ZeroCapacityHandoff(env, ledger)
    part = ledger.register(Part('a'), 'holder')
    event = handoff.put(part, 'holder')

    handoff.cancel_put(event)
    env.step()

    cancelled = event.value
    assert ledger.owner_of(part) == 'holder'
    assert cancelled.part is part
    assert cancelled.holder == 'holder'
    assert not handoff._puts


def test_pending_get_cancellation_means_no_part_was_held():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    handoff = ZeroCapacityHandoff(env, ledger)
    event = handoff.get('waiting-station')

    handoff.cancel_get(event)
    env.step()

    cancelled = event.value
    assert cancelled.part is None
    assert cancelled.holder is None
    assert not handoff._gets


def test_committed_handoff_cannot_be_undone():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    handoff = ZeroCapacityHandoff(env, ledger)
    part = ledger.register(Part('a'), 'sender')
    put = handoff.put(part, 'sender')
    get = handoff.get('receiver')

    env.run(until=1)

    assert ledger.owner_of(part) == 'receiver'
    with pytest.raises(simpy.HandoffCommitted):
        handoff.cancel_put(put)
    with pytest.raises(simpy.HandoffCommitted):
        handoff.cancel_get(get)


def test_processing_failure_reports_current_holder():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    inbound = ZeroCapacityHandoff(env, ledger)
    outbound = ZeroCapacityHandoff(env, ledger)
    part = Part('a')

    station = simpy.Workstation(
        env, 'M', inbound, outbound, broken_duration, ledger)
    env.process(offer_once(env, ledger, inbound, part))

    with pytest.raises(WorkstationFailure) as exc_info:
        env.run()

    assert station.state == 'failed'
    assert station.part is part
    assert exc_info.value.part is part
    assert exc_info.value.holder == 'M'
    assert ledger.owner_of(part) == 'M'


def test_processing_interruption_reports_current_holder():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    inbound = ZeroCapacityHandoff(env, ledger)
    outbound = ZeroCapacityHandoff(env, ledger)
    station = simpy.Workstation(env, 'M', inbound, outbound, 10, ledger)
    env.process(offer_once(env, ledger, inbound))
    env.process(interrupt_at(env, station.process, 1))

    with pytest.raises(WorkstationInterrupted) as exc_info:
        env.run()

    assert ledger.owner_of(exc_info.value.part) == 'M'
    assert exc_info.value.holder == 'M'


def test_interrupt_after_committed_delivery_keeps_downstream_owner():
    env = simpy.Environment()
    ledger = OwnershipLedger()
    inbound = ZeroCapacityHandoff(env, ledger)
    outbound = ZeroCapacityHandoff(env, ledger)
    part = Part('a')
    ledger.register(part, 'source')
    station = simpy.Workstation(
        env, 'M', inbound, outbound, 1, ledger)

    def script():
        yield inbound.put(part, 'source')
        yield env.timeout(1)
        yield env.timeout(0)
        receive = outbound.get('downstream')
        station.process.interrupt('too late to cancel')
        received_part = yield receive
        assert ledger.owner_of(received_part) == 'downstream'

    env.process(script())

    with pytest.raises(WorkstationInterrupted) as exc_info:
        env.run()

    assert ledger.owner_of(part) == 'downstream'
    assert exc_info.value.holder == 'downstream'
    assert station.part is None
    assert not outbound._puts
    assert not outbound._gets


def test_two_stage_line_matches_hand_calculation():
    from docs.examples.code.blocking_workstation import build_line

    env = simpy.Environment()
    ledger, upstream, downstream, departures = build_line(env)
    env.run()

    assert departures == [(1, 4), (2, 7), (3, 10), (4, 13), (5, 16)]
    assert upstream.processing_time == 5
    assert upstream.blocking_time == 8
    assert downstream.processing_time == 15
    assert downstream.blocking_time == 0

    locations = ledger.locations()
    assert set(locations) == {'sink'}
    assert [part.ident for part in locations['sink']] == [1, 2, 3, 4, 5]
    ledger.assert_unique_locations()
    assert not upstream.inbound._puts
    assert not upstream.outbound._puts
    assert not downstream.outbound._puts
    assert downstream.outbound._gets


class WorkstationWithState(object):
    def __init__(self, env, ledger, inbound, outbound):
        self.part = Part('a')
        ledger.register(self.part, 'source')
        self.station = simpy.Workstation(
            env, 'M', inbound, outbound, 2, ledger)


def offer_once(env, ledger, handoff, part=None):
    part = part or Part('a')
    ledger.register(part, 'source')
    yield handoff.put(part, 'source')


def broken_duration(part):
    raise RuntimeError('broken process')


def interrupt_at(env, process, at):
    yield env.timeout(at)
    process.interrupt('stop')
