"""
Tests for block-after-processing workstation primitives.
"""
import pytest

import simpy
from simpy.workstation import (
    HandoverPoint, Part, Workstation, TransferCancelled, TransferCommitted,
    WorkstationHalted, audit_locations)


def test_zero_capacity_handover_is_rendezvous(env):
    point = HandoverPoint(env, capacity=0, name='rendezvous')
    part = Part('p1')

    delivery = point.deliver(part, 'sender')
    assert not delivery.event.triggered
    assert point.blocked_deliveries == 1

    receipt = point.receive('receiver')
    env.run()

    assert delivery.event.value is part
    assert receipt.event.value is part
    assert point.items == []
    assert point.deliveries == []
    assert point.receipts == []


def test_workstation_requires_one_environment():
    first_env = simpy.Environment()
    second_env = simpy.Environment()
    source = HandoverPoint(first_env, capacity=1, name='source')
    output = HandoverPoint(second_env, capacity=0, name='output')

    with pytest.raises(ValueError):
        Workstation(
            first_env, process_time=1, input_point=source,
            output_point=output, name='M1')


def test_positive_capacity_buffer_blocks_when_full(env):
    point = HandoverPoint(env, capacity=1, name='buffer')
    first = Part('p1')
    second = Part('p2')

    first_delivery = point.deliver(first, 'sender')
    second_delivery = point.deliver(second, 'sender')
    env.run()

    assert first_delivery.event.triggered
    assert not second_delivery.event.triggered
    assert point.items == [first]

    receipt = point.receive('receiver')
    env.run()

    assert receipt.event.value is first
    assert second_delivery.event.value is second


def test_two_stage_line_timing_and_unique_locations(env):
    source = HandoverPoint(env, capacity=5, name='source')
    handover = HandoverPoint(env, capacity=0, name='handover')
    sink = HandoverPoint(env, capacity=5, name='sink')
    parts = [Part(i) for i in range(5)]

    for part in parts:
        source.deliver(part, 'feeder')

    first = Workstation(
        env, process_time=2, input_point=source, output_point=handover,
        name='M1')
    second = Workstation(
        env, process_time=5, input_point=handover, output_point=sink,
        name='M2')

    env.run()

    assert [record.departed for record in first.records] == [
        2, 7, 12, 17, 22]
    assert [record.departed for record in second.records] == [
        7, 12, 17, 22, 27]
    assert first.processing_time == 10
    assert first.blocked_time == 12
    assert second.processing_time == 25
    assert second.blocked_time == 0
    assert second.waiting_time == 2
    assert sink.items == parts

    locations = audit_locations(
        [source, handover, sink], [first, second])
    assert set(locations) == set(parts)
    assert set(locations.values()) == {sink}


def test_cancelled_blocked_delivery_reports_upstream_owner(env):
    source = HandoverPoint(env, capacity=1, name='source')
    output = HandoverPoint(env, capacity=0, name='output')
    part = Part('p1')
    source.deliver(part, 'feeder')
    station = Workstation(
        env, process_time=1, input_point=source, output_point=output,
        name='M1')

    def interrupt_after_processing():
        yield env.timeout(2)
        station.process.interrupt('stop handover')

    env.process(interrupt_after_processing())

    with pytest.raises(TransferCancelled) as excinfo:
        env.run()

    assert excinfo.value.owner is station
    assert excinfo.value.part is part
    assert audit_locations([source, output], [station]) == {part: station}


def test_processing_failure_leaves_part_with_station(env):
    source = HandoverPoint(env, capacity=1, name='source')
    output = HandoverPoint(env, capacity=0, name='output')
    part = Part('p1')
    source.deliver(part, 'feeder')

    def fail(workstation, part):
        raise RuntimeError('processing failed')

    station = Workstation(
        env, process_time=1, input_point=source, output_point=output,
        name='M1', process=fail)

    with pytest.raises(WorkstationHalted) as excinfo:
        env.run()

    assert excinfo.value.owner is station
    assert excinfo.value.part is part
    assert str(excinfo.value.cause) == 'processing failed'
    assert audit_locations([source, output], [station]) == {part: station}


def test_late_delivery_cancellation_reports_downstream_owner(env):
    source = HandoverPoint(env, capacity=1, name='source')
    output = HandoverPoint(env, capacity=0, name='output')
    part = Part('p1')
    source.deliver(part, 'feeder')
    station = Workstation(
        env, process_time=1, input_point=source, output_point=output,
        name='M1')
    held = []

    def downstream():
        yield env.timeout(2)
        receipt = output.receive('downstream')
        held.append((yield receipt.event))

    def interrupt_at_handover():
        yield env.timeout(2)
        station.process.interrupt('too late')

    env.process(downstream())
    env.process(interrupt_at_handover())

    with pytest.raises(TransferCommitted) as excinfo:
        env.run()

    assert excinfo.value.owner == 'downstream'
    assert excinfo.value.part is part
    assert held == [part]
    assert station.part is None


def test_audit_detects_duplicate_part_claims():
    env = simpy.Environment()
    point = HandoverPoint(env, capacity=1, name='point')
    part = Part('p1')
    point.items.append(part)

    class DuplicateHolder(object):
        pass

    holder = DuplicateHolder()
    holder.name = 'duplicate'
    holder.part = part

    with pytest.raises(RuntimeError):
        audit_locations([point], [holder])
