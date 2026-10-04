#!/usr/bin/env python
"""Two-stage line: fast upstream blocking on a zero-capacity handover."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import simpy
from simpy.workstation import HandoverPoint, Part, Workstation, audit_locations


def main():
    env = simpy.Environment()
    source = HandoverPoint(env, capacity=5, name='source')
    handover = HandoverPoint(env, capacity=0, name='M1->M2')
    sink = HandoverPoint(env, capacity=5, name='sink')

    parts = [Part(i + 1) for i in range(5)]
    for part in parts:
        source.deliver(part, 'feeder')

    first = Workstation(
        env, process_time=2, input_point=source, output_point=handover,
        name='M1(pt=2)')
    second = Workstation(
        env, process_time=5, input_point=handover, output_point=sink,
        name='M2(pt=5)')

    env.run()

    print('M1 handover times:', [
        record.departed for record in first.records])
    print('M2 line-exit times:', [
        record.departed for record in second.records])
    print('M1 processing=%d, blocked=%d' % (
        first.processing_time, first.blocked_time))
    print('M2 processing=%d, blocked=%d, input-wait=%d' % (
        second.processing_time, second.blocked_time,
        second.waiting_time))

    locations = audit_locations(
        [source, handover, sink], [first, second])
    print('final locations:', {
        part.id: owner.name for part, owner in locations.items()})


if __name__ == '__main__':
    main()
