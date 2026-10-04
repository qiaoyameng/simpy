"""
Blocking workstation example.

Two one-slot workstations are connected by zero-capacity handoff points.
The faster upstream machine keeps each finished part until the slower
downstream machine receives it.
"""
import simpy
from simpy.production import OwnershipLedger, Part, Workstation


PART_COUNT = 5
UPSTREAM_TIME = 1
DOWNSTREAM_TIME = 3


def source(env, count, handoff, ledger):
    for number in range(1, count + 1):
        part = Part(number)
        ledger.register(part, 'source')
        yield handoff.put(part, 'source')


def sink(env, handoff, ledger, departures):
    while True:
        event = handoff.get('sink')
        part = yield event
        ledger.assert_owner(part, 'sink')
        departures.append((part.ident, env.now))


def build_line(env):
    ledger = OwnershipLedger()
    source_handoff = simpy.ZeroCapacityHandoff(env, ledger)
    middle_handoff = simpy.ZeroCapacityHandoff(env, ledger)

    upstream = Workstation(
        env, 'M1', source_handoff, middle_handoff, UPSTREAM_TIME, ledger)
    downstream = Workstation(
        env, 'M2', middle_handoff,
        simpy.ZeroCapacityHandoff(env, ledger),
        DOWNSTREAM_TIME, ledger)

    departures = []
    env.process(source(env, PART_COUNT, source_handoff, ledger))
    env.process(sink(env, downstream.outbound, ledger, departures))
    return ledger, upstream, downstream, departures


def main():
    env = simpy.Environment()
    ledger, upstream, downstream, departures = build_line(env)
    env.run()

    expected = [(1, 4), (2, 7), (3, 10), (4, 13), (5, 16)]
    print('departures:', departures)
    print('expected:  ', expected)
    print('M1 processing: %.1f, blocking: %.1f' % (
        upstream.processing_time, upstream.blocking_time))
    print('M2 processing: %.1f, blocking: %.1f' % (
        downstream.processing_time, downstream.blocking_time))
    print('final locations:', {
        holder: [part.ident for part in parts]
        for holder, parts in ledger.locations().items()})


if __name__ == '__main__':
    main()
