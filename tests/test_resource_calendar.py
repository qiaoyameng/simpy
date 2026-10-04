"""Tests for capacity calendars (``simpy.resources.calendar``)."""
import pytest

import simpy
from simpy.resources.calendar import (
    CalendarEntry, CalendarPreemption, CalendarResource, CapacityCalendar)


#
# Calendar validation
#


def test_calendar_rejects_malformed_entries():
    pytest.raises(ValueError, CapacityCalendar, [(0, 4)])
    pytest.raises(ValueError, CapacityCalendar, [(0, 4, 1, 2)])
    pytest.raises(ValueError, CapacityCalendar, ['bogus'])


def test_calendar_rejects_invalid_intervals():
    # Empty or negative-span intervals
    pytest.raises(ValueError, CapacityCalendar, [(5, 5, 1)])
    pytest.raises(ValueError, CapacityCalendar, [(5, 4, 1)])
    # Entries must not start before simulation time 0
    pytest.raises(ValueError, CapacityCalendar, [(-1, 4, 1)])
    # Non-numeric times
    pytest.raises(ValueError, CapacityCalendar, [(None, 4, 1)])
    pytest.raises(ValueError, CapacityCalendar, [(0, '4', 1)])


def test_calendar_rejects_invalid_capacities():
    pytest.raises(ValueError, CapacityCalendar, [(0, 4, 0)])
    pytest.raises(ValueError, CapacityCalendar, [(0, 4, -2)])
    pytest.raises(ValueError, CapacityCalendar, [(0, 4, 1.5)])
    pytest.raises(ValueError, CapacityCalendar, [(0, 4, True)])
    pytest.raises(ValueError, CapacityCalendar, [(0, 4, '2')])


def test_calendar_rejects_overlapping_entries():
    pytest.raises(ValueError, CapacityCalendar, [(0, 8, 2), (4, 12, 1)])
    pytest.raises(ValueError, CapacityCalendar,
                  [(0, 8, 2), (8, 12, 1), (10, 14, 1)])


def test_calendar_allows_adjacent_entries():
    """Adjacent half-open intervals ([a, b), [b, c)) must not be reported
    as overlapping."""
    calendar = CapacityCalendar([(8, 16, 1), (0, 8, 2), (16, 24, 3)])
    # Entries are sorted by start time.
    assert list(calendar) == [CalendarEntry(0, 8, 2),
                              CalendarEntry(8, 16, 1),
                              CalendarEntry(16, 24, 3)]
    assert calendar.boundaries == (0, 8, 16, 24)


def test_calendar_capacity_at_half_open_intervals():
    calendar = CapacityCalendar([(8, 16, 3)])
    assert calendar.capacity_at(0) is None
    assert calendar.capacity_at(7.999) is None
    assert calendar.capacity_at(8) == 3  # start is inclusive
    assert calendar.capacity_at(15.5) == 3
    assert calendar.capacity_at(16) is None  # end is exclusive
    assert calendar.entry_at(10) == CalendarEntry(8, 16, 3)


#
# CalendarResource basics
#


def test_resource_rejects_invalid_strategy(env):
    pytest.raises(ValueError, CalendarResource, env, 2, [], 'bogus')


def test_resource_rejects_invalid_default_capacity(env):
    pytest.raises(ValueError, CalendarResource, env, 0, [])


def test_resource_attributes(env):
    calendar = CapacityCalendar([(8, 16, 1)])
    resource = simpy.CalendarResource(env, 2, calendar)
    assert resource.calendar is calendar
    assert resource.strategy == CalendarResource.DRAIN  # default
    assert resource.default_capacity == 2
    assert resource.capacity == 2


def test_initial_capacity_from_calendar(env):
    """An entry covering the construction time applies immediately."""
    resource = CalendarResource(env, 2, [(0, 8, 1)])
    assert resource.capacity == 1


def test_capacity_follows_calendar(env, log):
    """The effective capacity follows the half-open calendar entries and
    reverts to the default capacity outside of them."""
    resource = CalendarResource(env, 1, [(8, 16, 3)])

    def pem(env, resource, log):
        yield env.timeout(8)
        log.append(('t8', resource.capacity))
        yield env.timeout(8)
        log.append(('t16', resource.capacity))
        yield env.timeout(1)
        log.append(('t17', resource.capacity))

    env.process(pem(env, resource, log))
    env.run()

    # The shift events are processed before the process' timeouts that
    # expire at the same simulation time.
    assert log == [('t8', 3), ('t16', 1), ('t17', 1)]


#
# Graceful shrink (DRAIN)
#


def test_drain_allows_deficit_but_blocks_new_requests(env, log):
    """With the DRAIN strategy, running tasks finish before the capacity
    shrinks. A capacity deficit may exist meanwhile, but no new requests
    are granted."""
    resource = CalendarResource(env, 2, [(10, 20, 1)],
                                strategy=CalendarResource.DRAIN)

    def worker(env, name, resource, duration, log):
        with resource.request() as request:
            yield request
            log.append((name, 'start', env.now))
            yield env.timeout(duration)
            log.append((name, 'done', env.now))

    def latecomer(env, resource, log):
        yield env.timeout(10)
        env.process(worker(env, 'c', resource, 1, log))

    def monitor(env, resource):
        yield env.timeout(12)
        # Deficit while draining: 2 users but a capacity of only 1.
        assert resource.capacity == 1
        assert resource.count == 2

    env.process(worker(env, 'a', resource, 15, log))  # holds [0, 15]
    env.process(worker(env, 'b', resource, 25, log))  # holds [0, 25]
    env.process(latecomer(env, resource, log))
    env.process(monitor(env, resource))
    env.run()

    assert log == [
        ('a', 'start', 0),
        ('b', 'start', 0),
        ('a', 'done', 15),
        # c is not granted while the resource is drained (not even when a
        # releases at t=15, because the capacity is still exceeded). It is
        # granted once the capacity rises again at t=20.
        ('c', 'start', 20),
        ('c', 'done', 21),
        ('b', 'done', 25),
    ]


def test_drain_does_not_interrupt_users(env, log):
    resource = CalendarResource(env, 2, [(10, 20, 1)],
                                strategy=CalendarResource.DRAIN)

    def worker(env, name, resource, log):
        request = resource.request()
        yield request
        try:
            yield env.timeout(30)
            log.append((name, 'done', env.now))
        except simpy.Interrupt:
            log.append((name, 'interrupted', env.now))
        resource.release(request)

    env.process(worker(env, 'a', resource, log))
    env.process(worker(env, 'b', resource, log))
    env.run()

    assert log == [('a', 'done', 30), ('b', 'done', 30)]


#
# Immediate shrink (PREEMPT)
#


def test_preempt_selects_victims_by_priority_and_recency(env, log):
    """Victims of an immediate shrink are the users with the lowest
    priority (largest priority value); ties are broken by latest usage
    time (newest user first)."""
    resource = CalendarResource(env, 3, [(10, 30, 1)],
                                strategy=CalendarResource.PREEMPT)

    def worker(env, name, priority, delay, resource, log):
        yield env.timeout(delay)
        request = resource.request(priority=priority)
        yield request
        log.append((name, 'start', env.now))
        try:
            yield env.timeout(100)
            resource.release(request)
        except simpy.Interrupt as interrupt:
            log.append((name, 'preempted', env.now, interrupt.cause))

    env.process(worker(env, 'a', 0, 0, resource, log))
    env.process(worker(env, 'b', 1, 1, resource, log))
    env.process(worker(env, 'c', 1, 2, resource, log))

    def monitor(env, resource):
        yield env.timeout(11)
        assert resource.count == 1  # only a survives the shrink

    env.process(monitor(env, resource))
    env.run()

    starts = [entry for entry in log if entry[1] == 'start']
    assert starts == [('a', 'start', 0), ('b', 'start', 1),
                      ('c', 'start', 2)]

    preempted = [entry for entry in log if entry[1] == 'preempted']
    # Capacity 3 -> 1 preempts 2 users: c and b (both priority 1, newest
    # first). a (priority 0) survives.
    assert [(entry[0], entry[2]) for entry in preempted] == [('c', 10),
                                                             ('b', 10)]


def test_preemption_cause_names_calendar_reason(env, log):
    """The interrupt cause of a preemption names the calendar entry and
    the new capacity that caused it."""
    resource = CalendarResource(env, 2, [(10, 30, 1)],
                                strategy=CalendarResource.PREEMPT)

    def worker(env, name, delay, resource, log):
        yield env.timeout(delay)
        request = resource.request()
        yield request
        try:
            yield env.timeout(100)
            resource.release(request)
        except simpy.Interrupt as interrupt:
            log.append(interrupt.cause)

    env.process(worker(env, 'a', 0, resource, log))
    env.process(worker(env, 'b', 1, resource, log))
    env.run()

    assert len(log) == 1
    cause = log[0]
    assert isinstance(cause, CalendarPreemption)
    assert cause.resource is resource
    assert cause.capacity == 1
    assert cause.entry == CalendarEntry(10, 30, 1)
    assert cause.usage_since == 1  # b started using the resource at t=1
    assert cause.by is None  # not preempted by another process


def test_preempt_not_needed_when_capacity_suffices(env, log):
    resource = CalendarResource(env, 2, [(10, 30, 1)],
                                strategy=CalendarResource.PREEMPT)

    def worker(env, name, resource, log):
        request = resource.request()
        yield request
        try:
            yield env.timeout(5)
            log.append((name, 'done', env.now))
        except simpy.Interrupt:
            log.append((name, 'interrupted', env.now))
        resource.release(request)

    env.process(worker(env, 'a', resource, log))
    env.run()

    assert log == [('a', 'done', 5)]


#
# Capacity increases
#


def test_capacity_increase_wakes_only_valid_waiters(env, log):
    """When the capacity increases, only requests that are still waiting
    are woken up; cancelled requests are not triggered."""
    resource = CalendarResource(env, 1, [(10, 20, 3)])
    cancelled = []

    def holder(env, resource):
        with resource.request() as request:
            yield request
            yield env.timeout(30)

    def waiter(env, name, resource, log):
        request = resource.request()
        log.append((name, 'wait', env.now))
        yield request
        log.append((name, 'start', env.now))

    def canceller(env, resource, cancelled):
        request = resource.request()
        yield env.timeout(5)
        request.cancel()
        cancelled.append(request)
        log.append(('cancelled', env.now))

    env.process(holder(env, resource))
    env.process(waiter(env, 'w1', resource, log))
    env.process(waiter(env, 'w2', resource, log))
    env.process(canceller(env, resource, cancelled))
    env.run()

    # Both still-valid waiters are granted at t=10 (capacity 1 -> 3);
    # the cancelled request is never triggered.
    assert log == [('w1', 'wait', 0), ('w2', 'wait', 0), ('cancelled', 5),
                   ('w1', 'start', 10), ('w2', 'start', 10)]
    assert len(cancelled) == 1
    assert not cancelled[0].triggered


#
# Same-time ordering rules
#


def test_same_time_ordering(env, log):
    """If a shift, a task completion and a new request fall onto the same
    simulation time, the shift is applied first, then the completion,
    then the new request is evaluated against the new capacity."""
    resource = CalendarResource(env, 2, [(10, 20, 1)],
                                strategy=CalendarResource.PREEMPT)

    def worker(env, name, priority, resource, work, log):
        request = resource.request(priority=priority)
        yield request
        log.append((name, 'start', env.now))
        try:
            yield env.timeout(work)
            log.append((name, 'done', env.now))
            resource.release(request)
        except simpy.Interrupt as interrupt:
            log.append((name, 'preempted', env.now,
                        interrupt.cause.capacity))

    def latecomer(env, resource, log):
        yield env.timeout(10)
        request = resource.request()
        yield request
        log.append(('late', 'start', env.now))

    env.process(worker(env, 'a', 0, resource, 10, log))  # done at shift
    env.process(worker(env, 'b', 1, resource, 50, log))
    env.process(latecomer(env, resource, log))
    env.run()

    assert log == [
        ('a', 'start', 0),
        ('b', 'start', 0),
        # 1. The shift at t=10 preempts b (lowest priority) before a's
        #    completion is processed.
        ('b', 'preempted', 10, 1),
        # 2. a's task completion at t=10 is processed afterwards.
        ('a', 'done', 10),
        # 3. The new request at t=10 is evaluated against the new
        #    capacity and the slot freed by a.
        ('late', 'start', 10),
    ]
