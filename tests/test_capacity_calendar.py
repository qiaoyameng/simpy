import pytest

import simpy
from simpy.resources import calendar


def test_calendar_entries_are_half_open(env):
    resource = simpy.CalendarResource(
        env, capacity=1, calendar=[(0, 1, 2), (1, 2, 0)])

    assert resource.capacity == 2

    env.run(until=1)
    while env.peek() == 1:
        env.step()
    assert resource.capacity == 0

    env.run(until=2)
    assert resource.capacity == 1


def test_gentle_shrink_allows_deficit_but_no_new_grants(env):
    def worker(start):
        yield env.timeout(start)
        req = resource.request()
        yield req
        yield env.timeout(2)
        resource.release(req)

    resource = simpy.CalendarResource(
        env, capacity=2, calendar=[(1, 3, 1)])
    env.process(worker(0))
    env.process(worker(0))
    waiter = env.process(worker(1))

    env.run(until=1)
    while env.peek() == 1:
        env.step()
    assert resource.count == 2
    assert resource.deficit == 1
    assert [event.proc for event in resource.queue] == [waiter]

    env.run(until=2)
    assert resource.count == 2
    assert resource.deficit == 1
    assert [event.proc for event in resource.queue] == [waiter]

    env.run(until=3)
    while env.peek() == 3:
        env.step()
    assert resource.count == 1
    assert resource.capacity == 2
    assert resource.queue == []


def test_immediate_shrink_preempts_priority_and_fifo_tiebreak(
        env, log):
    def worker(name, priority):
        req = resource.request(priority=priority)
        try:
            yield req
            yield env.timeout(3)
        except simpy.Interrupt as interrupt:
            cause = interrupt.cause
            assert cause.reason == 'capacity_calendar'
            assert cause.calendar is resource.calendar
            assert cause.entry.capacity == 1
            assert cause.old_capacity == 3
            assert cause.new_capacity == 1
            log.append((env.now, name, 'preempted'))

    resource = simpy.CalendarResource(
        env, capacity=3, calendar=[(1, 2, 1)],
        shrink_policy=simpy.CapacityShrinkPolicy.IMMEDIATE)
    env.process(worker('same-old', 1))
    env.process(worker('same-new', 1))
    env.process(worker('high', 0))

    env.run()
    assert log == [(1, 'same-old', 'preempted'),
                   (1, 'same-new', 'preempted')]
    assert resource.count == 1
    assert resource.users[0].priority == 0
    assert [event.type for event in resource.events[:6]] == [
        calendar.REQUEST_GRANTED_EVENT,
        calendar.REQUEST_GRANTED_EVENT,
        calendar.REQUEST_GRANTED_EVENT,
        calendar.CAPACITY_CHANGE_EVENT,
        calendar.REQUEST_PREEMPTED_EVENT,
        calendar.REQUEST_PREEMPTED_EVENT,
    ]


def test_capacity_increase_wakes_only_still_valid_waiters(env):
    def waiter():
        req = resource.request()
        yield req
        resource.release(req)

    def canceller():
        req = resource.request()
        yield env.timeout(0)
        req.cancel()

    resource = simpy.CalendarResource(
        env, capacity=0, calendar=[(0, 1, 0), (1, 2, 1)])
    valid = env.process(waiter())
    cancelled = env.process(canceller())

    env.run(until=1)
    assert valid.is_alive
    assert not cancelled.is_alive
    assert resource.count == 1
    assert resource.queue == []


def test_capacity_increase_removes_interrupted_waiter(env):
    def waiter():
        req = resource.request()
        try:
            yield req
        except simpy.Interrupt:
            return
        resource.release(req)

    def interrupt_waiter(process):
        yield env.timeout(0)
        process.interrupt()

    resource = simpy.CalendarResource(
        env, capacity=0, calendar=[(0, 1, 0), (1, 2, 1)])
    waiting = env.process(waiter())
    env.process(interrupt_waiter(waiting))

    env.run()
    assert not waiting.is_alive
    assert resource.queue == []
    assert resource.count == 0


def test_capacity_change_records_policy_and_boundaries(env):
    resource = simpy.CalendarResource(
        env, capacity=2, calendar=[(1, 2, 1), (2, 3, 3)],
        shrink_policy=simpy.CapacityShrinkPolicy.IMMEDIATE)

    env.run(until=3)

    assert [
        (event.time, event.old_capacity, event.new_capacity, event.policy)
        for event in resource.calendar_events
    ] == [
        (1, 2, 1, simpy.CapacityShrinkPolicy.IMMEDIATE),
        (2, 1, 3, simpy.CapacityShrinkPolicy.IMMEDIATE),
        (3, 3, 2, simpy.CapacityShrinkPolicy.IMMEDIATE),
    ]


def test_same_capacity_boundary_is_not_recorded(env):
    resource = simpy.CalendarResource(
        env, capacity=1, calendar=[(1, 2, 1)])

    env.run()

    assert resource.calendar_events == []


def test_same_time_change_precedes_completion_and_new_request(
        env, log):
    def completing_worker(req):
        yield env.timeout(1)
        resource.release(req)
        log.append('completion')

    def new_applicant():
        yield env.timeout(1)
        req = resource.request()
        yield req
        log.append('new-request')
        resource.release(req)

    resource = simpy.CalendarResource(
        env, capacity=1, calendar=[(1, 2, 0)])
    initial_req = resource.request()
    env.process(completing_worker(initial_req))
    env.process(new_applicant())

    env.run(until=1)
    while env.peek() == 1:
        env.step()
    assert log == ['completion']
    assert [event.type for event in resource.events[1:]] == [
        calendar.CAPACITY_CHANGE_EVENT,
        calendar.REQUEST_RELEASED_EVENT,
    ]
    assert resource.count == 0
    assert len(resource.queue) == 1

    env.run()
    assert log[-1] == 'new-request'
    assert resource.events[-2].type == calendar.REQUEST_GRANTED_EVENT
    assert resource.events[-1].type == calendar.REQUEST_RELEASED_EVENT


@pytest.mark.parametrize(('entries', 'message'), [
    ([[1, 1, 1]], 'start < end'),
    ([[2, 1, 1]], 'start < end'),
    ([[0, 1, -1]], '>= 0'),
    ([[0, 1, 1.5]], 'integer'),
    ([[0, 2, 1], [1, 3, 1]], 'overlap'),
])
def test_invalid_calendar_entries(entries, message):
    with pytest.raises(ValueError) as excinfo:
        simpy.CapacityCalendar(entries)
    assert message in str(excinfo.value)


def test_adjacent_boundaries_are_allowed():
    calendar = simpy.CapacityCalendar([(0, 1, 1), (1, 2, 2)])
    assert calendar.capacity_at(1) == 2


def test_invalid_shrink_policy():
    with pytest.raises(ValueError):
        simpy.CapacityCalendar([(0, 1, 1)], shrink_policy='later')


def test_calendar_resource_rejects_invalid_baseline_capacity(env):
    with pytest.raises(ValueError):
        simpy.CalendarResource(env, capacity=-1, calendar=[(0, 1, 1)])

    with pytest.raises(ValueError):
        simpy.CalendarResource(env, capacity=1.5)


def test_capacity_change_has_no_wall_clock_dependency():
    source = open('src/simpy/resources/calendar.py').read()
    assert 'time.sleep' not in source
    assert 'RealtimeEnvironment' not in source
