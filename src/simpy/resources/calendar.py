"""
Time-dependent capacity calendars for resources.

A capacity calendar changes the number of available slots at simulation time.
Calendar intervals are half-open: an interval is active at its start time and
no longer active at its end time.
"""
from collections import namedtuple

from simpy.core import BoundClass
from simpy.events import Event
from simpy.resources import base
from simpy.resources.resource import (
    PriorityRequest, Request, Release, PriorityResource, Preempted)


CALENDAR_PRIORITY = -1
"""Scheduling priority used for capacity changes. It is earlier than SimPy's
urgent process events so a shift change is observed before completions and
requests that are scheduled for the same simulation time."""

GENTLE_SHRINK = 'gentle'
IMMEDIATE_SHRINK = 'immediate'


class CapacityShrinkPolicy(object):
    """Strategies used when a calendar interval reduces capacity."""

    GENTLE = GENTLE_SHRINK
    """Current users keep their slots. New users are not admitted while the
    resource has a capacity deficit."""

    IMMEDIATE = IMMEDIATE_SHRINK
    """Enough users are preempted immediately to reduce usage to the new
    capacity."""


CapacityCalendarEntry = namedtuple(
    'CapacityCalendarEntry', ('start', 'end', 'capacity'))
"""A half-open capacity interval ``[start, end)``."""

CapacityChange = namedtuple(
    'CapacityChange',
    ('time', 'old_capacity', 'new_capacity', 'policy', 'entry'))
"""Record of a capacity change that was applied at a simulation time."""

ResourceEvent = namedtuple(
    'ResourceEvent',
    ('time', 'type', 'capacity', 'count', 'request', 'detail'))
"""Resource-level event used to inspect calendar, grant, and release order."""

CAPACITY_CHANGE_EVENT = 'capacity_change'
REQUEST_GRANTED_EVENT = 'request_granted'
REQUEST_RELEASED_EVENT = 'request_released'
REQUEST_PREEMPTED_EVENT = 'request_preempted'


class CalendarPreempted(Preempted):
    """Cause of an interrupt caused by an immediate capacity reduction."""

    def __init__(self, resource, usage_since, change):
        super(CalendarPreempted, self).__init__(
            by=None, usage_since=usage_since, resource=resource)
        self.reason = 'capacity_calendar'
        """Reason identifying a capacity-calendar preemption."""
        self.calendar = resource.calendar
        """The capacity calendar that caused the preemption."""
        self.change = change
        """The :class:`CapacityChange` that caused the preemption."""
        self.entry = change.entry
        """The calendar entry that introduced the reduced capacity."""
        self.time = change.time
        """Simulation time at which the capacity change happened."""
        self.old_capacity = change.old_capacity
        self.new_capacity = change.new_capacity


class CapacityCalendar(object):
    """A validated, immutable sequence of half-open capacity intervals.

    *entries* are :class:`CapacityCalendarEntry` objects or ``(start, end,
    capacity)`` tuples. Intervals may touch at their boundaries but must not
    overlap. Times not covered by an interval use the resource's baseline
    capacity.
    """

    def __init__(self, entries=(),
                 shrink_policy=CapacityShrinkPolicy.GENTLE):
        entries = [self._coerce_entry(entry) for entry in entries]
        entries.sort(key=lambda entry: (entry.start, entry.end))

        for index, entry in enumerate(entries):
            self._validate_entry(entry)
            if index and entry.start < entries[index - 1].end:
                raise ValueError('Capacity calendar intervals must not '
                                 'overlap.')

        if shrink_policy not in (GENTLE_SHRINK, IMMEDIATE_SHRINK):
            raise ValueError('Invalid capacity shrink policy: %r'
                             % shrink_policy)

        self.entries = tuple(entries)
        self.shrink_policy = shrink_policy

    @staticmethod
    def _coerce_entry(entry):
        if isinstance(entry, CapacityCalendarEntry):
            return entry
        return CapacityCalendarEntry(*entry)

    @staticmethod
    def _validate_entry(entry):
        start, end, capacity = entry
        if not start < end:
            raise ValueError('Capacity calendar interval %r must use a valid '
                             'half-open range with start < end.' % (entry,))
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError('Capacity calendar capacity must be an integer: '
                             '%r' % (capacity,))
        if capacity < 0:
            raise ValueError('Capacity calendar capacity must be >= 0: %r'
                             % (capacity,))

    def capacity_at(self, time):
        """Return the calendar capacity active at *time*, or ``None`` when no
        interval covers that time."""
        for entry in self.entries:
            if entry.start <= time < entry.end:
                return entry.capacity
        return None

    def transitions_after(self, time):
        """Return ``(time, new_capacity, entry)`` transitions after *time*.

        A transition back to the baseline capacity has ``entry`` set to
        ``None``. Adjacent intervals are represented by one transition.
        """
        starts = {}
        ends = set()
        for entry in self.entries:
            if entry.start > time:
                starts[entry.start] = (entry.capacity, entry)
            if entry.end > time:
                ends.add(entry.end)

        points = set(starts)
        for point in ends:
            if point not in starts:
                points.add(point)

        transitions = []
        for point in sorted(points):
            if point in starts:
                transitions.append((point, starts[point][0], starts[point][1]))
            else:
                transitions.append((point, None, None))
        return tuple(transitions)


class CalendarRequest(PriorityRequest):
    """A priority request with an explicit tie-breaking sequence number."""

    def __init__(self, resource, priority=0, preempt=True):
        self.priority = priority
        self.preempt = preempt
        self.time = resource._env.now
        self.usage_since = None
        self.sequence = resource._next_request_sequence
        resource._next_request_sequence += 1
        self.key = (self.priority, self.time, self.sequence)
        Request.__init__(self, resource)


class CalendarRequestQueue(list):
    """Queue ordered by priority, request time, then request sequence."""

    def append(self, item):
        super(CalendarRequestQueue, self).append(item)
        super(CalendarRequestQueue, self).sort(
            key=lambda event: (event.priority, event.time, event.sequence))


class CalendarResource(PriorityResource):
    """A priority resource whose capacity follows a simulation-time calendar.

    *capacity* is the baseline capacity outside calendar intervals. Set
    *shrink_policy* to :attr:`CapacityShrinkPolicy.GENTLE` to let active work
    finish before reducing capacity, or to
    :attr:`CapacityShrinkPolicy.IMMEDIATE` to preempt users down to the new
    capacity.
    """

    PutQueue = CalendarRequestQueue

    def __init__(self, env, capacity=1, calendar=(),
                 shrink_policy=None):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError('"capacity" must be an integer.')
        if capacity < 0:
            raise ValueError('"capacity" must be >= 0.')

        if isinstance(calendar, CapacityCalendar):
            if shrink_policy is not None:
                calendar = CapacityCalendar(calendar.entries, shrink_policy)
        else:
            calendar = CapacityCalendar(
                calendar, shrink_policy or CapacityShrinkPolicy.GENTLE)

        active_capacity = calendar.capacity_at(env.now)
        initial_capacity = (capacity if active_capacity is None
                            else active_capacity)

        base.BaseResource.__init__(self, env, initial_capacity)
        self.users = []
        self.queue = self.put_queue
        self._baseline_capacity = capacity
        self.calendar = calendar
        self.calendar_events = []
        """Chronological records of applied capacity changes."""
        self.events = []
        """Chronological resource-level event records."""
        self._next_request_sequence = 0
        self._schedule_calendar_events()

    @property
    def baseline_capacity(self):
        """Capacity used when no calendar interval is active."""
        return self._baseline_capacity

    @property
    def deficit(self):
        """Users above the current capacity during a gentle shrink."""
        return max(0, len(self.users) - self.capacity)

    request = BoundClass(CalendarRequest)
    release = BoundClass(Release)

    def _trigger_put(self, get_event):
        for index in range(len(self.put_queue) - 1, -1, -1):
            request = self.put_queue[index]
            process = request.proc
            if (process is not None
                    and self._env.active_process is not process
                    and not self._is_valid_wait(request)):
                request.cancel()

        return super(CalendarResource, self)._trigger_put(get_event)

    def _schedule_calendar_events(self):
        for time, capacity, entry in self.calendar.transitions_after(
                self._env.now):
            new_capacity = (self._baseline_capacity if capacity is None
                            else capacity)
            event = Event(self._env)
            event._ok = True
            event._value = (new_capacity, entry)
            event.callbacks.append(self._apply_calendar_change)
            self._env.schedule(
                event, CALENDAR_PRIORITY, time - self._env.now)

    def _apply_calendar_change(self, event):
        new_capacity, entry = event.value
        old_capacity = self._capacity
        if new_capacity == old_capacity:
            return

        change = CapacityChange(
            self._env.now, old_capacity, new_capacity,
            self.calendar.shrink_policy, entry)
        self._capacity = new_capacity
        self.calendar_events.append(change)
        self.events.append(ResourceEvent(
            self._env.now, CAPACITY_CHANGE_EVENT, new_capacity,
            len(self.users), None, change))

        if new_capacity < old_capacity:
            if self.calendar.shrink_policy is IMMEDIATE_SHRINK:
                self._preempt_for_calendar(change)
        elif new_capacity > old_capacity:
            self._trigger_put(None)

    def _preempt_for_calendar(self, change):
        excess = len(self.users) - self._capacity
        if excess <= 0:
            return

        victims = sorted(
            self.users,
            key=lambda request: (request.priority, request.time,
                                 request.sequence))[-excess:]
        for request in victims:
            self.users.remove(request)
            cause = CalendarPreempted(self, request.usage_since, change)
            request.proc.interrupt(cause)
            self.events.append(ResourceEvent(
                self._env.now, REQUEST_PREEMPTED_EVENT, self.capacity,
                len(self.users), request, cause))

    def _do_put(self, event):
        if len(self.users) >= self.capacity:
            return False
        if not self._is_valid_wait(event):
            event.cancel()
            return True

        self.users.append(event)
        event.usage_since = self._env.now
        event.succeed()
        self.events.append(ResourceEvent(
            self._env.now, REQUEST_GRANTED_EVENT, self.capacity,
            len(self.users), event, None))
        return True

    def _do_get(self, event):
        try:
            self.users.remove(event.request)
            removed = True
        except ValueError:
            removed = False

        event.succeed()
        if removed:
            self.events.append(ResourceEvent(
                self._env.now, REQUEST_RELEASED_EVENT, self.capacity,
                len(self.users), event.request, None))

    @staticmethod
    def _is_valid_wait(event):
        process = event.proc
        if process is None or process.env.active_process is process:
            return True
        return process.is_alive and process._target is event
