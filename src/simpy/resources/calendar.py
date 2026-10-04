"""
Resources with a time-varying capacity.

A :class:`CalendarResource` changes its capacity over simulation time
according to a *capacity calendar*. This can, for example, model a workshop
that loses half of its workers at every shift change -- something a fixed
capacity cannot express.

A calendar (:class:`CapacityCalendar`) consists of
:class:`CalendarEntry` triples ``(start, end, capacity)``. Each entry is
effective within the half-open interval ``[start, end)``: it applies at
``start`` and stops applying exactly at ``end``. Entries must not overlap,
but adjacent entries (one ending where the next begins) are allowed.
Outside all entries, the resource's *default capacity* applies.

When a calendar shift reduces the capacity below the current number of
users, one of two strategies applies:

- ``CalendarResource.DRAIN`` (*graceful shrink*): current users keep the
  resource until they release it. A capacity deficit (more users than
  capacity) may exist while draining, but no new requests are granted as
  long as no free slot is available.
- ``CalendarResource.PREEMPT`` (*immediate shrink*): excess users are
  preempted immediately. Victims are chosen by lowest priority first
  (the largest ``priority`` value); ties are broken by latest usage time
  (the newest user first). Victims receive an
  :class:`~simpy.exceptions.Interrupt` whose cause is a
  :class:`CalendarPreemption` naming the calendar entry and the new
  capacity that caused the preemption.

When the capacity increases again, only requests that are still waiting in
the queue are woken up; cancelled requests have already been removed from
the queue and are not triggered.

All calendar shifts are driven by simulation time only. They are scheduled
as events with :data:`~simpy.events.URGENT` priority, so if a shift, a task
completion and a new request fall onto the same simulation time, they are
processed in a fixed order:

1. the capacity shift (including preemptions),
2. task completions and other normally scheduled events,
3. releases and grants of pending requests triggered by them.

No wall-clock timers are used (or added to the environment) for this.

"""
from collections import namedtuple
from numbers import Real

from simpy.events import Event, URGENT
from simpy.resources.resource import Preempted, PriorityResource


class CalendarEntry(namedtuple('CalendarEntry', ('start', 'end', 'capacity'))):
    """Capacity of a resource within the half-open interval ``[start, end)``.

    The entry applies at simulation time ``start`` and stops applying
    exactly at simulation time ``end``.

    """
    __slots__ = ()


class CapacityCalendar(object):
    """A set of non-overlapping capacity entries over simulation time.

    *entries* must be an iterable of ``(start, end, capacity)`` triples.
    Entries are stored sorted by their start time.

    Raise a :exc:`ValueError` if an entry is malformed, if its capacity is
    not a positive integer, if its interval is empty or starts before
    simulation time ``0``, or if two entries overlap. Adjacent entries
    (``end`` of one equals ``start`` of the next) do not count as
    overlapping.

    """
    def __init__(self, entries):
        parsed = []
        for entry in entries:
            try:
                start, end, capacity = entry
            except (TypeError, ValueError):
                raise ValueError(
                    'Calendar entries must be (start, end, capacity) '
                    'triples, got %r.' % (entry,))
            for time in (start, end):
                if isinstance(time, bool) or not isinstance(time, Real):
                    raise ValueError(
                        'Calendar times must be numbers, got %r.' % (time,))
            if start < 0:
                raise ValueError(
                    'Calendar entries must not start before 0, got %r.' %
                    (start,))
            if not start < end:
                raise ValueError(
                    'Calendar entry start (%r) must be before its end '
                    '(%r).' % (start, end))
            if (isinstance(capacity, bool) or
                    not isinstance(capacity, Real) or
                    capacity != int(capacity) or capacity < 1):
                raise ValueError(
                    'Calendar capacity must be a positive integer, got '
                    '%r.' % (capacity,))
            parsed.append(CalendarEntry(start, end, int(capacity)))

        parsed.sort()
        for previous, current in zip(parsed, parsed[1:]):
            if current.start < previous.end:
                raise ValueError(
                    'Calendar entries %r and %r overlap.' %
                    (previous, current))

        self._entries = tuple(parsed)
        self._boundaries = tuple(sorted(
            {time for entry in parsed for time in (entry.start, entry.end)}))

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self):
        """The sorted tuple of :class:`CalendarEntry` items."""
        return self._entries

    @property
    def boundaries(self):
        """Sorted tuple of all simulation times at which the capacity may
        change (all entry starts and ends)."""
        return self._boundaries

    def entry_at(self, time):
        """Return the :class:`CalendarEntry` effective at *time*, or
        ``None`` if no entry covers *time*."""
        for entry in self._entries:
            if entry.start <= time < entry.end:
                return entry
        return None

    def capacity_at(self, time):
        """Return the capacity effective at *time*, or ``None`` if no
        entry covers *time*."""
        entry = self.entry_at(time)
        return entry.capacity if entry is not None else None


class CalendarPreemption(Preempted):
    """Cause of an :class:`~simpy.exceptions.Interrupt` thrown into a
    process that is preempted by a capacity calendar shift.

    """
    def __init__(self, usage_since, resource, capacity, entry):
        super(CalendarPreemption, self).__init__(
            by=None, usage_since=usage_since, resource=resource)
        self.capacity = capacity
        """The reduced capacity that caused the preemption."""
        self.entry = entry
        """The :class:`CalendarEntry` that caused the preemption, or
        ``None`` if the capacity reverted to the resource's default
        capacity."""


class _CapacityShift(Event):
    """Event that applies a calendar boundary to its resource.

    It is scheduled with :data:`~simpy.events.URGENT` priority so that
    capacity changes take effect before all normally scheduled events
    (e.g. timeouts, releases or new requests) at the same simulation time.

    """
    def __init__(self, resource, time):
        super(_CapacityShift, self).__init__(resource._env)
        self.resource = resource
        """The :class:`CalendarResource` this shift belongs to."""
        self._ok = True
        self._value = None
        self.callbacks.append(resource._apply_shift)
        self.env.schedule(self, URGENT, time - self.env.now)


class CalendarResource(PriorityResource):
    """A :class:`~simpy.resources.resource.PriorityResource` whose capacity
    changes over simulation time according to a *calendar*.

    *capacity* is the default capacity that applies outside all calendar
    entries. *calendar* is a :class:`CapacityCalendar` (or an iterable of
    ``(start, end, capacity)`` triples). *strategy* must be
    :attr:`CalendarResource.DRAIN` or :attr:`CalendarResource.PREEMPT` and
    determines how a shrinking capacity is applied (see the module
    documentation).

    The :attr:`~simpy.resources.base.BaseResource.capacity` property always
    reports the capacity currently in effect.

    """
    DRAIN = 'drain'
    """Graceful shrink strategy: let current users finish; block new
    requests while no free slot is available."""

    PREEMPT = 'preempt'
    """Immediate shrink strategy: preempt excess users at once."""

    def __init__(self, env, capacity, calendar, strategy=DRAIN):
        if strategy not in (CalendarResource.DRAIN, CalendarResource.PREEMPT):
            raise ValueError(
                'strategy must be CalendarResource.DRAIN or '
                'CalendarResource.PREEMPT, got %r.' % (strategy,))
        if not isinstance(calendar, CapacityCalendar):
            calendar = CapacityCalendar(calendar)
        self._calendar = calendar
        self._strategy = strategy
        self._default_capacity = capacity

        super(CalendarResource, self).__init__(env, capacity)

        # Apply the capacity in effect right now and schedule one shift
        # event per future calendar boundary.
        entry = calendar.entry_at(env.now)
        if entry is not None:
            self._capacity = entry.capacity
        for boundary in calendar.boundaries:
            if boundary > env.now:
                _CapacityShift(self, boundary)

    @property
    def calendar(self):
        """The :class:`CapacityCalendar` of this resource."""
        return self._calendar

    @property
    def strategy(self):
        """The shrink strategy (:attr:`DRAIN` or :attr:`PREEMPT`)."""
        return self._strategy

    @property
    def default_capacity(self):
        """The capacity that applies outside all calendar entries."""
        return self._default_capacity

    def _apply_shift(self, event):
        """Apply the capacity effective at the current simulation time.

        Called as callback of the :class:`_CapacityShift` events.
        """
        entry = self._calendar.entry_at(self._env.now)
        new_capacity = (entry.capacity if entry is not None
                        else self._default_capacity)
        old_capacity = self._capacity
        if new_capacity == old_capacity:
            return
        self._capacity = new_capacity

        if new_capacity > old_capacity:
            # Wake up requests that are still waiting; cancelled requests
            # have already been removed from the put queue. _trigger_put()
            # grants at most one request per call for this resource type,
            # so repeat while free slots and waiters remain.
            while self.put_queue and len(self.users) < self._capacity:
                self._trigger_put(None)
        elif self._strategy == CalendarResource.PREEMPT:
            self._preempt_users(new_capacity, entry)
        # With the DRAIN strategy, current users keep the resource until
        # they release it. A capacity deficit (more users than capacity)
        # may exist while draining; _do_put() refuses to grant new
        # requests as long as len(self.users) >= self.capacity.

    def _preempt_users(self, capacity, entry):
        """Preempt excess users so that no more than *capacity* users
        remain. Victims receive an interrupt with a
        :class:`CalendarPreemption` cause."""
        excess = len(self.users) - capacity
        if excess <= 0:
            return
        # Victim selection: lowest priority (largest priority value)
        # first; ties are broken by latest usage time (newest user first).
        victims = sorted(
            self.users,
            key=lambda request: (-request.priority,
                                 -request.usage_since))[:excess]
        for victim in victims:
            self.users.remove(victim)
            if victim.proc is not None:
                victim.proc.interrupt(CalendarPreemption(
                    usage_since=victim.usage_since, resource=self,
                    capacity=capacity, entry=entry))
