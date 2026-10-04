"""
Staged (phased) execution environment for digital control models.

The ordinary :class:`~simpy.core.Environment` has a single notion of time.
Digital control models on the other hand distinguish the *physical time* at
which a clock tick occurs from the *micro steps* (delta cycles) that propagate
signals within that tick.  A tick is normally split into three stages:

- :attr:`Phase.INPUT`   -- compute the inputs of the model,
- :attr:`Phase.STATE`   -- update the internal state,
- :attr:`Phase.OUTPUT`  -- sample the (new) outputs.

Events that are produced in a later stage but target an earlier (or the same)
stage cannot be handled in the current round anymore.  They are deferred to
the next micro step, keeping the order in which they were submitted (FIFO).
Only events targeting a strictly later stage still happen in the current micro
step.

A normal :class:`~simpy.events.Timeout` with a positive delay advances the
physical time.  Whenever the simulation crosses to a new physical point in time
the micro step counter is reset to zero.  A limit can be imposed on the number
of micro steps that are allowed to happen at a single point in time.  If it is
reached -- which normally indicates a zero-delay feedback loop -- a
:class:`MicrostepLimit` exception is raised that reports the events which are
still cycling.  These events are *not* silently skipped; they remain in the
queue and can be inspected.

The default :class:`~simpy.core.Environment` and its queueing semantics are not
changed by this module.
"""
from heapq import heappush, heappop, heapify
from itertools import count

from simpy.core import Environment, Infinity, EmptySchedule
from simpy.events import NORMAL


class Phase(object):
    """The stages that make up a single micro step (delta cycle).

    The stages are ordered: :attr:`INPUT` < :attr:`STATE` < :attr:`OUTPUT`.
    They are plain integers so they can be compared and used directly as the
    *phase* argument of :meth:`PhasedEnvironment.schedule`,
    :meth:`PhasedEnvironment.timeout` and :meth:`simpy.events.Event.succeed`
    (and its siblings).

    Custom phases are possible by simply using other ``int`` values.
    """

    INPUT = 0
    """Compute the inputs of the model."""

    STATE = 1
    """Update the internal state."""

    OUTPUT = 2
    """Sample the outputs of the model."""

    _names = {0: 'INPUT', 1: 'STATE', 2: 'OUTPUT'}

    @classmethod
    def name(cls, phase):
        """Return a human readable *name* for a *phase*.

        Known phases (:attr:`INPUT`, :attr:`STATE` and :attr:`OUTPUT`) are
        rendered by their name, everything else as its integer value.
        """
        return cls._names.get(phase, str(phase))


class MicrostepLimit(RuntimeError):
    """Raised by a :class:`PhasedEnvironment` when more than the configured
    number of micro steps happen at a single point in physical time.

    This usually means that a zero-delay feedback loop keeps the model from
    settling.  The events that are still scheduled at the offending point in
    time are *not* discarded; they are available as :attr:`pending_events` for
    inspection.

    """

    def __init__(self, env):
        self.time = env.now
        """The physical time at which the limit was reached."""
        self.microstep = env.microstep
        """The (exceeded) micro step counter."""
        self.limit = env.microstep_limit
        """The configured maximum number of micro steps per physical time."""
        self.pending_events = [
            entry[2]
            for slot in env._slots.values()
            for entry in slot
        ]
        """The events still cycling at :attr:`time`."""
        super(MicrostepLimit, self).__init__(str(self))

    def __str__(self):
        pending = ', '.join(repr(event) for event in self.pending_events)
        if not pending:
            pending = '<none>'
        return (
            'Microstep limit of %s per physical time reached at time %s '
            '(microstep %s). %d event(s) are still cycling: %s'
            % (self.limit, self.time, self.microstep,
               len(self.pending_events), pending))

class PhasedEnvironment(Environment):
    """An :class:`~simpy.core.Environment` that separates the physical time
    from the micro step and processes every micro step in :class:`phases
    <Phase>`.

    *initial_time* is the physical time the environment starts at.
    *microstep_limit* is the maximum number of micro steps that may be
    processed at a single point in physical time.  It defaults to ``None``
    which disables the limit.  When the limit is reached, :meth:`step` raises
    a :class:`MicrostepLimit` exception describing the events which are still
    cycling.

    In addition to :attr:`~simpy.core.Environment.now` the environment exposes
    :attr:`microstep` and :attr:`phase`.

    Every micro step walks the stages in the order
    :attr:`~Phase.INPUT` -> :attr:`~Phase.STATE` -> :attr:`~Phase.OUTPUT`.
    Events scheduled for a strictly later stage of the micro step currently
    being processed are reachable in that micro step; events scheduled for an
    earlier stage or the stage currently being processed cannot be handled in
    this round anymore and are carried over to the next micro step, keeping
    their submission order (FIFO).  A *frontier* remembers the furthest stage
    reachable in the current micro step so that a following micro step cannot
    overtake the remaining stages of the current one.

    Internally the queue is a set of *slots*.  Each slot groups the events
    sharing the same ``(time, microstep, phase)`` and preserves their
    submission order (FIFO).  The slots themselves are ordered on a heap.
    """

    def __init__(self, initial_time=0, microstep_limit=None):
        super(PhasedEnvironment, self).__init__(initial_time)
        if microstep_limit is not None and microstep_limit < 0:
            raise ValueError(
                'microstep_limit(%s) must not be negative.'
                % microstep_limit)
        self._microstep_limit = microstep_limit
        self._microstep = 0
        self._phase = Phase.INPUT
        # Heap of slot keys (time, microstep, phase) and a dict mapping each
        # key to its FIFO list of (priority, sequence, event) entries.
        self._slot_keys = []
        self._slots = {}
        # Furthest (microstep, phase) reachable at the current physical time.
        # Only slots up to and including the frontier may be processed.
        self._frontier = (0, Phase.INPUT)

    @property
    def microstep(self):
        """The micro step (delta cycle) within the current physical time.

        It starts at ``0`` for every new physical point in time and grows each
        time an event that targets an earlier or the same :class:`Phase`
        defers the current round to another micro step.
        """
        return self._microstep

    @property
    def phase(self):
        """The :class:`Phase` that is currently being processed."""
        return self._phase

    @property
    def microstep_limit(self):
        """The maximum number of micro steps per physical time or ``None`` if
        no limit is set."""
        return self._microstep_limit

    def schedule(self, event, priority=NORMAL, delay=0, phase=None):
        """Schedule an *event* with a *priority* and a *delay*.

        If *phase* is ``None`` (the default) the event is appended to the
        current :attr:`phase` and micro step.  This keeps the FIFO behaviour
        of the plain :class:`~simpy.core.Environment` within that stage and,
        unlike an explicit phase, never opens a new micro step.

        If *phase* is given, the event is scheduled for that phase at the
        physical time ``now + delay``.  A phase strictly later than the
        current phase is still handled in the current micro step.  A phase
        equal to or earlier than the current phase cannot be handled in this
        round anymore and is carried over to the next micro step; events that
        land in the same phase and micro step keep their submission order
        (FIFO).
        """
        when = self._now + delay

        if when > self._now:
            # A timeout advances the physical time. The first round at the new
            # physical time starts at micro step 0; an explicit phase selects
            # the phase of that first round.
            microstep = 0
            if phase is None:
                phase = self._phase
        elif phase is None:
            # Stay in the currently processed slot (plain FIFO semantics).
            microstep = self._microstep
            phase = self._phase
        elif phase > self._phase:
            # A strictly later stage of the current round.
            microstep = self._microstep
        else:
            # The same or an earlier stage: defer to the next micro step.
            microstep = self._microstep + 1

        key = (when, microstep, phase)
        slot = self._slots.get(key)
        if slot is None:
            slot = []
            self._slots[key] = slot
            heappush(self._slot_keys, key)
        # Urgent events are handled before normal events within a stage; events
        # with the same priority keep their submission order via the id.
        slot.append((priority, next(self._eid), event))

        # Extend the reachable frontier. Events at a future physical time do
        # not affect the frontier of the current time.
        if when == self._now and (microstep, phase) > self._frontier:
            self._frontier = (microstep, phase)

    def peek(self):
        """Get the physical time of the next reachable event.  Return
        :data:`~simpy.core.Infinity` if there is no further event."""
        for key in self._slot_keys:
            when, microstep, phase = key
            if when > self._now or (microstep, phase) <= self._frontier:
                return when
        return Infinity

    def step(self):
        """Process the next event.

        Raise an :exc:`~simpy.core.EmptySchedule` if there are no further
        events.  Raise :class:`MicrostepLimit` if processing the next event
        would exceed the configured micro step limit.  In that case no event
        is removed from the queue.
        """
        key = self._next_key()

        # _next_key() raises EmptySchedule if nothing is reachable.
        when, microstep, phase = key

        if when > self._now:
            # A normal timeout advances the physical time; the micro step
            # count starts over.
            self._now = when
            self._microstep = 0
        elif microstep > self._microstep:
            # Crossing into a new micro step of the same physical time.
            if (self._microstep_limit is not None and
                    microstep > self._microstep_limit):
                raise MicrostepLimit(self)
            self._microstep = microstep
        self._phase = phase

        slot = self._slots[key]
        # Urgent events come first, then normal events; within a priority the
        # submission order (FIFO) is preserved.
        slot.sort()
        _, _, event = slot.pop(0)
        if not slot:
            del self._slots[key]
            # Rebuild the heap invariant after removing an arbitrary slot.
            # Slots are keyed by (time, microstep, phase); a simulation keeps
            # few of them, so the linear cost is negligible.
            self._slot_keys.remove(key)
            heapify(self._slot_keys)

        # Process callbacks of the event. Set the events callbacks to None
        # immediately to prevent concurrent modifications.
        callbacks, event.callbacks = event.callbacks, None
        for callback in callbacks:
            callback(event)

        if not event._ok and not hasattr(event, '_defused'):
            # The event has failed and has not been defused. Crash the
            # environment.
            # Create a copy of the failure exception with a new traceback.
            exc = type(event._value)(*event._value.args)
            exc.__cause__ = event._value
            raise exc

    def _next_key(self):
        """Return the key of the next reachable slot.

        Slots at the current physical time beyond :attr:`_frontier` are not
        yet reachable (they belong to a micro step whose preceding stages have
        not all been processed).  If only such slots -- or slots at a later
        physical time -- remain, advance the frontier accordingly.
        """
        # Discard stale keys (slots emptied without being the heap head).
        while (self._slot_keys and
               self._slot_keys[0] not in self._slots):
            heappop(self._slot_keys)

        if not self._slot_keys:
            raise EmptySchedule()

        head = self._slot_keys[0]
        when, microstep, phase = head

        if when > self._now:
            # The whole current physical time has settled. Move on.
            return head

        # Search the heap for the earliest reachable slot of the current
        # physical time.
        reachable = None
        for key in self._slot_keys:
            t, m, p = key
            if t > self._now:
                continue
            if (m, p) <= self._frontier:
                if reachable is None or key < reachable:
                    reachable = key

        if reachable is not None:
            return reachable

        # Nothing left within the current frontier. Any current-time slot is
        # part of a future micro step; open the smallest one.
        future_current = min(
            (key for key in self._slot_keys if key[0] == self._now),
            default=None)
        if future_current is not None:
            m, p = future_current[1], future_current[2]
            if (self._microstep_limit is not None and
                    m > self._microstep_limit):
                # Construct the error with the would-be micro step visible.
                self._microstep = m
                raise MicrostepLimit(self)
            self._frontier = (m, p)
            return future_current

        # Only later physical times remain.
        return head
