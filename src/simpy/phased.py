"""
Phased execution environment for delta-cycle style simulations.

Digital control models commonly need three stages within a single physical
point in time:

1. compute the inputs of a component,
2. update its state and
3. sample its outputs.

The same physical time may therefore be propagated through repeatedly.
:class:`PhasedEnvironment` keeps those *phases* separate from physical time:
time advances only through :class:`~simpy.events.Timeout` events, while
zero-delay events advance a *microstep* counter that is reset whenever the
environment moves on to a new physical time.

Within one microstep, phases are processed in the configured order (by
default ``compute``, ``update``, ``output``). Events produced while a later
phase is being processed but destined for an earlier phase are deferred to
the next microstep; events of the current phase stay in submission (FIFO)
order.

The regular :class:`~simpy.core.Environment` and its queueing rules are not
changed by this module.

"""
from heapq import heappush, heappop
from itertools import count

from simpy.core import (BoundClass, BaseEnvironment, EmptySchedule,
                        Infinity)
from simpy.events import (AllOf, AnyOf, Event, Process, Timeout, PENDING,
                          URGENT, NORMAL)
from simpy.exceptions import MicrostepLimit


class Phase(object):
    """A named phase within a single physical point in time.

    *index* determines in which order phases are processed within one
    microstep (lower values first). The default environment defines
    :data:`COMPUTE`, :data:`UPDATE` and :data:`OUTPUT`.

    """
    def __init__(self, index, name):
        self.index = index
        self.name = name

    def __repr__(self):
        return 'Phase(%d, %r)' % (self.index, self.name)

    def __eq__(self, other):
        return (isinstance(other, Phase) and self.index == other.index
                and self.name == other.name)

    def __hash__(self):
        return hash((self.index, self.name))


COMPUTE = Phase(0, 'compute')
"""First phase: compute component inputs from their signals."""

UPDATE = Phase(1, 'update')
"""Second phase: update component state."""

OUTPUT = Phase(2, 'output')
"""Third phase: sample component outputs."""

DEFAULT_PHASES = (COMPUTE, UPDATE, OUTPUT)


class PhasedEnvironment(BaseEnvironment):
    """Execution environment that separates physical time from phased
    microsteps.

    It behaves like :class:`~simpy.core.Environment`, but zero-delay events
    carry a :class:`Phase` and are ordered by ``(physical time, microstep,
    phase, submission order)`` instead of by a priority. The regular
    time-based queueing rules of :class:`~simpy.core.Environment` are left
    untouched; only this environment uses the phased scheme.

    *initial_time* is the physical time the simulation starts at.

    *phases* is the ordered tuple of phases used within each microstep and
    defaults to :data:`DEFAULT_PHASES`.

    *max_microsteps* optionally bounds the number of microsteps processed at
    a single physical time. If more would be needed (a typical symptom of an
    unbounded zero-delay feedback loop), a :class:`MicrostepLimit` is raised
    carrying a summary of the events still cycling. By default (``None``)
    there is no limit.

    """
    def __init__(self, initial_time=0, phases=None, max_microsteps=None):
        if phases is None:
            phases = DEFAULT_PHASES
        self._phases = tuple(phases)
        if (max_microsteps is not None
                and max_microsteps <= 0):
            raise ValueError('max_microsteps must be a positive integer or '
                             'None, got %r' % (max_microsteps,))
        self._max_microsteps = max_microsteps

        self._now = initial_time
        self._queue = []
        self._eid = count()
        self._active_proc = None

        # Cursor of the event currently being processed. Before the first
        # zero-delay event, microstep 0 / first phase are used as defaults.
        self._microstep = 0
        self._phase_index = self._phases[0].index
        self._active = False

        BoundClass.bind_early(self)

    @property
    def now(self):
        """The current physical simulation time."""
        return self._now

    @property
    def microstep(self):
        """The microstep currently being processed at :attr:`now` (0-based).

        The counter is reset to ``0`` whenever physical time advances.

        """
        return self._microstep

    @property
    def phase(self):
        """The :class:`Phase` currently being processed."""
        return self._phases[self._phase_index]

    @property
    def phases(self):
        """Tuple of phases processed within each microstep, in order."""
        return self._phases

    @property
    def active_process(self):
        """The currently active process of the environment."""
        return self._active_proc

    process = BoundClass(Process)
    timeout = BoundClass(Timeout)
    event = BoundClass(PhasedEvent)
    all_of = BoundClass(AllOf)
    any_of = BoundClass(AnyOf)

    def _resolve_phase(self, phase):
        """Return the index of *phase* in :attr:`phases`.

        *phase* may be a :class:`Phase` instance or an integer index.

        """
        if isinstance(phase, Phase):
            key = phase.index
        else:
            key = phase
        try:
            resolved = self._phases[key]
        except (IndexError, TypeError):
            raise ValueError('%r is not one of the environment phases %r'
                             % (phase, self._phases))
        return resolved.index

    def schedule(self, event, priority=NORMAL, delay=0, phase=None):
        """Schedule an *event* with a given *priority* and *delay*.

        *priority* is accepted for compatibility with
        :class:`~simpy.core.BaseEnvironment` but ignored: in this environment
        zero-delay events are ordered by phase and submission order.

        *delay* works like in the regular environment: an event scheduled
        with a positive delay becomes the first microstep (``compute``
        phase) of the target physical time.

        Zero-delay events are scheduled according to *phase*:

        - If *phase* is ``None`` and this call happens while an event is
          being processed, the event joins the current phase of the current
          microstep. This keeps plain
          :class:`~simpy.events.Event`-style chains in submission (FIFO)
          order.
        - If *phase* is an earlier phase than the one currently processed,
          the event is deferred to the next microstep.
        - If it is the same or a later phase, it is processed in the current
          microstep.

        """
        at = self._now + delay
        if delay > 0 or not self._active:
            microstep = 0
            phase_index = (self._resolve_phase(phase) if phase is not None
                           else self._phases[0].index)
        else:
            if phase is None:
                microstep = self._microstep
                phase_index = self._phase_index
            else:
                phase_index = self._resolve_phase(phase)
                if phase_index < self._phase_index:
                    microstep = self._microstep + 1
                else:
                    microstep = self._microstep

        heappush(self._queue,
                 (at, microstep, phase_index, next(self._eid), event))

    def peek(self):
        """Get the physical time of the next scheduled event. Return
        :data:`~simpy.core.Infinity` if there is no further event."""
        try:
            return self._queue[0][0]
        except IndexError:
            return Infinity

    def step(self):
        """Process the next event.

        Raise an :class:`~simpy.core.EmptySchedule` if no further events are
        available. Raise a :class:`MicrostepLimit` if processing the next
        event at the current physical time would exceed the configured
        microstep limit; the events still cycling remain in the queue and are
        reported on the exception instead of being skipped.

        """
        try:
            at, microstep, phase_index, _eid, event = heappop(self._queue)
        except IndexError:
            raise EmptySchedule()

        if at == self._now and self._active:
            if (self._max_microsteps is not None
                    and microstep >= self._max_microsteps):
                entries = [entry for entry in self._queue
                           if entry[0] == at]
                entries.insert(0, (at, microstep, phase_index, _eid, event))
                raise MicrostepLimit(at, microstep,
                                     self._max_microsteps, entries,
                                     self._phases)

        if at != self._now:
            # A normal Timeout advanced physical time: start a fresh
            # microstep count at the new time.
            self._now = at
            self._microstep = 0
        else:
            self._microstep = microstep

        self._phase_index = phase_index

        # Process callbacks of the event. Set the events callbacks to None
        # immediately to prevent concurrent modifications.
        self._active = True
        callbacks, event.callbacks = event.callbacks, None
        try:
            for callback in callbacks:
                callback(event)
        finally:
            self._active = False

        if not event._ok and not hasattr(event, '_defused'):
            # The event has failed and has not been defused. Crash the
            # environment.
            # Create a copy of the failure exception with a new traceback.
            exc = type(event._value)(*event._value.args)
            exc.__cause__ = event._value
            raise exc


class PhasedEvent(Event):
    """An :class:`~simpy.events.Event` that is processed at a fixed
    :class:`Phase` of a :class:`PhasedEnvironment`.

    The event is processed in *phase* (defaulting to the middle phase,
    :data:`UPDATE`) whenever it is triggered with a zero delay. If that
    phase has already passed in the current microstep, it is deferred to the
    next microstep.

    """
    def __init__(self, env, phase=None):
        super(PhasedEvent, self).__init__(env)
        if phase is None:
            phase = env.phases[len(env.phases) // 2]
        self._phase = env._resolve_phase(phase)

    @property
    def phase(self):
        """The :class:`Phase` at which this event is processed."""
        return self.env.phases[self._phase]

    def trigger(self, event):
        """Trigger the event at its configured phase."""
        self._ok = event._ok
        self._value = event._value
        self.env.schedule(self, phase=self.phase)

    def succeed(self, value=None):
        """Mark the event successful and schedule it at its phase."""
        if self._value is not PENDING:
            raise RuntimeError('%s has already been triggered' % self)

        self._ok = True
        self._value = value
        self.env.schedule(self, phase=self.phase)
        return self

    def fail(self, exception):
        """Mark the event failed and schedule it at its phase."""
        if self._value is not PENDING:
            raise RuntimeError('%s has already been triggered' % self)
        if not isinstance(exception, BaseException):
            raise ValueError('%s is not an exception.' % exception)
        self._ok = False
        self._value = exception
        self.env.schedule(self, phase=self.phase)
        return self
