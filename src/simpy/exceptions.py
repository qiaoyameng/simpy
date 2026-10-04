"""
SimPy specific exeptions.

"""


class SimPyException(Exception):
    """Base class for all SimPy specific exceptions."""


class Interrupt(SimPyException):
    """Exception thrown into a process if it is interrupted (see
    :func:`~simpy.events.Process.interrupt()`).

    :attr:`cause` provides the reason for the interrupt, if any.

    If a process is interrupted concurrently, all interrupts will be thrown
    into the process in the same order as they occurred.


    """
    def __init__(self, cause):
        super(Interrupt, self).__init__(cause)

    def __str__(self):
        return '%s(%r)' % (self.__class__.__name__, self.cause)

    @property
    def cause(self):
        """The cause of the interrupt or ``None`` if no cause was provided."""
        return self.args[0]


class StopProcess(SimPyException):
    """Raised to stop a SimPy process (similar to :exc:`StopIteration`).

    In Python 2, a ``return value`` inside generator functions is not allowed.
    The fall-back was raising ``StopIteration(value)`` instead.  However, this
    is deprecated_ now, so we need a custom exception type for this.

    .. _deprecated: https://www.python.org/dev/peps/pep-0479/

    """
    def __init__(self, value):
        super(StopProcess, self).__init__(value)

    def __str__(self):
        return '%s(%r)' % (self.__class__.__name__, self.value)

    @property
    def value(self):
        """The process' return value."""
        return self.args[0]


class MicrostepLimit(SimPyException):
    """Raised by a phased environment (e.g.
    :class:`~simpy.phased.PhasedEnvironment`) when the number of zero-delay
    microsteps at one physical time exceeds its configured limit.

    This usually indicates an unbounded zero-delay feedback loop. The events
    that are still cycling at that physical time are *not* silently skipped;
    they are attached to the exception as :attr:`remaining` and summarized in
    its message.

    """
    def __init__(self, time, microstep, limit, entries, phases):
        super(MicrostepLimit, self).__init__()
        #: Physical time at which the limit was reached.
        self.time = time
        #: Microstep index that would be processed next (0-based).
        self.microstep = microstep
        #: Configured maximum number of microsteps per physical time.
        self.limit = limit
        #: Events that are still queued at *time* and thus keep cycling.
        self.remaining = [entry[4] for entry in entries]
        self._entries = entries
        self._phases = phases

    def __str__(self):
        counts = {}
        for _t, _ms, phase_idx, _seq, _event in self._entries:
            name = self._phases[phase_idx].name
            counts[name] = counts.get(name, 0) + 1
        summary = ', '.join('%s: %d' % (name, count)
                            for name, count in sorted(counts.items()))
        examples = '; '.join(repr(entry[4])
                             for entry in self._entries[:5])
        return (
            'Microstep limit of %d per physical time reached at time %s '
            '(next microstep %d); %d event(s) still cycling at that time '
            '(%s). First events: %s'
            % (self.limit, self.time, self.microstep,
               len(self.remaining), summary, examples))
