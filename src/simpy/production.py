"""
Blocking production-line primitives.

A workstation is modelled as a state machine instead of a plain timeout.
A finished part remains with its workstation until a downstream receiver
accepts it, so a full downstream stage blocks the upstream workstation.
"""
from simpy.events import Event
from simpy.exceptions import Interrupt, SimPyException


WAITING = 'waiting'
PROCESSING = 'processing'
HANDING_OFF = 'handing_off'
FAILED = 'failed'


class ProductionError(SimPyException):
    """Base class for production-line errors."""


class Part(object):
    """A work item whose identity survives every transfer."""

    def __init__(self, ident):
        self.ident = ident

    def __repr__(self):
        return 'Part(%r)' % (self.ident,)


class OwnershipLedger(object):
    """Tracks exactly one current holder for every registered part."""

    def __init__(self):
        self._owners = {}

    def register(self, part, holder):
        if part in self._owners:
            raise ProductionError('%s is already registered' % part)
        self._owners[part] = holder
        return part

    def owner_of(self, part):
        try:
            return self._owners[part]
        except KeyError:
            raise ProductionError('%s is not registered' % part)

    def assert_owner(self, part, holder):
        owner = self.owner_of(part)
        if owner != holder:
            raise ProductionError(
                '%s is held by %r, not %r' % (part, owner, holder))

    def transfer(self, part, sender, receiver):
        self.assert_owner(part, sender)
        if receiver == sender:
            raise ProductionError('sender and receiver are both %r'
                                  % receiver)
        self._owners[part] = receiver

    def locations(self):
        result = {}
        for part, holder in self._owners.items():
            result.setdefault(holder, []).append(part)
        return result

    def assert_unique_locations(self):
        seen = {}
        for part in self._owners:
            ident = part.ident if hasattr(part, 'ident') else id(part)
            if ident in seen:
                raise ProductionError(
                    'part identity %r occurs more than once' % ident)
            seen[ident] = part
        return seen


class HandoffError(ProductionError):
    """Reports the current holder after a handoff cancellation."""

    def __init__(self, part, holder, handoff):
        super(HandoffError, self).__init__(part, holder, handoff)
        self.part = part
        self.holder = holder
        self.handoff = handoff

    def __str__(self):
        return '%s is held by %r' % (self.part, self.holder)


class HandoffCanceled(HandoffError):
    """A pending handoff was canceled before ownership moved."""


class HandoffCommitted(HandoffError):
    """Cancellation arrived after the receiver accepted the part."""


class ZeroCapacityHandoff(object):
    """A zero-capacity, rendezvous-style part handoff point.

    The handoff never stores a part. Both sides must be present at the same
    simulation time, and ownership is transferred before either side resumes.
    """

    capacity = 0

    def __init__(self, env, ledger):
        self.env = env
        self.ledger = ledger
        self._puts = []
        self._gets = []

    def put(self, part, holder):
        """Offer *part* without releasing ownership of it."""
        self.ledger.assert_owner(part, holder)
        self._assert_no_duplicate(self._puts, part, holder)
        event = Event(self.env)
        entry = (event, part, holder)
        event.part = part
        event.holder = holder
        event.committed = False

        if self._gets:
            get_event, receiver = self._gets.pop(0)
            self._commit(entry, get_event, receiver)
        else:
            self._puts.append(entry)
        return event

    def get(self, receiver):
        """Wait until an offered part is accepted by *receiver*."""
        self._assert_receiver_absent(receiver)
        event = Event(self.env)
        event.receiver = receiver
        event.part = None
        event.committed = False

        if self._puts:
            entry = self._puts.pop(0)
            self._commit(entry, event, receiver)
        else:
            self._gets.append((event, receiver))
        return event

    def cancel_put(self, event):
        """Remove a pending offer; a committed offer cannot be undone."""
        if event.triggered:
            if event.committed:
                raise HandoffCommitted(
                    event.part, self.ledger.owner_of(event.part), self)
            raise HandoffCanceled(event.part, event.holder, self)

        for index, entry in enumerate(self._puts):
            if entry[0] is event:
                del self._puts[index]
                self._fail_event(
                    event, HandoffCanceled(event.part, event.holder, self))
                return
        raise ProductionError('unknown put event')

    def cancel_get(self, event):
        """Remove a pending receive request."""
        if event.triggered:
            if event.committed:
                raise HandoffCommitted(
                    event.part, self.ledger.owner_of(event.part), self)
            raise HandoffCanceled(event.part, None, self)

        for index, entry in enumerate(self._gets):
            if entry[0] is event:
                del self._gets[index]
                self._fail_event(
                    event, HandoffCanceled(event.part, None, self))
                return
        raise ProductionError('unknown get event')

    def _commit(self, put_entry, get_event, receiver):
        put_event, part, sender = put_entry
        self.ledger.transfer(part, sender, receiver)
        put_event.committed = True
        get_event.committed = True
        get_event.part = part
        put_event.succeed(part)
        get_event.succeed(part)

    def _fail_event(self, event, error):
        event._ok = False
        event._value = error
        event._defused = True
        self.env.schedule(event)

    def _assert_no_duplicate(self, entries, part, holder):
        for _, queued_part, queued_holder in entries:
            if queued_part is part or queued_holder == holder:
                raise ProductionError(
                    '%r already has a pending handoff' % holder)

    def _assert_receiver_absent(self, receiver):
        for _, queued_receiver in self._gets:
            if queued_receiver == receiver:
                raise ProductionError(
                    '%r already has a pending receive' % receiver)


class WorkstationError(ProductionError):
    """Reports part ownership after an interruption or failure."""

    def __init__(self, station, part, holder, state, cause):
        super(WorkstationError, self).__init__(
            station, part, holder, state, cause)
        self.station = station
        self.holder = holder
        self.part = part
        self.state = state
        self.cause = cause

    def __str__(self):
        if self.part is None:
            return '%s failed in state %r without a part' % (
                self.station.name, self.state)
        return '%s failed in state %r; %s is held by %r' % (
            self.station.name, self.state, self.part, self.holder)


class WorkstationInterrupted(WorkstationError):
    """The workstation process was externally interrupted."""


class WorkstationFailure(WorkstationError):
    """Processing raised an ordinary exception."""


class Workstation(object):
    """A one-slot composable workstation.

    States are waiting, processing and handing_off. The workstation owns a
    part while processing it and continues to own it during handoff until the
    downstream side accepts it.
    """

    def __init__(self, env, name, inbound, outbound, duration, ledger):
        if inbound.env is not env or outbound.env is not env:
            raise ValueError('all handoffs must use the workstation env')
        if inbound.ledger is not ledger or outbound.ledger is not ledger:
            raise ValueError('all handoffs must share the ownership ledger')

        self.env = env
        self.name = name
        self.inbound = inbound
        self.outbound = outbound
        self.duration = duration
        self.ledger = ledger
        self.state = WAITING
        self.part = None
        self.processing_time = 0.0
        self.blocking_time = 0.0
        self._state_since = env.now
        self.process = env.process(self._run())

    def _run(self):
        while True:
            self._enter(WAITING)
            receive = self.inbound.get(self.name)
            try:
                part = yield receive
            except HandoffCanceled as exc:
                self._fail(None, exc, WorkstationInterrupted)
            except Interrupt as exc:
                if receive.committed:
                    self._fail(receive.part, exc, WorkstationInterrupted)
                self.inbound.cancel_get(receive)
                self._fail(None, exc, WorkstationInterrupted)
            except Exception as exc:
                self._fail(None, exc, WorkstationFailure)

            self.part = part
            self.ledger.assert_owner(part, self.name)

            self._enter(PROCESSING)
            try:
                if callable(self.duration):
                    delay = self.duration(part)
                else:
                    delay = self.duration
                yield self.env.timeout(delay)
            except Interrupt as exc:
                self._fail(part, exc, WorkstationInterrupted)
            except Exception as exc:
                self._fail(part, exc, WorkstationFailure)

            self._enter(HANDING_OFF)
            delivery = self.outbound.put(part, self.name)
            try:
                yield delivery
            except HandoffCanceled as exc:
                self._fail(part, exc, WorkstationInterrupted)
            except Interrupt as exc:
                try:
                    self.outbound.cancel_put(delivery)
                except HandoffCommitted as committed:
                    self.part = None
                    self._fail(part, committed, WorkstationInterrupted)
                self._fail(part, exc, WorkstationInterrupted)
            except Exception as exc:
                self._fail(part, exc, WorkstationFailure)
            self.part = None

    def _enter(self, state):
        now = self.env.now
        elapsed = now - self._state_since
        if self.state == PROCESSING:
            self.processing_time += elapsed
        elif self.state == HANDING_OFF:
            self.blocking_time += elapsed
        self.state = state
        self._state_since = now

    def elapsed_in_state(self):
        return self.env.now - self._state_since

    def _fail(self, part, cause, error_type):
        self._enter(FAILED)
        holder = self.ledger.owner_of(part) if part is not None else None
        self.part = part if holder == self.name else None
        raise error_type(self, part, holder, FAILED, cause)
