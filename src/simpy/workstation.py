"""
Composable workstation primitives with explicit part ownership.

A :class:`Workstation` waits for a part, processes it, and then keeps the part
while waiting for the downstream side to receive it.  A zero-capacity
:class:`HandoverPoint` is therefore a rendezvous: the upstream workstation
remains occupied until the downstream workstation is actually committed to take
the part.
"""

from collections import namedtuple


WAITING = 'waiting'
PROCESSING = 'processing'
WAITING_HANDOVER = 'waiting_handover'
BLOCKED = WAITING_HANDOVER


class WorkstationError(Exception):
    """Base class for workstation failures."""


class TransferCancelled(WorkstationError):
    """A pending transfer request was cancelled.

    :attr:`owner` identifies who held the part while the request was pending.
    A cancelled delivery keeps the part at the sender.  A cancelled receive
    request never claimed a part; undelivered parts remain with their senders.
    """

    def __init__(self, message, owner=None, part=None):
        super(TransferCancelled, self).__init__(message, owner, part)
        self.message = message
        self.owner = owner
        self.part = part

    def __str__(self):
        return self.message


class TransferCommitted(WorkstationError):
    """Cancellation arrived after a transfer had already been committed.

    The transfer cannot be undone.  :attr:`owner` is the downstream receiver
    or positive-capacity handover point that now holds the part.
    """

    def __init__(self, owner, part):
        message = 'Transfer is committed; %r now holds %r.' % (owner, part)
        super(TransferCommitted, self).__init__(owner, part)
        self.message = message
        self.owner = owner
        self.part = part

    def __str__(self):
        return self.message


class WorkstationHalted(WorkstationError):
    """A workstation stopped while it owned a part.

    ``owner`` is always the halted workstation, so callers do not have to infer
    ownership from the simulation phase in which the failure occurred.
    """

    def __init__(self, workstation, part, cause):
        message = '%s halted while holding %r' % (workstation.name, part)
        super(WorkstationHalted, self).__init__(workstation, part, cause)
        self.message = message
        self.workstation = workstation
        self.owner = workstation
        self.part = part
        self.cause = cause

    def __str__(self):
        return self.message


class Part(namedtuple('Part', ('id', 'payload'))):
    """An identifiable work item.

    ``payload`` is optional and may be ``None``.  The same object is passed
    through every handover; the primitives never copy it.
    """

    __slots__ = ()

    def __new__(cls, id, payload=None):
        return super(Part, cls).__new__(cls, id, payload)

    def __repr__(self):
        return 'Part(%r)' % (self.id,)


class _Transfer(object):
    __slots__ = ('event', 'part', 'owner', 'receiver', 'new_owner',
                 'cancelled')

    def __init__(self, event, part=None, owner=None, receiver=None):
        self.event = event
        self.part = part
        self.owner = owner
        self.receiver = receiver
        self.new_owner = None
        self.cancelled = False


class HandoverPoint(object):
    """A bounded handover point for identifiable parts.

    ``capacity=0`` models blocking after processing.  No part is stored at the
    point.  Delivery and receipt are matched in FIFO order, and both requests
    become committed at the same simulation time.

    ``capacity > 0`` models a finite buffer.  Once a delivery is accepted, the
    handover point itself owns the part until a receiver removes it.
    """

    def __init__(self, env, capacity=0, name='handover'):
        if capacity < 0:
            raise ValueError('"capacity" must be >= 0.')

        self.env = env
        self.capacity = capacity
        self.name = name
        self.items = []
        """Parts currently stored at a positive-capacity handover point."""
        self.deliveries = []
        """Pending delivery transfers, oldest first."""
        self.receipts = []
        """Pending receipt transfers, oldest first."""
        self.committed_receipts = []
        """Receipts committed but not yet resumed by their SimPy processes."""

    @property
    def blocked_deliveries(self):
        """Number of upstream owners waiting to deliver a part."""
        return len(self.deliveries)

    def deliver(self, part, owner):
        """Begin delivering *part* from *owner*.

        The returned transfer succeeds only when the part is accepted.  Until
        it succeeds, ownership remains with *owner*.
        """
        transfer = _Transfer(self.env.event(), part=part, owner=owner)
        self.deliveries.append(transfer)
        self._match()
        return transfer

    def receive(self, receiver):
        """Begin receiving a part for *receiver*.

        The returned transfer succeeds with the same part object supplied by
        the matched delivery.
        """
        event = self.env.event()
        transfer = _Transfer(event, receiver=receiver)
        event.callbacks.append(
            lambda event: self._clear_committed_receipt(transfer))
        self.receipts.append(transfer)
        self._match()
        return transfer

    def cancel(self, transfer):
        """Cancel an unmatched delivery or receipt.

        A matched or buffered transfer cannot be cancelled because ownership
        has already moved.
        """
        if transfer.cancelled:
            return

        if transfer.event.triggered:
            raise RuntimeError('Matched transfer cannot be cancelled.')

        if transfer in self.deliveries:
            self.deliveries.remove(transfer)
        elif transfer in self.receipts:
            self.receipts.remove(transfer)
        else:
            raise RuntimeError('Unknown transfer.')

        transfer.cancelled = True
        transfer.event.fail(TransferCancelled(
            self._cancellation_message(transfer), transfer.owner,
            transfer.part))

    def _cancellation_message(self, transfer):
        if transfer.receiver is not None:
            return ('%s cancelled receipt before claiming a part; '
                    'undelivered parts remain with their senders.' %
                    self.name)
        return '%s cancelled delivery; %r retains the part.' % (
            self.name, transfer.owner)

    def _release_sender(self, delivery):
        release = getattr(delivery.owner, '_release_part', None)
        if callable(release):
            release(delivery.part)

    def _clear_committed_receipt(self, receipt):
        if not receipt.event.ok:
            return
        try:
            self.committed_receipts.remove(receipt)
        except ValueError:
            pass

    def _commit_direct_handover(self, receipt, delivery):
        part = delivery.part
        delivery.new_owner = receipt.receiver
        receipt.part = part
        receipt.owner = receipt.receiver
        receipt.new_owner = receipt.receiver
        self._release_sender(delivery)
        self.deliveries.remove(delivery)
        delivery.event.succeed(part)

    def _commit_buffered_part(self, receipt):
        part = self.items.pop(0)
        receipt.part = part
        receipt.owner = self
        receipt.new_owner = receipt.receiver

    def _commit_receipt(self, receipt):
        self.receipts.remove(receipt)
        self.committed_receipts.append(receipt)
        receipt.event.succeed(receipt.part)

    def _match(self):
        while self.receipts and (self.items or self.deliveries):
            receipt = self.receipts[0]
            if self.items:
                self._commit_buffered_part(receipt)
            else:
                self._commit_direct_handover(receipt, self.deliveries[0])
            self._commit_receipt(receipt)

        if self.capacity:
            while self.deliveries and len(self.items) < self.capacity:
                delivery = self.deliveries[0]
                delivery.new_owner = self
                self._release_sender(delivery)
                self.items.append(delivery.part)
                self.deliveries.pop(0)
                delivery.event.succeed(delivery.part)


class WorkRecord(namedtuple('WorkRecord', (
        'part', 'input_wait', 'processing', 'blocked', 'started',
        'processing_ended', 'departed'))):
    """Timing information for one part at one workstation."""

    __slots__ = ()


class Workstation(object):
    """A sequentially occupied processing workstation.

    ``process_time`` may be a number or a callable receiving
    ``(workstation, part)`` and returning the processing duration.  The
    optional ``process`` callable receives the same arguments and may mutate
    the part payload before delivery.
    """

    def __init__(self, env, process_time, input_point, output_point,
                 name='workstation', process=None):
        if input_point.env is not env or output_point.env is not env:
            raise ValueError('All handover points must use this environment.')

        self.env = env
        self.name = name
        self.process_time = process_time
        self.input_point = input_point
        self.output_point = output_point
        self.process_function = process
        self.state = None
        self.part = None
        self.records = []
        self.process = env.process(self._run())

    @property
    def waiting_time(self):
        return sum(record.input_wait for record in self.records)

    @property
    def processing_time(self):
        return sum(record.processing for record in self.records)

    @property
    def blocked_time(self):
        return sum(record.blocked for record in self.records)

    def duration_for(self, part):
        if callable(self.process_time):
            return self.process_time(self, part)
        return self.process_time

    def _release_part(self, part):
        if self.part is not part:
            raise RuntimeError('%s does not hold %r.' % (self.name, part))
        self.part = None
        self.state = None

    def _record_departure(self, part, wait_started, started,
                          processing_ended, departed):
        self.records.append(WorkRecord(
            part, started - wait_started, processing_ended - started,
            departed - processing_ended, started, processing_ended, departed))

    def _run(self):
        while True:
            self.state = WAITING
            self.part = None
            receipt = self.input_point.receive(self)
            wait_started = self.env.now
            try:
                part = yield receipt.event
            except Exception as cause:
                if receipt.event.triggered and receipt.event.ok:
                    if not receipt.event.processed:
                        yield self.env.timeout(0)
                    part = receipt.event.value
                    self.state = PROCESSING
                    self.part = part
                    raise WorkstationHalted(self, part, cause)

                if not receipt.cancelled:
                    self.input_point.cancel(receipt)
                raise receipt.event._value

            started = self.env.now
            self.state = PROCESSING
            self.part = part

            try:
                yield self.env.timeout(self.duration_for(part))
                if self.process_function is not None:
                    self.process_function(self, part)
            except Exception as cause:
                raise WorkstationHalted(self, part, cause)

            processing_ended = self.env.now
            self.state = WAITING_HANDOVER
            delivery = self.output_point.deliver(part, self)
            try:
                yield delivery.event
            except Exception as cause:
                if delivery.event.triggered and delivery.event.ok:
                    if not delivery.event.processed:
                        yield self.env.timeout(0)
                    departed = self.env.now
                    self._record_departure(
                        part, wait_started, started, processing_ended,
                        departed)
                    raise TransferCommitted(delivery.new_owner, part)

                if not delivery.cancelled:
                    self.output_point.cancel(delivery)
                if delivery.event.triggered:
                    raise delivery.event._value
                raise WorkstationHalted(self, part, cause)

            departed = self.env.now
            self._record_departure(
                part, wait_started, started, processing_ended, departed)


def audit_locations(points, workstations=()):
    """Return ``{part: owner}`` and fail if a part occupies two locations.

    ``points`` may contain :class:`HandoverPoint` instances or SimPy stores.
    Direct deliveries still pending are claimed by their sender.  Receipts
    committed but awaiting process resumption are claimed by their receiver.
    Workstations holding an undelivered part are also included.
    """
    locations = {}

    def claim(part, owner):
        if part in locations:
            raise RuntimeError(
                '%r is held by both %r and %r.' %
                (part, locations[part], owner))
        locations[part] = owner

    for point in points:
        for part in getattr(point, 'items', ()):
            claim(part, point)
        for transfer in getattr(point, 'committed_receipts', ()):
            claim(transfer.part, transfer.owner)
        for transfer in getattr(point, 'deliveries', ()):
            claim(transfer.part, transfer.owner)

    for workstation in workstations:
        if workstation.part is not None:
            claim(workstation.part, workstation)

    return locations
