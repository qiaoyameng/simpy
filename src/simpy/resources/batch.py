"""
Batch processing resource.

:class:`BatchResource` models a server that processes items in batches
(e.g., an oven that can bake up to *max_batch* items at once).  A batch
starts as soon as *max_batch* items are waiting.  Otherwise a deadline is
computed from the arrival of the first queued item and the batch starts
with whatever is queued once *max_wait* has elapsed.

"""
from simpy.events import Event


class Withdrawn(Exception):
    """Raised into processes whose item was withdrawn while waiting."""

    def __init__(self, request):
        super(Withdrawn, self).__init__(request)
        self.request = request

    def __str__(self):
        return ('Item %r was withdrawn before its batch started.' %
                (self.request.item,))


class BatchRequest(Event):
    """Event returned by :meth:`BatchResource.put`.

    It is triggered with the item as its value once the batch containing
    the item has finished its service.  If the item is withdrawn while
    waiting, this event fails with a :class:`Withdrawn` exception.

    """

    def __init__(self, resource, item):
        super(BatchRequest, self).__init__(resource._env)
        self.resource = resource
        self.item = item


class _Slot(object):
    """Bookkeeping for a queued item."""

    __slots__ = ('item', 'arrival', 'request')

    def __init__(self, item, arrival, request):
        self.item = item
        self.arrival = arrival
        self.request = request


class BatchResource(object):
    """Resource that serves items in batches.

    A batch starts when one of the following conditions holds:

    - *max_batch* items are queued, or
    - the oldest queued item has waited for *max_wait* time units and at
      least *min_batch* items are queued.

    If the waiting time expires while fewer than *min_batch* items are
    queued, the resource does **not** start an undersized batch and does
    **not** restart the timer.  Instead it keeps the (already expired)
    deadline and starts the batch immediately once the *min_batch*-th item
    arrives.

    Items can be withdrawn while they are waiting.  The deadline is always
    recomputed as ``earliest arrival + max_wait`` of the remaining items,
    so withdrawing the first item moves the deadline to the next earliest
    arrival -- it is never restarted from "now".

    Once a batch starts its service, its members are fixed: items arriving
    during the service join the next batch.  The service duration is
    charged once per batch, no matter how many items it contains.

    *service_time* is either a number or a callable receiving the list of
    batched items and returning the service duration.  It is evaluated
    exactly once per batch.

    """

    def __init__(self, env, min_batch=1, max_batch=8, max_wait=10.0,
                 service_time=0.0):
        if not 1 <= min_batch <= max_batch:
            raise ValueError('need 1 <= min_batch <= max_batch')
        if max_wait < 0:
            raise ValueError('max_wait must be >= 0')
        self._env = env
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.service_time = service_time
        self._waiting = []
        self._changed = None
        env.process(self._run())

    @property
    def queue(self):
        """Items currently waiting for a batch to start."""
        return [slot.item for slot in self._waiting]

    def put(self, item):
        """Queue *item* for the next batch.

        Returns a :class:`BatchRequest` event that is triggered with the
        item once its batch has been served.

        """
        request = BatchRequest(self, item)
        self._waiting.append(_Slot(item, self._env.now, request))
        self._notify()
        return request

    def withdraw(self, item):
        """Remove a waiting *item* (or its :class:`BatchRequest`).

        Returns ``True`` if the item was still waiting and has been
        removed.  Returns ``False`` if the item is unknown or its batch
        has already started -- in-service batches cannot be changed.

        """
        for slot in self._waiting:
            if slot.item is item or slot.request is item:
                self._waiting.remove(slot)
                slot.request.fail(Withdrawn(slot.request))
                if not slot.request.callbacks:
                    # Nobody is waiting on the request; don't let the
                    # unhandled failure crash the environment.
                    slot.request.defused = True
                self._notify()
                return True
        return False

    def _notify(self):
        if self._changed is not None and not self._changed.triggered:
            self._changed.succeed()

    def _run(self):
        env = self._env
        while True:
            # Wait for at least one item.
            while not self._waiting:
                self._changed = env.event()
                yield self._changed

            # Collect phase: gather items until the batch is full or the
            # deadline of the oldest item is reached.  The deadline is
            # recomputed on every arrival/withdrawal, so a stale timer
            # left over from a previous iteration can never start a batch.
            while 0 < len(self._waiting) < self.max_batch:
                deadline = min(s.arrival for s in self._waiting) + self.max_wait
                remaining = deadline - env.now
                if remaining <= 0:
                    if len(self._waiting) >= self.min_batch:
                        break
                    # Timed out below min_batch: keep the expired deadline
                    # and start as soon as enough items are queued.
                    while 0 < len(self._waiting) < self.min_batch:
                        self._changed = env.event()
                        yield self._changed
                    break
                timer = env.timeout(remaining)
                self._changed = env.event()
                yield timer | self._changed

            if not self._waiting:
                continue  # Everything was withdrawn; wait for new arrivals.

            # The members of this batch are now fixed; later arrivals wait
            # for the next batch.
            batch = self._waiting[:self.max_batch]
            del self._waiting[:len(batch)]
            items = [slot.item for slot in batch]
            if callable(self.service_time):
                duration = self.service_time(items)
            else:
                duration = self.service_time
            yield env.timeout(duration)
            for slot in batch:
                slot.request.succeed(slot.item)
