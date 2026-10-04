"""
Batching resources for processing groups of requests.

A :class:`BatchResource` collects requests until a batch is full or until a
maximum waiting time has elapsed since the oldest request in the current
batch. Full batches are fixed immediately. Timed-out, non-empty batches are
also fixed, even when they contain fewer requests than ``min_batch_size``.

"""
from functools import partial

from simpy.core import BoundClass
from simpy.events import Event


class BatchRequest(Event):
    """Request to process *item* in a batch.

    The event is triggered after the batch containing the item has been
    processed. The event's value is a tuple of the items in that batch.

    A waiting request may be cancelled with :meth:`cancel`, for example from
    a ``with`` statement. Use :meth:`BatchResource.withdraw` to explicitly
    withdraw an item and fail this event.

    """
    def __init__(self, resource, item):
        super(BatchRequest, self).__init__(resource._env)
        self.resource = resource
        self.proc = self.env.active_process
        self.item = item
        """The item represented by this request."""
        self.time = self.env.now
        """The simulation time at which this item was submitted."""
        self.batch = None
        """The fixed batch containing this request, or ``None`` while
        waiting."""
        resource._add_request(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cancel()

    def cancel(self):
        """Remove this request if it is still waiting for a batch."""
        if self.batch is None and not self.triggered:
            self.resource._cancel_request(self)


class BatchResource(object):
    """Process up to ``max_batch_size`` requests as one batch.

    A batch is fixed as soon as it contains ``max_batch_size`` items. If it
    does not fill up, its deadline is ``max_wait_time`` after the arrival of
    its oldest currently waiting item. When the deadline is reached, the
    non-empty batch starts with its current size, even if it is smaller than
    ``min_batch_size``. Thus, ``min_batch_size`` describes the normal minimum
    load, but never causes items to wait past the configured timeout.

    Once a batch is fixed, its members cannot change. Items arriving while a
    batch is queued or being processed join a subsequent batch.

    ``service_time`` may be a number or a callable. A callable receives the
    tuple of items in the batch and must return the processing duration. It is
    called exactly once per batch, rather than once per item.

    """
    def __init__(self, env, max_batch_size=8, min_batch_size=1,
                 max_wait_time=1, service_time=0, capacity=None):
        if capacity is not None:
            max_batch_size = capacity
        if max_batch_size <= 0:
            raise ValueError('"max_batch_size" must be > 0.')
        if min_batch_size <= 0:
            raise ValueError('"min_batch_size" must be > 0.')
        if min_batch_size > max_batch_size:
            raise ValueError('"min_batch_size" must not exceed '
                             '"max_batch_size".')
        if max_wait_time < 0:
            raise ValueError('"max_wait_time" must be >= 0.')

        self._env = env
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size
        self._max_wait_time = max_wait_time
        self._service_time = service_time

        self._waiting = []
        self._queued_batches = []
        self._active_batch = ()
        self._deadline = None
        self._timer_id = 0
        self._batch_ready = env.event()
        self._worker = env.process(self._serve_batches())

    @property
    def capacity(self):
        """Maximum number of items in one batch."""
        return self._max_batch_size

    @property
    def max_batch_size(self):
        """Maximum number of items in one batch."""
        return self._max_batch_size

    @property
    def min_batch_size(self):
        """Normal minimum number of items in one batch."""
        return self._min_batch_size

    @property
    def max_wait_time(self):
        """Maximum time from the oldest waiting item to batch closure."""
        return self._max_wait_time

    @property
    def queue(self):
        """Requests currently eligible for the batch being collected."""
        return self._waiting

    @property
    def active_batch(self):
        """Requests in the batch currently being processed."""
        return self._active_batch

    submit = BoundClass(BatchRequest)
    """Submit an item for batch processing."""

    add = BoundClass(BatchRequest)
    """Alias for :meth:`submit`."""

    put = BoundClass(BatchRequest)
    """Alias for :meth:`submit`."""

    request = BoundClass(BatchRequest)
    """Alias for :meth:`submit`."""

    def withdraw(self, request):
        """Withdraw a waiting item and fail its request.

        Raise :exc:`RuntimeError` if the request already belongs to a fixed
        batch.

        """
        if request.batch is not None or request.triggered:
            raise RuntimeError('Cannot withdraw an item from a fixed batch.')
        self._cancel_request(request)
        request._defused = True
        request.fail(RuntimeError('Batch request was withdrawn.'))
        return request

    def _add_request(self, request):
        is_first = not self._waiting
        self._waiting.append(request)

        if is_first:
            self._schedule_deadline()

        if len(self._waiting) >= self._max_batch_size:
            self._close_batch()

    def _cancel_request(self, request):
        try:
            index = self._waiting.index(request)
        except ValueError:
            raise ValueError('Request is not waiting for a batch.')

        self._waiting.pop(index)
        if not self._waiting:
            self._invalidate_deadline()
        elif index == 0:
            self._schedule_deadline()

    def _invalidate_deadline(self):
        self._timer_id += 1
        self._deadline = None

    def _schedule_deadline(self):
        if not self._waiting:
            self._invalidate_deadline()
            return

        arrival_time = self._waiting[0].time
        self._deadline = arrival_time + self._max_wait_time
        delay = max(0, self._deadline - self._env.now)
        self._timer_id += 1
        timer_id = self._timer_id
        timeout = self._env.timeout(delay)
        timeout.callbacks.append(partial(self._on_deadline, timer_id))

    def _on_deadline(self, timer_id, event):
        if timer_id != self._timer_id or not self._waiting:
            return
        self._close_batch()

    def _close_batch(self):
        requests = self._waiting
        self._waiting = []
        self._invalidate_deadline()

        items = tuple(request.item for request in requests)
        for request in requests:
            request.batch = items
        self._queued_batches.append(requests)

        if not self._batch_ready.triggered:
            ready = self._batch_ready
            self._batch_ready = self._env.event()
            ready.succeed()

    def _serve_batches(self):
        while True:
            if not self._queued_batches:
                self._batch_ready = self._env.event()
                yield self._batch_ready

            requests = self._queued_batches.pop(0)
            items = tuple(request.item for request in requests)
            self._active_batch = requests

            if callable(self._service_time):
                duration = self._service_time(items)
            else:
                duration = self._service_time
            yield self._env.timeout(duration)

            self._active_batch = ()
            for request in requests:
                request.succeed(items)
