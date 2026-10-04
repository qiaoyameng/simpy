"""
Passive time-weighted statistics for resources.

The observers in this module integrate piecewise constant resource state over
simulation time.  They do not schedule timeouts or any other events.
"""
from collections import namedtuple
from functools import wraps


_RESULT_FIELDS = (
    'start', 'end', 'elapsed', 'mean_queue_length', 'mean_usage',
    'mean_utilization',
)


class ResourceStatsResult(namedtuple('ResourceStatsResult', _RESULT_FIELDS)):
    """Time-weighted resource statistics for one observation interval."""
    __slots__ = ()

    @property
    def queue_length(self):
        """Alias for :attr:`mean_queue_length`."""
        return self.mean_queue_length

    @property
    def usage(self):
        """Alias for :attr:`mean_usage`."""
        return self.mean_usage

    @property
    def count(self):
        """Alias for :attr:`mean_usage`."""
        return self.mean_usage

    @property
    def mean_count(self):
        """Alias for :attr:`mean_usage`."""
        return self.mean_usage


_BATCH_FIELDS = (
    'index', 'start', 'end', 'elapsed', 'mean_queue_length', 'mean_usage',
    'mean_utilization', 'complete',
)


class ResourceStatsBatch(
        namedtuple('ResourceStatsBatch', _BATCH_FIELDS)):
    """Time-weighted statistics for a single fixed-length batch."""
    __slots__ = ()

    @property
    def queue_length(self):
        """Alias for :attr:`mean_queue_length`."""
        return self.mean_queue_length

    @property
    def usage(self):
        """Alias for :attr:`mean_usage`."""
        return self.mean_usage

    @property
    def count(self):
        """Alias for :attr:`mean_usage`."""
        return self.mean_usage

    @property
    def partial(self):
        """``True`` if the batch did not cover its full interval."""
        return not self.complete


class ResourceStats(object):
    """Observe time-weighted queue length and usage of a resource.

    *resource* must expose ``queue``, ``count`` and ``capacity``, as
    :class:`~simpy.resources.resource.Resource` does.  The observer patches
    the resource instance and collects state only when an operation or an
    already scheduled resource event is processed.

    *start* (or its alias *warmup*) is the beginning of the observation
    window.  *end* optionally limits the window.  *batch_size* enables fixed
    simulation-time batch means.  Batch means are descriptive means only; this
    class does not calculate variances or confidence intervals.
    """
    def __init__(self, resource, start=None, warmup=None, end=None,
                 batch_size=None):
        if start is not None and warmup is not None and start != warmup:
            raise ValueError('"start" and "warmup" must be equal.')
        if start is None:
            start = warmup
        if start is None:
            start = resource._env.now
        if end is not None and end < start:
            raise ValueError('"end" must not be before "start".')
        if batch_size is not None and batch_size <= 0:
            raise ValueError('"batch_size" must be > 0.')

        self.resource = resource
        self.env = resource._env
        if start < self.env.now:
            raise ValueError('"start" must not be in the past.')
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.capacity = float(resource.capacity)

        self._known_state = self._state()
        self._last_state = None
        self._last_time = None
        self._batch_index = None
        self._closed = False
        self._queue_area = 0.0
        self._usage_area = 0.0
        self._batch_queue_area = 0.0
        self._batch_usage_area = 0.0
        self._batches = []

        self._patch_resource()

    def _state(self):
        return float(len(self.resource.queue)), float(self.resource.count)

    def _patch_resource(self):
        for name in ('request', 'release', 'put', 'get'):
            original = getattr(self.resource, name, None)
            if not callable(original):
                continue

            def make_wrapper(original):
                @wraps(original)
                def wrapper(*args, **kwargs):
                    self._record(self._state())
                    result = original(*args, **kwargs)
                    self._record(self._state())
                    if getattr(result, 'callbacks', None) is not None:
                        result.callbacks.append(self._event_processed)
                        self._wrap_cancel(result)
                    return result

                wrapper.__wrapped_operation__ = original
                return wrapper

            setattr(self.resource, name, make_wrapper(original))

    def _wrap_cancel(self, event):
        if event.triggered or not callable(getattr(event, 'cancel', None)):
            return
        original = event.cancel

        @wraps(original)
        def cancel_wrapper(*args, **kwargs):
            self._record(self._state())
            try:
                return original(*args, **kwargs)
            finally:
                self._record(self._state())

        event.cancel = cancel_wrapper

    def _event_processed(self, event):
        self._record(self._state())

    def _record(self, state):
        now = self.env.now
        self._known_state = state
        if now < self.start or self._closed:
            return
        if self._last_state is None:
            self._last_state = state
            self._last_time = self.start
            self._batch_index = 0

        window_end = now if self.end is None else min(now, self.end)
        self._accumulate(window_end, state)
        self._last_state = state
        self._last_time = now

        if self.end is not None and now >= self.end:
            boundary = self._batch_boundary(self._batch_index)
            complete = self.batch_size is None or self.end >= boundary
            self._finish_batch(self.end, complete=complete)
            self._closed = True

    def _accumulate(self, time, state):
        """Accumulate the current constant segment up to *time*."""
        if time <= self._last_time:
            return

        queue_length, usage = self._last_state
        while self._last_time < time:
            boundary = self._batch_boundary(self._batch_index)
            split = time if self.end is None else min(time, self.end)
            if self.batch_size is not None:
                split = min(split, boundary)
            duration = split - self._last_time

            self._queue_area += queue_length * duration
            self._usage_area += usage * duration
            self._batch_queue_area += queue_length * duration
            self._batch_usage_area += usage * duration
            self._last_time = split

            if self.batch_size is not None and split == boundary:
                self._finish_batch(boundary, complete=split < time or
                                   self.end is None or self.end >= boundary)
                self._batch_index += 1
                self._batch_queue_area = 0.0
                self._batch_usage_area = 0.0

        self._last_state = state

    def _batch_boundary(self, index):
        if self.batch_size is None:
            return None
        return self.start + (index + 1) * self.batch_size

    def _batch_start(self, index):
        if self.batch_size is None:
            return self.start
        return self.start + index * self.batch_size

    def _finish_batch(self, end, complete):
        start = self._batch_start(self._batch_index)
        if end <= start:
            return
        elapsed = end - start
        complete = bool(complete and end >= self._batch_boundary(
            self._batch_index))
        self._batches.append(self._batch(start, end, elapsed,
                                         self._batch_queue_area,
                                         self._batch_usage_area, complete))

    def _result(self, start, end, queue_area, usage_area, elapsed):
        if elapsed == 0:
            mean_queue = mean_usage = mean_utilization = 0.0
        else:
            mean_queue = queue_area / elapsed
            mean_usage = usage_area / elapsed
            mean_utilization = usage_area / (self.capacity * elapsed)
        return ResourceStatsResult(start, end, elapsed, mean_queue,
                                   mean_usage, mean_utilization)

    def _batch(self, start, end, elapsed, queue_area, usage_area, complete):
        mean_queue = queue_area / elapsed
        mean_usage = usage_area / elapsed
        mean_utilization = usage_area / (self.capacity * elapsed)
        return ResourceStatsBatch(
            len(self._batches), start, end, elapsed, mean_queue, mean_usage,
            mean_utilization, complete)

    def _observation_end(self, at=None):
        if at is None:
            at = self.env.now
        if self.end is not None:
            at = min(at, self.end)
        return at

    def result(self, at=None):
        """Return integrated means without advancing simulation time."""
        end = self._observation_end(at)
        if end < self.start:
            raise ValueError('"at" must not be before the observation '
                             'start.')
        if end == self.start:
            return self._result(self.start, self.start, 0.0, 0.0, 0.0)
        if self._closed:
            elapsed = end - self.start
            return self._result(self.start, end, self._queue_area,
                                self._usage_area, elapsed)

        saved = self.__dict__.copy()
        try:
            if self._last_state is None:
                self._last_state = self._known_state
                self._last_time = self.start
                self._batch_index = 0
            if end > self._last_time:
                self._accumulate(end, self._last_state)
            elif end < self._last_time:
                raise ValueError('"at" must not be before a recorded event.')
            elapsed = end - self.start
            return self._result(self.start, end, self._queue_area,
                                self._usage_area, elapsed)
        finally:
            self.__dict__.clear()
            self.__dict__.update(saved)

    def batch_means(self, at=None):
        """Return fixed-length batch means up to the current time."""
        if self.batch_size is None:
            raise ValueError('No "batch_size" was configured.')

        end = self._observation_end(at)
        if end < self.start:
            raise ValueError('"at" must not be before the observation '
                             'start.')
        if end == self.start:
            return []
        if self._closed:
            return list(self._batches)

        saved = self.__dict__.copy()
        try:
            if self._last_state is None:
                self._last_state = self._known_state
                self._last_time = self.start
                self._batch_index = 0
            if end > self._last_time:
                self._accumulate(end, self._last_state)
            elif end < self._last_time:
                raise ValueError('"at" must not be before a recorded event.')

            batches = list(self._batches)
            batch_start = self._batch_start(self._batch_index)
            if not self._closed and end > batch_start:
                boundary = self._batch_boundary(self._batch_index)
                elapsed = end - batch_start
                partial = self._batch(
                    batch_start, end, elapsed, self._batch_queue_area,
                    self._batch_usage_area, end >= boundary)
                batches.append(partial)
            return batches
        finally:
            self.__dict__.clear()
            self.__dict__.update(saved)

    @property
    def mean_queue_length(self):
        """Time-weighted mean queue length at the current time."""
        return self.result().mean_queue_length

    @property
    def mean_usage(self):
        """Time-weighted mean resource usage at the current time."""
        return self.result().mean_usage

    @property
    def mean_utilization(self):
        """Time-weighted mean resource utilization at the current time."""
        return self.result().mean_utilization

    @property
    def elapsed(self):
        """Observed elapsed simulation time."""
        return self.result().elapsed


ResourceObserver = ResourceStats
