"""Time-weighted statistics for :class:`~simpy.Resource` instances."""

from collections import namedtuple
from bisect import bisect_right
from functools import wraps
import math

from simpy.resources.resource import Resource


class ResourceTimeAverage(namedtuple('ResourceTimeAverage',
                                    ('start', 'end', 'queue_length',
                                     'occupancy'))):
    """Time average of a resource's queue length and occupancy."""
    __slots__ = ()

    @property
    def count(self):
        """Alias for :attr:`occupancy`."""
        return self.occupancy


class ResourceArea(namedtuple('ResourceArea',
                              ('duration', 'queue_length', 'occupancy'))):
    """Integrated queue length and occupancy over an observation interval."""
    __slots__ = ()

    @property
    def count(self):
        """Alias for :attr:`occupancy`."""
        return self.occupancy


class ResourceBatch(namedtuple('ResourceBatch',
                               ('index', 'start', 'end', 'queue_length',
                                'occupancy', 'complete'))):
    """Time averages for one fixed-length simulation-time batch."""
    __slots__ = ()

    @property
    def count(self):
        """Alias for :attr:`occupancy`."""
        return self.occupancy

    @property
    def partial(self):
        """``True`` if the batch does not span a complete interval."""
        return not self.complete


def _make_method_wrapper(resource, original):
    @wraps(original)
    def wrapper(*args, **kwargs):
        observers = list(resource._simpy_observers)
        try:
            event = original(*args, **kwargs)
        except Exception:
            for observer in observers:
                observer._record()
            raise

        for observer in observers:
            observer._observe_event(event)
        return event

    return wrapper


class ResourceObserver(object):
    """Observe time-weighted queue length and occupancy of *resource*.

    The observer records state after resource operations and their event
    callbacks have been processed. Multiple state changes at the same
    simulation time contribute no area; only the final state at that time
    is retained.

    *start* may discard a warm-up period. The observer still has to be
    attached before that period because it needs the initial resource state.
    Reading statistics neither schedules events nor advances simulation time.
    """
    def __init__(self, resource, start=None):
        if not isinstance(resource, Resource):
            raise TypeError('resource must be a Resource instance.')

        self._resource = resource
        self._env = resource._env
        attach_time = self._env.now

        if start is None:
            start = attach_time
        elif start < attach_time:
            raise ValueError(
                'start(=%s) must not be earlier than the attachment '
                'time(=%s).' % (start, attach_time))

        self._window_start = start
        self._attach_time = attach_time
        self._closed = False
        self._closed_at = None

        self._times = [attach_time]
        self._queue_lengths = [len(resource.queue)]
        self._occupancies = [resource.count]

        if not hasattr(resource, '_simpy_observers'):
            originals = {}
            for name in ('request', 'release'):
                originals[name] = getattr(resource, name)

            resource._simpy_observers = []
            resource._simpy_observer_originals = originals
            for name, original in originals.items():
                setattr(resource, name,
                        _make_method_wrapper(resource, original))

        resource._simpy_observers.append(self)

        for event in list(resource.queue):
            self._observe_event(event)

    @property
    def start(self):
        """Default start time for statistics."""
        return self._window_start

    @property
    def resource(self):
        """The observed resource."""
        return self._resource

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Stop observing and restore the resource's original methods."""
        if self._closed:
            return

        self._closed = True
        self._closed_at = self._env.now
        resource = self._resource
        resource._simpy_observers.remove(self)

        if not resource._simpy_observers:
            for name, original in \
                    resource._simpy_observer_originals.items():
                setattr(resource, name, original)
            del resource._simpy_observer_originals
            del resource._simpy_observers

    def _observe_event(self, event):
        if self._closed:
            return

        self._record()

        if event.callbacks is not None:
            event.callbacks.append(self._event_callback)

        original_cancel = event.cancel

        @wraps(original_cancel)
        def cancel_wrapper(*args, **kwargs):
            try:
                return original_cancel(*args, **kwargs)
            finally:
                self._record()

        event.cancel = cancel_wrapper

    def _event_callback(self, event):
        self._record()

    def _record(self):
        if self._closed:
            return

        now = self._env.now
        queue_length = len(self._resource.queue)
        occupancy = self._resource.count

        if now == self._times[-1]:
            self._queue_lengths[-1] = queue_length
            self._occupancies[-1] = occupancy
        else:
            self._times.append(now)
            self._queue_lengths.append(queue_length)
            self._occupancies.append(occupancy)

    def _resolve_interval(self, start, end):
        if start is None:
            start = self._window_start
        if end is None:
            end = self._env.now

        if start < self._attach_time:
            raise ValueError(
                'start(=%s) must not be earlier than the attachment '
                'time(=%s).' % (start, self._attach_time))
        if end <= start:
            raise ValueError('end(=%s) must be later than start(=%s).'
                             % (end, start))
        if self._closed_at is not None and end > self._closed_at:
            raise RuntimeError(
                'end(=%s) is after the observer was closed at %s.'
                % (end, self._closed_at))
        if end > self._env.now:
            next_event = self._env.peek()
            if end > next_event:
                raise RuntimeError(
                    'end(=%s) crosses an unprocessed event at %s. Run the '
                    'simulation to the end of the observation window first.'
                    % (end, next_event))

        return start, end

    def _areas_for_interval(self, start, end, state_index=None):
        queue_area = 0.0
        occupancy_area = 0.0
        if state_index is None:
            state_index = bisect_right(self._times, start) - 1
        if state_index < 0:
            raise ValueError(
                'start(=%s) is earlier than the first observation.' % start)

        current = start
        while current < end:
            if state_index + 1 < len(self._times):
                next_change = self._times[state_index + 1]
            else:
                next_change = float('inf')

            segment_end = min(end, next_change)
            if segment_end > current:
                duration = segment_end - current
                queue_area += (self._queue_lengths[state_index] *
                               duration)
                occupancy_area += (self._occupancies[state_index] *
                                   duration)
                current = segment_end

            if next_change <= current:
                state_index += 1

        return (ResourceArea(end - start, queue_area, occupancy_area),
                state_index)

    def areas(self, start=None, end=None):
        """Return integrated queue-length-time and occupancy-time."""
        start, end = self._resolve_interval(start, end)
        return self._areas_for_interval(start, end)[0]

    def time_average(self, start=None, end=None):
        """Return queue length and occupancy averaged by state duration."""
        start, end = self._resolve_interval(start, end)
        areas = self._areas_for_interval(start, end)[0]
        duration = areas.duration
        return ResourceTimeAverage(
            start, end,
            areas.queue_length / duration,
            areas.occupancy / duration)

    def batch_time_averages(self, interval, start=None, end=None):
        """Return time averages for fixed simulation-time batches."""
        if interval <= 0:
            raise ValueError('interval(=%s) must be > 0.' % interval)
        if math.isinf(float(interval)):
            raise ValueError('interval must be finite.')

        start, end = self._resolve_interval(start, end)
        duration = end - start
        ratio = duration / interval
        full_batches = int(math.floor(ratio))

        rounded = int(round(ratio))
        if rounded >= 1 and abs(ratio - rounded) <= 1e-12:
            full_batches = rounded
            if full_batches:
                end = start + full_batches * interval

        batches = []
        state_index = bisect_right(self._times, start) - 1
        if state_index < 0:
            raise ValueError(
                'start(=%s) is earlier than the first observation.' % start)

        for batch_index in range(full_batches):
            batch_start = start + batch_index * interval
            batch_end = min(end, batch_start + interval)
            areas, state_index = self._areas_for_interval(
                batch_start, batch_end, state_index)
            batches.append(ResourceBatch(
                batch_index, batch_start, batch_end,
                areas.queue_length / areas.duration,
                areas.occupancy / areas.duration,
                True))

        final_start = start + full_batches * interval
        if end - final_start > interval * 1e-12:
            areas, state_index = self._areas_for_interval(
                final_start, end, state_index)
            batches.append(ResourceBatch(
                full_batches, final_start, end,
                areas.queue_length / areas.duration,
                areas.occupancy / areas.duration,
                False))

        return batches

    mean = time_average
    batch_means = batch_time_averages
