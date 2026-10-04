"""
Shared resource with processor sharing scheduling.

Every active request owns an amount of work and receives a share of the
resource capacity proportional to its positive weight.  Unlike
:class:`~simpy.resources.resource.Resource`, requests do not occupy exclusive
slots; all active requests progress concurrently.

"""
import math

from heapq import heapify

from simpy.events import Event, NORMAL
from simpy.exceptions import SimPyException


EPSILON = 1e-12


class ProcessorSharingCanceled(SimPyException):
    """Raised when a processor sharing request is canceled before completion.

    """
    def __init__(self, request):
        self.request = request
        """The canceled request."""
        super(ProcessorSharingCanceled, self).__init__(request)


def _positive_number(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError('"%s" must be a positive number.' % name)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('"%s" must be a positive number.' % name)
    return float(value)


def _non_negative_capacity(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError('"capacity" must be >= 0.')
    if not math.isfinite(value) or value < 0:
        raise ValueError('"capacity" must be >= 0.')
    return float(value)


class ProcessorSharingRequest(Event):
    """An amount of work scheduled on a
    :class:`ProcessorSharingResource`.

    """
    def __init__(self, resource, work, weight=1):
        super(ProcessorSharingRequest, self).__init__(resource._env)
        self.resource = resource
        self.proc = self.env.active_process
        self.work = _positive_number(work, 'work')
        """Total amount of work requested."""
        self.weight = _positive_number(weight, 'weight')
        """Relative share of resource capacity."""
        self.remaining = self.work
        """Amount of work that has not yet been processed."""
        self.service = 0.0
        """Amount of capacity-time already received."""

        resource._add_request(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not GeneratorExit and not self.triggered:
            self.resource.cancel(self)


class _CapacityChange(Event):
    def __init__(self, env, resource, at, capacity):
        super(_CapacityChange, self).__init__(env)
        self._ok = True
        self._value = None
        self.at = at
        self.capacity = capacity
        self.callbacks.append(resource._on_capacity_change)
        env.schedule(self, NORMAL, at - env.now)


class _Completion(Event):
    def __init__(self, env, resource, delay):
        super(_Completion, self).__init__(env)
        self._ok = True
        self._value = None
        self.callbacks.append(resource._on_completion)
        env.schedule(self, NORMAL, delay)


class ProcessorSharingResource(object):
    """Process all active requests concurrently in proportion to weights.

    Capacity may be set to zero to pause processing and later restored.

    """
    def __init__(self, env, capacity=1):
        self._env = env
        self._capacity = _non_negative_capacity(capacity)
        self.users = []
        """Requests that still have outstanding work."""
        self.queue = []
        """Compatibility alias. Active requests never wait for a slot."""
        self._last_update = env.now
        self._next_completion = None
        self._capacity_changes = set()

    @property
    def count(self):
        """Number of requests currently receiving service."""
        return len(self.users)

    @property
    def capacity(self):
        """Total processing capacity."""
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self.set_capacity(value)

    def request(self, work, weight=1):
        """Request processing for *work* with a positive *weight*."""
        return ProcessorSharingRequest(self, work, weight)

    def cancel(self, request):
        """Cancel an unfinished request and fail its event."""
        if request.triggered or request not in self.users:
            return

        self._settle()
        if request.triggered or request not in self.users:
            return

        self.users.remove(request)
        request.fail(ProcessorSharingCanceled(request))
        request._defused = True
        self._schedule_completion()

    release = cancel

    def set_capacity(self, capacity, at=None):
        """Change capacity immediately or at a future simulation time."""
        capacity = _non_negative_capacity(capacity)
        if at is None:
            at = self._env.now
        else:
            if not isinstance(at, (int, float)) or isinstance(at, bool):
                raise ValueError('"at" must be a number.')
            if not math.isfinite(at) or at < self._env.now:
                raise ValueError('"at" must not be in the past.')
            at = float(at)

        if at == self._env.now:
            self._invalidate_capacity_changes(at)
            self._settle()
            self._capacity = capacity
            self._schedule_completion()
            return None

        self._invalidate_capacity_changes(at)
        event = _CapacityChange(self._env, self, at, capacity)
        self._capacity_changes.add(event)
        return event

    def _add_request(self, request):
        self._settle()
        self.users.append(request)
        self._schedule_completion()

    def _settle(self):
        now = self._env.now
        elapsed = now - self._last_update
        self._last_update = now

        if elapsed <= 0 or self._capacity <= 0 or not self.users:
            return

        total_weight = sum(user.weight for user in self.users)
        for user in self.users:
            service = (elapsed * self._capacity * user.weight /
                       total_weight)
            service = min(service, user.remaining)
            user.service += service
            user.remaining -= service

    def _durations(self, total_weight):
        durations = []
        for user in self.users:
            duration = (user.remaining * total_weight /
                        (self._capacity * user.weight))
            durations.append(duration)
        return durations

    def _schedule_completion(self):
        self._invalidate_next_completion()
        if self._capacity <= 0 or not self.users:
            return

        total_weight = sum(user.weight for user in self.users)
        delay = min(self._durations(total_weight))
        if delay <= EPSILON:
            self._complete_due_requests()
            return

        event = _Completion(self._env, self, delay)
        self._next_completion = event

    def _invalidate_next_completion(self):
        event = self._next_completion
        self._next_completion = None
        if event is None or event.processed:
            return

        event.callbacks = []
        for item in self._env._queue:
            if item[3] is event:
                self._env._queue.remove(item)
                heapify(self._env._queue)
                return

        raise RuntimeError('Scheduled completion event not found.')

    def _invalidate_capacity_changes(self, at):
        for event in list(self._capacity_changes):
            if event.at != at or event.processed:
                continue

            event.callbacks = []
            for item in self._env._queue:
                if item[3] is event:
                    self._env._queue.remove(item)
                    heapify(self._env._queue)
                    break
            else:
                raise RuntimeError(
                    'Scheduled capacity change event not found.')
            self._capacity_changes.remove(event)

    def _complete_due_requests(self):
        if self._capacity <= 0 or not self.users:
            return

        total_weight = sum(user.weight for user in self.users)
        durations = self._durations(total_weight)
        nearest = min(durations)
        tolerance = EPSILON * max(1.0, nearest)
        due = [
            user for user, duration in zip(self.users, durations)
            if duration <= nearest + tolerance
        ]

        for user in due:
            self.users.remove(user)
            user.service = user.work
            user.remaining = 0.0
            user.succeed()

    def _on_completion(self, event):
        if event is not self._next_completion:
            return

        self._next_completion = None
        self._settle()
        self._complete_due_requests()
        self._schedule_completion()

    def _on_capacity_change(self, event):
        self._capacity_changes.discard(event)
        self._settle()
        self._capacity = event.capacity
        self._schedule_completion()
