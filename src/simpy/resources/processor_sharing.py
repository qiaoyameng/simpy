"""
Weighted processor-sharing resource.

A :class:`ProcessorSharingResource` models a shared processing node (e.g. a
CPU or a network link) with a total service *capacity* that is shared
*simultaneously* between all active requests. Each request carries an amount
of *work* (e.g. FLOPs or bytes) and a positive *weight*. At any moment, a
request receives service at the rate::

    capacity * request.weight / sum(weights of all active requests)

The request event is triggered once its complete work amount has been
served. Requests can be cancelled at any time; the freed capacity is
redistributed among the surviving requests. The total capacity can be
changed at any simulation time via :meth:`ProcessorSharingResource.set_capacity`;
setting it to ``0`` pauses all service until a positive capacity is
restored.

"""
from simpy.core import BoundClass
from simpy.events import Event


class ProcessorSharingRequest(Event):
    """Request *work* units of service from the *resource* with a positive
    *weight*. Subclass of :class:`simpy.events.Event`.

    The event is triggered once *work* units of service have been delivered.
    While the request is active, :attr:`remaining` tracks the amount of work
    still to be served and :attr:`served` the amount already delivered.

    The request can be used as a context manager; leaving the
    :keyword:`with` block cancels the request if it is still pending:

    .. code-block:: python

        with node.request(work=5, weight=2) as req:
            yield req

    """
    def __init__(self, resource, work, weight):
        if work < 0:
            raise ValueError('work(=%s) must be >= 0.' % work)
        if weight <= 0:
            raise ValueError('weight(=%s) must be > 0.' % weight)
        super(ProcessorSharingRequest, self).__init__(resource._env)
        self.resource = resource
        """The :class:`ProcessorSharingResource` the request belongs to."""
        self.proc = self.env.active_process
        """The process that created the request."""
        self.work = work
        """The total amount of work requested."""
        self.weight = weight
        """The positive sharing weight of the request."""
        self.remaining = work
        """The amount of work not yet served."""

        resource._add(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cancel()

    @property
    def served(self):
        """The amount of work already served to this request."""
        return self.work - self.remaining

    def cancel(self):
        """Cancel the request.

        The request is removed from the resource and its share of the
        capacity is redistributed among the surviving requests. The event
        itself is never triggered, so it must no longer be yielded.

        """
        if not self.triggered:
            self.resource._remove(self)


class ProcessorSharingResource(object):
    """A resource that shares its total service *capacity* between all
    active requests proportionally to their weights.

    *capacity* is the total service rate (work units per simulation time
    unit). It may be ``0``, in which case all service is paused. Use
    :meth:`set_capacity` to change the capacity at the current or at a
    scheduled future simulation time.

    """
    def __init__(self, env, capacity):
        if capacity < 0:
            raise ValueError('capacity(=%s) must be >= 0.' % capacity)
        self._env = env
        self._capacity = capacity
        self._requests = []
        self._total_weight = 0.0
        self._last_update = env.now
        # Bumped on every state change to invalidate scheduled events.
        self._generation = 0

        BoundClass.bind_early(self)

    @property
    def capacity(self):
        """The current total service rate of the resource."""
        return self._capacity

    @property
    def count(self):
        """The number of active requests."""
        return len(self._requests)

    @property
    def requests(self):
        """The list of active :class:`ProcessorSharingRequest` events."""
        return list(self._requests)

    request = BoundClass(ProcessorSharingRequest)
    """Request service from the resource. Returns a
    :class:`ProcessorSharingRequest` event that is triggered once the
    requested amount of work has been served."""

    def set_capacity(self, capacity, at=None):
        """Set the total service rate to *capacity*.

        The change is applied at the simulation time *at* (default: now).
        *at* must not lie in the past. Setting the capacity to ``0`` pauses
        all service; restoring a positive capacity resumes it.

        """
        if capacity < 0:
            raise ValueError('capacity(=%s) must be >= 0.' % capacity)
        now = self._env.now
        if at is None or at == now:
            self._set_capacity(capacity)
        elif at < now:
            raise ValueError('at(=%s) must be >= the current simulation '
                             'time.' % at)
        else:
            event = self._env.timeout(at - now)
            event.callbacks.append(
                lambda event, cap=capacity: self._set_capacity(cap))

    def _settle(self):
        """Account for the service delivered since the last state change.

        Deduct the work served during the elapsed interval at the rates
        valid during that interval from all active requests.

        """
        now = self._env.now
        elapsed = now - self._last_update
        if elapsed > 0 and self._capacity > 0 and self._total_weight > 0:
            for req in self._requests:
                # Divide last to keep the result exact for nice numbers.
                req.remaining -= (self._capacity * elapsed * req.weight /
                                  self._total_weight)
        self._last_update = now

    def _reschedule(self):
        """Schedule the next completion event.

        Invalidates any previously scheduled completion event and computes
        the nearest point in time at which one or more requests complete
        under the current rates. All requests completing at that same
        instant are collected so they can be settled together.

        """
        self._generation += 1
        if self._capacity <= 0 or not self._requests:
            return
        total_weight = self._total_weight
        capacity = self._capacity
        delays = [req.remaining * total_weight / (capacity * req.weight)
                  for req in self._requests]
        delay = min(delays)
        # Collect all requests completing at (nearly) the same instant.
        tol = abs(delay) * 1e-12 + 1e-15
        due = [req for req, req_delay in zip(self._requests, delays)
               if req_delay <= delay + tol]
        generation = self._generation
        event = self._env.timeout(max(0.0, delay))
        event.callbacks.append(
            lambda event, gen=generation, due=due: self._complete(gen, due))

    def _complete(self, generation, due):
        """Settle all requests in *due* that complete at the current time.

        Events scheduled before the most recent state change are stale and
        must not trigger any request.

        """
        if generation != self._generation:
            return  # Stale event; the state has changed since it was
            # scheduled and a newer completion event exists.
        self._settle()
        for req in due:
            if req in self._requests:
                self._requests.remove(req)
                self._total_weight -= req.weight
                req.remaining = 0.0
                req.succeed()
        self._reschedule()

    def _add(self, req):
        """Activate *req* and redistribute the capacity."""
        self._settle()
        if req.work == 0:
            req.succeed()
            return
        self._requests.append(req)
        self._total_weight += req.weight
        self._reschedule()

    def _remove(self, req):
        """Remove *req* and redistribute its share of the capacity."""
        self._settle()
        self._requests.remove(req)
        self._total_weight -= req.weight
        self._reschedule()

    def _set_capacity(self, capacity):
        """Apply a capacity change at the current simulation time."""
        self._settle()
        self._capacity = capacity
        self._reschedule()
