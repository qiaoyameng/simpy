"""
Resource for storing perishable goods in batches with individual expiration
times (e.g. crates in a cold-chain warehouse).

A :class:`PerishableStore` issues goods in FEFO (first-expired, first-out)
order.  A get request for a quantity may be fulfilled from several batches and
receives a breakdown of the batches actually used.  Batches whose expiration
time is reached are scrapped automatically.

"""
from heapq import heappush, heappop
from itertools import count

from simpy.core import BoundClass
from simpy.events import Event, URGENT
from simpy.resources import base


class Batch(object):
    """A batch of *quantity* units of perishable goods expiring at *expiry*.

    Batches are created by :meth:`PerishableStore.put`.  The :attr:`quantity`
    attribute always reflects the remaining (not yet issued or scrapped)
    amount of the batch.

    """
    __slots__ = ('id', 'expiry', 'quantity')

    def __init__(self, id, expiry, quantity):
        self.id = id
        """Sequence number of the batch within its store."""
        self.expiry = expiry
        """Simulation time at which the batch perishes."""
        self.quantity = quantity
        """Remaining quantity of the batch."""

    def __repr__(self):
        return '%s(id=%s, expiry=%s, quantity=%s)' % (
            self.__class__.__name__, self.id, self.expiry, self.quantity)


class PerishablePut(base.Put):
    """Request to put *amount* units expiring at *expiry* into the *store*.
    The request is triggered once there is enough space in the store.  Its
    value is the created :class:`Batch`.

    Raise a :exc:`ValueError` if ``amount <= 0`` or ``expiry <= env.now``.

    """
    def __init__(self, store, amount, expiry):
        if amount <= 0:
            raise ValueError('amount(=%s) must be > 0.' % amount)
        if expiry <= store._env.now:
            raise ValueError('expiry(=%s) must be > the current time (%s).' %
                             (expiry, store._env.now))
        self.amount = amount
        """The amount of goods to be put into the store."""
        self.expiry = expiry
        """The expiration time of the goods to be put into the store."""

        super(PerishablePut, self).__init__(store)


class PerishableGet(base.Get):
    """Request to get *amount* units from the *store* in FEFO order.

    The request is triggered once enough unexpired goods are available.  It is
    fulfilled atomically: either the whole *amount* is issued or the request
    keeps waiting; no partial amount is ever deducted for a pending request.
    Its value is a list of ``(batch, amount)`` tuples describing which
    batches the goods were taken from.

    Raise a :exc:`ValueError` if ``amount <= 0``.

    """
    def __init__(self, store, amount):
        if amount <= 0:
            raise ValueError('amount(=%s) must be > 0.' % amount)
        self.amount = amount
        """The amount of goods to be taken out of the store."""

        super(PerishableGet, self).__init__(store)


class PerishableStore(base.BaseResource):
    """Resource storing perishable goods in batches with individual
    expiration times.

    Goods are issued in FEFO (first-expired, first-out) order: a get request
    for an amount is fulfilled from the earliest expiring batches first and
    may be split across several batches.  The value of a successful get
    request is the breakdown of the batches used, a list of
    ``(batch, amount)`` tuples.

    Batches perish automatically once the simulation time reaches their
    *expiry*.  Expiration at a time *t* is processed before any get request
    fulfilled at the same time *t*, so goods expiring at *t* can no longer be
    issued at *t*.  Scrapped and fully issued batches are removed from the
    store, so their metadata does not accumulate.

    The *env* parameter is the :class:`~simpy.core.Environment` instance the
    store is bound to.  The *capacity* limits the total amount of goods in
    the store and is unlimited by default.

    The amounts of all inbound (:attr:`total_put`), outbound
    (:attr:`total_get`) and scrapped (:attr:`total_scrapped`) goods are
    tracked so that the conservation invariant ``total_put == total_get +
    total_scrapped + level`` can be checked at any time.

    Raise a :exc:`ValueError` if ``capacity <= 0``.

    """
    def __init__(self, env, capacity=float('inf')):
        if capacity <= 0:
            raise ValueError('"capacity" must be > 0.')

        super(PerishableStore, self).__init__(env, capacity)

        self._batches = []  # Heap of (expiry, seq, batch) of live batches.
        self._seq = count()  # Tie-breaker for batches with the same expiry.
        self._level = 0
        self._expiry_event = None  # Pending event for the nearest expiry.
        self._expiry_event_time = None

        self.total_put = 0
        """Total amount of goods ever put into the store."""
        self.total_get = 0
        """Total amount of goods ever issued by the store."""
        self.total_scrapped = 0
        """Total amount of goods ever scrapped because they perished."""

    @property
    def level(self):
        """The current total amount of unexpired goods in the store."""
        return self._level

    @property
    def batches(self):
        """The live batches in the store, ordered by expiration time."""
        return [batch for expiry, seq, batch in sorted(self._batches)]

    put = BoundClass(PerishablePut)
    """Request to put *amount* units expiring at *expiry* into the store."""

    get = BoundClass(PerishableGet)
    """Request to get *amount* units out of the store in FEFO order."""

    def _do_put(self, event):
        if self._capacity - self._level >= event.amount:
            batch = Batch(next(self._seq), event.expiry, event.amount)
            heappush(self._batches, (batch.expiry, batch.id, batch))
            self._level += event.amount
            self.total_put += event.amount
            if (self._expiry_event is None or
                    batch.expiry < self._expiry_event_time):
                self._schedule_expiry(batch.expiry)
            event.succeed(batch)
            return True

    def _do_get(self, event):
        if self._level >= event.amount:
            # Deduct atomically in FEFO order.  Batches are only touched now
            # that the whole amount is known to be available, so a request
            # that cannot be fulfilled never reserves or deducts anything.
            remaining = event.amount
            breakdown = []
            while remaining > 0:
                expiry, seq, batch = heappop(self._batches)
                taken = min(batch.quantity, remaining)
                batch.quantity -= taken
                remaining -= taken
                breakdown.append((batch, taken))
                if batch.quantity > 0:
                    heappush(self._batches, (expiry, seq, batch))
                # Fully used batches are dropped, releasing their metadata.
            self._level -= event.amount
            self.total_get += event.amount
            event.succeed(breakdown)
            return True

    def _schedule_expiry(self, at):
        """Schedule the expiry event for the nearest expiration time *at*.

        At most one expiry event is pending at any time.  If an event for a
        later expiration is already pending, it is superseded and becomes a
        no-op once processed.

        """
        expiry_event = Event(self._env)
        expiry_event._ok = True
        expiry_event._value = None
        expiry_event.callbacks.append(self._expire)
        # URGENT priority makes sure that goods expiring at a time t are
        # scrapped before processes (which resume with NORMAL priority) can
        # issue get requests at t.
        self._env.schedule(expiry_event, URGENT, at - self._env.now)
        self._expiry_event = expiry_event
        self._expiry_event_time = at

    def _expire(self, event):
        if event is not self._expiry_event:
            return  # Superseded by an earlier scheduled expiry event.

        now = self._env.now
        while self._batches and self._batches[0][0] <= now:
            expiry, seq, batch = heappop(self._batches)
            self._level -= batch.quantity
            self.total_scrapped += batch.quantity
            # Scrapped batches are dropped, releasing their metadata.

        if self._batches:
            self._schedule_expiry(self._batches[0][0])
        else:
            self._expiry_event = None
            self._expiry_event_time = None
