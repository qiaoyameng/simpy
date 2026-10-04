"""
Tests for the FEFO (first-expired, first-out) perishable inventory resource.

"""
import pytest

import simpy


def test_put_get_bounds(env):
    store = simpy.PerishableStore(env)
    pytest.raises(ValueError, simpy.PerishableStore, env, 0)
    pytest.raises(ValueError, store.put, 0, 10)
    pytest.raises(ValueError, store.put, -1, 10)
    pytest.raises(ValueError, store.put, 1, 0)  # expiry must be > env.now
    pytest.raises(ValueError, store.get, 0)
    pytest.raises(ValueError, store.get, -1)


def test_fefo_order_and_breakdown(env):
    """Goods are issued earliest-expiry first, possibly split over batches,
    and the get request returns the batches actually used."""
    store = simpy.PerishableStore(env)
    log = []

    def pem():
        b1 = yield store.put(5, expiry=30)
        b2 = yield store.put(5, expiry=10)
        b3 = yield store.put(5, expiry=20)
        breakdown = yield store.get(8)
        log.append([(batch.id, amount) for batch, amount in breakdown])
        assert [batch for batch, amount in breakdown] == [b2, b3]
        assert [amount for batch, amount in breakdown] == [5, 3]
        assert b2.quantity == 0
        assert b3.quantity == 2
        assert store.level == 7

    env.process(pem())
    env.run()
    assert log == [[(1, 5), (2, 3)]]


def test_same_expiry_stable_order(env):
    """Batches with the same expiry are used in the order they were put."""
    store = simpy.PerishableStore(env)
    log = []

    def pem():
        b1 = yield store.put(4, expiry=10)
        b2 = yield store.put(4, expiry=10)
        breakdown = yield store.get(6)
        log.append([batch.id for batch, amount in breakdown])
        assert [batch for batch, amount in breakdown] == [b1, b2]
        assert [amount for batch, amount in breakdown] == [4, 2]

    env.process(pem())
    env.run()
    assert log == [[0, 1]]


def test_get_is_atomic(env):
    """A get that cannot be fulfilled completely deducts nothing and waits
    until enough goods are available."""
    store = simpy.PerishableStore(env)
    log = []

    def stocker():
        yield store.put(3, expiry=50)
        yield env.timeout(5)
        yield store.put(4, expiry=60)

    def getter():
        breakdown = yield store.get(5)
        log.append((env.now, [amount for batch, amount in breakdown]))

    def probe():
        yield env.timeout(1)
        assert store.level == 3  # nothing deducted for the pending get
        assert store.total_get == 0

    env.process(stocker())
    env.process(getter())
    env.process(probe())
    env.run(until=10)
    assert log == [(5, [3, 2])]
    assert store.level == 2


def test_get_exceeding_stock_waits(env):
    """A get for more than the available stock stays pending and does not
    touch the stock."""
    store = simpy.PerishableStore(env)

    def pem():
        yield store.put(6, expiry=50)
        get = store.get(10)
        yield env.timeout(5)
        assert not get.triggered
        assert store.level == 6
        get.cancel()

    env.process(pem())
    env.run(until=10)
    assert store.level == 6
    assert store.total_get == 0


def test_cancelled_get_reserves_nothing(env):
    """Cancelling a waiting get leaves all stock available for others."""
    store = simpy.PerishableStore(env)

    def pem():
        yield store.put(6, expiry=50)
        get = store.get(10)  # cannot be fulfilled from current stock
        yield env.timeout(1)
        get.cancel()
        yield env.timeout(1)
        breakdown = yield store.get(6)
        assert sum(amount for batch, amount in breakdown) == 6

    env.process(pem())
    env.run()
    assert store.level == 0
    assert store.total_get == 6


def test_expiry(env):
    """Batches are scrapped automatically when they perish."""
    store = simpy.PerishableStore(env)

    def pem():
        yield store.put(4, expiry=5)
        yield store.put(6, expiry=10)
        yield env.timeout(5)
        assert store.level == 6
        assert store.total_scrapped == 4
        assert len(store.batches) == 1
        yield env.timeout(5)
        assert store.level == 0
        assert store.total_scrapped == 10
        assert store.batches == []

    env.process(pem())
    env.run()


def test_same_expiry_multiple_batches(env):
    """All batches expiring at the same time are scrapped together."""
    store = simpy.PerishableStore(env)

    def pem():
        yield store.put(3, expiry=5)
        yield store.put(7, expiry=5)
        yield store.put(2, expiry=8)
        yield env.timeout(5)
        assert store.level == 2
        assert store.total_scrapped == 10
        assert [batch.expiry for batch in store.batches] == [8]

    env.process(pem())
    env.run()


def test_expiration_precedes_same_time_get(env):
    """Goods expiring at time t are scrapped before gets fulfilled at t."""
    store = simpy.PerishableStore(env)
    log = []

    def stocker():
        yield store.put(5, expiry=10)

    def getter():
        yield env.timeout(10)
        breakdown = yield store.get(5)
        log.append(breakdown)

    env.process(stocker())
    env.process(getter())
    env.run(until=20)
    assert log == []  # the get is still pending
    assert store.total_scrapped == 5
    assert store.level == 0


def test_get_before_expiry_uses_batch(env):
    """A get fulfilled strictly before the expiry uses the batch."""
    store = simpy.PerishableStore(env)
    log = []

    def stocker():
        yield store.put(5, expiry=10)

    def getter():
        yield env.timeout(9)
        breakdown = yield store.get(5)
        log.append([amount for batch, amount in breakdown])

    env.process(stocker())
    env.process(getter())
    env.run()
    assert log == [[5]]
    assert store.total_scrapped == 0


def test_expired_batches_release_metadata(env):
    """Scrapped batches do not accumulate in the store."""
    store = simpy.PerishableStore(env)

    def pem():
        for i in range(100):
            yield store.put(1, expiry=env.now + 1)
            yield env.timeout(1)

    env.process(pem())
    env.run()
    assert store.batches == []
    assert store.level == 0
    assert store.total_put == 100
    assert store.total_scrapped == 100


def test_consumed_batches_release_metadata(env):
    """Fully issued batches are removed from the store."""
    store = simpy.PerishableStore(env)

    def pem():
        for i in range(100):
            yield store.put(1, expiry=1000)
        breakdown = yield store.get(100)
        assert len(breakdown) == 100

    env.process(pem())
    env.run()
    assert store.batches == []
    assert store.level == 0


def test_capacity(env):
    store = simpy.PerishableStore(env, capacity=5)

    def pem():
        yield store.put(5, expiry=10)
        put = store.put(1, expiry=10)
        yield env.timeout(1)
        assert not put.triggered
        yield store.get(2)
        yield put  # now there is space

    env.process(pem())
    env.run(until=5)
    assert store.level == 4


def test_conservation(env):
    """Inbound = outbound + scrapped + stock on hand, at all times."""
    store = simpy.PerishableStore(env)
    gets = []

    def check():
        assert (store.total_put ==
                store.total_get + store.total_scrapped + store.level)

    def stocker():
        yield store.put(10, expiry=8)
        yield env.timeout(2)
        yield store.put(6, expiry=20)
        yield env.timeout(2)
        # Earlier expiry than the pending expiry event: reschedules it.
        yield store.put(4, expiry=6)

    def getter():
        yield env.timeout(1)
        gets.append((env.now, (yield store.get(5))))
        check()
        yield env.timeout(10)
        gets.append((env.now, (yield store.get(3))))
        check()

    def auditor():
        for i in range(29):
            yield env.timeout(1)
            check()

    env.process(stocker())
    env.process(getter())
    env.process(auditor())
    env.run(until=30)

    assert store.total_put == 20
    assert store.total_get == 8
    assert store.total_scrapped == 12  # 4 @t=6, 5 @t=8, 3 @t=20
    assert store.level == 0
    assert store.total_put == (store.total_get + store.total_scrapped +
                               store.level)
    # The get at t=1 came from the batch expiring at 8, the one at t=11
    # from the batch expiring at 20.
    assert [amount for batch, amount in gets[0][1]] == [5]
    assert gets[0][1][0][0].expiry == 8
    assert [amount for batch, amount in gets[1][1]] == [3]
    assert gets[1][1][0][0].expiry == 20
