"""Tests for :class:`simpy.resources.batch.BatchResource`."""
import pytest

from simpy.resources.batch import BatchResource, Withdrawn


def serve(env, oven, item, done, withdrawn):
    """Put *item* into the oven and record when its batch completes."""
    request = oven.put(item)
    try:
        yield request
        done.append((item, env.now))
    except Withdrawn:
        withdrawn.append(item)


def test_dense_arrivals(env):
    """Dense arrivals: full batches start immediately on the 8th item,
    the remainder starts after max_wait from its first arrival."""
    batches = []

    def service_time(items):
        batches.append(list(items))
        return 2

    oven = BatchResource(env, min_batch=2, max_batch=8, max_wait=5,
                         service_time=service_time)
    done, withdrawn = [], []

    def source():
        for i in range(20):
            env.process(serve(env, oven, 'item-%d' % i, done, withdrawn))
            yield env.timeout(0.5)

    env.process(source())
    env.run()

    # 8 items per full batch, remainder of 4 via timeout.
    assert batches == [
        ['item-%d' % i for i in range(0, 8)],
        ['item-%d' % i for i in range(8, 16)],
        ['item-%d' % i for i in range(16, 20)],
    ]
    # Service time is charged once per batch.
    assert len(batches) == 3
    # Every item is processed exactly once.
    assert sorted(item for item, _ in done) == sorted(
        'item-%d' % i for i in range(20))
    assert len(done) == 20
    assert not withdrawn
    # Batch 1: 8th item at t=3.5, done at 5.5.  Batch 2: starts at 7.5,
    # done at 9.5.  Batch 3: first arrival at 8.0 + max_wait 5 -> 13.0,
    # done at 15.0.
    for item, now in done:
        i = int(item.split('-')[1])
        expected = 5.5 if i < 8 else (9.5 if i < 16 else 15.0)
        assert now == expected


def test_sparse_arrivals(env):
    """Sparse arrivals: timeout with fewer than min_batch items does not
    start an undersized batch; it starts as soon as min_batch is reached."""
    batches = []

    def service_time(items):
        batches.append((env.now, list(items)))
        return 1

    oven = BatchResource(env, min_batch=3, max_batch=8, max_wait=4,
                         service_time=service_time)
    done, withdrawn = [], []

    def source():
        for i, at in enumerate([0, 3, 6, 9, 12, 14]):
            yield env.timeout(at - env.now)
            env.process(serve(env, oven, 'item-%d' % i, done, withdrawn))

    env.process(source())
    env.run()

    # t=4: only 2 items (< min_batch) -> keep waiting; 3rd item at t=6
    # starts the batch immediately.  Same for the second batch at t=14.
    assert batches == [
        (6, ['item-0', 'item-1', 'item-2']),
        (14, ['item-3', 'item-4', 'item-5']),
    ]
    assert sorted(item for item, _ in done) == sorted(
        'item-%d' % i for i in range(6))
    assert len(done) == 6
    assert sorted(now for _, now in done) == [7, 7, 7, 15, 15, 15]


def test_withdraw_recomputes_deadline(env):
    """Withdrawing the first item recomputes the deadline from the
    earliest *remaining* arrival, not from "now"."""
    batches = []

    def service_time(items):
        batches.append((env.now, list(items)))
        return 1

    oven = BatchResource(env, min_batch=1, max_batch=8, max_wait=10,
                         service_time=service_time)
    done, withdrawn = [], []

    env.process(serve(env, oven, 'A', done, withdrawn))       # t=0

    def later():
        yield env.timeout(4)
        env.process(serve(env, oven, 'B', done, withdrawn))   # t=4
        yield env.timeout(2)
        assert oven.withdraw('A')                             # t=6

    env.process(later())
    env.run()

    # Deadline is 4 + 10 = 14, not 6 + 10 = 16.
    assert batches == [(14, ['B'])]
    assert done == [('B', 15)]
    assert withdrawn == ['A']


def test_stale_timer_starts_no_empty_batch(env):
    """Discarded timer events (from recomputed deadlines and emptied
    queues) must never start an empty batch."""
    batches = []

    def service_time(items):
        batches.append((env.now, list(items)))
        return 2

    oven = BatchResource(env, min_batch=1, max_batch=8, max_wait=5,
                         service_time=service_time)
    done, withdrawn = [], []

    env.process(serve(env, oven, 'A', done, withdrawn))       # t=0

    def script():
        yield env.timeout(2)
        env.process(serve(env, oven, 'B', done, withdrawn))   # t=2
        yield env.timeout(1)
        assert oven.withdraw('A')                             # t=3
        yield env.timeout(1)
        assert oven.withdraw('B')                             # t=4, queue empty
        yield env.timeout(5)
        env.process(serve(env, oven, 'C', done, withdrawn))   # t=9

    env.process(script())
    env.run()

    # Stale timers at t=5 and t=7 fired while the queue was empty and
    # started nothing.  Only C is served, once, after its own max_wait.
    assert batches == [(14, ['C'])]
    assert done == [('C', 16)]
    assert sorted(withdrawn) == ['A', 'B']


def test_in_service_batch_is_fixed(env):
    """Items arriving during a service join the next batch; members of a
    running batch can no longer be withdrawn."""
    oven = BatchResource(env, min_batch=1, max_batch=2, max_wait=10,
                         service_time=5)
    done, withdrawn = [], []

    env.process(serve(env, oven, 'A', done, withdrawn))       # t=0

    def script():
        yield env.timeout(1)
        env.process(serve(env, oven, 'B', done, withdrawn))   # t=1
        yield env.timeout(1)
        # t=2: batch [A, B] is in service (until t=6).
        assert not oven.withdraw('A')
        assert not oven.withdraw('B')
        env.process(serve(env, oven, 'C', done, withdrawn))   # next batch

    env.process(script())
    env.run()

    # [A, B] start at t=1 (max_batch reached), done at t=6.  C waits its
    # own max_wait from t=2 and is done at 12 + 5 = 17.
    assert sorted(done) == [('A', 6), ('B', 6), ('C', 17)]
    assert not withdrawn


def test_invalid_configuration(env):
    with pytest.raises(ValueError):
        BatchResource(env, min_batch=0)
    with pytest.raises(ValueError):
        BatchResource(env, min_batch=9, max_batch=8)
    with pytest.raises(ValueError):
        BatchResource(env, max_wait=-1)
