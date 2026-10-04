import pytest

import simpy


def submit(env, resource, item, completed, withdrawn=None):
    request = resource.submit(item)
    try:
        batch = yield request
        completed.append((item, env.now, batch))
    except RuntimeError:
        if withdrawn is not None:
            withdrawn.append(item)
        else:
            raise


def test_dense_arrivals_form_full_fixed_batches(env):
    completed = []
    service_calls = []

    def service_time(batch):
        service_calls.append(batch)
        return 2

    def setup():
        resource = simpy.BatchResource(
            env, max_batch_size=8, min_batch_size=4,
            max_wait_time=1, service_time=service_time)
        requests = [resource.submit(item) for item in range(20)]
        yield env.all_of(requests)
        for item in range(20):
            batch = yield requests[item]
            completed.append((item, batch))

    env.process(setup())
    env.run()

    processed_items = [item for item, _ in completed]
    assert processed_items == list(range(20))

    batches = list(dict.fromkeys(
        (batch for _, batch in completed)))
    assert sorted(len(batch) for batch in batches) == [4, 8, 8]
    assert sorted(batches, key=len, reverse=True) == [
        tuple(range(0, 8)), tuple(range(8, 16)), tuple(range(16, 20))]
    assert service_calls == [
        tuple(range(0, 8)), tuple(range(8, 16)), tuple(range(16, 20))]


def test_sparse_timeout_processes_batch_below_minimum(env):
    completed = []
    started = []

    def service_time(batch):
        started.append((env.now, batch))
        return 1

    def arrivals():
        resource = simpy.BatchResource(
            env, max_batch_size=8, min_batch_size=3,
            max_wait_time=2, service_time=service_time)
        procs = []
        for arrival, item in [(0, 'a'), (4, 'b'), (8, 'c')]:
            yield env.timeout(arrival - env.now)
            procs.append(env.process(
                submit(env, resource, item, completed)))
        yield env.all_of(procs)

    env.process(arrivals())
    env.run()

    assert started == [
        (2, ('a',)), (6, ('b',)), (10, ('c',))]
    assert [(item, time) for item, time, _ in completed] == [
        ('a', 3), ('b', 7), ('c', 11)]


def test_withdraw_earliest_recomputes_deadline_from_next_arrival(env):
    completed = []
    resource = simpy.BatchResource(
        env, max_batch_size=8, min_batch_size=3, max_wait_time=5)
    requests = {}
    withdrawn = []

    def scenario():
        def submit_named(item):
            request = resource.submit(item)
            requests[item] = request
            try:
                yield request
            except RuntimeError:
                withdrawn.append(item)

        first = env.process(submit_named('first'))
        yield env.timeout(1)
        second = env.process(submit(env, resource, 'second', completed,
                                    withdrawn))
        yield env.timeout(1)
        resource.withdraw(requests['first'])
        yield env.timeout(4)
        assert not completed
        assert env.now == 6
        yield env.timeout(1)

        assert completed == [('second', 6, ('second',))]
        assert not requests['first'].ok
        assert requests['first'].defused
        assert first.ok
        assert withdrawn == ['first']

    env.process(scenario())
    env.run()


def test_stale_timeout_after_withdrawal_does_not_open_empty_batch(env):
    batches = []
    resource = simpy.BatchResource(
        env, max_batch_size=8, min_batch_size=3, max_wait_time=5,
        service_time=lambda batch: batches.append(batch) or 0)

    def scenario():
        first = resource.submit('first')
        yield env.timeout(2)
        resource.withdraw(first)
        yield env.timeout(3)
        assert not batches
        assert env.now == 5

        request = resource.submit('second')
        yield request
        assert batches == [('second',)]
        assert env.now == 10

    env.process(scenario())
    env.run()


def test_withdraw_later_item_keeps_earliest_deadline(env):
    closed = []
    resource = simpy.BatchResource(
        env, max_batch_size=8, min_batch_size=3, max_wait_time=5,
        service_time=lambda batch: closed.append((env.now, batch)) or 0)

    def scenario():
        first = resource.submit('first')
        yield env.timeout(1)
        second = resource.submit('second')
        yield env.timeout(1)
        resource.withdraw(second)
        yield env.timeout(2)

        assert env.now == 4
        assert not closed
        yield env.timeout(1)
        yield first
        assert closed == [(5, ('first',))]

    env.process(scenario())
    env.run()


def test_later_arrivals_join_next_batch_and_cannot_be_withdrawn(env):
    completed = []
    resource = simpy.BatchResource(
        env, max_batch_size=2, max_wait_time=5, service_time=3)

    first = resource.submit('first')
    second = resource.submit('second')
    third = resource.submit('third')

    assert first.batch == ('first', 'second')
    assert third.batch is None
    with pytest.raises(RuntimeError):
        resource.withdraw(first)

    def scenario():
        yield env.timeout(5)
        third_batch = yield third
        assert third_batch == ('third',)
        assert env.now == 8

        first_batch = yield first
        second_batch = yield second
        assert first_batch == second_batch == ('first', 'second')
        completed.extend([first_batch, third_batch])
        assert len(completed) == 2

    env.process(scenario())
    env.run()


@pytest.mark.parametrize('kwargs,message', [
    ({'max_batch_size': 0}, '"max_batch_size" must be > 0.'),
    ({'min_batch_size': 0}, '"min_batch_size" must be > 0.'),
    ({'max_batch_size': 2, 'min_batch_size': 3},
     '"min_batch_size" must not exceed "max_batch_size".'),
    ({'max_wait_time': -1}, '"max_wait_time" must be >= 0.'),
])
def test_batch_resource_arguments(env, kwargs, message):
    with pytest.raises(ValueError) as excinfo:
        simpy.BatchResource(env, **kwargs)
    assert str(excinfo.value) == message
