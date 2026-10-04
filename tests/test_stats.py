import simpy


def build_staircase(noise=True, batch_size=None, end=None):
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=3)
    stats = simpy.ResourceStats(
        resource, batch_size=batch_size, end=end)

    permanent = resource.request()
    first = resource.request()

    def hold_first():
        yield env.timeout(2)
        resource.release(first)

    def user_a():
        yield env.timeout(1)
        request = resource.request()
        yield request
        yield env.event()

    def noisy_cancel():
        yield env.timeout(2.5)
        first_request = resource.request()
        second_request = resource.request()
        second_request.cancel()
        first_request.cancel()

    def user_e():
        yield env.timeout(4)
        request = resource.request()
        yield request
        yield env.event()

    def user_b():
        yield env.timeout(5)
        request = resource.request()
        yield request
        yield env.timeout(1)

    env.process(hold_first())
    env.process(user_a())
    if noise:
        env.process(noisy_cancel())
    env.process(user_e())
    env.process(user_b())
    env.run(until=6)
    return env, resource, stats, permanent


def test_resource_stats_integrate_step_curve():
    env, resource, stats, permanent = build_staircase(
        noise=False, end=6)

    # Queue length is one on [1, 2]; the next request enters at time 5.
    # Usage is two on [0, 2], three on [2, 6].
    result = stats.result()
    assert result.elapsed == 6
    assert result.mean_queue_length == 1.0 / 6
    assert result.mean_usage == 2.5
    assert result.mean_utilization == 2.5 / 3


def run_redundant_event_scenario(redundant):
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=2)
    stats = simpy.ResourceStats(resource)
    permanent = resource.request()

    def user():
        yield env.timeout(1)
        request = resource.request()
        yield request
        yield env.timeout(3)
        resource.release(request)

    def redundant_events():
        if not redundant:
            return
        yield env.timeout(3)
        first = resource.request()
        second = resource.request()
        second.cancel()
        first.cancel()

    env.process(user())
    env.process(redundant_events())
    env.run(until=5)
    return env, resource, stats, permanent


def test_resource_stats_ignore_redundant_same_time_events():
    _, _, with_events, permanent = run_redundant_event_scenario(True)
    _, _, without_events, other_permanent = run_redundant_event_scenario(
        False)

    assert with_events.result() == without_events.result()
    # Usage is one on [0, 1], two on [1, 4], one on [4, 5].
    assert with_events.result().mean_usage == 1.6

    with_events.resource.release(permanent)
    without_events.resource.release(other_permanent)


def test_resource_stats_warmup_and_end_inside_constant_intervals():
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=1)
    stats = simpy.ResourceStats(resource, start=1.5, end=5.5)

    first = resource.request()

    def first_user():
        yield env.timeout(2)
        resource.release(first)

    def second_user():
        request = resource.request()
        yield request
        yield env.timeout(4)
        resource.release(request)

    env.process(first_user())
    env.process(second_user())
    env.run(until=6)

    result = stats.result()
    assert result.start == 1.5
    assert result.end == 5.5
    assert result.elapsed == 4
    assert result.mean_usage == 1
    assert result.mean_queue_length == 0.5 / 4


def test_resource_stats_start_mid_run():
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=1)

    def user():
        request = resource.request()
        yield request
        yield env.timeout(4)
        resource.release(request)

    env.process(user())
    env.run(until=2)
    stats = simpy.ResourceStats(resource)
    env.run(until=4)

    assert stats.elapsed == 2
    assert stats.mean_usage == 1


def test_resource_stats_batches_mark_short_final_interval():
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=1)
    stats = simpy.ResourceStats(resource, batch_size=2)

    def user(start, duration):
        yield env.timeout(start)
        request = resource.request()
        yield request
        yield env.timeout(duration)
        resource.release(request)

    env.process(user(0, 2))
    env.process(user(2, 2))
    env.process(user(4, 2))

    scheduled = [0]
    original_schedule = env.schedule

    def counting_schedule(*args, **kwargs):
        scheduled[0] += 1
        return original_schedule(*args, **kwargs)

    env.schedule = counting_schedule
    env.run(until=5)
    before_read = scheduled[0]
    now_before_read = env.now
    batches = stats.batch_means()

    assert env.now == now_before_read == 5
    assert scheduled[0] == before_read
    assert len(batches) == 3
    assert [batch.start for batch in batches] == [0, 2, 4]
    assert [batch.end for batch in batches] == [2, 4, 5]
    assert [batch.elapsed for batch in batches] == [2, 2, 1]
    assert [batch.mean_usage for batch in batches] == [1, 1, 1]
    assert [batch.complete for batch in batches] == [
        True, True, False]
    assert batches[-1].partial
    assert stats.result() == stats.result()


def test_resource_stats_batches_complete_window_without_boundary_event():
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=1)
    stats = simpy.ResourceStats(resource, batch_size=2, end=6)

    def user():
        request = resource.request()
        yield request
        yield env.timeout(5)
        resource.release(request)

    env.process(user())
    env.run(until=7)

    batches = stats.batch_means()
    assert [batch.end for batch in batches] == [2, 4, 6]
    assert all(batch.complete for batch in batches)


def test_resource_stats_do_not_schedule_observation_events():
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=1)

    def user():
        request = resource.request()
        yield request
        yield env.timeout(2)
        resource.release(request)

    env.process(user())
    env.run()

    scheduled = [0]
    original_schedule = env.schedule

    def counting_schedule(*args, **kwargs):
        scheduled[0] += 1
        return original_schedule(*args, **kwargs)

    env.schedule = counting_schedule
    stats = simpy.ResourceStats(resource)

    assert scheduled[0] == 0
    assert stats.mean_usage == 0
    assert stats.mean_queue_length == 0
    assert scheduled[0] == 0
