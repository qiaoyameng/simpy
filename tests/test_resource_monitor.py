"""Tests for time-weighted resource statistics."""

import pytest

import simpy
from simpy import ResourceObserver


def run_step_curve(env, resource):
    """Queue (2,1,0), occupancy (1,1,1,0) on half-time intervals."""
    requests = [resource.request() for _ in range(3)]

    def release(request, delay):
        yield env.timeout(delay)
        resource.release(request)

    for index, request in enumerate(requests):
        env.process(release(request, 0.5 * (index + 1)))


def test_time_weighted_average_step_curve(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource)
    run_step_curve(env, resource)
    env.run(until=2.0)

    areas = observer.areas()

    assert areas.duration == pytest.approx(2.0)
    assert areas.queue_length == pytest.approx(1.5)
    assert areas.occupancy == pytest.approx(1.5)

    average = observer.time_average()

    assert average.start == 0.0
    assert average.end == 2.0
    assert average.queue_length == pytest.approx(0.75)
    assert average.occupancy == pytest.approx(0.75)
    assert average.count == average.occupancy


def test_warmup_clips_only_previous_area(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource, start=0.75)
    run_step_curve(env, resource)
    env.run(until=2.0)

    areas = observer.areas()
    average = observer.time_average()

    assert areas.duration == pytest.approx(1.25)
    assert areas.queue_length == pytest.approx(
        0.25 * 1 + 0.5 * 0 + 0.5 * 0)
    assert areas.occupancy == pytest.approx(0.75)
    assert average.start == pytest.approx(0.75)
    assert average.queue_length == pytest.approx(0.2)
    assert average.occupancy == pytest.approx(0.6)


def test_window_may_start_and_end_between_events(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource)
    run_step_curve(env, resource)
    env.run(until=2.0)

    average = observer.time_average(0.25, 1.25)

    assert average.start == pytest.approx(0.25)
    assert average.end == pytest.approx(1.25)
    assert average.queue_length == pytest.approx(
        (0.25 * 2 + 0.5 * 1 + 0.25 * 0) / 1.0)
    assert average.occupancy == pytest.approx(1.0)


def test_batches_identify_short_final_interval(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource)
    run_step_curve(env, resource)
    env.run(until=2.0)

    batches = observer.batch_time_averages(0.75)

    assert len(batches) == 3
    assert [(batch.index, batch.complete, batch.partial)
            for batch in batches] == [(0, True, False),
                                      (1, True, False),
                                      (2, False, True)]
    assert batches[0].start == 0.0
    assert batches[0].end == pytest.approx(0.75)
    assert batches[1].start == pytest.approx(0.75)
    assert batches[1].end == pytest.approx(1.5)
    assert batches[2].start == pytest.approx(1.5)
    assert batches[2].end == 2.0
    assert batches[2].occupancy == pytest.approx(0.0)
    assert batches[2].queue_length == 0.0

    assert batches[0].queue_length == pytest.approx(5.0 / 3.0)
    assert batches[0].occupancy == pytest.approx(1.0)
    assert batches[1].queue_length == pytest.approx(1.0 / 3.0)
    assert batches[1].occupancy == pytest.approx(1.0)

    assert observer.batch_means(0.75) == batches


def test_fixed_batches_start_at_warmup(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource, start=0.5)
    run_step_curve(env, resource)
    env.run(until=2.0)

    batches = observer.batch_time_averages(0.5)

    assert [(batch.start, batch.end, batch.complete)
            for batch in batches] == [
                (0.5, 1.0, True),
                (1.0, 1.5, True),
                (1.5, 2.0, True),
            ]


def test_same_time_state_changes_count_once(env):
    resource = simpy.Resource(env, capacity=3)
    observer = ResourceObserver(resource)

    requests = [resource.request() for _ in range(3)]
    env.run(until=1.0)
    for request in requests:
        resource.release(request)
    env.run(until=2.0)

    assert observer.time_average().occupancy == pytest.approx(1.5)


def test_changing_number_of_same_time_events_keeps_area():
    def run(noisy):
        env = simpy.Environment()
        resource = simpy.Resource(env, capacity=3)
        observer = ResourceObserver(resource)
        request = resource.request()

        def change_state():
            yield env.timeout(1.0)
            if noisy:
                transient_requests = [resource.request()
                                      for _ in range(2)]
                for transient_request in transient_requests:
                    resource.release(transient_request)
            resource.release(request)

        env.process(change_state())
        env.run(until=2.0)
        return observer.areas()

    assert run(False) == run(True)


def test_instantaneous_intermediate_state_does_not_affect_area(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource)

    long_request = resource.request()

    def noise():
        yield env.timeout(1.0)
        queued = resource.request()
        resource.release(long_request)
        yield queued
        resource.release(queued)

    env.process(noise())
    env.run(until=2.0)

    average = observer.time_average()

    assert average.occupancy == pytest.approx(0.5)
    assert average.queue_length == 0.0


def test_observer_may_attach_mid_run(env):
    resource = simpy.Resource(env, capacity=1)
    request = resource.request()

    def release():
        yield env.timeout(1.0)
        resource.release(request)

    env.process(release())
    env.run(until=0.5)

    observer = ResourceObserver(resource)
    env.run(until=2.0)

    average = observer.time_average()

    assert average.start == 0.5
    assert average.occupancy == pytest.approx(1.0 / 3.0)


def test_reading_statistics_does_not_schedule_events(env):
    resource = simpy.Resource(env, capacity=1)
    request = resource.request()

    def release():
        yield env.timeout(2.0)
        resource.release(request)

    env.process(release())
    observer = ResourceObserver(resource)
    env.run(until=1.0)

    before = list(env._queue)
    average = observer.time_average(end=1.5)
    batches = observer.batch_time_averages(0.5, end=1.5)
    areas = observer.areas(end=1.5)

    assert env.now == 1.0
    assert env._queue == before
    assert len(batches) == 3
    assert average.occupancy == 1.0
    assert areas.occupancy == pytest.approx(1.5)


def test_constructing_observer_does_not_schedule_events(env):
    resource = simpy.Resource(env, capacity=1)

    ResourceObserver(resource)

    assert env._queue == []
    assert env.peek() == float('inf')


def test_canceled_queued_request_updates_state(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource)
    active_request = resource.request()
    queued_request = resource.request()

    def cancel_request():
        yield env.timeout(1.0)
        queued_request.cancel()

    def release_request():
        yield env.timeout(2.0)
        resource.release(active_request)

    env.process(cancel_request())
    env.process(release_request())
    env.run(until=3.0)

    areas = observer.areas()

    assert areas.queue_length == pytest.approx(1.0)
    assert areas.occupancy == pytest.approx(2.0)


def test_invalid_windows_and_batch_interval(env):
    resource = simpy.Resource(env, capacity=1)
    observer = ResourceObserver(resource)

    pytest.raises(ValueError, ResourceObserver, resource, start=-1.0)
    pytest.raises(ValueError, observer.time_average, 0.0, 0.0)
    pytest.raises(ValueError, observer.batch_time_averages, 0.0)
    pytest.raises(ValueError, observer.batch_time_averages,
                  float('inf'))


def test_close_restores_resource_methods(env):
    resource = simpy.Resource(env, capacity=1)
    request_method = resource.request
    release_method = resource.release

    with ResourceObserver(resource) as observer:
        assert resource.request != request_method
        assert resource.release != release_method
        request = resource.request()
        assert observer._times[-1] == 0.0

    assert resource.request == request_method
    assert resource.release == release_method

    resource.release(request)
    env.run()
    assert observer._closed is True


def test_closed_observer_does_not_extrapolate_later_state(env):
    resource = simpy.Resource(env, capacity=1)
    request = resource.request()
    observer = ResourceObserver(resource)

    def release_request():
        yield env.timeout(1.0)
        resource.release(request)

    env.process(release_request())
    env.run(until=0.5)
    observer.close()
    env.run(until=2.0)

    assert observer.areas(end=0.5).occupancy == pytest.approx(0.5)
    pytest.raises(RuntimeError, observer.areas, None, 1.0)
