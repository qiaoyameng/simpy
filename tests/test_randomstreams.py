"""
Tests for deterministic simulation random streams.

"""
import pytest

import simpy
from simpy.randomstreams import RandomStreamManager


CUSTOMER_COUNT = 6


def run_branch_model(trial_seed, extra_branch):
    if isinstance(trial_seed, RandomStreamManager):
        manager = trial_seed
    else:
        manager = RandomStreamManager(trial_seed)

    env = simpy.Environment()
    arrivals = []
    interarrivals = []
    demands = {}

    def source():
        arrival_rng = manager.stream('source', 'interarrival')
        for customer_id in range(CUSTOMER_COUNT):
            interarrival = arrival_rng.expovariate(2.0)
            interarrivals.append(interarrival)
            yield env.timeout(interarrival)
            arrivals.append(env.now)
            env.process(customer(customer_id))

    def customer(customer_id):
        service_rng = manager.stream(
            'customer-%d' % customer_id, 'service'
        )
        demands[customer_id] = service_rng.uniform(1.0, 10.0)

        if extra_branch:
            branch_rng = manager.stream('alternate-route', 'delay')
            yield env.timeout(branch_rng.uniform(0.1, 1.0))
        else:
            yield env.timeout(0)

    env.process(source())
    env.run()
    return arrivals, interarrivals, demands


def test_streams_are_stable_and_deterministic():
    first = RandomStreamManager(12345)
    second = RandomStreamManager(12345)

    assert first.stream('entity', 'purpose') is first.stream(
        'entity', 'purpose'
    )
    assert first.stream('entity', 'purpose') is not second.stream(
        'entity', 'purpose'
    )

    expected = [
        first.stream('entity', 'purpose').random() for _ in range(5)
    ]
    actual = [
        second.stream('entity', 'purpose').random() for _ in range(5)
    ]

    assert actual == expected


def test_other_streams_do_not_shift_sampling_order():
    baseline = RandomStreamManager(987)
    expected = [
        baseline.stream('customer', 'service').random() for _ in range(5)
    ]

    interleaved = RandomStreamManager(987)
    observed = []
    for index in range(5):
        observed.append(interleaved.stream('customer', 'service').random())
        interleaved.stream('new-customer-%d' % index, 'service').random()
        interleaved.stream('customer', 'delay').random()

    assert observed == expected


def test_keys_with_same_textual_form_but_different_types_distinct():
    manager = RandomStreamManager('trial')
    values = []

    for entity_key, purpose_key in (
        (1, 'draw'),
        ('1', 'draw'),
        (1.0, 'draw'),
        (True, 'draw'),
        (b'1', 'draw'),
        ('1', b'draw'),
    ):
        values.append(manager.stream(entity_key, purpose_key).random())

    assert len(set(values)) == len(values)


def test_invalid_keys_are_rejected():
    manager = RandomStreamManager(1)

    for bad_key in (('tuple',), ['list'], {'dict': 1}, bytearray(b'x')):
        with pytest.raises(TypeError):
            manager.stream(bad_key, 'purpose')
        with pytest.raises(TypeError):
            manager.stream('entity', bad_key)

    with pytest.raises(TypeError):
        RandomStreamManager(object())


def test_copy_initial_layout_ignores_current_draws():
    manager = RandomStreamManager('experiment')
    manager.stream('entity', 'purpose').random()
    manager.stream('entity', 'purpose').random()

    copied = manager.copy_initial_layout()
    fresh = RandomStreamManager('experiment')

    assert copied.stream('entity', 'purpose').random() == (
        fresh.stream('entity', 'purpose').random()
    )


def test_reset_restores_initial_stream_state():
    manager = RandomStreamManager(42)
    rng = manager.stream('entity', 'purpose')
    initial = rng.random()
    rng.random()
    rng.random()

    manager.reset()

    assert manager.stream('entity', 'purpose').random() == initial


def test_model_branches_share_common_random_layout():
    manager = RandomStreamManager(20261007)
    manager.stream('unrelated', 'draw').random()

    branch_arrivals, branch_interarrivals, branch_demands = run_branch_model(
        manager.copy_initial_layout(), True
    )
    baseline_arrivals, baseline_interarrivals, baseline_demands = (
        run_branch_model(manager.copy_initial_layout(), False)
    )

    assert branch_interarrivals == baseline_interarrivals
    assert branch_demands == baseline_demands
    assert branch_arrivals == baseline_arrivals
    assert len(branch_interarrivals) == CUSTOMER_COUNT


def test_different_trial_seed_changes_common_sequences():
    _, first_interarrivals, first_demands = run_branch_model(20261007, True)
    _, second_interarrivals, second_demands = run_branch_model(
        20261008, True
    )

    assert second_interarrivals != first_interarrivals
    assert second_demands != first_demands
