# encoding: utf-8
"""
Tests for :mod:`simpy.rng` -- explicit random stream management.

"""
import os
import random
import subprocess
import sys

import pytest

import simpy
from simpy.rng import RandomStreamManager


def test_same_seed_same_streams():
    """Managers with equal trial seeds produce identical sub-streams."""
    first = RandomStreamManager(42)
    second = RandomStreamManager(42)
    for entity, purpose in [('machine-1', 'service'), ('machine-2', 'iat')]:
        seq_a = [first.stream(entity, purpose).random() for _ in range(10)]
        seq_b = [second.stream(entity, purpose).random() for _ in range(10)]
        assert seq_a == seq_b


def test_different_seed_different_streams():
    """A different trial seed yields a different set of sequences."""
    first = RandomStreamManager(42)
    second = RandomStreamManager(43)
    seq_a = [first.stream('machine-1', 'service').random() for _ in range(10)]
    seq_b = [second.stream('machine-1', 'service').random() for _ in range(10)]
    assert seq_a != seq_b


def test_streams_advance_independently():
    """Each sub-stream advances in its own sampling order only."""
    manager = RandomStreamManager(7)
    reference = RandomStreamManager(7)
    expected = [reference.stream('alpha', 'service').random()
                for _ in range(5)]

    observed = [manager.stream('alpha', 'service').random()]
    # Interleave draws from other entities and purposes, and add entirely
    # new entities: the alpha/service stream must not notice.
    manager.stream('beta', 'service').random()
    manager.stream('alpha', 'interarrival').random()
    manager.stream('gamma', 'service').uniform(0, 1)
    observed.extend(manager.stream('alpha', 'service').random()
                    for _ in range(4))
    assert observed == expected


def test_stream_returns_random_instance():
    manager = RandomStreamManager(0)
    assert isinstance(manager.stream('e', 'p'), random.Random)


def test_same_shaped_keys_of_different_types_are_distinct():
    """1, 1.0, True, '1' and b'1' must not collapse into one key."""
    manager = RandomStreamManager(0)
    sequences = []
    for key in (1, 1.0, True, '1', b'1'):
        rng = manager.stream(key, 'purpose')
        sequences.append(tuple(rng.random() for _ in range(5)))
    for i, seq in enumerate(sequences):
        for other in sequences[i + 1:]:
            assert seq != other


@pytest.mark.parametrize('key', [
    (1, 2), [1, 2], {'a': 1}, object(), 1j,
])
def test_non_basic_keys_are_rejected(key):
    """Keys without a deterministic encoding raise :exc:`TypeError`."""
    manager = RandomStreamManager(0)
    with pytest.raises(TypeError):
        manager.stream(key, 'purpose')
    with pytest.raises(TypeError):
        manager.stream('entity', key)
    with pytest.raises(TypeError):
        RandomStreamManager(key)


def test_streams_do_not_depend_on_process_hash():
    """Derivation is stable across PYTHONHASHSEED values."""
    code = (
        'from simpy.rng import RandomStreamManager;'
        'm = RandomStreamManager("trial");'
        'print(m.stream("entity", "purpose").random())'
    )
    outputs = set()
    for hashseed in ('0', '1', '42'):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.check_output([sys.executable, '-c', code], env=env)
        outputs.add(out.strip())
    assert len(outputs) == 1


def test_copy_initial_reproduces_trial_layout():
    """copy_initial() restores every stream to its first-use state."""
    manager = RandomStreamManager('trial-3')
    expected = [manager.stream('e1', 'a').random() for _ in range(5)]
    manager.stream('e2', 'b').random()

    clone = manager.copy_initial()
    assert clone.trial_seed == 'trial-3'
    assert [clone.stream('e1', 'a').random() for _ in range(5)] == expected
    # Streams never requested before the copy derive identically anyway.
    assert (clone.stream('late', 'x').random() ==
            RandomStreamManager('trial-3').stream('late', 'x').random())
    # The original manager keeps its own sampling position.
    assert manager.stream('e1', 'a').random() != expected[0]


def _run_shop(trial_seed, variant, until=50.0):
    """Run a tiny job shop with three machine entities.

    The two variants differ only in the gamma branch: the changed model
    uses a different mean service time and draws extra inspection values
    for gamma jobs.

    """
    env = simpy.Environment()
    manager = RandomStreamManager(trial_seed)
    log = {'alpha': [], 'beta': [], 'gamma': []}

    def machine(entity, mean_service, inspect):
        interarrival = manager.stream(entity, 'interarrival')
        service = manager.stream(entity, 'service')
        inspection = (manager.stream(entity, 'inspection')
                      if inspect else None)
        while True:
            gap = interarrival.expovariate(1.0)
            yield env.timeout(gap)
            demand = service.uniform(0.5, 1.5) * mean_service
            log[entity].append((gap, demand))
            yield env.timeout(demand)
            if inspection is not None:
                # Branch-specific extra sampling; must not leak into the
                # streams of the other entities.
                yield env.timeout(inspection.uniform(0.0, 0.2))

    env.process(machine('alpha', 1.0, False))
    env.process(machine('beta', 1.0, False))
    if variant == 'baseline':
        env.process(machine('gamma', 1.0, False))
    else:
        env.process(machine('gamma', 2.0, True))
    env.run(until=until)
    return log


def test_branch_change_keeps_common_entities_identical():
    """Two models differing in one branch agree on all common entities."""
    baseline = _run_shop(7, 'baseline')
    changed = _run_shop(7, 'changed')
    for entity in ('alpha', 'beta'):
        assert len(baseline[entity]) > 10
        # Inter-arrival gaps and service demands match item by item.
        assert baseline[entity] == changed[entity]
    # The changed branch really did produce different gamma demands.
    assert baseline['gamma'] != changed['gamma']


def test_trial_seed_changes_all_sequences():
    """Switching the trial seed produces a different set of sequences."""
    run_a = _run_shop(7, 'baseline')
    run_b = _run_shop(8, 'baseline')
    for entity in ('alpha', 'beta', 'gamma'):
        assert run_a[entity] != run_b[entity]


def test_trial_is_reproducible_via_copy_initial():
    """A copied initial layout reproduces the original trial exactly."""
    env = simpy.Environment()
    manager = RandomStreamManager(11)
    drawn = []

    def source():
        rng = manager.stream('source', 'gap')
        while True:
            gap = rng.expovariate(1.0)
            drawn.append(gap)
            yield env.timeout(gap)

    env.process(source())
    env.run(until=20.0)

    clone = manager.copy_initial()
    replay = [clone.stream('source', 'gap').expovariate(1.0)
              for _ in range(len(drawn))]
    assert replay == drawn
