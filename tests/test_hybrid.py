"""Tests for the hybrid (continuous/discrete) simulation environment."""
import math

import pytest

from simpy.hybrid import HybridEnvironment, HybridIntegrationError


def test_exponential_cooling_matches_analytic_solution():
    """Newton cooling: y' = -k (y - t_amb) has the closed-form solution
    y(t) = t_amb + (y0 - t_amb) exp(-k t)."""
    k, t_amb, y0 = 0.5, 20.0, 100.0
    env = HybridEnvironment([y0], lambda t, y: [-k * (y[0] - t_amb)],
                            rtol=1e-9, atol=1e-12)
    env.run(until=10.0)
    expected = t_amb + (y0 - t_amb) * math.exp(-k * 10.0)
    assert env.now == 10.0
    assert env.state[0] == pytest.approx(expected, abs=1e-6)


def test_multidimensional_state():
    """Harmonic oscillator: x' = v, v' = -x with x(0)=1, v(0)=0 gives
    x(t) = cos(t), v(t) = -sin(t)."""
    env = HybridEnvironment([1.0, 0.0], lambda t, y: [y[1], -y[0]],
                            rtol=1e-9, atol=1e-12)
    env.run(until=1.0)
    assert env.state[0] == pytest.approx(math.cos(1.0), abs=1e-6)
    assert env.state[1] == pytest.approx(-math.sin(1.0), abs=1e-6)


def test_threshold_down_crossing_time():
    """Cooling through a threshold: locate the crossing time analytically."""
    k, t_amb, y0, level = 0.5, 20.0, 90.0, 60.0
    env = HybridEnvironment([y0], lambda t, y: [-k * (y[0] - t_amb)],
                            rtol=1e-9, atol=1e-12)
    hits = []
    env.add_threshold(lambda t, y: y[0] - level,
                      lambda env, th: hits.append(env.now), direction='down')
    env.run(until=10.0)
    # 60 = 20 + 70 exp(-0.5 t)  =>  t = 2 ln(70/40)
    expected = 2.0 * math.log(70.0 / 40.0)
    assert len(hits) == 1
    assert hits[0] == pytest.approx(expected, abs=1e-5)


def test_threshold_direction_is_respected():
    """A 'down' threshold must not fire on an up-crossing and vice versa."""
    # Heating: y grows linearly through the level.
    env = HybridEnvironment([0.0], lambda t, y: [1.0])
    hits = []
    env.add_threshold(lambda t, y: y[0] - 5.0,
                      lambda env, th: hits.append(env.now), direction='down')
    env.run(until=10.0)
    assert hits == []

    # Same trajectory, 'up' threshold fires.
    env = HybridEnvironment([0.0], lambda t, y: [1.0])
    env.add_threshold(lambda t, y: y[0] - 5.0,
                      lambda env, th: hits.append(env.now), direction='up')
    env.run(until=10.0)
    assert hits == pytest.approx([5.0], abs=1e-8)

    # 'any' fires on crossings in both directions: y = sin(t) crosses zero
    # downwards at pi and upwards at 2 pi.
    env = HybridEnvironment([0.0], lambda t, y: [math.cos(t)],
                            rtol=1e-9, atol=1e-12)
    any_hits = []
    env.add_threshold(lambda t, y: y[0],
                      lambda env, th: any_hits.append(env.now),
                      direction='any')
    env.run(until=7.0)
    assert any_hits == pytest.approx([math.pi, 2 * math.pi], abs=1e-5)


def test_threshold_callback_may_modify_state():
    """Thermostat: constant heating, reset the state on every crossing."""
    env = HybridEnvironment([50.0], lambda t, y: [10.0])
    hits = []

    def reset(env, th):
        hits.append(env.now)
        env.state[0] = 50.0

    env.add_threshold(lambda t, y: y[0] - 80.0, reset, direction='up')
    env.run(until=10.0)
    assert hits == pytest.approx([3.0, 6.0, 9.0], abs=1e-8)
    # After the last reset at t=9 the state heats up for one more time unit.
    assert env.state[0] == pytest.approx(60.0, abs=1e-6)


def test_scheduled_heating_jump_and_event_ordering():
    """A discrete process switches the dynamics at t=5.  The integration
    must not step past that event, and the threshold must fire at the
    analytically correct time afterwards, in the right order."""
    k, t_cold, t_hot = 0.5, 20.0, 100.0
    mode = {'ambient': t_cold}

    def deriv(t, y):
        return [-k * (y[0] - mode['ambient'])]

    env = HybridEnvironment([90.0], deriv, rtol=1e-9, atol=1e-12)
    log = []

    def controller(env):
        yield env.timeout(5.0)
        # The state at the discrete event must reflect pure cooling.
        log.append(('heating-on', env.now, env.state[0]))
        mode['ambient'] = t_hot

    env.process(controller(env))

    def on_threshold(env, th):
        log.append(('threshold', env.now, env.state[0]))

    # Cooling from 90 crosses 60 downwards first (must not fire), then the
    # heating phase crosses 60 upwards.
    env.add_threshold(lambda t, y: y[0] - 60.0, on_threshold, direction='up')
    env.run(until=10.0)

    y5 = t_cold + 70.0 * math.exp(-k * 5.0)
    t_event = 5.0 + 2.0 * math.log((t_hot - y5) / (t_hot - 60.0))

    assert [entry[0] for entry in log] == ['heating-on', 'threshold']
    assert log[0][1] == 5.0
    assert log[0][2] == pytest.approx(y5, abs=1e-6)
    assert log[1][1] == pytest.approx(t_event, abs=1e-5)
    assert log[1][2] == pytest.approx(60.0, abs=1e-4)


def test_threshold_fires_before_queued_discrete_event():
    """A threshold crossing ahead of the next discrete event is handled
    first; the discrete event is processed afterwards."""
    env = HybridEnvironment([0.0], lambda t, y: [1.0])
    log = []
    env.add_threshold(lambda t, y: y[0] - 3.0,
                      lambda env, th: log.append(('threshold', env.now)),
                      direction='up')

    def probe(env):
        yield env.timeout(5.0)
        log.append(('timeout', env.now))

    env.process(probe(env))
    env.run(until=10.0)
    assert [entry[0] for entry in log] == ['threshold', 'timeout']
    assert log[0][1] == pytest.approx(3.0, abs=1e-8)
    assert log[1][1] == 5.0


def test_minimum_step_size_failure():
    """If the required step size falls below h_min, fail explicitly."""
    env = HybridEnvironment([1.0], lambda t, y: [-y[0]],
                            rtol=1e-13, atol=1e-16, h_min=0.5)
    with pytest.raises(HybridIntegrationError):
        env.run(until=10.0)


def test_root_iteration_limit_failure():
    """If the crossing cannot be located within the iteration budget,
    fail explicitly."""
    env = HybridEnvironment([0.0], lambda t, y: [1.0], h_initial=10.0,
                            event_tol=1e-12, max_root_iterations=2)
    env.add_threshold(lambda t, y: y[0] - 3.3, lambda env, th: None,
                      direction='up')
    with pytest.raises(HybridIntegrationError):
        env.run(until=10.0)


def test_invalid_direction_rejected():
    env = HybridEnvironment([0.0], lambda t, y: [1.0])
    with pytest.raises(ValueError):
        env.add_threshold(lambda t, y: y[0], lambda env, th: None,
                          direction='sideways')
