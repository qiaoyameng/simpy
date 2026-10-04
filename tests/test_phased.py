"""
Tests for the staged ``simpy.phased.PhasedEnvironment``.

The digital control model distinguishes the physical time of a clock tick
from the micro steps (delta cycles) that propagate signals within that tick.
Each micro step runs through the three stages INPUT, STATE and OUTPUT.
"""
import pytest

import simpy
from simpy import Phase
from simpy.phased import MicrostepLimit


@pytest.fixture
def env():
    return simpy.PhasedEnvironment()


def trace(log, env, name):
    """Return a callback that records at which physical time, micro step and
    phase it was invoked."""
    def callback(event):
        log.append((env.now, env.microstep, env.phase, name))
    return callback


def test_initial_state():
    """A fresh environment starts at the input stage of micro step zero."""
    env = simpy.PhasedEnvironment(initial_time=5)
    assert env.now == 5
    assert env.microstep == 0
    assert env.phase == Phase.INPUT
    assert env.microstep_limit is None
    assert env.peek() == simpy.core.Infinity


def test_negative_microstep_limit():
    pytest.raises(ValueError, simpy.PhasedEnvironment, microstep_limit=-1)


def test_later_phase_stays_in_round(env, log):
    """Events targeting a strictly later phase are still processed in the
    current micro step, in ascending phase order."""
    def starter(event):
        out = env.event()
        out.callbacks.append(trace(log, env, 'out'))
        out.succeed(phase=Phase.OUTPUT)

        state = env.event()
        state.callbacks.append(trace(log, env, 'state'))
        state.succeed(phase=Phase.STATE)

    kick = env.event()
    kick.callbacks.append(trace(log, env, 'kick'))
    kick.callbacks.append(starter)
    kick.succeed()  # default phase -> INPUT of micro step 0
    env.run()

    assert log == [
        (0, 0, Phase.INPUT, 'kick'),
        (0, 0, Phase.STATE, 'state'),
        (0, 0, Phase.OUTPUT, 'out'),
    ]


def test_earlier_and_same_phase_go_to_next_round(env, log):
    """Events produced in a later phase for an earlier (or the same) phase
    are deferred to the next micro step.  Events sharing the resulting slot
    keep their submission order (FIFO)."""
    def starter(event):
        first = env.event()
        first.callbacks.append(trace(log, env, 'first'))
        first.succeed(phase=Phase.INPUT)

        second = env.event()
        second.callbacks.append(trace(log, env, 'second'))
        second.succeed(phase=Phase.INPUT)

        earlier = env.event()
        earlier.callbacks.append(trace(log, env, 'earlier'))
        earlier.succeed(phase=Phase.STATE)

    kick = env.event()
    kick.callbacks.append(trace(log, env, 'kick'))
    kick.callbacks.append(starter)
    kick.succeed(phase=Phase.OUTPUT)
    env.run()

    assert [(m, p, n) for _, m, p, n in log] == [
        (0, Phase.OUTPUT, 'kick'),
        (1, Phase.INPUT, 'first'),
        (1, Phase.INPUT, 'second'),
        (1, Phase.STATE, 'earlier'),
    ]


def test_timeout_advances_time_and_resets_microstep(env, log):
    """A normal Timeout advances the physical time and the micro step count
    starts over at the new physical point in time."""
    def feedback(event):
        # OUTPUT -> INPUT pushes the model into micro step 1.
        delayed = env.event()
        delayed.callbacks.append(trace(log, env, 'same_tick'))
        delayed.succeed(phase=Phase.INPUT)

    def clock(event):
        log.append(('timeout-created', env.now, env.microstep))
        env.timeout(3)

    fb = env.event()
    fb.callbacks.append(trace(log, env, 'fb'))
    fb.callbacks.append(feedback)
    fb.succeed(phase=Phase.OUTPUT)

    tick = env.event()
    tick.callbacks.append(clock)
    tick.succeed(phase=Phase.OUTPUT)

    env.run()

    assert log[:3] == [
        (0, 0, Phase.OUTPUT, 'fb'),
        ('timeout-created', 0, 0),
        (0, 1, Phase.INPUT, 'same_tick'),
    ]
    assert env.now == 3
    assert env.microstep == 0
    assert env.peek() == simpy.core.Infinity


def test_timeout_can_target_phase_of_next_tick(env, log):
    """An explicit phase on a positive delay selects the phase in the first
    micro step of the new physical time."""
    event = env.timeout(2, phase=Phase.STATE)
    event.callbacks.append(trace(log, env, 'tick'))
    env.run()
    assert log == [(2, 0, Phase.STATE, 'tick')]


def test_microstep_limit_reports_cycling_events():
    """Hitting the micro step limit raises and keeps the cycling events in
    the queue instead of silently dropping them."""
    env = simpy.PhasedEnvironment(microstep_limit=2)
    fired = [0]

    def make_loop():
        event = env.event()

        def loop(_):
            fired[0] += 1
            nxt = env.event()
            nxt.callbacks.append(loop)
            # OUTPUT feeding back into INPUT forces a new micro step.
            nxt.succeed(phase=Phase.INPUT)

        event.callbacks.append(loop)
        event.succeed(phase=Phase.OUTPUT)

    make_loop()

    with pytest.raises(MicrostepLimit) as excinfo:
        env.run()

    error = excinfo.value
    assert error.time == 0
    assert error.microstep == 2
    assert error.limit == 2
    assert len(error.pending_events) == 1
    # The event was not silently skipped: it is still scheduled.
    assert env.peek() == 0
    assert 'still cycling' in str(error)


def test_microstep_limit_is_per_physical_time():
    """The micro step budget is renewed whenever physical time advances."""
    env = simpy.PhasedEnvironment(microstep_limit=1)

    def tick(env):
        for current in range(3):
            event = env.event()
            event.callbacks.append(lambda _: None)
            event.succeed(phase=Phase.INPUT)  # one feedback micro step
            yield env.timeout(1)

    env.process(tick(env))
    env.run()
    assert env.now == 3


# ---------------------------------------------------------------------------
# Two-level combinational logic
# ---------------------------------------------------------------------------

class CombinationalModel(object):
    """Two inverters in series driven by one input.

    ``input`` drives gate ``g1`` (level 1) which drives gate ``g2``
    (level 2); ``g2`` produces ``output``.  Every gate recomputes in the
    INPUT phase, so a change of the input needs two micro steps to reach the
    output.  An optional feedback wire connects the output back to the input
    with zero delay.
    """

    def __init__(self, env, log, feedback=False):
        self.env = env
        self.log = log
        self.feedback = feedback
        # Initial stable state: input=0 -> g1=1 -> g2=0 -> output=0.
        self.signals = {'input': 0, 'g1': 1, 'g2': 0, 'output': 0}

    def _drive(self, wire, value, phase=Phase.INPUT):
        event = self.env.event()
        event.callbacks.append(lambda _: self._update(wire, value))
        event.succeed(phase=phase)

    def _update(self, wire, value):
        if self.signals[wire] == value:
            return
        self.signals[wire] = value
        self.log.append((self.env.now, self.env.microstep, wire, value))

        if wire == 'input':
            self._drive('g1', 1 - value)
        elif wire == 'g1':
            self._drive('g2', 1 - value)
        elif wire == 'g2':
            # Two inversions in series: the output equals the gate-2 signal.
            self._drive('output', value, phase=Phase.OUTPUT)
        elif wire == 'output' and self.feedback:
            # Zero-delay feedback of the output into the combinational input.
            self._drive('input', 1 - value)

    def set_input(self, value):
        # External stimulus is applied directly to the current INPUT stage.
        event = self.env.event()
        event.callbacks.append(lambda _: self._update('input', value))
        event.succeed()


def test_two_level_combinational_logic_settles():
    """A two-level circuit reaches a stable output within a finite number of
    micro steps: input=1 -> g1=0 -> g2=1 -> output=1."""
    env = simpy.PhasedEnvironment()
    log = []
    model = CombinationalModel(env, log)

    model.set_input(1)
    env.run()

    assert model.signals == {'input': 1, 'g1': 0, 'g2': 1, 'output': 1}
    # input at m0/INPUT, g1 at m1/INPUT, g2 at m2/INPUT, output sampled in
    # m2's OUTPUT stage.
    assert log == [
        (0, 0, 'input', 1),
        (0, 1, 'g1', 0),
        (0, 2, 'g2', 1),
        (0, 2, 'output', 1),
    ]
    # Once settled there is nothing left to do.
    assert env.peek() == simpy.core.Infinity


def test_zero_delay_feedback_is_diagnosed_in_finite_steps():
    """Closing the loop with zero-delay feedback keeps the combinational
    logic oscillating.  The micro step limit diagnoses it in finite steps and
    reports the events that are still cycling."""
    env = simpy.PhasedEnvironment(microstep_limit=5)
    log = []
    model = CombinationalModel(env, log, feedback=True)

    model.set_input(1)
    with pytest.raises(MicrostepLimit) as excinfo:
        env.run()

    error = excinfo.value
    assert error.time == 0
    assert error.microstep == 5
    # The output starts oscillating instead of settling to a stable value.
    outputs = [value for _, _, wire, value in log if wire == 'output']
    assert outputs[:2] == [1, 0]
    # Signals keep changing up to the limit (the loop never reaches a fixed
    # point in a finite number of micro steps).
    assert len(log) > 4
    # Cycling events are reported, not silently skipped.
    assert error.pending_events
    assert env.peek() == 0


def test_default_environment_queueing_unchanged():
    """The plain Environment must not expose physical micro steps and must
    keep its original FIFO queueing; the new *phase* argument is ignored."""
    env = simpy.Environment()
    assert not hasattr(env, 'microstep')

    order = []
    for name in ('a', 'b', 'c'):
        event = env.event()
        event.callbacks.append(lambda _, n=name: order.append(n))
        event.succeed(phase=Phase.OUTPUT)  # phase is a no-op here
    env.run()
    assert order == ['a', 'b', 'c']


def test_plain_processes_work_in_phased_environment():
    """Ordinary SimPy processes (which never request a phase) keep running in
    the current phase and micro step and do not accumulate micro steps."""
    env = simpy.PhasedEnvironment(microstep_limit=10)
    seen = []

    def pem(env):
        for expected in range(3):
            assert env.microstep == 0
            seen.append((env.now, env.phase))
            yield env.timeout(1)

    env.process(pem(env))
    env.run()
    assert seen == [(0, Phase.INPUT), (1, Phase.INPUT), (2, Phase.INPUT)]
