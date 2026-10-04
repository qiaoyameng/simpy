"""
Hybrid (continuous/discrete) simulation environment.

This module provides :class:`HybridEnvironment`, an optional extension of
:class:`simpy.core.Environment` that manages a finite-dimensional continuous
state alongside the usual discrete event queue.

The continuous state evolves according to a caller-provided derivative
function and is integrated with an adaptive Runge-Kutta scheme (the
Dormand-Prince RK45 pair) that uses an embedded error estimate for step-size
control.  Integration never steps past the next queued discrete event, so
discrete events can safely modify the state or the dynamics.

Threshold (zero-crossing) functions can be registered with
:meth:`HybridEnvironment.add_threshold`.  They support up-crossings,
down-crossings and crossings in any direction.  Once a sign change is
detected within an accepted step, the event time is located inside that step
and the step is truncated there.  The associated discrete callback may then
modify the state or the dynamics, and integration continues from the new
state.

"""
from simpy.core import Environment, EmptySchedule, Infinity
from simpy.exceptions import SimPyException


class HybridIntegrationError(SimPyException):
    """Raised when the continuous integration cannot meet its accuracy
    requirements, e.g. when the required step size falls below the minimum
    step size or when a threshold event cannot be located within the allowed
    number of root-finding iterations."""


#: Valid crossing directions for threshold functions.
DIRECTIONS = ('up', 'down', 'any')

# Dormand-Prince RK45 coefficients.
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_C = (0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1)
_B = (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0)
_E = (71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525,
      -1 / 40)

# Step-size controller tuning.
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0


class Threshold(object):
    """A registered threshold (zero-crossing) monitor.

    Instances are created via :meth:`HybridEnvironment.add_threshold`.

    """

    def __init__(self, env, func, callback, direction):
        self.env = env
        self.func = func
        self.callback = callback
        self.direction = direction
        #: Time of the last detected crossing (``None`` if never triggered).
        self.time = None
        self._g = func(env.now, env.state)

    def value(self):
        """Evaluate the threshold function at the environment's current
        time and state."""
        return self.func(self.env.now, self.env.state)

    def crossed(self, g1):
        """Check whether a crossing in the configured direction occurred
        between the memorized value and *g1*."""
        g0 = self._g
        if g0 == 0:
            # Exactly on the threshold; wait until the sign is defined.
            return False
        if self.direction == 'up':
            return g0 < 0 <= g1
        if self.direction == 'down':
            return g0 > 0 >= g1
        return (g0 < 0 <= g1) or (g0 > 0 >= g1)


class HybridEnvironment(Environment):
    """Simulation environment with an additional continuous state.

    The *initial_state* is a finite sequence of floats and *derivative* is a
    callable ``derivative(t, state)`` returning the time derivative of the
    state as a sequence of the same length.

    Integration uses the Dormand-Prince RK45 pair with adaptive step-size
    control based on the embedded error estimate (*rtol* and *atol*).  The
    step size is bounded below by *h_min* and above by *h_max*; if a step
    smaller than *h_min* would be required to meet the error tolerance, a
    :class:`HybridIntegrationError` is raised.

    Threshold events are located by bisection on the dense (Hermite) output
    of an accepted step until the bracket is smaller than *event_tol*.  At
    most *max_root_iterations* bisection iterations are performed before a
    :class:`HybridIntegrationError` is raised.

    """

    def __init__(self, initial_state, derivative, initial_time=0,
                 rtol=1e-6, atol=1e-9, h_initial=None, h_min=1e-12,
                 h_max=Infinity, event_tol=1e-9, max_root_iterations=50):
        super(HybridEnvironment, self).__init__(initial_time)
        self.state = [float(v) for v in initial_state]
        self.derivative = derivative
        self.rtol = rtol
        self.atol = atol
        self.h_min = h_min
        self.h_max = h_max
        self.event_tol = event_tol
        self.max_root_iterations = max_root_iterations
        self._h = h_initial
        self._thresholds = []

    def add_threshold(self, func, callback, direction='any'):
        """Monitor the threshold function *func* for zero-crossings.

        *func* is a callable ``func(t, state)`` returning a float.  When its
        sign changes in the given *direction* (``'up'``, ``'down'`` or
        ``'any'``), the crossing time is located within the current
        integration step, the environment's time and state are advanced to
        that point, and *callback* is invoked as ``callback(env, threshold)``.
        The callback may modify ``env.state`` or the dynamics; integration
        then continues from the new state.

        Returns the created :class:`Threshold` instance.

        """
        if direction not in DIRECTIONS:
            raise ValueError('direction must be one of %r, not %r' %
                             (DIRECTIONS, direction))
        threshold = Threshold(self, func, callback, direction)
        self._thresholds.append(threshold)
        return threshold

    def remove_threshold(self, threshold):
        """Stop monitoring a previously registered *threshold*."""
        self._thresholds.remove(threshold)

    def step(self):
        """Integrate up to the next discrete event and process it.

        If a threshold crossing occurs before the next queued discrete
        event, advance time and state to the crossing and invoke the
        threshold callback instead.

        """
        t_next = self.peek()
        if t_next == Infinity:
            # No discrete event bounds the integration interval; there is
            # nothing to advance towards.
            raise EmptySchedule()
        hit = self._integrate_to(t_next)
        if hit is None:
            super(HybridEnvironment, self).step()
        else:
            threshold, t_event = hit
            threshold.time = t_event
            threshold.callback(self, threshold)
            # The callback may have changed the state or the dynamics;
            # refresh all memorized threshold signs.
            for th in self._thresholds:
                th._g = th.func(self._now, self.state)

    # -- Continuous integration -------------------------------------------

    def _deriv(self, t, y):
        return [float(v) for v in self.derivative(t, y)]

    def _rk45_step(self, t, y, h):
        """Perform one Dormand-Prince RK45 step.

        Return ``(y_new, err, f0, f1)`` where *err* is the embedded error
        estimate and *f0*/*f1* are the derivatives at the step endpoints
        (reused for dense output).

        """
        f = self._deriv
        k = [f(t, y)]
        for i in range(1, 7):
            ti = t + _C[i] * h
            yi = [y[j] + h * sum(_A[i][m] * k[m][j]
                                 for m in range(i))
                  for j in range(len(y))]
            k.append(f(ti, yi))
        y_new = [y[j] + h * sum(_B[m] * k[m][j] for m in range(7))
                 for j in range(len(y))]
        err = [h * sum(_E[m] * k[m][j] for m in range(7))
               for j in range(len(y))]
        return y_new, err, k[0], k[6]

    def _error_norm(self, y, y_new, err):
        """RMS norm of the error estimate weighted by the tolerances."""
        total = 0.0
        for y0, y1, e in zip(y, y_new, err):
            scale = self.atol + self.rtol * max(abs(y0), abs(y1))
            total += (e / scale) ** 2
        return (total / len(err)) ** 0.5

    def _integrate_to(self, t_end):
        """Integrate from the current time to *t_end*.

        Return ``(threshold, t_event)`` if a threshold crossing was detected
        and located at *t_event* (time and state are then left at the
        crossing), or ``None`` if *t_end* was reached.

        """
        if self._h is None:
            span = t_end - self._now
            self._h = min(self.h_max, max(self.h_min, 0.1 * span))
        while self._now < t_end:
            h = min(self._h, t_end - self._now, self.h_max)
            if h < self.h_min:
                raise HybridIntegrationError(
                    'Step size %g fell below the minimum step size %g at '
                    't=%g; required accuracy cannot be reached.' %
                    (h, self.h_min, self._now))
            t0, y0 = self._now, self.state
            y1, err, f0, f1 = self._rk45_step(t0, y0, h)
            norm = self._error_norm(y0, y1, err)
            if norm <= 1.0:
                # Accept the step.
                t1 = t0 + h
                hit = self._check_thresholds(t0, y0, f0, t1, y1, f1)
                self._update_step_size(norm, h)
                if hit is not None:
                    threshold, t_event = hit
                    self._now = t_event
                    self.state = self._interpolate(t0, y0, f0, t1, y1, f1,
                                                   t_event)
                    return hit
                self._now = t1
                self.state = y1
            else:
                self._update_step_size(norm, h)
        return None

    def _update_step_size(self, norm, h):
        if norm == 0:
            factor = _MAX_FACTOR
        else:
            factor = _SAFETY * norm ** -0.2
            factor = min(_MAX_FACTOR, max(_MIN_FACTOR, factor))
        self._h = h * factor

    def _check_thresholds(self, t0, y0, f0, t1, y1, f1):
        """Check all thresholds for a crossing within the accepted step
        ``[t0, t1]``.  Return ``(threshold, t_event)`` for the earliest
        crossing, if any."""
        earliest = None
        for th in self._thresholds:
            g1 = th.func(t1, y1)
            if th.crossed(g1):
                if g1 == 0:
                    # The step lands exactly on the threshold.
                    t_event = t1
                else:
                    t_event = self._locate(th, t0, y0, f0, t1, y1, f1, g1)
                if earliest is None or t_event < earliest[1]:
                    earliest = (th, t_event)
            else:
                th._g = g1
        return earliest

    def _locate(self, threshold, t0, y0, f0, t1, y1, f1, g1):
        """Locate the crossing time of *threshold* within ``[t0, t1]`` by
        bisection on the dense output of the step."""
        lo, hi = t0, t1
        # Sign of the threshold function after the crossing.
        post_sign = 1.0 if g1 > 0 else -1.0
        for _ in range(self.max_root_iterations):
            if hi - lo <= self.event_tol:
                break
            mid = 0.5 * (lo + hi)
            y_mid = self._interpolate(t0, y0, f0, t1, y1, f1, mid)
            g_mid = threshold.func(mid, y_mid)
            if g_mid == 0:
                return mid
            if (g_mid < 0) == (g1 < 0):
                hi = mid
            else:
                lo = mid
        else:
            raise HybridIntegrationError(
                'Could not locate threshold event within %d iterations '
                '(t in [%g, %g]).' % (self.max_root_iterations, lo, hi))
        # Return the bracket endpoint that lies on the post-crossing side so
        # that the memorized sign does not immediately re-trigger.
        y_lo = self._interpolate(t0, y0, f0, t1, y1, f1, lo)
        if (threshold.func(lo, y_lo) < 0) == (post_sign < 0):
            return lo
        return hi

    @staticmethod
    def _interpolate(t0, y0, f0, t1, y1, f1, t):
        """Cubic Hermite dense output of the accepted step at time *t*."""
        h = t1 - t0
        theta = (t - t0) / h
        t2 = theta * theta
        t3 = t2 * theta
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + theta
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return [h00 * y0[j] + h10 * h * f0[j] + h01 * y1[j] + h11 * h * f1[j]
                for j in range(len(y0))]
