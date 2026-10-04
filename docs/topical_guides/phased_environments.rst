.. _phased-environments:

==================================
Phased Environments and Micro Steps
==================================

The normal :class:`~simpy.core.Environment` knows a single clock: the
simulation time.  That is a good match for queuing networks and similar
models, but digital control models use two different notions of "when":

- the *physical time* of a clock tick, and
- the *micro steps* (also called *delta cycles*) in which signals propagate
  *within* a single tick.

SimPy provides the optional
:class:`~simpy.phased.PhasedEnvironment` for such models.  It keeps the
physical time (``now``) separate from a :attr:`~simpy.phased.PhasedEnvironment.microstep`
counter and splits every micro step into three ordered
:class:`~simpy.phased.Phase` stages:

- :attr:`~simpy.phased.Phase.INPUT` -- compute the inputs,
- :attr:`~simpy.phased.Phase.STATE` -- update the state,
- :attr:`~simpy.phased.Phase.OUTPUT` -- sample the outputs.

Scheduling by phase
===================

An event is placed into a stage by passing the ``phase`` argument to
:meth:`~simpy.events.Event.succeed`, :meth:`~simpy.events.Event.trigger`,
:meth:`~simpy.events.Event.fail` or to :meth:`timeout
<simpy.phased.PhasedEnvironment.timeout>`::

    input_event.succeed(phase=simpy.Phase.STATE)

The rules are:

- an event for a **strictly later** stage of the round currently being
  processed is handled in the same micro step;
- an event for the **same** or an **earlier** stage cannot be handled in this
  round anymore and is carried over to the **next micro step**;
- events landing in the same stage and micro step keep the order in which they
  were submitted (FIFO).

Not passing a phase (``phase=None``) keeps the normal SimPy behaviour: the
event is appended to the stage and micro step that are currently being
processed and never opens a new micro step by itself.  That is why ordinary
SimPy processes keep working unchanged inside a ``PhasedEnvironment``.

Physical time and the micro step limit
======================================

A normal :class:`~simpy.events.Timeout` with a positive delay advances the
physical time.  Whenever the simulation crosses to a new physical point in
time, the micro step counter is reset to zero.

Zero-delay feedback is useful, but if a combinational part never settles the
signals keep oscillating forever.  To get a finite diagnosis instead of an
infinite loop, pass ``microstep_limit`` to the environment::

    env = simpy.PhasedEnvironment(microstep_limit=100)

If more than ``microstep_limit`` micro steps happen at a single point in
physical time, :meth:`~simpy.phased.PhasedEnvironment.step` raises a
:class:`~simpy.phased.MicrostepLimit`.  The exception exposes the physical
:attr:`~simpy.phased.MicrostepLimit.time`, the reached
:attr:`~simpy.phased.MicrostepLimit.microstep`, the configured
:attr:`~simpy.phased.MicrostepLimit.limit` and the
:attr:`~simpy.phased.MicrostepLimit.pending_events` that are still cycling.
These events are **not** silently skipped; they remain in the queue so the
model can be inspected.

The budget is per physical time and is renewed whenever a timeout advances the
clock.

The default :class:`~simpy.core.Environment` is unaffected: it does not expose
micro steps and ignores the optional ``phase`` argument, so its queueing rules
do not change.
