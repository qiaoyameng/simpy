.. _cosim:

=================================================================
Conservative co-simulation of independent SimPy environments
=================================================================

:mod:`simpy.cosim` runs several models — each with its **own** SimPy
:class:`~simpy.core.Environment` and its own event heap — in a single process,
without ever merging the per-environment heaps, without advancing a model past
a promised message time, and without ever rolling a clock back.

The protocol is the classic conservative (Chandy–Misra style) scheme driven
purely by the lookahead declared on point-to-point connections.

Connections and minimum delay
==============================

A cross-model event may only travel through a declared
:class:`~simpy.cosim.Connection`. Every connection states a **strictly
positive** ``min_delay``::

    conn = Connection('A->B', model_a, model_b, min_delay=2)

While model A has not passed time ``t``, nothing it sends on this connection
can arrive earlier than ``t + 2``. That promise is the only thing the
coordinator needs.

Safe-horizon rule
==================

For a model ``M`` the coordinator computes, over every incoming connection
``c`` with source ``s`` and bound ``L``::

    safe_horizon(M) = min( earliest_message_already_in_transit(c),
                           s.now + L )

When the source process has terminated it can never send again, so the bound
collapses to the earliest concrete in-transit arrival. Models with no incoming
connections have an infinite horizon.

A model may only be chosen when it has a **concrete trigger** at or below its
horizon: a local event, or an in-transit message whose arrival time is now
safe. Advancing an empty clock merely to widen someone else's bound is an
unbounded null advance and is treated as a *stall* instead.

Same-time messages
===================

All messages with ``arrival <= t`` for a chosen step are delivered in the
public, stable order::

    (arrival_time, connection_name, per_connection_sequence)

Within one connection this is FIFO; between connections at the same arrival
time the globally unique connection name breaks ties deterministically.
Local events already scheduled at ``t`` always fire before the messages
delivered at ``t``. This makes every model's local trace independent of the
order in which the coordinator picks simultaneously eligible models.

Stopping
=========

``Coordinator.run()`` returns a :class:`~simpy.cosim.RunResult` whose status is
either ``simpy.cosim.COMPLETED`` (every process on every model terminated and
no messages in transit) or ``simpy.cosim.STALLED``, with a human-readable
``reason`` naming the blocked models, in-transit messages and local times.
Pass ``on_stall='raise'`` to get :class:`~simpy.cosim.exceptions.
ProtocolStalledError` instead.

Violation checks
=================

* :class:`~simpy.cosim.exceptions.ConnectionDeclarationError` — zero, negative
  or non-numeric ``min_delay``, a self-loop, or an undeclared endpoint.
* :class:`~simpy.cosim.exceptions.MinimumDelayError` — a send whose delay or
  explicit arrival time beats the declared bound.
* :class:`~simpy.cosim.exceptions.PastMessageError` — an arrival time behind
  the receiver's current clock.
* :class:`~simpy.cosim.exceptions.CrossModelScheduleError` — scheduling an
  event directly into another model's environment instead of going through a
  connection.

Selection policies
===================

``select`` accepts ``'round-robin'`` (default), ``'name'``, ``'reverse'``,
``'earliest'``, or a callable ``f(eligible_list) -> Eligible``. See
``examples/cosim_ring.py`` for a three-model delayed ring whose business
trajectory is identical under every policy, and ``examples/cosim_stall.py``
for a genuine lookahead deadlock reported as a stall.
