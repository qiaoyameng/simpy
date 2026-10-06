.. _random_streams:

================
Random Streams
================

When comparing two model variants, a global :mod:`random` seed is fragile:
adding one draw in a branch shifts every later draw in the same trial.  The
apparent diff then contains unrelated sampling noise.

:class:`~simpy.RandomStreamManager` gives each entity and sampling purpose
its own stateful stream.  Streams are derived from the trial seed, an entity
key, and a purpose key::

    from simpy import RandomStreamManager

    streams = RandomStreamManager(trial_seed=20261007)
    arrivals = streams.stream('source', 'interarrival')
    service = streams.stream('customer-3', 'service')

Repeated calls with the same keys return the same stream.  Sampling another
stream, including a stream used only by an alternative model branch, does
not move this stream's sequence.

Keys must be ``None``, ``bool``, ``int``, ``float``, ``str``, or ``bytes``.
The manager encodes values together with their type, so ``1`` and ``'1'``
cannot become the same key.  It never relies on Python's process-randomized
``hash()``.

To compare scenarios from identical initial randomness, copy the trial
layout before drawing values::

    baseline = streams.copy_initial_layout()
    alternative = streams.copy_initial_layout()

Run each model with its own manager.  Common entities then draw the same
interarrival times and service demands item by item, while an extra branch
only advances that branch's separate streams.  Changing ``trial_seed``
produces a different set of sequences.

.. warning::

   Stream seeds are derived with HMAC-SHA256 to get deterministic,
   well-separated streams.  This facility is for reproducible simulations,
   not cryptography.  The generated :class:`random.Random` objects are not
   unpredictable and must not be used for passwords, tokens, or secrets.
