"""
Explicit random stream management for reproducible simulations.

Comparing two scheduling policies with a single shared (or global) random
generator is fragile: as soon as one branch of a model draws one more (or
one fewer) random number than the other, every subsequent draw is shifted
and the observed differences are polluted by sampling noise.

:class:`RandomStreamManager` solves this by deriving an independent
sub-stream (a dedicated :class:`random.Random` instance) for every
``(entity_key, purpose_key)`` pair from the *trial seed*.  Each sub-stream
advances strictly in its own sampling order, so adding entities or drawing
extra values in one branch of the model never perturbs the streams of the
other entities.  Two models that differ in only one branch therefore see
bit-for-bit identical random numbers for their common entities.

The derivation is a deterministic function of the trial seed and the keys
(SHA-256 over a canonical, type-tagged encoding).  It is stable across
processes, platforms and ``PYTHONHASHSEED`` values because it never relies
on the process-randomized builtin :func:`hash`.  It is meant for
*reproducible simulation only* -- it is **not** a cryptographic primitive
and must not be used to derive secrets, tokens or passwords.

"""
import hashlib
import random
import struct


#: Domain separator baked into every derived seed, so that accidental
#: collisions with other SHA-256 based derivations are avoided.
_DOMAIN = b'simpy-random-stream-v1\x00'


def _encode_key(value):
    """Encode *value* into an unambiguous, deterministic byte string.

    Only basic types with a deterministic encoding are accepted:
    ``None``, ``bool``, ``int``, ``float``, ``str`` and ``bytes``.  Every
    encoding carries a type tag and a length prefix, so same-shaped values
    of different types (e.g. ``1``, ``1.0``, ``True`` and ``'1'``) never
    collapse into the same key, and concatenated keys can always be split
    unambiguously.

    """
    if value is None:
        payload = b'N'
    elif isinstance(value, bool):
        # ``bool`` must be checked before ``int`` (it is a subclass).
        payload = b'B' + (b'\x01' if value else b'\x00')
    elif isinstance(value, int):
        payload = b'I' + str(value).encode('ascii')
    elif isinstance(value, float):
        payload = b'F' + struct.pack('>d', value)
    elif isinstance(value, str):
        payload = b'S' + value.encode('utf-8')
    elif isinstance(value, bytes):
        payload = b'Y' + value
    else:
        raise TypeError(
            'random stream keys must be None, bool, int, float, str or '
            'bytes; got %s (%r).  Compound or mutable values have no '
            'deterministic encoding -- build a basic-type key from them '
            'yourself.' % (type(value).__name__, value))
    return struct.pack('>I', len(payload)) + payload


def _derive_seed(trial_seed_enc, entity_enc, purpose_enc):
    """Derive an integer seed from the encoded trial seed and keys."""
    digest = hashlib.sha256(
        _DOMAIN + trial_seed_enc + entity_enc + purpose_enc).digest()
    seed = 0
    for byte in bytearray(digest):
        seed = (seed << 8) | byte
    return seed


class RandomStreamManager(object):
    """Manages independent random sub-streams for one simulation trial.

    *trial_seed* identifies the trial (replication) and must itself be a
    valid key (``None``, ``bool``, ``int``, ``float``, ``str`` or
    ``bytes``).  Use :meth:`stream` to obtain the sub-stream for a given
    entity and purpose::

        manager = RandomStreamManager(trial_seed=42)
        arrivals = manager.stream('machine-1', 'interarrival')
        service = manager.stream('machine-1', 'service')

    Each call with the same keys returns the same :class:`random.Random`
    instance, which advances only when *it* is sampled.  Streams for other
    entities or purposes are completely independent, so structural changes
    elsewhere in the model cannot shift their sequences.

    """

    def __init__(self, trial_seed):
        self._trial_seed = trial_seed
        self._trial_seed_enc = _encode_key(trial_seed)
        self._streams = {}
        self._initial_states = {}

    @property
    def trial_seed(self):
        """The trial seed this manager was created with."""
        return self._trial_seed

    def stream(self, entity_key, purpose_key):
        """Return the sub-stream for *entity_key* and *purpose_key*.

        Both keys must be basic types with a deterministic encoding (see
        :class:`RandomStreamManager`); anything else raises
        :exc:`TypeError`.  Repeated calls with equal keys return the same
        :class:`random.Random` instance.

        """
        cache_key = (_encode_key(entity_key), _encode_key(purpose_key))
        rng = self._streams.get(cache_key)
        if rng is None:
            rng = random.Random(_derive_seed(
                self._trial_seed_enc, cache_key[0], cache_key[1]))
            self._streams[cache_key] = rng
            # Remember the pristine state so copy_initial() can reproduce
            # the exact random layout of this trial later on.
            self._initial_states[cache_key] = rng.getstate()
        return rng

    def copy_initial(self):
        """Return a copy of this trial's *initial* random layout.

        The returned manager contains every sub-stream created so far, but
        reset to the state it had when it was first requested -- no matter
        how many values have been drawn from this manager's streams in the
        meantime.  This lets you re-run a trial (or a variant of the
        model) with exactly the same random layout as the original run.

        """
        clone = type(self)(self._trial_seed)
        for cache_key, state in self._initial_states.items():
            rng = random.Random()
            rng.setstate(state)
            clone._streams[cache_key] = rng
            clone._initial_states[cache_key] = state
        return clone
