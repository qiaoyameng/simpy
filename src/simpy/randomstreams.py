"""
Deterministic random streams for simulations.

This module is intended for reproducible simulation experiments.  It derives
an independent random number generator for every ``(entity_key,
purpose_key)`` pair, so adding or changing another model branch cannot shift
the draws used by existing entities.

The keys and seed are encoded with explicit type tags and length prefixes.
Values that merely look the same after conversion to text are kept apart.
Python's process-randomized ``hash()`` is deliberately not used.

Although HMAC-SHA256 is used to derive stream seeds, this is not a security
or cryptographic facility.  In particular, :class:`random.Random` is not
unpredictable and should not be used for secrets.

"""
import hashlib
import hmac
import random
import struct


__all__ = ['RandomStreamManager']


_DOMAIN = b'simpy.randomstreams.v1'
_LENGTH = struct.Struct('>Q')

_NONE = b'n'
_BOOL = b'b'
_INT = b'i'
_FLOAT = b'f'
_STRING = b's'
_BYTES = b'y'


def _frame(tag, payload=b''):
    return tag + _LENGTH.pack(len(payload)) + payload


def _encode(value):
    """Return a canonical, collision-resistant encoding of a basic key."""
    value_type = type(value)

    if value is None:
        return _frame(_NONE)
    if value_type is bool:
        return _frame(_BOOL, b'\x01' if value else b'\x00')
    if value_type is int:
        return _frame(_INT, str(value).encode('ascii'))
    if value_type is float:
        return _frame(_FLOAT, struct.pack('>d', value))
    if value_type is str:
        return _frame(_STRING, value.encode('utf-8'))
    if value_type is bytes:
        return _frame(_BYTES, value)

    raise TypeError(
        'random stream keys and trial seeds must be one of None, bool, int, '
        'float, str, or bytes; got %s' % value_type.__name__
    )


class RandomStreamManager(object):
    """Manage independent, deterministically seeded simulation streams.

    :param trial_seed: A ``None``, :class:`bool`, :class:`int`,
        :class:`float`, :class:`str`, or :class:`bytes` value identifying the
        simulation trial.

    Call :meth:`stream` with an entity key and a purpose key to retrieve the
    corresponding :class:`random.Random` instance.  Repeated calls return the
    same stream, while different key pairs get independent streams.

    """

    def __init__(self, trial_seed):
        self._trial_seed = trial_seed
        self._trial_seed_key = _encode(trial_seed)
        self._streams = {}

    @property
    def trial_seed(self):
        """The seed identifying this experiment."""
        return self._trial_seed

    def stream(self, entity_key, purpose_key):
        """Return the RNG for one entity and sampling purpose.

        The returned object is a normal :class:`random.Random`.  It advances
        only when that particular stream is sampled.

        """
        encoded_entity = _encode(entity_key)
        encoded_purpose = _encode(purpose_key)
        cache_key = (encoded_entity, encoded_purpose)

        rng = self._streams.get(cache_key)
        if rng is None:
            message = (
                _DOMAIN
                + _frame(b'entity', encoded_entity)
                + _frame(b'purpose', encoded_purpose)
            )
            digest = hmac.new(
                self._trial_seed_key, message, hashlib.sha256
            ).digest()
            stream_seed = int.from_bytes(digest, byteorder='big')
            rng = random.Random(stream_seed)
            self._streams[cache_key] = rng

        return rng

    def copy_initial_layout(self):
        """Return a manager with this trial's untouched random streams.

        Use this to run two model variants from identical initial randomness.
        The returned manager is independent from this manager, even if some
        streams in the current manager have already been sampled.

        """
        return self.__class__(self._trial_seed)

    def reset(self):
        """Restore all streams to their initial trial state."""
        self._streams.clear()
