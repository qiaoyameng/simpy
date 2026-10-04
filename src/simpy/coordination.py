"""Conservative coordination of independent SimPy environments."""
from collections import namedtuple
from heapq import heappop, heappush
from itertools import count

from simpy.core import Environment, Infinity
from simpy.events import Event, NORMAL, URGENT
from simpy.exceptions import SimPyException


COORDINATED = URGENT - 1


COMPLETED = 'completed'
DEADLOCKED = 'deadlocked'


class CoordinationError(SimPyException):
    """Base class for coordination protocol errors."""


class IllegalCrossEnvironmentEvent(CoordinationError):
    """Raised when an event crosses environments without a connection."""


class InvalidConnection(CoordinationError):
    """Raised when a connection declaration is invalid."""


Message = namedtuple('Message', (
    'time', 'sequence', 'source', 'target', 'value', 'connection'
))


CoordinatorResult = namedtuple('CoordinatorResult',
                               ('status', 'reason', 'times'))


def message_order(message):
    """Public, stable ordering key for cross-model messages."""
    return (message.time, message.connection.index, message.sequence)


class Connection(object):
    """A directed, delay-constrained link between two models."""
    def __init__(self, source, target, delay):
        self.source = source
        self.target = target
        self.delay = float(delay)
        self._coordinator = None
        self._index = None
        self._sequences = count()
        self._inbox = []
        self._pending = []
        self._waiters = []

    @property
    def index(self):
        """Position in coordinator declaration order."""
        return self._index

    @property
    def inbox(self):
        """Messages received but not consumed by a process."""
        return self._inbox

    def send(self, value=None, delay=None):
        """Send *value* after at least the connection's minimum delay."""
        return self._coordinator._send(self, None, delay, value)

    def send_at(self, value=None, at=None):
        """Send *value* for delivery at absolute time *at*."""
        return self._coordinator._send(self, at, None, value)

    def recv(self):
        """Return a target-environment event for the next message."""
        return self._coordinator._receive(self)

    def _deliver(self, message):
        target = self._coordinator._model(self.target).environment
        if self._waiters:
            receipt = self._waiters.pop(0)
        else:
            receipt = Event(target)
            receipt._coordinated_delivery = True
            receipt.callbacks.append(self._store_message)

        receipt._ok = True
        receipt._value = message.value
        receipt._coordinated_delivery = True
        receipt._coordinated_message = message
        target.schedule(receipt, COORDINATED, message.time - target.now)

    def _store_message(self, event):
        value = event._value
        if self._waiters:
            receipt = self._waiters.pop(0)
            receipt._ok = True
            receipt._value = value
            receipt._coordinated_delivery = True
            self._coordinator._model(self.target).environment.schedule(
                receipt, COORDINATED)
        else:
            self._inbox.append(value)

    def __repr__(self):
        return 'Connection(%r -> %r, delay=%s)' % (
            self.source, self.target, self.delay)


class _ModelState(object):
    def __init__(self, model_id, environment):
        self.id = model_id
        self.environment = environment
        self.outgoing = []
        self.incoming = []


class Coordinator(object):
    """Conservatively coordinate a collection of private SimPy queues."""

    COMPLETED = COMPLETED
    DEADLOCKED = DEADLOCKED

    def __init__(self, environments=None, connections=None, order=None):
        self._models = []
        self._by_id = {}
        self._by_env = {}
        self._connections = []
        self._running = False
        self._active_environment = None

        if environments is not None:
            for environment in environments:
                self.add_model(environment)
        if connections is not None:
            for connection in connections:
                self.add_connection(connection=connection)
        self._order = None if order is None else list(order)

    @property
    def order(self):
        """Model selection order used when models are equally safe."""
        if self._order is None:
            return [model.id for model in self._models]
        return list(self._order)

    @property
    def connections(self):
        """Connections in their public declaration order."""
        return list(self._connections)

    def add_model(self, environment, model_id=None):
        """Register an environment and protect its event boundary."""
        if not isinstance(environment, Environment):
            raise CoordinationError('Model must be a SimPy Environment')
        if environment in self._by_env:
            raise CoordinationError('Environment is already registered')

        if model_id is None:
            model_id = str(len(self._models))
        model_id = str(model_id)
        if model_id in self._by_id:
            raise CoordinationError('Duplicate model id %r' % model_id)

        self._install_boundary(environment)
        state = _ModelState(model_id, environment)
        self._models.append(state)
        self._by_id[model_id] = state
        self._by_env[environment] = state
        return model_id

    def add_connection(self, source=None, target=None, delay=None,
                       connection=None):
        """Declare a directed connection with a strictly positive delay."""
        if connection is None:
            if source is None or target is None or delay is None:
                raise InvalidConnection(
                    'A connection needs source, target and delay')
            connection = Connection(source, target, delay)
        elif source is not None or target is not None or delay is not None:
            raise InvalidConnection(
                'Do not pass both a connection and connection fields')

        if not isinstance(connection, Connection):
            raise InvalidConnection('Expected a Connection instance')
        if connection._coordinator is not None:
            raise InvalidConnection('Connection is already registered')
        if not connection.delay > 0:
            raise InvalidConnection('Connection delay must be positive')

        source_id = self._resolve_id(connection.source)
        target_id = self._resolve_id(connection.target)
        if source_id == target_id:
            raise InvalidConnection('A connection must cross models')

        connection.source = source_id
        connection.target = target_id
        connection._index = len(self._connections)
        connection._coordinator = self

        source_state = self._by_id[source_id]
        target_state = self._by_id[target_id]
        source_state.outgoing.append(connection)
        target_state.incoming.append(connection)
        self._connections.append(connection)
        return connection

    def run(self, order=None, until=None):
        """Run all models until completion, *until*, or protocol deadlock."""
        if not self._models:
            return self._result(COMPLETED, 'No models registered')

        order_ids = self.order if order is None else [
            self._resolve_id(item) for item in order]
        if len(order_ids) != len(self._by_id) or \
                set(order_ids) != set(self._by_id):
            raise CoordinationError('Run order must contain every model once')

        if until is not None:
            until = float(until)
            for model in self._models:
                if until <= model.environment.now:
                    raise ValueError(
                        'until(=%s) should be > every current model time' %
                        until)

        self._running = True
        try:
            while True:
                earliest, safe = self._progress_bounds()
                for model in self._models:
                    self._inject_messages(model, safe[model.id])

                candidates = []
                for model_id in order_ids:
                    model = self._by_id[model_id]
                    next_time = model.environment.peek()
                    if until is not None and next_time >= until:
                        continue
                    external_ready = self._coordinated_event_is_next(model)
                    if next_time < safe[model.id]:
                        candidates.append(('strict', next_time, None, model))
                    elif next_time <= safe[model.id] and external_ready:
                        event = model.environment._queue[0][3]
                        candidates.append(('boundary', next_time,
                                           message_order(
                                               event._coordinated_message),
                                           model))

                selected = None
                strict = [item for item in candidates if item[0] == 'strict']
                if strict:
                    selected = min(strict, key=lambda item: (
                        item[1], order_ids.index(item[3].id)))[3]
                elif candidates:
                    selected = min(candidates, key=lambda item: (
                        item[1], item[2]))[3]

                if selected is not None:
                    self._active_environment = selected.environment
                    try:
                        selected.environment.step()
                    finally:
                        self._active_environment = None
                    continue

                if self._is_complete(until):
                    if until is None:
                        reason = 'All models completed'
                    else:
                        reason = 'All models reached time %s' % until
                    return self._result(COMPLETED, reason)

                return self._deadlock_result(until, earliest, safe)
        finally:
            self._running = False

    def _is_complete(self, until):
        if until is not None:
            if not all(model.environment.now >= until
                       for model in self._models):
                return False
        elif any(model.environment.peek() < Infinity
                 for model in self._models):
            return False
        for model in self._models:
            pending = any(connection._pending
                          for connection in model.incoming)
            waiting = any(self._receiver_ready(connection)
                          for connection in model.incoming)
            if pending or waiting:
                return False
        return True

    def _deadlock_result(self, until, earliest, safe):
        details = []
        for model in self._models:
            next_time = model.environment.peek()
            if next_time < Infinity and next_time >= safe[model.id]:
                details.append('model %s waits at %s for bound %s' % (
                    model.id, model.environment.now, safe[model.id]))
            elif any(connection._pending for connection in model.incoming):
                pending = min(connection._pending[0][1].time
                              for connection in model.incoming
                              if connection._pending)
                details.append('model %s cannot consume message at %s' % (
                    model.id, pending))
            elif until is not None and model.environment.now < until:
                details.append('model %s stopped at %s' % (
                    model.id, model.environment.now))
        reason = 'No model can safely advance; ' + ', '.join(
            details or ['all models are idle'])
        return self._result(DEADLOCKED, reason)

    def _result(self, status, reason):
        times = dict((model.id, model.environment.now)
                     for model in self._models)
        return CoordinatorResult(status, reason, times)

    def _send(self, connection, at, delay, value):
        source_state = self._model(connection.source)
        target_state = self._model(connection.target)
        source = source_state.environment
        target = target_state.environment

        if not self._running:
            raise CoordinationError(
                'Messages can only be sent while the coordinator is running')
        if source.active_process is None or source.active_process.env is \
                not source:
            raise CoordinationError(
                'Connection.send must be called from a source-model process')

        now = source.now
        if at is None:
            if delay is None:
                delay = connection.delay
            delay = float(delay)
            if not delay >= connection.delay:
                raise CoordinationError(
                    'Requested delay %s is below minimum %s' %
                    (delay, connection.delay))
            at = now + delay
        else:
            at = float(at)
            if not at >= now:
                raise CoordinationError('Cannot send a message into the past')
            minimum_time = now + connection.delay
            if not at >= minimum_time:
                raise CoordinationError(
                    'Requested arrival time %s is below minimum %s' %
                    (at, minimum_time))
        if not at >= target.now:
            raise CoordinationError(
                'Target model has already passed the requested arrival time')

        message = Message(at, next(connection._sequences),
                          connection.source, connection.target, value,
                          connection)
        entry = (message_order(message), message)
        heappush(connection._pending, entry)
        return message

    def _receive(self, connection):
        target = self._model(connection.target).environment
        active = target.active_process
        if active is not None and active.env is not target:
            raise CoordinationError(
                'Connection.recv must be called by the target model')

        if connection._inbox:
            value = connection._inbox.pop(0)
            receipt = Event(target)
            receipt._ok = True
            receipt._value = value
            receipt._coordinated_delivery = True
            target.schedule(receipt, COORDINATED)
            return receipt

        receipt = Event(target)
        connection._waiters.append(receipt)
        return receipt

    def _progress_bounds(self):
        earliest = {}
        for model in self._models:
            value = model.environment.peek()
            for connection in model.incoming:
                if self._receiver_ready(connection) and connection._pending:
                    value = min(value, connection._pending[0][1].time)
            earliest[model.id] = value

        # Waiting processes make a model reachable through chains of future
        # sends. Relax the finite connection graph until bounds stop moving.
        for _ in range(len(self._models)):
            updated = dict(earliest)
            for model in self._models:
                for connection in model.incoming:
                    if self._receiver_ready(connection):
                        source_time = earliest[connection.source]
                        if source_time < Infinity:
                            updated[model.id] = min(
                                updated[model.id],
                                source_time + connection.delay)
            earliest = updated

        safe = {}
        for model in self._models:
            value = Infinity
            for connection in model.incoming:
                source_time = earliest[connection.source]
                if source_time < Infinity:
                    value = min(value, source_time + connection.delay)
                if connection._pending:
                    value = min(value, connection._pending[0][1].time)
            safe[model.id] = value
        return earliest, safe

    def _receiver_ready(self, connection):
        return bool(connection._waiters and
                    connection._waiters[0].callbacks)

    def _coordinated_event_is_next(self, model):
        queue = model.environment._queue
        return bool(queue and getattr(queue[0][3],
                                      '_coordinated_delivery', False))

    def _inject_messages(self, model, safe_time):
        if safe_time == Infinity:
            due = []
        else:
            due = []
            for connection in model.incoming:
                while connection._pending and \
                        connection._pending[0][1].time <= safe_time:
                    due.append(heappop(connection._pending))

        for _, message in sorted(due, key=lambda item: item[0]):
            message.connection._deliver(message)

    def _install_boundary(self, environment):
        current = getattr(environment, '_simpy_coordinator', None)
        if current is not None and current is not self:
            raise CoordinationError(
                'Environment is already coordinated by another coordinator')
        if current is self:
            return

        environment._simpy_coordinator = self
        original_schedule = environment.schedule

        def guarded_schedule(event, priority=NORMAL, delay=0):
            event_environment = getattr(event, 'env', None)
            coordinated = getattr(event, '_coordinated_delivery', False)
            active_environment = self._active_environment
            if not coordinated and (
                    event_environment is not environment or
                    (active_environment is not None and
                     active_environment is not environment)):
                raise IllegalCrossEnvironmentEvent(
                    'Cross-environment events must use a Connection')
            return original_schedule(event, priority, delay)

        def validate_process_event(event):
            event_environment = getattr(event, 'env', None)
            if event_environment is not None and \
                    event_environment is not environment and not getattr(
                        event, '_coordinated_delivery', False):
                raise IllegalCrossEnvironmentEvent(
                    'Process yielded an event from another Environment')
            return event

        environment.schedule = guarded_schedule
        environment._validate_process_event = validate_process_event

    def _resolve_id(self, model):
        if isinstance(model, Environment):
            try:
                return self._by_env[model].id
            except KeyError:
                raise CoordinationError('Unknown model environment')

        model_id = str(model)
        if model_id not in self._by_id:
            raise CoordinationError('Unknown model %r' % model_id)
        return model_id

    def _model(self, model):
        return self._by_id[self._resolve_id(model)]


ConservativeCoordinator = Coordinator
DelayedConnection = Connection
CoordinatorStopped = CoordinatorResult
