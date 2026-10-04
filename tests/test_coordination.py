import pytest
import simpy

from simpy import Connection, Coordinator
from simpy.coordination import COMPLETED, DEADLOCKED, Message, message_order


def make_ring(delays=(1, 1, 1)):
    envs = tuple(simpy.Environment() for _ in range(3))
    coordinator = Coordinator(environments=envs)
    links = tuple(
        coordinator.add_connection(
            source=index,
            target=(index + 1) % 3,
            delay=delay)
        for index, delay in enumerate(delays))
    return envs, coordinator, links


def incoming_link(links, model):
    return links[(model - 1) % len(links)]


def test_ring_trace_is_independent_of_model_order():
    traces = []
    orders = (
        ('0', '1', '2'),
        ('0', '2', '1'),
        ('1', '0', '2'),
        ('1', '2', '0'),
        ('2', '0', '1'),
        ('2', '1', '0'),
    )

    for order in orders:
        envs, coordinator, links = make_ring((1, 2, 3))
        trace = []

        def make_process(index):
            def process(env):
                if index == 0:
                    yield env.timeout(1)
                    links[0].send('seed')
                message = yield incoming_link(links, index).recv()
                trace.append((env.now, index, message))
                if index != 0:
                    links[index].send(message)
            return process

        for index, env in enumerate(envs):
            env.process(make_process(index)(env))

        result = coordinator.run(order=order)
        assert result.status == COMPLETED
        traces.append(trace)

    assert traces[0] == traces[1] == traces[2]
    assert traces[0] == traces[3] == traces[4] == traces[5]
    assert traces[0] == [
        (2.0, 1, 'seed'), (4.0, 2, 'seed'), (7.0, 0, 'seed')]


def test_zero_delay_connection_is_rejected():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    coordinator = Coordinator((env_a, env_b))

    with pytest.raises(Exception, match='positive'):
        coordinator.add_connection(0, 1, 0)


def test_nan_delay_connection_is_rejected():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    coordinator = Coordinator((env_a, env_b))

    with pytest.raises(Exception, match='positive'):
        coordinator.add_connection(0, 1, float('nan'))


def test_message_order_is_public_and_stable():
    envs = tuple(simpy.Environment() for _ in range(3))
    coordinator = Coordinator(envs)
    first = coordinator.add_connection(0, 2, 1)
    second = coordinator.add_connection(1, 2, 1)
    messages = (
        Message(2, 1, '1', '2', 'b', second),
        Message(2, 0, '0', '2', 'a', first),
        Message(1, 0, '0', '2', 'old', first),
    )

    assert [message.value for message in sorted(messages,
                                                key=message_order)] == [
        'old', 'a', 'b']


def test_direct_cross_environment_yield_is_rejected():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    coordinator = Coordinator((env_a, env_b))

    def process(env, foreign_event):
        yield foreign_event

    env_a.process(process(env_a, env_b.event()))

    with pytest.raises(Exception, match='another Environment'):
        coordinator.run()


def test_direct_cross_environment_schedule_is_rejected():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    coordinator = Coordinator((env_a, env_b))

    def process(env):
        yield env.timeout(1)
        env_b.schedule(env.event())

    env_a.process(process(env_a))

    with pytest.raises(Exception, match='must use a Connection'):
        coordinator.run()


def test_foreign_event_scheduled_before_run_is_rejected():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    Coordinator((env_a, env_b))

    with pytest.raises(Exception, match='must use a Connection'):
        env_b.schedule(env_a.event())


def test_foreign_event_triggered_in_target_environment_is_rejected():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    coordinator = Coordinator((env_a, env_b))
    foreign_event = env_a.event()

    def process(env):
        yield env.timeout(1)
        foreign_event.succeed()

    env_b.process(process(env_b))

    with pytest.raises(Exception, match='must use a Connection'):
        coordinator.run()


def test_send_in_past_is_rejected():
    envs, coordinator, links = make_ring()

    def send_late(env):
        yield env.timeout(3)
        with pytest.raises(Exception, match='Cannot send.*past'):
            links[0].send_at('late', at=1)

    def receive(env):
        yield incoming_link(links, 1).recv()

    def receive_two(env):
        yield incoming_link(links, 2).recv()

    envs[0].process(send_late(envs[0]))
    envs[1].process(receive(envs[1]))
    envs[2].process(receive_two(envs[2]))

    result = coordinator.run()
    assert result.status == DEADLOCKED


def test_requested_delay_below_minimum_is_rejected():
    envs, coordinator, links = make_ring()

    def sender(env):
        yield env.timeout(1)
        links[0].send('fast', delay=0.5)

    def wait(model):
        def process(env):
            yield incoming_link(links, model).recv()
        return process

    envs[0].process(sender(envs[0]))
    envs[1].process(wait(1)(envs[1]))
    envs[2].process(wait(2)(envs[2]))

    with pytest.raises(Exception, match='below minimum'):
        coordinator.run()


def test_same_time_messages_use_public_stable_order():
    envs = tuple(simpy.Environment() for _ in range(3))
    coordinator = Coordinator(envs)
    first = coordinator.add_connection(0, 2, 1)
    second = coordinator.add_connection(1, 2, 1)
    received = []

    def sender(env, link, value):
        yield env.timeout(1)
        link.send(value)

    def receiver(env):
        first_value = yield first.recv()
        received.append((env.now, first_value))
        second_value = yield second.recv()
        received.append((env.now, second_value))

    envs[0].process(sender(envs[0], first, 'first'))
    envs[1].process(sender(envs[1], second, 'second'))
    envs[2].process(receiver(envs[2]))

    result = coordinator.run(order=('1', '0', '2'))
    assert result.status == COMPLETED
    assert received == [(2, 'first'), (2, 'second')]
    assert first.index < second.index


def test_message_arrives_before_a_receiver_is_created():
    env_a = simpy.Environment()
    env_b = simpy.Environment()
    coordinator = Coordinator((env_a, env_b))
    link = coordinator.add_connection(0, 1, 2)

    def sender(env):
        yield env.timeout(1)
        link.send('stored')

    def receiver(env):
        yield env.timeout(4)
        value = yield link.recv()
        return (env.now, value)

    sender_process = env_a.process(sender(env_a))
    receiver_process = env_b.process(receiver(env_b))
    result = coordinator.run()

    assert result.status == COMPLETED
    assert result.times == {'0': 1.0, '1': 4.0}
    assert sender_process.value is None
    assert receiver_process.value == (4, 'stored')


def test_deadlock_is_reported_without_rollback():
    envs, coordinator, links = make_ring()
    times = []

    def wait(index):
        def process(env):
            times.append(env.now)
            yield incoming_link(links, index).recv()
        return process

    for index, env in enumerate(envs):
        env.process(wait(index)(env))

    result = coordinator.run(order=('2', '1', '0'))
    assert result.status == DEADLOCKED
    assert 'No model can safely advance' in result.reason
    assert result.times == {'0': 0, '1': 0, '2': 0}
