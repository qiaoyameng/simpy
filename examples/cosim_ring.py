"""Three independent SimPy models in a delayed ring.

三个模型 A -> B -> C -> A 组成带严格正延迟的环。
用不同的协调器选择次序推进,业务轨迹(谁在什么时刻收到哪个令牌)
必须完全相同。

Run with::

    python examples/cosim_ring.py
"""
from simpy.cosim import Connection, Coordinator, Model


# Per-hop minimum transmission delays. 每一跳的最小传输延迟。
DELAYS = {('A', 'B'): 2, ('B', 'C'): 1, ('C', 'A'): 3}
TOTAL_HOPS = 6


def build_federation():
    models = {name: Model(name) for name in ('A', 'B', 'C')}
    connections = {}
    for (src, dst), delay in DELAYS.items():
        conn = Connection('%s->%s' % (src, dst), models[src], models[dst],
                          min_delay=delay)
        connections[(src, dst)] = conn

    def make_process(model):
        outgoing = {(s, d): c for (s, d), c in connections.items()
                    if s == model.name}

        def pem():
            # A seeds the token at time 0; everyone else waits on inbox.
            if model.name == 'A':
                token = {'id': 'token-0', 'hops': 0}
            else:
                token = (yield model.inbox.get()).payload
            next_name = {'A': 'B', 'B': 'C', 'C': 'A'}
            while True:
                if token.get('shutdown'):
                    # Propagate the shutdown one hop further, then end.
                    dst = next_name[model.name]
                    outgoing[(model.name, dst)].send({'shutdown': True})
                    model.trace.append((model.env.now, 'shutdown', None))
                    break
                model.trace.append((model.env.now, token['id'],
                                    token['hops']))
                if token['hops'] >= TOTAL_HOPS:
                    # A owns the last hop: start the shutdown cascade.
                    dst = next_name[model.name]
                    outgoing[(model.name, dst)].send({'shutdown': True})
                    break
                dst = next_name[model.name]
                outgoing[(model.name, dst)].send(
                    {'id': token['id'],
                     'hops': token['hops'] + 1})
                token = (yield model.inbox.get()).payload
        return pem

    for model in models.values():
        model.trace = []
        model.process(make_process(model)())
    return models, connections


def business_trace(models):
    """Ordered (time, receiver, token-id, hop) business trajectory."""
    rows = []
    for name in ('A', 'B', 'C'):
        for time, token_id, hops in models[name].trace:
            if token_id == 'shutdown':
                continue
            rows.append((time, name, token_id, hops))
    return sorted(rows)


def main():
    policies = ('round-robin', 'name', 'reverse', 'earliest')
    traces = {}
    for policy in policies:
        models, connections = build_federation()
        coordinator = Coordinator(
            list(models.values()), list(connections.values()),
            select=policy)
        result = coordinator.run()
        traces[policy] = business_trace(models)
        print('select=%-11s -> %s, local times %s'
              % (policy, result.status, result.times))

    reference = traces['round-robin']
    print()
    for row in reference:
        print('  t=%2d  %s receives %s (hop %d)'
              % (row[0], row[1], row[2], row[3]))
    print()
    assert all(traces[p] == reference for p in policies), traces
    print('OK: changing the advancement order did not change the business '
          'trajectory (%d receipts).' % len(reference))


if __name__ == '__main__':
    main()
