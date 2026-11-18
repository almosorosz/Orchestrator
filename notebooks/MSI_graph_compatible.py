"""Standalone replacement for the original MSI_encoder notebook classes.

Dependencies: networkx, matplotlib, pyvis. No CrystaLoop imports are required.
Use %run MSI_graph_compatible.py in a notebook, or paste this file into a cell.
"""
import hashlib
import colorsys
import html
import re
import uuid
from pathlib import Path

import networkx as nx
from pyvis.network import Network


def _items(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def _parse_body(body):
    if body.count('___') != 1:
        raise ValueError('Expected unit definitions followed by ___S_identifier[[sensors]]')
    units_text, sensor_text = body.split('___')
    match = re.fullmatch(r'(S_[^\[\]]+)\[\[([^\[\]]*)\]\]', sensor_text.strip())
    if not match:
        raise ValueError('Invalid sensor-system definition')
    units = []
    for entry in units_text.split('__') if units_text.strip() else []:
        unit = re.fullmatch(r'(U_[^(){}\[\],]+)\(([^()]*)\)\{([^{}]*)\}\[([^\[\]]*)\]', entry.strip())
        if not unit:
            raise ValueError(f'Invalid unit definition: {entry}')
        identifier, read_only, controlled, connections = unit.groups()
        reads, controls = _items(read_only), _items(controlled)
        if any(not value.startswith('R_') for value in reads):
            raise ValueError(f'{identifier}: read-only probes must start with R_')
        if any(not value.startswith('C_') for value in controls):
            raise ValueError(f'{identifier}: control probes must start with C_')
        units.append({'id': identifier, 'read': reads, 'control': controls, 'out': _items(connections)})
    group, sensors = match.groups()
    sensor_ids = _items(sensors)
    if any(not value.startswith('S_') for value in sensor_ids):
        raise ValueError('Sensor-system probes must start with S_')
    definitions = [u['id'] for u in units] + [p for u in units for p in u['read'] + u['control']]
    definitions += ([group] + sensor_ids) if sensor_ids else []
    if len(definitions) != len(set(definitions)):
        raise ValueError('Duplicate unit/probe identifier')
    return units, group, sensor_ids


class MSI_encoder:
    def __init__(self, MSI_code):
        self.MSI_code = MSI_code.strip()
        self.units, self.sensor_syst, self.sensor_ids = _parse_body(self.MSI_code)
        self.code = self.MSI_code.split('___')[0]
        self.sensors_raw = [self.MSI_code.split('___')[1]]
        self.sensor_lists = [self.sensor_ids]
        self.split_raw = self.code.split('__') if self.code else []
        self.unitops = [[u['id']] for u in self.units]
        self.reado_probs = [u['read'] for u in self.units]
        self.c_probs = [u['control'] for u in self.units]
        self.unit_conn = [u['out'] for u in self.units]
        self.sequence = self.seq_gen()
        self.coledg = ['#FFD700', '#1AFF8D', '#FF1493', '#DAF7A6', '#FFC300', '#FF33A6', '#00FF7F', '#FFC0CB']
        self._module_colors = {}

    def ext_struct(self):
        # Include isolated units as well as connected ones.
        return {u['id']: list(u['out']) for u in self.units}

    def ext_graph(self):
        import matplotlib.pyplot as plt
        graph = nx.DiGraph(self.ext_struct())
        nx.draw(graph, pos=nx.spring_layout(graph, seed=42), edgecolors='black', linewidths=2,
                width=2, with_labels=True, font_size=10, node_size=1000, node_color='lightblue')
        return plt.gcf()

    def general_struct(self):
        graph = {u['id']: list(dict.fromkeys(u['out'] + u['read'])) for u in self.units}
        for unit in self.units:
            for probe in unit['read']:
                graph[probe] = []
            for control in unit['control']:
                graph[control] = [unit['id']]
        if self.sensor_ids:
            graph[self.sensor_syst] = list(self.sensor_ids)
            graph.update({sensor: [] for sensor in self.sensor_ids})
        return graph

    def seq_gen(self):
        return [u['id'] for u in self.units] + [p for u in self.units for p in u['read'] + u['control']]

    def _graph(self):
        graph = nx.DiGraph()
        for unit in self.units:
            graph.add_node(unit['id'], role='unit')
        for unit in self.units:
            for target in unit['out']:
                if target not in graph:
                    graph.add_node(target, role='unit')
                graph.add_edge(unit['id'], target,
                               kind='attachment' if unit['id'].startswith('U_S_') else 'flow')
            for probe in unit['read']:
                graph.add_node(probe, role='read_only')
                graph.add_edge(unit['id'], probe, kind='read_only')
            for control in unit['control']:
                graph.add_node(control, role='controlled')
                graph.add_edge(control, unit['id'], kind='controlled')
        if self.sensor_ids:
            graph.add_node(self.sensor_syst, role='sensor_group')
            for sensor in self.sensor_ids:
                graph.add_node(sensor, role='sensor')
                graph.add_edge(self.sensor_syst, sensor, kind='sensor')
        return graph

    def _color(self, group):
        if group in self._module_colors:
            return self._module_colors[group]
        # Stable for arbitrary identifiers; independent of Python's randomized hash().
        index = int.from_bytes(hashlib.sha256(group.encode('utf-8')).digest()[:4], 'big')
        return self.coledg[index % len(self.coledg)]

    def _network(self, graph, font_color='black'):
        network = Network(notebook=False, cdn_resources='in_line', bgcolor='transparent',
                          font_color=font_color, height='650px', width='100%', directed=True,
                          neighborhood_highlight=True)
        # Add explicitly: preserves isolated nodes and avoids changing graph attributes.
        for identifier, attributes in graph.nodes(data=True):
            role = attributes.get('role', 'unit')
            group = attributes.get('module', identifier)
            color = self._color(group)
            shape, size = 'square', 14
            if role == 'read_only':
                shape, size, color = 'dot', 8, '#FF0000'
            elif role == 'controlled':
                shape, size, color = 'dot', 8, '#FFEC8B'
            elif role in {'sensor', 'sensor_group'}:
                shape, size = 'triangle', 12
            network.add_node(identifier, label=identifier, title=f'{identifier} | {role} | {group}',
                             shape=shape, size=size, color=color)
        for source, target, attributes in graph.edges(data=True):
            kind = attributes.get('kind', 'flow')
            group = graph.nodes[source].get('module', source)
            color = '#94a3b8' if kind == 'attachment' else self._color(group)
            network.add_edge(source, target, color=color, title=kind, arrows='to',
                             dashes=kind in {'controlled', 'attachment'})
        # Stable initial positions; PyVis physics remains interactive after opening.
        positions = nx.spring_layout(graph, seed=42, scale=500) if graph else {}
        if len(positions) > 1:
            width = max(p[0] for p in positions.values()) - min(p[0] for p in positions.values())
            height = max(p[1] for p in positions.values()) - min(p[1] for p in positions.values())
            if width > 0:
                for position in positions.values():
                    position[0] *= max(1., 1.9 * height / width)
        for node in network.nodes:
            node['x'], node['y'] = map(float, positions[node['id']])
        return network

    def _document(self, network):
        identifier = 'msi_' + uuid.uuid4().hex
        document = network.generate_html(notebook=False).replace('mynetwork', identifier)
        # The vis library is embedded. Remove optional Bootstrap CDN assets and
        # old commented local resources so both notebook/app run offline.
        document = re.sub(r'<!--.*?-->', '', document, flags=re.S)
        document = re.sub(r'<script\b[^>]*\bsrc=[^>]*>.*?</script>', '', document, flags=re.S | re.I)
        document = re.sub(r'<link\b[^>]*\bhref=[^>]*>', '', document, flags=re.S | re.I)
        style = (f'<style>html,body{{height:100%;margin:0;overflow:hidden;background:transparent}}'
                 f'.card,.card-body{{height:100%;margin:0;padding:0;border:0;background:transparent}}'
                 f'#{identifier}{{height:100vh!important;width:100%!important;border:0!important}}</style>')
        landscape = '''<script>
        let autoArrangePending = true;
        window.msiLandscapeFit = function () {
            if (!autoArrangePending || !window.network || !window.nodes) return;
            autoArrangePending = false;
            network.setOptions({physics:{enabled:false}});
            var positions = network.getPositions(), ids = nodes.getIds();
            if (ids.length > 1) {
                var xs = ids.map(id => positions[id].x), ys = ids.map(id => positions[id].y);
                var width = Math.max(...xs)-Math.min(...xs), height = Math.max(...ys)-Math.min(...ys);
                var factor = width > 1 ? Math.max(1,1.9*height/width) : 1;
                nodes.update(ids.map(id => ({id:id,x:positions[id].x*factor,y:positions[id].y})));
            }
            network.fit({animation:false});
        };
        network.on('stabilized', window.msiLandscapeFit);
        network.on('stabilizationIterationsDone', window.msiLandscapeFit);
        network.setOptions({interaction:{dragNodes:true,dragView:true}});
        network.on('dragStart', function (event) {
            if (!event.nodes.length) return;
            autoArrangePending = false;
            network.setOptions({physics:{enabled:false}});
        });
        network.on('dragEnd', function (event) {
            if (event.nodes.length) network.storePositions();
        });
        window.msiAutoArrange = function () {
            autoArrangePending = true;
            network.setOptions({physics:{enabled:true}});
            network.stabilize(150);
        };
        </script>'''
        return document.replace('</head>', style + '</head>').replace('</body>', landscape + '</body>'), identifier

    def _render(self, graph, filename, notebook):
        network = self._network(graph)
        # PyVis.write_html uses the Windows default encoding in some versions.
        document, identifier = self._document(network)
        Path(filename).write_text(document, encoding='utf-8')
        if notebook:
            from IPython.display import HTML
            # Each cell gets a separate document and JavaScript scope. Returning
            # raw PyVis HTML reuses global variables/container IDs across cells.
            return HTML(f'<iframe id="frame_{identifier}" title="MSI connectivity graph" '
                        f'style="width:100%;height:690px;border:0" '
                        f'srcdoc="{html.escape(document, quote=True)}"></iframe>')
        return str(Path(filename).resolve())

    def adv_graph(self, filename='graph.html', notebook=True):
        return self._render(self._graph(), filename, notebook)


class MSI_encoder_w_modules(MSI_encoder):
    def __init__(self, MSI_code):
        self.MSI_module_code = MSI_code.strip()
        self.code_raw = [chunk.strip() for chunk in self.MSI_module_code.split('--')]
        self.code_modules, names, self.modules_conn = [], [], []
        for chunk in self.code_raw:
            match = re.fullmatch(r'(M_[^{}()]+)\{\{(.*?)\}\}\(\((.*?)\)\)', chunk, re.S)
            if not match:
                raise ValueError('Invalid MSI module syntax')
            name, body, outputs = match.groups()
            names.append(name)
            self.code_modules.append(body)
            self.modules_conn.append(_items(outputs))
        if len(names) != len(set(names)):
            raise ValueError('Duplicate module identifier')
        self.modules = [[name] for name in names]
        super().__init__(self.code_modules[0])
        self.MSI_code = self.MSI_module_code
        self._encoders = [MSI_encoder(body) for body in self.code_modules]
        owners = {}
        for name, encoder in zip(names, self._encoders):
            declared = encoder.seq_gen() + ([encoder.sensor_syst] + encoder.sensor_ids if encoder.sensor_ids else [])
            for identifier in declared:
                if identifier in owners:
                    raise ValueError(f'Duplicate MSI identifier: {identifier}')
                owners[identifier] = name
        for outputs in self.modules_conn:
            if any(target not in names for target in outputs):
                raise ValueError('Unknown target module')
        for encoder in self._encoders:
            for targets in encoder.unit_conn:
                if any(target not in owners or not target.startswith('U_') for target in targets):
                    raise ValueError('Unknown connected unit operation')
        self._owners = owners
        for index, name in enumerate(sorted(names)):
            if index < len(self.coledg):
                self._module_colors[name] = self.coledg[index]
            else:
                rgb = colorsys.hsv_to_rgb((index * .61803398875) % 1, .45, .95)
                self._module_colors[name] = '#' + ''.join(f'{round(value * 255):02x}' for value in rgb)

    def _module(self, n):
        if not isinstance(n, int) or not 1 <= n <= len(self._encoders):
            raise ValueError(f'Module number must be between 1 and {len(self._encoders)}')
        return self._encoders[n - 1]

    def ext_module_struct(self):
        return {name[0]: list(outputs) for name, outputs in zip(self.modules, self.modules_conn)}

    def ext_module_graph(self, filename='modules.html', notebook=True):
        graph = nx.DiGraph(self.ext_module_struct())
        for identifier in graph:
            graph.nodes[identifier].update(role='unit', module=identifier)
        return self._render(graph, filename, notebook)

    def int_module_graph(self, n, filename=None, notebook=True):
        encoder = self._module(n)
        graph = encoder._graph()
        for identifier in graph:
            graph.nodes[identifier]['module'] = self._owners.get(identifier, self.modules[n - 1][0])
        return self._render(graph, filename or f'module_{n}.html', notebook)

    def sensors(self, n):
        return list(self._module(n).sensor_ids)

    def modular_system_struct(self):
        return {key: value for encoder in self._encoders for key, value in encoder.general_struct().items()}

    def modular_system_graph(self, filename='graph.html', notebook=True):
        return self._render(self.network_graph(), filename, notebook)

    def network_graph(self):
        graph = nx.compose_all([encoder._graph() for encoder in self._encoders])
        for identifier in graph:
            graph.nodes[identifier]['module'] = self._owners[identifier]
        return graph

    def build_network(self, font_color='black'):
        """Same styled PyVis network used by the notebook and desktop app."""
        return self._network(self.network_graph(), font_color=font_color)

    def graph_document(self):
        return self._document(self.build_network())[0]
