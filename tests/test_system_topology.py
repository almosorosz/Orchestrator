import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-gpu-compositing')
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from PySide6.QtWidgets import QApplication

from experiment.system_topology import encode_msi, parse_msi, system_graph, token, validate_topology
from experiment.models import ExperimentDefinition
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from hardware_devices.simulated import SimulatedDevice

APP = QApplication.instance() or QApplication([])


def fixture():
    topology = {"version": 1, "modules": [{"code": "Feed", "name": "Feed module"}, {"code": "Cryst", "name": "Crystallization"}],
        "unit_operations": [{"code": "AS_tank", "name": "Antisolvent tank", "kind": "Container", "module": "Feed"},
                            {"code": "C1", "name": "Crystallizer", "kind": "MSMPR", "module": "Cryst"},
                            {"code": "Product", "name": "Product receiver", "kind": "Container", "module": "Cryst"}]}
    configs = {
        "PUMP": {"name": "AS pump", "type": "masterflex", "connection": {}, "simulated": True,
                 "connectivity": {"role": "controlled", "module": "Feed", "source": "AS_tank", "target": "C1"}},
        "T": {"name": "Huber temperature", "type": "huber", "connection": {}, "simulated": True,
              "connectivity": {"role": "read_only", "module": "Cryst", "unit": "C1"}},
        "RPM": {"name": "Stirrer", "type": "jkem", "connection": {}, "simulated": True,
                "connectivity": {"role": "controlled", "module": "Cryst", "unit": "C1"}}}
    return topology, configs


class TopologyTests(unittest.TestCase):
    def test_module_sensors_fill_msi_and_graph(self):
        topology, configs = fixture()
        topology['modules'][0]['sensors'] = [{'code': 'ambientT', 'name': 'Ambient temperature'},
                                             {'code': 'pressure', 'name': 'Room pressure'}]
        code, graph = system_graph(topology, configs)
        group, sensor = token('S_', 'Feed'), token('S_', 'ambientT')
        self.assertIn(f"___{group}[[{sensor},{token('S_', 'pressure')}]]", code)
        nodes = {node['id']: node for node in graph['nodes']}
        self.assertEqual(nodes[sensor]['module'], token('M_', 'Feed'))
        self.assertIn('Ambient temperature', nodes[sensor]['label'])
        self.assertIn({'source': group, 'target': sensor, 'kind': 'sensor'}, graph['edges'])
        self.assertNotIn(token('S_', 'Cryst'), nodes)
        topology['modules'][1]['sensors'] = [{'code': 'ambientT', 'name': 'Duplicate'}]
        with self.assertRaisesRegex(ValueError, 'unique'):
            encode_msi(topology, configs)

    def test_module_sensor_editor_generates_and_validates_module_references(self):
        from gui.system_topology_dialog import SystemTopologyDialog
        topology, configs = fixture()
        dialog = SystemTopologyDialog(topology, configs, initial_tab=1)
        try:
            self.assertEqual(dialog.tabs.currentIndex(), 1)
            dialog.add_row(dialog.sensors, ['airT', 'Air temperature', 'Feed'])
            dialog.generate()
            self.assertIsNotNone(dialog.graph_data, dialog.status.text())
            self.assertIn('S_airT', dialog.code.toPlainText())
            self.assertEqual(dialog.tabs.currentIndex(), dialog.graph_tab_index)
            restored = dialog.read_definition()
            self.assertEqual(restored['modules'][0]['sensors'], [{'code': 'airT', 'name': 'Air temperature'}])
            dialog.sensors.item(0, 2).setText('missing')
            with self.assertRaisesRegex(ValueError, 'existing module'):
                dialog.read_definition()
        finally:
            dialog.close()

    def test_module_equipment_and_flow_use_distinct_colors(self):
        from notebooks.MSI_graph_compatible import MSI_encoder_w_modules
        topology, configs = fixture()
        encoder = MSI_encoder_w_modules(encode_msi(topology, configs))
        network = encoder.build_network()
        nodes = {node['id']: node for node in network.nodes}
        self.assertNotEqual(nodes[token('U_', 'AS_tank')]['color'], nodes[token('U_', 'C1')]['color'])
        for edge in network.edges:
            if edge['title'] == 'flow':
                self.assertEqual(edge['color'], nodes[edge['from']]['color'])

    def test_pump_module_is_independent_of_both_endpoints(self):
        topology, configs = fixture()
        topology['modules'].append({'code': 'Transfer', 'name': 'Pump module'})
        configs['PUMP']['connectivity']['module'] = 'Transfer'
        code, graph = system_graph(topology, configs)
        modules = {m['id']: m['outputs'] for m in graph['modules']}
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(nodes[token('U_', 'AS_tank')]['module'], token('M_', 'Feed'))
        self.assertEqual(nodes[token('U_P_', 'PUMP')]['module'], token('M_', 'Transfer'))
        self.assertEqual(nodes[token('C_F_', 'PUMP')]['module'], token('M_', 'Transfer'))
        self.assertEqual(modules[token('M_', 'Feed')], [token('M_', 'Transfer')])
        self.assertEqual(modules[token('M_', 'Transfer')], [token('M_', 'Cryst')])
        self.assertEqual(parse_msi(code)['edges'], graph['edges'])
        configs['PUMP']['connectivity']['module'] = 'Cryst'
        self.assertIn(token('M_', 'Cryst'), next(m for m in system_graph(topology, configs)[1]['modules']
                      if m['id'] == token('M_', 'Feed'))['outputs'])

    def test_selecting_source_preserves_explicit_pump_module(self):
        from gui.device_connectivity import DeviceConnectivity
        topology, configs = fixture()
        config = deepcopy(configs['PUMP'])
        fields = DeviceConnectivity(config)
        try:
            fields.set_definition(topology)
            fields.module.setCurrentIndex(fields.module.findData('Cryst'))
            fields.source.setCurrentIndex(fields.source.findData('Product'))
            fields.source.setCurrentIndex(fields.source.findData('AS_tank'))
            self.assertEqual(config['connectivity']['module'], 'Cryst')
            self.assertEqual(config['connectivity']['source'], 'AS_tank')
            self.assertIn('module Feed', fields.source.currentText())
            fields.module.setCurrentIndex(0)
            fields.source.setCurrentIndex(fields.source.findData('Product'))
            self.assertEqual(config['connectivity']['module'], 'Cryst')
        finally:
            fields.close()

    def test_pumps_and_stirrers_are_units_with_their_own_signals(self):
        from notebooks.MSI_graph_compatible import MSI_encoder_w_modules
        topology, configs = fixture()
        code, graph = system_graph(topology, configs)
        nodes = {node['id']: node for node in graph['nodes']}
        pump, stirrer = token('U_P_', 'PUMP'), token('U_S_', 'RPM')
        vessel = token('U_', 'C1')
        flow, speed, torque = token('C_F_', 'PUMP'), token('C_SR_', 'RPM'), token('R_TQ_', 'RPM')
        self.assertEqual(nodes[pump]['role'], 'unit')
        self.assertEqual(nodes[stirrer]['role'], 'unit')
        self.assertIn({'source': flow, 'target': pump, 'kind': 'controlled'}, graph['edges'])
        self.assertIn({'source': speed, 'target': stirrer, 'kind': 'controlled'}, graph['edges'])
        self.assertIn({'source': stirrer, 'target': torque, 'kind': 'read_only'}, graph['edges'])
        self.assertIn({'source': stirrer, 'target': vessel, 'kind': 'attachment'}, graph['edges'])
        self.assertNotIn({'source': stirrer, 'target': vessel, 'kind': 'flow'}, graph['edges'])
        self.assertNotIn(token('C_', 'RPM'), nodes)
        self.assertNotIn(token('R_TQ_', 'PUMP'), nodes)
        self.assertEqual(parse_msi(code)['edges'], graph['edges'])
        styled = {node['id']: node for node in MSI_encoder_w_modules(code).build_network().nodes}
        self.assertEqual(styled[pump]['shape'], 'square')
        self.assertEqual(styled[stirrer]['shape'], styled[vessel]['shape'])
        self.assertEqual(styled[torque]['shape'], 'dot')
        self.assertEqual(styled[torque]['color'], '#FF0000')

    def test_read_only_stirrer_has_measurements_and_no_control_node(self):
        topology, configs = fixture()
        configs['RPM']['connectivity']['role'] = 'read_only'
        _, graph = system_graph(topology, configs)
        nodes = {node['id'] for node in graph['nodes']}
        self.assertIn(token('R_TQ_', 'RPM'), nodes)
        self.assertIn(token('R_SR_', 'RPM'), nodes)
        self.assertNotIn(token('C_SR_', 'RPM'), nodes)
        configs['RPM']['connectivity']['role'] = 'excluded'
        _, graph = system_graph(topology, configs)
        self.assertNotIn(token('U_S_', 'RPM'), {node['id'] for node in graph['nodes']})

    def test_round_trip_keeps_isolated_units_flow_and_module_edges(self):
        topology, configs = fixture()
        code, graph = system_graph(topology, configs)
        self.assertEqual(parse_msi(code)["edges"], graph["edges"])
        self.assertIn(token("U_", "Product"), {n["id"] for n in graph["nodes"]})
        self.assertIn({"source": token("U_", "AS_tank"), "target": token("U_P_", "PUMP"), "kind": "flow"}, graph["edges"])
        self.assertIn({"source": token("U_P_", "PUMP"), "target": token("U_", "C1"), "kind": "flow"}, graph["edges"])
        self.assertIn({"source": token("U_", "C1"), "target": token("R_", "T"), "kind": "read_only"}, graph["edges"])
        feed = next(m for m in graph["modules"] if m["id"] == token("M_", "Feed"))
        self.assertEqual(feed["outputs"], [token("M_", "Cryst")])
        reversed_configs = dict(reversed(list(configs.items())))
        self.assertEqual(encode_msi(topology, reversed_configs), code)

    def test_notebook_grammar_supports_multiple_module_outputs_and_sensors(self):
        code = "M_m1{{U_a11(R_t11){C_f11}[U_c22,U_c33]___S_s1[[S_p1,S_v1]]}}((M_m2,M_m3))--M_m2{{U_c22(){}[]___S_s2[[]]}}(())--M_m3{{U_c33(){}[U_a11]___S_s3[[]]}}((M_m1))"
        graph = parse_msi(code)
        self.assertEqual(graph["modules"][0]["outputs"], ["M_m2", "M_m3"])
        self.assertEqual(len([e for e in graph["edges"] if e["kind"] == "flow"]), 3)
        self.assertIn("S_p1", {n["id"] for n in graph["nodes"]})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse_msi(code.replace("U_c33(){}[U_a11]", "U_c33(){}[U_missing]"))

    def test_incomplete_invalid_and_cross_module_device_placements_rejected(self):
        topology, configs = fixture()
        configs["PUMP"]["connectivity"]["target"] = "missing"
        with self.assertRaisesRegex(ValueError, "source and target"):
            encode_msi(topology, configs)
        validate_topology(topology, configs, complete=False)
        _, configs = fixture()
        configs["T"]["connectivity"]["module"] = "Feed"
        with self.assertRaisesRegex(ValueError, "same module"):
            encode_msi(topology, configs)

    def test_graph_is_offline_and_nodes_can_be_moved(self):
        import time
        from gui.system_graph_view import SystemGraphView
        topology, configs = fixture()
        view = SystemGraphView()
        try:
            view.resize(1000, 650)
            code, graph = system_graph(topology, configs)
            view.draw_graph(graph, code)
            from notebooks.MSI_graph_compatible import MSI_encoder_w_modules
            expected = MSI_encoder_w_modules(code).build_network(font_color=view.palette().windowText().color().name())
            self.assertEqual(view.network.nodes, expected.nodes)
            self.assertEqual(view.network.edges, expected.edges)
            self.assertTrue(view.html_file.exists())
            document = view.html_file.read_text(encoding='utf-8')
            self.assertNotIn('cdn.jsdelivr.net', document)
            self.assertIn('neighbourhoodHighlight', document)
            view.show()
            deadline = time.monotonic() + 15
            while not view._ready and time.monotonic() < deadline:
                APP.processEvents()
                time.sleep(.01)
            self.assertTrue(view._ready, 'Embedded PyVis document did not load')
            results = []
            view.web.page().runJavaScript("""
                const movedId = nodes.getIds()[0];
                network.body.emitter.emit('dragStart', {nodes: [movedId]});
                network.moveNode(movedId, 123, 456);
                network.body.emitter.emit('dragEnd', {nodes: [movedId]});
                JSON.stringify({count: nodes.length, drag: network.interactionHandler.options.dragNodes,
                                x: nodes.get(movedId).x, y: nodes.get(movedId).y});
            """, results.append)
            while not results and time.monotonic() < deadline:
                APP.processEvents()
                time.sleep(.01)
            result = json.loads(results[0])
            self.assertEqual(result['count'], len(graph['nodes']))
            self.assertTrue(result['drag'])
            self.assertEqual((result['x'], result['y']), (123, 456))
            self.assertFalse(view.grab().isNull())
        finally:
            view.close()

    def test_monitor_only_huber_is_never_actuated(self):
        _, configs = fixture()
        devices = {key: SimulatedDevice(config["name"], config["type"]) for key, config in configs.items()}
        for device in devices.values():
            device.connect()
        huber = devices["T"]
        for method in ("start", "stop", "set_target", "set_jacket_control", "set_process_control", "release_remote_mode"):
            setattr(huber, method, Mock(side_effect=AssertionError("Monitor-only Huber was actuated")))
        worker = ExperimentWorker(devices, configs, [pd.DataFrame({"time_s": [0.], "PUMP": [1.], "RPM": [100.]})], end_hold_s=0)
        worker.start_experiment()
        self.assertEqual(worker.outcome.value, "completed", worker.error_message)
        huber.start.assert_not_called()
        huber.stop.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Read-only"):
            ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0.], "T": [25.]}), configs)

    def test_control_panel_config_and_device_fields_preserve_physical_setup(self):
        from MAIN_WINDOW import MainWindow
        topology, configs = fixture()
        configs['PUMP']['connectivity']['module'] = 'Cryst'
        topology['modules'][0]['sensors'] = [{'code': 'airT', 'name': 'Air temperature'}]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root, patch("gui.recording_settings_dialog.load_recording_settings",
                return_value={"output_folder": root, "enabled": True, "sql_enabled": False, "remote_sql_url": ""}):
            window = MainWindow()
            try:
                panel = window.control_panel
                panel.apply_loaded_configuration(configs, {}, [], root, topology)
                self.assertEqual(panel.device_widgets["PUMP"].connectivity_fields.source.currentData(), "AS_tank")
                self.assertEqual(panel.device_widgets["PUMP"].connectivity_fields.module.currentData(), "Cryst")
                self.assertEqual(panel.device_widgets["T"].connectivity_fields.role.currentData(), "read_only")
                filename = str(Path(root) / "control.json")
                with patch("control_panel.control_panel.QFileDialog.getSaveFileName", return_value=(filename, "")), patch("control_panel.control_panel.QMessageBox.information"):
                    panel.save_configuration()
                saved = json.loads(Path(filename).read_text())
                self.assertEqual(saved['system_topology']['modules'][0]['sensors'], topology['modules'][0]['sensors'])
                self.assertEqual(saved["system_topology"]["msi_code"], encode_msi(topology, configs))
                panel.apply_loaded_configuration(saved["devices"], {}, [], root, saved["system_topology"])
                self.assertEqual(panel.current_system_topology()["msi_code"], saved["system_topology"]["msi_code"])
            finally:
                window.close()
                APP.processEvents()
