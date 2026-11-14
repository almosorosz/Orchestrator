import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('networkx', 'pyvis'))


@unittest.skipUnless(AVAILABLE, 'Optional notebook dependencies: networkx and pyvis')
class NotebookCompatibilityTests(unittest.TestCase):
    def test_two_module_example_returns_independent_notebook_outputs(self):
        import html
        import re
        from notebooks.MSI_graph_compatible import MSI_encoder_w_modules
        code = ('M_m1{{U_a11(){}[U_p12]__U_p12(){C_f12}[U_c13]__U_c13(R_f13,R_c13,R_p13){C_f13}[U_p21]'
                '___S_s1[[S_a1,S_p1,S_v1]]}}((M_m2))--M_m2{{U_p21(){C_f21}[U_c22]'
                '__U_c22(R_c22,R_p22,R_f22){C_f22}[U_p23]__U_p23(){C_f23}[U_f24]'
                '__U_f24(){}[]___S_s2[[S_p2]]}}(())')
        encoder = MSI_encoder_w_modules(code)
        self.assertEqual(encoder.ext_module_struct(), {'M_m1': ['M_m2'], 'M_m2': []})
        self.assertNotEqual(encoder._color('M_m1'), encoder._color('M_m2'))
        self.assertIn('U_p21', encoder.modular_system_struct()['U_c13'])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            system = encoder.modular_system_graph(Path(root) / 'system.html')
            first = encoder.int_module_graph(1, Path(root) / 'first.html')
            second = encoder.int_module_graph(2, Path(root) / 'second.html')
            outputs = [system.data, first.data, second.data]
            frames = [re.search(r'id="(frame_[^"]+)"', value).group(1) for value in outputs]
            self.assertEqual(len(set(frames)), 3)
            self.assertTrue(all(value.startswith('<iframe ') for value in outputs))
            self.assertIn('S_v1', html.unescape(first.data))
            self.assertNotIn('S_v1', html.unescape(second.data))
            self.assertIn('S_p2', html.unescape(second.data))
            self.assertNotIn('mynetwork', system.data)

    def test_shared_code_generates_full_internal_and_module_graphs(self):
        from notebooks.MSI_graph_compatible import MSI_encoder_w_modules
        code = ('M_m1{{U_AS1(R_WAS1){}[U_P_PAS1]__U_C1(R_D101,R_D501,R_D901,R_FBRM1,R_IR1){C_T1,C_V1}[]'
                '__U_CLEN1(){}[U_P_PCLEN1]__U_P_PAS1(){C_F_PAS1}[U_C1]__U_P_PCLEN1(){C_F_PCLEN1}[U_C1]'
                '__U_P_PSEED1(){C_F_PSEED1}[U_C1]__U_P_PSOLU1(){C_F_PSOLU1}[U_C1]__U_P_PSOLV1(){C_F_PSOLV1}[U_C1]'
                '__U_SEED1(){}[U_P_PSEED1]__U_SOLU1(){}[U_P_PSOLU1]__U_SOLV1(){}[U_P_PSOLV1]'
                '__U_S_SMAIN1(R_TQ_SMAIN1){C_SR_SMAIN1}[U_C1]__U_S_SSEED1(R_TQ_SSEED1){C_SR_SSEED1}[U_SEED1]'
                '__U_S_SSOLU1(R_TQ_SSOLU1){C_SR_SSOLU1}[U_SOLU1]___S_m1[[]]}}(())')
        encoder = MSI_encoder_w_modules(code)
        self.assertEqual(encoder.sensors(1), [])
        adjacency = encoder.modular_system_struct()
        self.assertIn('C_F_PAS1', adjacency)
        self.assertEqual(adjacency['C_SR_SMAIN1'], ['U_S_SMAIN1'])
        self.assertNotIn('C_SR_SMAIN1', adjacency['U_S_SMAIN1'])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            for method, filename in ((encoder.modular_system_graph, 'system.html'),
                                     (encoder.ext_module_graph, 'modules.html')):
                output = Path(method(Path(root) / filename, notebook=False))
                text = output.read_text(encoding='utf-8')
                self.assertIn('M_m1' if filename == 'modules.html' else 'R_TQ_SMAIN1', text)
                self.assertNotIn('int(nt.nodes', text)
            output = encoder.int_module_graph(1, Path(root) / 'inside.html', notebook=True)
            self.assertIn('R_TQ_SMAIN1', output.data)

    def test_legacy_multiple_modules_empty_lists_and_isolated_nodes(self):
        from notebooks.MSI_graph_compatible import MSI_encoder_w_modules
        code = ('M_m1{{U_a11(R_t11){C_f11}[U_c22,U_c33]___S_s1[[S_p1,S_v1]]}}((M_m2,M_m3))'
                '--M_m2{{U_c22(){}[]__U_isolated(){}[]___S_s2[[]]}}(())'
                '--M_m3{{U_c33(){}[U_a11]___S_s3[[]]}}((M_m1))')
        encoder = MSI_encoder_w_modules(code)
        self.assertEqual(encoder.ext_module_struct()['M_m1'], ['M_m2', 'M_m3'])
        self.assertIn('M_m2', encoder.ext_module_struct())
        self.assertIn('U_isolated', encoder.modular_system_struct())
        self.assertEqual(encoder.sensors(1), ['S_p1', 'S_v1'])
        self.assertEqual(encoder.sensors(2), [])
        self.assertEqual(encoder._color('U_AS1'), encoder._color('U_AS1'))
        with self.assertRaises(ValueError):
            encoder.int_module_graph(0)

    def test_readonly_torque_is_output_and_stirrer_connection_is_attachment(self):
        from notebooks.MSI_graph_compatible import MSI_encoder
        encoder = MSI_encoder('U_C1(){}[]__U_S_SS1(R_TQ_SS1){C_SR_SS1}[U_C1]___S_m1[[]]')
        graph = encoder._graph()
        self.assertEqual(graph['U_S_SS1']['U_C1']['kind'], 'attachment')
        self.assertEqual(graph['U_S_SS1']['R_TQ_SS1']['kind'], 'read_only')
        self.assertEqual(graph['C_SR_SS1']['U_S_SS1']['kind'], 'controlled')
        self.assertFalse(graph.has_edge('U_S_SS1', 'C_SR_SS1'))
