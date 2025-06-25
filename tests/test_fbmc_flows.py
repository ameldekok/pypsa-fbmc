import unittest
import pandas as pd
import numpy as np
import pypsa

import src.fbmc.parameters.flows as fbmc_flow

class TestCalculateFlowReliabilityMargin(unittest.TestCase):
    
    def setUp(self):
        # Sample data for testing
        self.line_capacities = pd.Series([100, 200, 300, 400], index=['line1', 'line2', 'line3', 'line4'])

    def test_calculate_flow_reliability_margin(self):
        frm = fbmc_flow.calculate_flow_reliability_margin(self.line_capacities)
        expected_frm = pd.Series([10.0, 20.0, 30.0, 40.0], index=['line1', 'line2', 'line3', 'line4'])
        pd.testing.assert_series_equal(frm, expected_frm)

    def test_reliability_margin_out_of_bounds(self):
        with self.assertRaises(ValueError):
            fbmc_flow.calculate_flow_reliability_margin(self.line_capacities, reliability_margin_factor=1.5)
    
    def test_non_positive_line_capacities(self):
        line_capacities = pd.Series([100, 200, -10, 400], index=['line1', 'line2', 'line3', 'line4'])
        with self.assertRaises(ValueError):
            fbmc_flow.calculate_flow_reliability_margin(line_capacities)

class TestGetBaseFlows(unittest.TestCase):
    
    def setUp(self):
        # Sample data for testing
        self.basecase = pypsa.Network()
        self.basecase.set_snapshots(range(2))
        self.basecase.add("Bus", "bus1")
        self.basecase.add("Bus", "bus2")
        self.basecase.add("Link", "link1", bus0="bus1", bus1="bus2", p_nom=300)
        self.basecase.add("Line", "line1", bus0="bus1", bus1="bus2", s_nom=400)
        self.basecase.add("Transformer", "trans1", bus0="bus1", bus1="bus2", p_nom=500)
        self.basecase.links_t.p0 = pd.DataFrame({
            'link1': [50, 60]
        })
        self.basecase.lines_t.p0 = pd.DataFrame({
            'line1': [70, 80]
        })
        self.basecase.transformers_t.p0 = pd.DataFrame({
            'trans1': [90, 100]
        })

    def test_get_base_flows(self):
        base_flows = fbmc_flow.get_base_flows(self.basecase)
        expected_base_flows = pd.DataFrame({
            'trans1': [90, 100],
            'link1': [50, 60],
            'line1': [70, 80]            
        }).T
        pd.testing.assert_frame_equal(base_flows, expected_base_flows)

if __name__ == '__main__':
    unittest.main()