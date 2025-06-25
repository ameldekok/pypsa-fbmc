import unittest
import pandas as pd
import numpy as np
import pypsa
import src.fbmc.main as fbmc_main
import src.fbmc.parameters.cne as cne_params
import src.fbmc.parameters as fbmc_params

class TestDetermineCnes(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.mean_absolute_flow = pd.Series([100, 200, 300, 400], index=['line1', 'line2', 'line3', 'line4'])
        self.line_capacity = pd.Series([500, 500, 500, 500], index=['line1', 'line2', 'line3', 'line4'])
        self.line_usage_threshold = 0.3

    def test_determine_cnes(self):
        # Test with default threshold
        cnes = fbmc_params.determine_cnes(self.mean_absolute_flow, self.line_capacity, line_usage_threshold=self.line_usage_threshold)
        self.assertEqual(cnes, ['line2', 'line3', 'line4'])

    def test_determine_cnes_custom_threshold(self):
        # Test with custom threshold
        cnes = fbmc_params.determine_cnes(self.mean_absolute_flow, self.line_capacity, line_usage_threshold=0.5)
        self.assertEqual(cnes, ['line3', 'line4'])

    def test_threshold_out_of_bounds(self):
        # Test with threshold out of bounds
        with self.assertRaises(AssertionError):
            fbmc_params.determine_cnes(self.mean_absolute_flow, self.line_capacity, line_usage_threshold=1.5)

    def test_indices_mismatch(self):
        # Test with mismatched indices
        mean_absolute_flow = pd.Series([100, 200, 300, 400], index=['line1', 'line2', 'line3', 'line5'])
        with self.assertRaises(AssertionError):
            fbmc_params.determine_cnes(mean_absolute_flow, self.line_capacity, line_usage_threshold=self.line_usage_threshold)

    def test_non_positive_mean_absolute_flow(self):
        # Test with non-positive mean absolute flow
        mean_absolute_flow = pd.Series([100, -200, 300, 400], index=['line1', 'line2', 'line3', 'line4'])
        with self.assertRaises(AssertionError):
            fbmc_params.determine_cnes(mean_absolute_flow, self.line_capacity, line_usage_threshold=self.line_usage_threshold)

    def test_non_positive_line_capacity(self):
        # Test with non-positive line capacity
        line_capacity = pd.Series([500, 500, 0, 500], index=['line1', 'line2', 'line3', 'line4'])
        with self.assertRaises(AssertionError):
            fbmc_params.determine_cnes(self.mean_absolute_flow, line_capacity, line_usage_threshold=self.line_usage_threshold)

    def test_no_cnes(self):
        # Test with no CNEs
        mean_absolute_flow = pd.Series([100, 100, 100, 100], index=['line1', 'line2', 'line3', 'line4'])
        with self.assertRaises(AssertionError):
            fbmc_params.determine_cnes(mean_absolute_flow, self.line_capacity, line_usage_threshold=0.5)
class TestGetNetworkPtdf(unittest.TestCase):

    def setUp(self):
        self.basecase_network = pypsa.Network()
        self.basecase_network.set_snapshots(range(2))
        
        # Add three buses in a triangle configuration
        self.basecase_network.add("Bus", "bus1", v_nom=220)
        self.basecase_network.add("Bus", "bus2", v_nom=220)
        self.basecase_network.add("Bus", "bus3", v_nom=220)
        
        # Add lines forming a triangle
        self.basecase_network.add("Line", "line1", 
            bus0="bus1",
            bus1="bus2",
            x=0.2,      # reactance
            r=0.01,     # resistance
            b=0.001,    # susceptance
            s_nom=300   # thermal rating
        )
        self.basecase_network.add("Line", "line2",
            bus0 = "bus1",
            bus1 = "bus3",
            x = 0.2,
            r = 0.01,
            b = 0.001,
            s_nom = 300
        )

        self.basecase_network.add("Line", "line3", 
            bus0="bus2",
            bus1="bus3",
            x=0.2,
            r=0.01,
            b=0.001,
            s_nom=200
        )
        
    def test_get_network_ptdf(self):
        ptdf, _ = fbmc_params.get_network_ptdf(self.basecase_network)
        self.assertEqual(ptdf.columns.tolist(), ['bus1', 'bus2', 'bus3'])
        self.assertEqual(ptdf.index.tolist(), ['line1', 'line2', 'line3'])
        
        # Check that PTDF matrix has expected properties
        self.assertFalse((ptdf == 0).all().all(), "PTDF matrix should not be all zeros")
        
        # Check row sums based on line connectivity to reference bus
        ref_bus = 'bus1'  # The reference bus used in the PTDF calculation
        for line_name in ptdf.index:
            line = self.basecase_network.lines.loc[line_name]
            if line.bus0 == ref_bus or line.bus1 == ref_bus:
                # Lines connected to reference bus should sum to approximately -1
                self.assertTrue(np.isclose(ptdf.loc[line_name].sum(), -1.0), 
                               f"Sum of PTDF row for {line_name} connected to reference bus should be close to -1")
            else:
                # Lines not connected to reference bus should sum to approximately 0
                self.assertTrue(np.isclose(ptdf.loc[line_name].sum(), 0.0), 
                               f"Sum of PTDF row for {line_name} not connected to reference bus should be close to 0")

if __name__ == '__main__':
    unittest.main()