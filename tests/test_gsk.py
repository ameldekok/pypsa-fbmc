import unittest
import pandas as pd
import numpy as np
import pypsa
import os
import sys
from unittest.mock import patch, MagicMock

# Add src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the suppress_warnings_and_info function before importing the GSK module
# This addresses the ImportError with the relative import
sys.modules['src.utils'] = MagicMock()
sys.modules['src.utils'].suppress_warnings_and_info = MagicMock()
sys.modules['src.fbmc.parameters.gsk.suppress_warnings_and_info'] = MagicMock()

# Import the GSK module
from src.fbmc.parameters.gsk import (
    calculate_gsk, 
    gsk_adjustable_cap, 
    gsk_iterative_uncertainty,
    gsk_current_generation,
    get_uncertain_elements,
    initialize_gen_difference,
    introduce_variation_to_network,
    calculate_generation_difference,
    process_generation_difference
)
from src.fbmc.config import FBMCConfig


class TestGskAdjustableCap(unittest.TestCase):
    """Test the adjustable capacity-based GSK calculation method."""

    def setUp(self):
        # Set up test data for adjustable capacity method
        self.generators = pd.DataFrame({
            'bus': ['bus1', 'bus2', 'bus3', 'bus4', 'bus5'],
            'carrier': ['hydropower', 'hydropower', 'coal', 'hydropower', 'gas'],
            'p_nom': [100, 200, 300, 400, 500]
        }, index=['gen1', 'gen2', 'gen3', 'gen4', 'gen5'])

        self.buses = pd.DataFrame({
            'zone_name': ['zone1', 'zone1', 'zone2', 'zone2', 'zone3']
        }, index=['bus1', 'bus2', 'bus3', 'bus4', 'bus5'])

    def test_basic_gsk_calculation(self):
        """Test basic GSK calculation with adjustable generators."""
        gsk = gsk_adjustable_cap(self.generators, self.buses)
        
        # Check that GSK values are correct for zone1 (all adjustable)
        self.assertAlmostEqual(gsk.loc['zone1', 'bus1'], 100/300)  # 100/(100+200)
        self.assertAlmostEqual(gsk.loc['zone1', 'bus2'], 200/300)  # 200/(100+200)
        
        # Check zone2 (mixed adjustable and non-adjustable)
        self.assertAlmostEqual(gsk.loc['zone2', 'bus3'], 0)  # Non-adjustable should have 0
        self.assertAlmostEqual(gsk.loc['zone2', 'bus4'], 1.0)  # Only adjustable in zone2
        
        # Check zone3 (no adjustable) - should distribute evenly
        self.assertAlmostEqual(gsk.loc['zone3', 'bus5'], 1.0)  # Only one generator

    def test_gsk_sum_equals_one(self):
        """Test that GSK values in each zone sum to 1."""
        gsk = gsk_adjustable_cap(self.generators, self.buses)
        
        # For each zone, sum should be close to 1
        for zone in gsk.index:
            self.assertAlmostEqual(gsk.loc[zone].sum(), 1.0, 
                                  msg=f"Sum of GSK for {zone} should be 1.0")

    def test_missing_adjustable(self):
        """Test GSK calculation when no adjustable generators are present."""
        # Create data with no adjustable generators
        generators_no_adjustable = self.generators.copy()
        generators_no_adjustable['carrier'] = 'coal'
        
        with self.assertRaises(ValueError) as context:
            gsk_adjustable_cap(generators_no_adjustable, self.buses)
        
        self.assertIn("No adjustable generators found", str(context.exception))

    def test_empty_zone(self):
        """Test GSK calculation with a zone that has no generators."""
        # Create a zone with no generators
        buses_with_empty_zone = self.buses.copy()
        buses_with_empty_zone.loc['bus6'] = {'zone_name': 'zone4'}
        
        gsk = gsk_adjustable_cap(self.generators, buses_with_empty_zone)
        
        # Check that zone4 is in the GSK matrix but has all zeros
        self.assertIn('zone4', gsk.index)
        self.assertTrue((gsk.loc['zone4'] == 0).all())

    def test_with_mixed_generator_types(self):
        """Test GSK calculation with mixed generator types."""
        # Add generators with different carriers
        mixed_gens = self.generators.copy()
        mixed_gens.loc['gen6'] = {'bus': 'bus3', 'carrier': 'hydropower', 'p_nom': 150}
        mixed_gens.loc['gen7'] = {'bus': 'bus5', 'carrier': 'hydropower', 'p_nom': 250}
        
        gsk = gsk_adjustable_cap(mixed_gens, self.buses)
        
        # Check that zone2 now has non-zero GSK for bus3
        self.assertGreater(gsk.loc['zone2', 'bus3'], 0)
        
        # Check that zone3 now uses adjustable capacity instead of even distribution
        self.assertAlmostEqual(gsk.loc['zone3', 'bus5'], 1.0)


class TestGskIterativeUncertainty(unittest.TestCase):
    """Test the iterative uncertainty-based GSK calculation method."""

    def setUp(self):
        # Create a simple test network
        self.network = pypsa.Network()
        self.network.set_snapshots(['2023-01-01 12:00'])
        
        # Add buses with zones
        for i in range(1, 6):
            self.network.add("Bus", f"bus{i}", v_nom=220)
        
        # Set zone names for buses
        self.network.buses['zone_name'] = ['zone1', 'zone1', 'zone2', 'zone2', 'zone3']
        
        # Add generators
        gen_data = [
            {"name": "gen1", "bus": "bus1", "carrier": "offshore-wind", "p_nom": 100},
            {"name": "gen2", "bus": "bus2", "carrier": "hydropower", "p_nom": 200},
            {"name": "gen3", "bus": "bus3", "carrier": "coal", "p_nom": 300},
            {"name": "gen4", "bus": "bus4", "carrier": "onshore-wind", "p_nom": 150},
            {"name": "gen5", "bus": "bus5", "carrier": "gas", "p_nom": 250}
        ]
        
        for gen in gen_data:
            self.network.add("Generator", gen["name"], 
                            bus=gen["bus"], 
                            carrier=gen["carrier"], 
                            p_nom=gen["p_nom"])
        
        # Add loads
        for i in range(1, 6):
            self.network.add("Load", f"load{i}", bus=f"bus{i}", p_set=50)
        
        # Add some lines
        line_data = [
            {"name": "line1", "bus0": "bus1", "bus1": "bus2", "x": 0.1, "r": 0.01, "s_nom": 200},
            {"name": "line2", "bus0": "bus2", "bus1": "bus3", "x": 0.15, "r": 0.015, "s_nom": 200},
            {"name": "line3", "bus0": "bus3", "bus1": "bus4", "x": 0.2, "r": 0.02, "s_nom": 200},
            {"name": "line4", "bus0": "bus4", "bus1": "bus5", "x": 0.25, "r": 0.025, "s_nom": 200},
            {"name": "line5", "bus0": "bus5", "bus1": "bus1", "x": 0.3, "r": 0.03, "s_nom": 200}
        ]
        
        for line in line_data:
            self.network.add("Line", line["name"],
                           bus0=line["bus0"],
                           bus1=line["bus1"],
                           x=line["x"],
                           r=line["r"],
                           s_nom=line["s_nom"])
        
        # Initialize generators_t.p
        self.network.generators_t.p = pd.DataFrame(
            index=self.network.snapshots,
            columns=self.network.generators.index,
            data=[[50, 100, 150, 75, 125]]  # Initial generation values
        )
        
        # Initialize generators_t.p_max_pu
        self.network.generators_t.p_max_pu = pd.DataFrame(
            index=self.network.snapshots,
            columns=self.network.generators.index,
            data=[[0.5, 0.5, 0.5, 0.5, 0.5]]  # 50% max capacity
        )

    @patch('pypsa.Network.optimize')
    def test_get_uncertain_elements(self, mock_optimize):
        """Test extraction of uncertain elements from the network."""
        uncertain_carriers = ['offshore-wind', 'onshore-wind']
        uncertain_gens, uncertain_loads = get_uncertain_elements(self.network, uncertain_carriers)
        
        # Check that only wind generators are included in uncertain_gens
        self.assertEqual(len(uncertain_gens), 2)
        self.assertIn('gen1', uncertain_gens.index)
        self.assertIn('gen4', uncertain_gens.index)
        
        # Check that all loads are included
        self.assertEqual(len(uncertain_loads), 5)

    def test_initialize_gen_difference(self):
        """Test initialization of generation difference array."""
        num_iterations = 5
        result = initialize_gen_difference(self.network, num_iterations)
        
        # Check dimensions
        self.assertEqual(result.shape, (5, 5, 1))  # (iterations, generators, snapshots)
        
        # Check that it's initialized to zeros
        self.assertTrue(np.all(result.values == 0))

    @patch('numpy.random.normal')
    def test_introduce_variation_to_network(self, mock_normal):
        """Test introduction of variations to the network."""
        # Mock random variations
        mock_normal.return_value = np.array([[10, 20, 30, 40, 50]])
        
        uncertain_gens, uncertain_loads = get_uncertain_elements(
            self.network, ['offshore-wind', 'onshore-wind'])
        
        # Take a copy of the original values
        original_gen_values = self.network.generators_t.p.copy()
        
        # Introduce variations
        introduce_variation_to_network(
            self.network,
            uncertain_gens,
            uncertain_loads,
            0.1,
            0.1
        )
        
        # Only wind generators should be changed
        for gen in self.network.generators.index:
            if gen in uncertain_gens.index:
                self.assertNotEqual(
                    self.network.generators_t.p.loc[self.network.snapshots[0], gen],
                    original_gen_values.loc[self.network.snapshots[0], gen]
                )

    @patch('pypsa.Network.optimize')
    def test_calculate_generation_difference(self, mock_optimize):
        """Test calculation of generation differences after optimization."""
        # Setup mock values for post-optimization
        mock_optimize.side_effect = lambda **kwargs: setattr(
            self.network.generators_t, 'p',
            pd.DataFrame(
                index=self.network.snapshots,
                columns=self.network.generators.index,
                data=[[60, 110, 140, 85, 135]]  # Different from initial values
            )
        )
        
        # Get the differences
        diff = calculate_generation_difference(self.network)
        
        # Check dimensions
        self.assertEqual(diff.shape, (5, 1))  # (generators, snapshots)
        
        # Check expected differences
        expected_diffs = np.array([[10], [10], [-10], [10], [10]])  # Each generator change
        self.assertTrue(np.array_equal(diff, expected_diffs))

    @patch('src.fbmc.parameters.gsk.silence_output')
    @patch('pypsa.Network.optimize')
    def test_gsk_iterative_uncertainty(self, mock_optimize, mock_silence):
        """Test the full GSK iterative uncertainty calculation."""
        # Mock the optimization to avoid actual solver calls
        mock_optimize.return_value = None
        
        # Setup post-optimization generator values (different for each iteration)
        def side_effect(**kwargs):
            # Simulating different generator responses for each optimization
            random_adjustments = np.random.rand(5) * 20 - 10  # Random values between -10 and 10
            new_values = self.network.generators_t.p.values[0] + random_adjustments
            self.network.generators_t.p = pd.DataFrame(
                index=self.network.snapshots,
                columns=self.network.generators.index,
                data=[new_values]
            )
        
        # Replace the optimize method with our mock version
        mock_optimize.side_effect = side_effect
        
        # Calculate GSK with iterative uncertainty
        np.random.seed(42)  # For reproducibility
        result = gsk_iterative_uncertainty(
            self.network,
            uncertain_carriers=['offshore-wind', 'onshore-wind'],
            num_iterations=3,
            gen_variation_std_dev=0.1,
            load_variation_std_dev=0.1
        )
        
        # Check that result is a dictionary (one GSK matrix per snapshot)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)  # One snapshot in test network
        
        # Check GSK values for the snapshot
        gsk_snapshot = result[self.network.snapshots[0]]
        
        # GSK should be a DataFrame with zones as index and buses as columns
        self.assertIsInstance(gsk_snapshot, pd.DataFrame)
        self.assertEqual(set(gsk_snapshot.index), {'zone1', 'zone2', 'zone3'})
        self.assertEqual(set(gsk_snapshot.columns), {'bus1', 'bus2', 'bus3', 'bus4', 'bus5'})
        
        # Check that all GSK values are between 0 and 1
        self.assertTrue((gsk_snapshot >= 0).all().all())
        self.assertTrue((gsk_snapshot <= 1).all().all())
        
        # Check that GSKs sum to 1 for each zone
        for zone in gsk_snapshot.index:
            self.assertAlmostEqual(gsk_snapshot.loc[zone].sum(), 1.0)


class TestCalculateGsk(unittest.TestCase):
    """Test the main calculate_gsk function."""

    def setUp(self):
        # Create a simple test network
        self.network = pypsa.Network()
        
        # Add buses with zone names
        for i in range(1, 4):
            self.network.add("Bus", f"bus{i}", v_nom=220)
        
        self.network.buses['zone_name'] = ['zone1', 'zone1', 'zone2']
        
        # Add generators
        self.network.add("Generator", "gen1", bus="bus1", carrier="hydropower", p_nom=100)
        self.network.add("Generator", "gen2", bus="bus2", carrier="offshore-wind", p_nom=200)
        self.network.add("Generator", "gen3", bus="bus3", carrier="coal", p_nom=300)
        
        # Create config objects
        self.hydro_config = FBMCConfig(gsk_method="ADJUSTABLE_CAP")
        self.iterative_config = FBMCConfig(
            gsk_method="ITERATIVE_UNCERTAINTY",
            uncertain_carriers=["offshore-wind"],
            num_scenarios=2
        )

    @patch('src.fbmc.parameters.gsk.gsk_adjustable_cap')
    def test_calculate_gsk_adjustable_method(self, mock_adjustable_gsk):
        """Test that calculate_gsk calls the correct method with ADJUSTABLE_CAP."""
        mock_adjustable_gsk.return_value = "adjustable_gsk_result"
        
        result = calculate_gsk(self.network, self.hydro_config)
        
        # Check that gsk_adjustable_cap was called with correct arguments
        mock_adjustable_gsk.assert_called_once()
        args = mock_adjustable_gsk.call_args[0]
        self.assertEqual(args[0].equals(self.network.generators), True)
        self.assertEqual(args[1].equals(self.network.buses), True)
        
        # Check that result is from the mock
        self.assertEqual(result, "adjustable_gsk_result")

    @patch('src.fbmc.parameters.gsk.gsk_iterative_uncertainty')
    def test_calculate_gsk_iterative_method(self, mock_iterative_gsk):
        """Test that calculate_gsk calls the correct method with ITERATIVE_UNCERTAINTY."""
        mock_iterative_gsk.return_value = "iterative_gsk_result"
        
        result = calculate_gsk(self.network, self.iterative_config)
        
        # Check that gsk_iterative_uncertainty was called with correct arguments
        mock_iterative_gsk.assert_called_once_with(
            self.network,
            uncertain_carriers=["offshore-wind"],
            num_iterations=2,
            gen_variation_std_dev=0.1,
            load_variation_std_dev=0.1
        )
        
        # Check that result is from the mock
        self.assertEqual(result, "iterative_gsk_result")

    def test_calculate_gsk_unknown_method(self):
        """Test that calculate_gsk raises an error for unknown methods."""
        invalid_config = FBMCConfig(gsk_method="UNKNOWN_METHOD")
        
        with self.assertRaises(ValueError) as context:
            calculate_gsk(self.network, invalid_config)
        
        self.assertIn("Unknown method", str(context.exception))

    def test_calculate_gsk_no_generators(self):
        """Test that calculate_gsk raises an error when network has no generators."""
        network_no_gens = pypsa.Network()
        network_no_gens.add("Bus", "bus1", v_nom=220)
        network_no_gens.buses['zone_name'] = ['zone1']
        
        with self.assertRaises(ValueError) as context:
            calculate_gsk(network_no_gens, self.hydro_config)
        
        self.assertIn("Network contains no generators", str(context.exception))

    def test_calculate_gsk_no_zone_names(self):
        """Test that calculate_gsk raises an error when buses have no zone_name."""
        network_no_zones = pypsa.Network()
        network_no_zones.add("Bus", "bus1", v_nom=220)
        network_no_zones.add("Generator", "gen1", bus="bus1", carrier="hydropower", p_nom=100)
        
        with self.assertRaises(ValueError) as context:
            calculate_gsk(network_no_zones, self.hydro_config)
        
        self.assertIn("zone_name", str(context.exception))


class TestGskCurrentGeneration(unittest.TestCase):
    """Test the current generation-based GSK calculation method."""

    def setUp(self):
        """Set up test data for current generation GSK method."""
        # Create timestamps for multiple snapshots
        self.snapshots = pd.to_datetime(['2023-01-01 12:00', '2023-01-01 13:00', '2023-01-01 14:00'])
        
        # Create generators data
        self.generators = pd.DataFrame({
            'bus': ['bus1', 'bus2', 'bus3', 'bus4', 'bus5', 'bus1'], # gen6 is also on bus1
        }, index=['gen1', 'gen2', 'gen3', 'gen4', 'gen5', 'gen6'])

        # Create buses data with zone mapping
        self.buses = pd.DataFrame({
            'zone_name': ['zone1', 'zone1', 'zone2', 'zone2', 'zone3']
        }, index=['bus1', 'bus2', 'bus3', 'bus4', 'bus5'])

        # Generation output for 3 snapshots
        self.generators_t_p = pd.DataFrame({
            'gen1': [50, 60, 0],   # bus1, zone1
            'gen2': [100, 120, 0], # bus2, zone1
            'gen3': [150, 0, 0],   # bus3, zone2
            'gen4': [75, 80, 10],  # bus4, zone2
            'gen5': [0, 0, 0],     # bus5, zone3
            'gen6': [25, 30, 0]    # bus1, zone1
        }, index=self.snapshots)

    def test_basic_calculation_multi_snapshot(self):
        """Test GSK calculation based on generation share across multiple snapshots."""
        gsk_dict = gsk_current_generation(self.generators, self.generators_t_p, self.buses)

        # Check result structure
        self.assertIsInstance(gsk_dict, dict)
        self.assertEqual(len(gsk_dict), 3) # One entry per snapshot

        # --- Snapshot 1: 2023-01-01 12:00 ---
        gsk1 = gsk_dict[self.snapshots[0]]
        
        # Zone 1: gen1(50)+gen6(25) on bus1, gen2(100) on bus2. Total = 175
        # bus1 share = 75/175, bus2 share = 100/175
        self.assertAlmostEqual(gsk1.loc['zone1', 'bus1'], 75/175)
        self.assertAlmostEqual(gsk1.loc['zone1', 'bus2'], 100/175)
        self.assertAlmostEqual(gsk1.loc['zone1', ['bus3', 'bus4', 'bus5']].sum(), 0) # Other buses in zone1
        
        # Zone 2: gen3(150) on bus3, gen4(75) on bus4. Total = 225
        self.assertAlmostEqual(gsk1.loc['zone2', 'bus3'], 150/225)
        self.assertAlmostEqual(gsk1.loc['zone2', 'bus4'], 75/225)
        self.assertAlmostEqual(gsk1.loc['zone2', ['bus1', 'bus2', 'bus5']].sum(), 0)
        
        # Zone 3: gen5(0) on bus5. Total = 0. Should distribute evenly. Only bus5 in zone3.
        self.assertAlmostEqual(gsk1.loc['zone3', 'bus5'], 1.0)
        self.assertAlmostEqual(gsk1.loc['zone3', ['bus1', 'bus2', 'bus3', 'bus4']].sum(), 0)

        # --- Snapshot 2: 2023-01-01 13:00 ---
        gsk2 = gsk_dict[self.snapshots[1]]
        
        # Zone 1: gen1(60)+gen6(30) on bus1, gen2(120) on bus2. Total = 210
        # bus1 share = 90/210, bus2 share = 120/210
        self.assertAlmostEqual(gsk2.loc['zone1', 'bus1'], 90/210)
        self.assertAlmostEqual(gsk2.loc['zone1', 'bus2'], 120/210)
        
        # Zone 2: gen3(0) on bus3, gen4(80) on bus4. Total = 80
        self.assertAlmostEqual(gsk2.loc['zone2', 'bus3'], 0) # No generation on bus3
        self.assertAlmostEqual(gsk2.loc['zone2', 'bus4'], 1.0) # All generation on bus4
        
        # --- Snapshot 3: 2023-01-01 14:00 (zero generation in zone1) ---
        gsk3 = gsk_dict[self.snapshots[2]]
        
        # Zone 1: gen1(0)+gen6(0) on bus1, gen2(0) on bus2. Total = 0
        # Should distribute evenly between bus1 and bus2
        self.assertAlmostEqual(gsk3.loc['zone1', 'bus1'], 0.5)
        self.assertAlmostEqual(gsk3.loc['zone1', 'bus2'], 0.5)
        
        # Zone 2: gen3(0) on bus3, gen4(10) on bus4. Total = 10
        self.assertAlmostEqual(gsk3.loc['zone2', 'bus3'], 0)
        self.assertAlmostEqual(gsk3.loc['zone2', 'bus4'], 1.0)
        
        # Verify all zones sum to 1.0 in all snapshots
        for snapshot in [gsk1, gsk2, gsk3]:
            for zone in snapshot.index:
                self.assertAlmostEqual(snapshot.loc[zone].sum(), 1.0)

    def test_empty_generation_input(self):
        """Test error handling for empty generators_t_p."""
        empty_gen_t_p = pd.DataFrame(index=self.snapshots, columns=self.generators.index)
        with self.assertRaisesRegex(ValueError, "generators_t_p DataFrame cannot be empty"):
            gsk_current_generation(self.generators, empty_gen_t_p, self.buses)

    def test_missing_columns(self):
        """Test error handling for missing required columns."""
        # Missing 'bus' in generators
        generators_no_bus = self.generators.drop(columns=['bus'])
        with self.assertRaisesRegex(ValueError, "Generators DataFrame must include 'bus' column"):
            gsk_current_generation(generators_no_bus, self.generators_t_p, self.buses)

        # Missing 'zone_name' in buses
        buses_no_zone = self.buses.drop(columns=['zone_name'])
        with self.assertRaisesRegex(ValueError, "Buses DataFrame must include 'zone_name' column"):
            gsk_current_generation(self.generators, self.generators_t_p, buses_no_zone)

    def test_generators_not_in_time_series(self):
        """Test handling of generators present in static data but not in time series."""
        generators_extra = self.generators.copy()
        generators_extra.loc['gen_extra'] = {'bus': 'bus1'} # Add a generator not in generators_t_p
        
        # Should run without error, ignoring 'gen_extra'
        gsk_dict = gsk_current_generation(generators_extra, self.generators_t_p, self.buses)
        
        # Check snapshot 1, should be same as in basic test
        gsk1 = gsk_dict[self.snapshots[0]]
        self.assertAlmostEqual(gsk1.loc['zone1', 'bus1'], 75/175)
        self.assertAlmostEqual(gsk1.loc['zone1', 'bus2'], 100/175)
        
    def test_zero_generation_zones(self):
        """Test handling of zones with zero generation."""
        # Create a data set with zone2 having zero generation for all snapshots
        zero_gen_data = self.generators_t_p.copy()
        zero_gen_data.loc[:, 'gen3'] = 0
        zero_gen_data.loc[:, 'gen4'] = 0
        
        gsk_dict = gsk_current_generation(self.generators, zero_gen_data, self.buses)
        
        # Check that for all snapshots, zone2's GSK is distributed equally among its buses
        for snapshot in self.snapshots:
            gsk = gsk_dict[snapshot]
            self.assertAlmostEqual(gsk.loc['zone2', 'bus3'], 0.5)
            self.assertAlmostEqual(gsk.loc['zone2', 'bus4'], 0.5)
            
    def test_with_network_integration(self):
        """Test integration with PyPSA network."""
        # Create a PyPSA network with the test data
        network = pypsa.Network()
        network.set_snapshots(self.snapshots)
        
        # Add buses with zones
        for i, bus_id in enumerate(self.buses.index):
            network.add("Bus", bus_id, v_nom=220)
        network.buses['zone_name'] = self.buses['zone_name'].values
            
        # Add generators
        for gen_id, gen_data in self.generators.iterrows():
            network.add("Generator", gen_id, bus=gen_data['bus'], p_nom=100)
            
        # Set generation values
        network.generators_t.p = self.generators_t_p
        
        # Create GSK config
        current_gen_config = FBMCConfig(gsk_method="CURRENT_GENERATION")
        
        # Calculate GSK with the calculate_gsk function
        gsk_result = calculate_gsk(network, current_gen_config)
        
        # Verify that result matches direct call to gsk_current_generation
        direct_result = gsk_current_generation(network.generators, network.generators_t.p, network.buses)
        
        # Compare first snapshot
        snapshot = self.snapshots[0]
        pd.testing.assert_frame_equal(gsk_result[snapshot], direct_result[snapshot])


if __name__ == '__main__':
    unittest.main()