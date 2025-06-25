"""Helper functions for GSK calculations."""

import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pypsa
import sys
import io
import contextlib
from typing import Tuple, Dict, List, Union

def get_uncertain_elements(network: pypsa.Network, uncertain_carriers: list) -> tuple:
    """
    Extract uncertain generators and loads from the network.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    uncertain_carriers : list
        List of generator carriers with uncertainty.

    Returns
    -------
    tuple
        A tuple containing uncertain generators and loads.
    """
    uncertain_gens = network.generators[network.generators["carrier"].isin(uncertain_carriers)]
    uncertain_loads = network.loads.copy()
    return uncertain_gens, uncertain_loads

def initialize_gen_difference(network: pypsa.Network, num_iterations: int) -> xr.DataArray:
    """
    Initialize the generation difference array.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    num_iterations : int
        Number of iterations for stochastic sampling.

    Returns
    -------
    xr.DataArray
        An xarray DataArray to store generation differences.
    """
    return xr.DataArray(
        np.zeros((num_iterations, len(network.generators.index), len(network.snapshots))),
        dims=["iteration", "Generator", "snapshot"],
        coords={
            "iteration": range(num_iterations),
            "Generator": network.generators.index,
            "snapshot": network.snapshots,
        },
    )

def introduce_variation_to_network(
    network: pypsa.Network, 
    uncertain_gens: pd.DataFrame, 
    uncertain_loads: pd.DataFrame, 
    gen_variation_std_dev: float, 
    load_variation_std_dev: float
) -> None:
    """
    Apply stochastic variation to generators and loads to model uncertainty.

    This function introduces random variations to:
    1. Generator outputs (within their capacity limits)
    2. Load demands (ensuring they remain positive)

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object to modify.
    uncertain_gens : pd.DataFrame
        DataFrame containing generators that have uncertainty.
    uncertain_loads : pd.DataFrame
        DataFrame containing loads that have uncertainty.
    gen_variation_std_dev : float
        Standard deviation for generator variation as fraction of nominal power.
    load_variation_std_dev : float
        Standard deviation for load variation as fraction of nominal power.
    """
    # Apply uncertainty to generators
    if len(uncertain_gens) > 0:
        try:
            gen_scales = uncertain_gens["p_nom"].values
            gen_means = network.generators_t.p.loc[:, uncertain_gens.index].values
            
            # Create normally distributed variations
            gen_variations = np.random.normal(
                loc=gen_means,
                scale=gen_variation_std_dev * gen_scales[np.newaxis, :]
            )
            
            # Ensure variations stay within bounds [0, p_nom]
            gen_variations = np.clip(gen_variations, 0, gen_scales[np.newaxis, :])
            
            # Update generator values
            network.generators_t.p.loc[:, uncertain_gens.index] = gen_variations
            
            # Update p_max_pu consistently
            valid_p_nom_mask = uncertain_gens["p_nom"].values != 0
            for i, gen_idx in enumerate(uncertain_gens.index):
                if valid_p_nom_mask[i]:
                    network.generators_t.p_max_pu.loc[:, gen_idx] = (
                        network.generators_t.p.loc[:, gen_idx] / uncertain_gens.at[gen_idx, "p_nom"]
                    )
                else:
                    network.generators_t.p_max_pu.loc[:, gen_idx] = 0
        except Exception as e:
            raise RuntimeError(f"Error applying generator uncertainty: {e}")

    # Apply uncertainty to loads
    if len(uncertain_loads) > 0:
        try:
            load_scales = uncertain_loads["p_set"].values
            load_means = np.zeros((len(network.snapshots), len(uncertain_loads.index)))
            
            # Check if loads_t.p_set exists and has values
            has_load_data = (hasattr(network, 'loads_t') and 
                             hasattr(network.loads_t, 'p_set') and 
                             not network.loads_t.p_set.empty)
            
            if has_load_data:
                # Get intersection of uncertain loads and loads in the network
                existing_loads = [load for load in uncertain_loads.index 
                                 if load in network.loads_t.p_set.columns]
                
                # Fill load_means with existing load values
                if existing_loads:
                    temp_values = network.loads_t.p_set.loc[:, existing_loads].values
                    for i, load_idx in enumerate(uncertain_loads.index):
                        if load_idx in existing_loads:
                            existing_idx = existing_loads.index(load_idx)
                            load_means[:, i] = temp_values[:, existing_idx]
            
            # Create normally distributed variations (ensure non-negative loads)
            load_variations = np.random.normal(
                loc=load_means,
                scale=load_variation_std_dev * load_scales[np.newaxis, :]
            )
            load_variations = np.clip(load_variations, 0, None)
            
            # Ensure loads_t.p_set exists
            if not hasattr(network, 'loads_t'):
                network.loads_t = type('loads_t', (), {})()
            
            # Create or update loads_t.p_set
            network.loads_t.p_set = pd.DataFrame(
                load_variations,
                index=network.snapshots,
                columns=uncertain_loads.index
            )
                    
        except Exception as e:
            raise RuntimeError(f"Error applying load uncertainty: {e}")

def calculate_generation_difference(network: pypsa.Network) -> np.ndarray:
    """
    Calculate the difference in generation after optimization.
    
    This function:
    1. Stores the current generator values
    2. Optimizes the network
    3. Calculates the difference before and after optimization

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object to optimize.

    Returns
    -------
    np.ndarray
        A NumPy array containing the generation differences.
        Shape: (n_generators, n_snapshots)
        
    Raises
    ------
    RuntimeError
        If optimization fails.
    """
    # Store generator values before optimization
    gen_t_before_optimization = network.generators_t.p.copy()
    
    try:
        # Run optimization silently using the silence_output context manager
        with silence_output():
            network.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
    except Exception as e:
        raise RuntimeError(f"Network optimization failed: {e}")
    
    # Calculate generation differences (transpose to match expected dimensions)
    return (network.generators_t.p - gen_t_before_optimization).values.T

def process_generation_difference(gen_difference, network):
    """
    Process the generation differences and calculate the GSK.
    
    This function:
    1. Checks for NaN values in the generation differences
    2. Calculates GSK values per iteration as fraction of zonal change
    3. Takes the mean of these fractions across iterations
    4. Maps generators to their respective zones and buses
    5. Aggregates GSK values by bus for final output
    
    Parameters
    ----------
    gen_difference : xr.DataArray
        An xarray DataArray containing generation differences for all iterations.
        Dimensions: (iteration, Generator, snapshot)
    network : pypsa.Network
        The PyPSA network object with buses and their zone mapping.
        
    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        If using the iterative uncertainty method, returns a dictionary of DataFrames
        with snapshots as keys, each DataFrame containing the GSKs for each bus-zone pair.
        If using static GSK method, returns a single DataFrame with zones as rows and buses as columns.
    
    Raises
    ------
    ValueError
        If the generation differences contain NaN values or if zone mapping fails.
    """
    # Check for NaN values which would indicate problems in the optimization
    if np.isnan(gen_difference).any():
        raise ValueError("Generation differences contain NaN values. Check optimization results.")
    
    # Create mapping from generators to zones and buses
    try:
        bus_to_zone = network.buses["zone_name"].to_dict()
        gen_to_bus = network.generators["bus"].to_dict()
        
        # Create mappings from generator to zone and bus
        gen_to_zone = {gen: bus_to_zone.get(gen_to_bus.get(gen)) 
                      for gen in network.generators.index}
        
        gen_to_bus = {gen: gen_to_bus.get(gen) for gen in network.generators.index}
    except KeyError as e:
        raise ValueError(f"Failed to map generators to zones or buses: {e}")
    
    # Get all unique zones, buses, and snapshots for consistent dimensions
    all_zones = sorted(network.buses["zone_name"].unique())
    all_buses = sorted(network.buses.index)
    all_snapshots = sorted(network.snapshots)
    
    # Create a mapping from bus to zone for all buses
    bus_zone_mapping = network.buses["zone_name"].to_dict()
    
    # Create a dictionary to store GSK matrices for each snapshot
    gsk_matrices = {}
    
    # Process each snapshot separately
    for s, snapshot in enumerate(all_snapshots):
        # Initialize an empty GSK matrix for this snapshot
        gsk_matrix = pd.DataFrame(0.0, index=all_zones, columns=all_buses)
        
        # For each iteration, calculate the GSK values and then average them
        iteration_gsks = []
        
        for i in range(gen_difference.shape[0]):  # Iterate through iterations
            # Extract the generation differences for this iteration and snapshot
            iter_gen_diff = gen_difference[i, :, s].values
            
            # Create a DataFrame with generator differences for this iteration
            iter_df = pd.DataFrame({
                'Generator': network.generators.index,
                'gen_dif': iter_gen_diff,
                'zone': [gen_to_zone.get(gen) for gen in network.generators.index],
                'bus': [gen_to_bus.get(gen) for gen in network.generators.index]
            })
            
            # Drop generators that couldn't be mapped to a zone (should be none if mapping is correct)
            iter_df = iter_df.dropna(subset=['zone', 'bus'])
            
            # Calculate zonal sum and count for each zone
            zonal_sums = iter_df.groupby('zone')['gen_dif'].sum().to_dict()
            zonal_counts = iter_df.groupby('zone').size().to_dict()
            
            # Calculate GSK for each generator in this iteration
            for _, row in iter_df.iterrows():
                zone = row['zone']
                bus = row['bus']
                zonal_sum = zonal_sums[zone]
                
                # Calculate GSK: if zonal sum is non-zero, use ratio; otherwise use equal distribution
                if abs(zonal_sum) > 1e-6:  # Using a small threshold to avoid division by very small numbers
                    gsk_value = row['gen_dif'] / zonal_sum
                else:
                    # If sum is nearly zero, distribute equally among generators in the zone
                    gsk_value = 1.0 / zonal_counts[zone]
                
                # Store GSK value keyed by (zone, bus) for this iteration
                iter_df.loc[iter_df['Generator'] == row['Generator'], 'gsk'] = gsk_value
            
            # Add this iteration's GSK values to our collection
            iteration_gsks.append(iter_df)
        
        # Combine all iterations into one DataFrame
        all_iter_df = pd.concat(iteration_gsks, ignore_index=True)
        
        # Group by zone and bus, and calculate mean GSK across iterations
        mean_gsks = all_iter_df.groupby(['zone', 'bus'])['gsk'].mean().reset_index()
        
        # Fill the GSK matrix with the mean values
        for _, row in mean_gsks.iterrows():
            zone = row['zone']
            bus = row['bus']
            gsk_value = row['gsk']
            
            if pd.notna(zone) and pd.notna(bus):
                gsk_matrix.at[zone, bus] = gsk_value
        
        # Normalize GSK values to ensure each zone's GSKs sum to 1
        # This handles any numerical issues from averaging
        for zone in all_zones:
            zone_sum = gsk_matrix.loc[zone].sum()
            if zone_sum > 0:
                gsk_matrix.loc[zone] = gsk_matrix.loc[zone] / zone_sum
            else:
                # If no GSK values for this zone, distribute equally among its buses
                zone_buses = [bus for bus, bus_zone in bus_zone_mapping.items() if bus_zone == zone]
                if zone_buses:
                    gsk_matrix.loc[zone, zone_buses] = 1.0 / len(zone_buses)
        
        # Store the GSK matrix for this snapshot
        gsk_matrices[snapshot] = gsk_matrix
    
    return gsk_matrices

@contextlib.contextmanager
def silence_output():
    """
    Context manager to silence all output (logging, stdout, stderr).
    Use with 'with' statement to suppress output during a block of code.
    """
    # Silence logging via the existing utility function
    suppress_warnings_and_info()
    
    # Additionally silence linopy logging which may not be caught by the general function
    import logging
    original_levels = {}
    for logger_name in ['linopy', 'linopy.model', 'linopy.io', 'linopy.constants']:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)
    
    # Redirect stdout and stderr
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        # Restore stdout and stderr
        sys.stdout = save_stdout
        sys.stderr = save_stderr
        
        # Restore original logging levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def suppress_warnings_and_info():
    """
    Taken from PowerVision code @wouterko
    Suppresses warnings and INFO messages from PyPSA and related libraries.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="cartopy") # suppress irrelevant cartopy plotting warnings resulting from PyPSA itself
    # Configure logging to suppress INFO messages
    logging.getLogger('pypsa.pf').setLevel(logging.WARNING)
    logging.getLogger('pypsa.io').setLevel(logging.WARNING)
    logging.getLogger('gurobipy').setLevel(logging.WARNING)
    logging.getLogger('pypsa.optimization.optimize').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    logging.getLogger('pypsa.consistency').setLevel(logging.WARNING)
    return 