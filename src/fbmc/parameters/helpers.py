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

def initialize_nodal_injection_difference(network: pypsa.Network, num_iterations: int) -> xr.DataArray:
    """
    Initialize the nodal injection difference array.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    num_iterations : int
        Number of iterations for stochastic sampling.

    Returns
    -------
    xr.DataArray
        An xarray DataArray to store nodal injection differences.
    """
    return xr.DataArray(
        np.zeros((num_iterations, len(network.buses.index), len(network.snapshots))),
        dims=["iteration", "Bus", "snapshot"],
        coords={
            "iteration": range(num_iterations),
            "Bus": network.buses.index,
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

def silent_run_opf(network: pypsa.Network) -> None:
    """
    Run the optimal power flow (OPF) optimization silently.

    This function uses the silence_output context manager to suppress all output
    during the optimization process.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object to optimize.
    
    Raises
    ------
    RuntimeError
        If optimization fails.
    """
    try:
        with silence_output():
            network.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
    except Exception as e:
        raise RuntimeError(f"Network optimization failed: {e}")

def calculate_nodal_injection_difference(network: pypsa.Network) -> np.ndarray:
    """
    Calculate the difference in nodal injections after optimization.

    This function:
    1. Stores the current nodal injections
    2. Optimizes the network
    3. Calculates the difference before and after optimization

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object to optimize.

    Returns
    -------
    np.ndarray
        A NumPy array containing the nodal injection differences.
        Shape: (n_buses, n_snapshots)

    Raises
    ------
    RuntimeError
        If optimization fails.
    """
    # Store nodal injections before optimization
    nodal_injections_before_optimization = network.buses_t.p.copy()

    try:
        # Run optimization silently using the silence_output context manager
        with silence_output():
            network.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
    except Exception as e:
        raise RuntimeError(f"Network optimization failed: {e}")

    # Calculate nodal injection differences (transpose to match expected dimensions)
    return (network.buses_t.p - nodal_injections_before_optimization).values.T

import numpy as np
import pandas as pd


def calculate_gsk_per_bus(nodal_diff, network):
    """
    Process nodal injection differences and calculate the GSK per bus.

    Parameters
    ----------
    nodal_diff : xr.DataArray
        Nodal injection changes for all iterations.
        Dimensions: (iteration, bus, snapshot)
    network : pypsa.Network
        The PyPSA network object, with each bus carrying a 'zone_name' attribute.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary keyed by snapshot, each value a DataFrame of shape
        (zones × buses) containing the averaged GSKs.
    """
    # 1) Sanity check
    if np.isnan(nodal_diff.values).any():
        raise ValueError("Nodal differences contain NaNs.")
    
    # 2) Extract mappings and index lists
    bus_zones = network.buses["zone_name"].to_dict()
    all_buses = list(network.buses.index)
    all_zones = sorted(network.buses["zone_name"].unique())
    all_snaps = list(nodal_diff.coords["snapshot"].values)
    n_iters   = nodal_diff.sizes["iteration"]

    # 3) Prepare output container
    gsk_per_snapshot = {}

    # 4) Loop over snapshots
    for si, snap in enumerate(all_snaps):
        # accumulators per zone×bus and valid‐count per zone
        accum = {z: {b: 0.0 for b in all_buses} for z in all_zones}
        valid_iters = {z: 0 for z in all_zones}
        
        # 5) Loop over iterations
        for it in range(n_iters):
            diffs = nodal_diff.isel(iteration=it, snapshot=si).values
            df = pd.DataFrame({"bus": all_buses, "diff": diffs})
            df["zone"] = df["bus"].map(bus_zones)

            zonal_sum = df.groupby("zone")["diff"].sum()

            # skip zones with zero net change
            nonzero_zones = [z for z, s in zonal_sum.items() if abs(s) > 1e-8]
            if not nonzero_zones:
                continue

            # compute per-bus GSK for non-zero zones
            for z in nonzero_zones:
                zsum = zonal_sum[z]
                subset = df[df["zone"] == z]
                for _, row in subset.iterrows():
                    accum[z][row["bus"]] += row["diff"] / zsum
                valid_iters[z] += 1
        
        # 6) average over *valid* iterations and normalize
        mat = pd.DataFrame(0.0, index=all_zones, columns=all_buses)
        for z in all_zones:
            if valid_iters[z] > 0:
                for b in all_buses:
                    mat.at[z, b] = accum[z][b] / valid_iters[z]
            else:
                # no valid iteration → uniform
                buses_in_z = [b for b, zz in bus_zones.items() if zz == z]
                for b in buses_in_z:
                    mat.at[z, b] = 1.0 / len(buses_in_z)

            # finally renormalize row to sum to 1 exactly
            row_sum = mat.loc[z].sum()
            if row_sum > 1e-8:
                mat.loc[z] /= row_sum

        gsk_per_snapshot[snap] = mat

    return gsk_per_snapshot


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