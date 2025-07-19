import numpy as np
import pandas as pd
import pypsa
from typing import List, Dict, Tuple, Union, Optional

from ..config import FBMCConfig, GSKMethod
from .helpers import (
    get_uncertain_elements, 
    initialize_gen_difference,
    introduce_variation_to_network,
    calculate_generation_difference,
    process_generation_difference,
    silence_output
)

def calculate_gsk(network: pypsa.Network, 
                  config: FBMCConfig = FBMCConfig()) -> Union[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Calculate the Generator Shift Key (GSK) of every node in the network.
    
    The GSK represents how changes in a zone's net position are distributed among
    its nodes. It is used to transform nodal PTDFs to zonal PTDFs for flow-based
    market coupling.
    
    Parameters
    ----------
    network : pypsa.Network
        The network object containing generators and buses.
    config : FBMCConfig
        Configuration object containing GSK calculation parameters.
    
    Returns
    -------
    pd.DataFrame or Dict[pd.Timestamp, pd.DataFrame]
        - For static GSK methods: A DataFrame containing the GSKs for each bus-zone pair.
          Index: zones, Columns: buses, Values: share of zone's change allocated to bus
        - For dynamic GSK methods: A dictionary mapping timestamps to GSK DataFrames.
    
    Raises
    ------
    ValueError
        If an unknown GSK method is specified or if required network components are missing.
    """
    # Validate network has required components
    if len(network.generators) == 0:
        raise ValueError("Network contains no generators. Cannot calculate GSK.")
    
    if 'zone_name' not in network.buses.columns:
        raise ValueError("Buses in network must have 'zone_name' attribute for GSK calculation.")

    # Select method based on config
    if config.gsk_method == GSKMethod.ADJUSTABLE_CAP:
        return gsk_adjustable_cap(network.generators, network.buses)
    elif config.gsk_method == GSKMethod.ITERATIVE_UNCERTAINTY:
        return gsk_iterative_uncertainty(
            network,
            uncertain_carriers=config.uncertain_carriers,
            num_iterations=config.num_scenarios,
            gen_variation_std_dev=config.gen_variation_std_dev,
            load_variation_std_dev=config.load_variation_std_dev,
        )
    elif config.gsk_method == GSKMethod.CURRENT_GENERATION:
        return gsk_current_generation(network.generators, network.generators_t.p, network.buses)
    elif config.gsk_method == GSKMethod.ITERATIVE_FBMC:
        return gsk_iterative_fbmc(
            network,
            config=config,
            uncertain_carriers=config.uncertain_carriers,
            num_iterations=config.num_scenarios,
            max_gsk_iterations=config.max_gsk_iterations,
            gen_variation_std_dev=config.gen_variation_std_dev,
            load_variation_std_dev=config.load_variation_std_dev,
            initial_gsk_method=config.initial_gsk_method,
        )
    elif config.gsk_method == GSKMethod.MERIT_ORDER:
        return calc_merit_order_based_gsk(network, standard_deviation=config.gsk_std_dev)
    else:
        raise ValueError(f"Unknown method: {config.gsk_method}. Supported methods are: 'MERIT_ORDER','ADJUSTABLE_CAP', 'ITERATIVE_UNCERTAINTY', 'CURRENT_GENERATION', 'ITERATIVE_FBMC'.")
    

def gsk_iterative_uncertainty(
    network: pypsa.Network,
    uncertain_carriers: List[str] = ['offshore-wind', 'onshore-wind'],
    num_iterations: int = 10,
    gen_variation_std_dev: float = 0.1,
    load_variation_std_dev: float = 0.1
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Calculate GSK using Monte Carlo simulation with uncertainty in generation and load.
    
    This method:
    1. Creates variations of the base case with uncertainty in generation and load
    2. Optimizes each variation to see how generation responds
    3. Uses these responses to determine which generators are most flexible
    4. Constructs GSK values based on the average response of generators
    
    Parameters
    ----------
    network : pypsa.Network
        The network to calculate GSKs for
    uncertain_carriers : list
        List of generation technologies with uncertainty (e.g. wind, solar)
    num_iterations : int
        Number of Monte Carlo iterations (higher = more stable results)
    gen_variation_std_dev : float
        Standard deviation for generator variation as fraction of nominal power
    load_variation_std_dev : float
        Standard deviation for load variation as fraction of nominal power
        
    Returns
    -------
    Dict[pd.Timestamp, pd.DataFrame]
        Dictionary of DataFrames containing GSK values for each bus and zone, one per snapshot
    """
    # Validate inputs
    if num_iterations < 1:
        raise ValueError("Number of iterations must be at least 1")
    if gen_variation_std_dev < 0 or load_variation_std_dev < 0:
        raise ValueError("Standard deviations must be non-negative")

    # Run the network once to get initial generation and load values
    try:
        with silence_output():
            network.optimize(solver_name='gurobi')
    except Exception as e:
        raise RuntimeError(f"Failed to optimize initial network state: {e}")

    # Collect the loads and gens (wind) with uncertainty
    uncertain_gens, uncertain_loads = get_uncertain_elements(network, uncertain_carriers)
    
    # Initialize the generation difference array
    gen_difference = initialize_gen_difference(network, num_iterations)

    # Perform stochastic sampling
    for i in range(num_iterations):
        # Create a fresh copy for this iteration
        stochastic_network = network.copy(snapshots=network.snapshots)
        
        # Apply uncertainty to generators and loads
        introduce_variation_to_network(
            stochastic_network,
            uncertain_gens,
            uncertain_loads,
            gen_variation_std_dev,
            load_variation_std_dev,
        )

        # Calculate generation differences after optimization
        gen_difference[i,:,:] = calculate_generation_difference(stochastic_network)
        
    # Process results and calculate GSK
    return process_generation_difference(gen_difference, network)

def gsk_iterative_merit_order(
    network: pypsa.Network,
    uncertain_carriers: List[str] = ['offshore-wind', 'onshore-wind'],
    num_iterations: int = 10,
    gen_variation_std_dev: float = 0.1,
    load_variation_std_dev: float = 0.1
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Calculate GSK using Monte Carlo simulation with uncertainty in generation and load.
    
    This method:
    1. Creates variations of the base case with uncertainty in generation and load
    2. Optimizes each variation to see how generation responds
    3. Uses these responses to determine which generators are most flexible
    4. Constructs GSK values based on the average response of generators
    
    Parameters
    ----------
    network : pypsa.Network
        The network to calculate GSKs for
    uncertain_carriers : list
        List of generation technologies with uncertainty (e.g. wind, solar)
    num_iterations : int
        Number of Monte Carlo iterations (higher = more stable results)
    gen_variation_std_dev : float
        Standard deviation for generator variation as fraction of nominal power
    load_variation_std_dev : float
        Standard deviation for load variation as fraction of nominal power
        
    Returns
    -------
    Dict[pd.Timestamp, pd.DataFrame]
        Dictionary of DataFrames containing GSK values for each bus and zone, one per snapshot
    """
    # Validate inputs
    if num_iterations < 1:
        raise ValueError("Number of iterations must be at least 1")
    if gen_variation_std_dev < 0 or load_variation_std_dev < 0:
        raise ValueError("Standard deviations must be non-negative")

    # Run the network once to get initial generation and load values
    try:
        with silence_output():
            network.optimize(solver_name='gurobi')
    except Exception as e:
        raise RuntimeError(f"Failed to optimize initial network state: {e}")

    # Collect the loads and gens (wind) with uncertainty
    uncertain_gens, uncertain_loads = get_uncertain_elements(network, uncertain_carriers)
    
    # Initialize the generation difference array
    gen_difference = initialize_gen_difference(network, num_iterations)

    # Perform stochastic sampling
    for i in range(num_iterations):
        # Create a fresh copy for this iteration
        stochastic_network = network.copy(snapshots=network.snapshots)
        
        # Apply uncertainty to generators and loads
        introduce_variation_to_network(
            stochastic_network,
            uncertain_gens,
            uncertain_loads,
            gen_variation_std_dev,
            load_variation_std_dev,
        )

        # Store generator values before optimization
        gen_t_before_optimization = stochastic_network.generators_t.p.copy()

        stochastic_network.optimize.create_model()
        

        try:
            # Run optimization silently using the silence_output context manager
            with silence_output():
                stochastic_network.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
        except Exception as e:
            raise RuntimeError(f"Network optimization failed: {e}")
        
        # Calculate generation differences (transpose to match expected dimensions)
        gen_difference[i,:,:] = (stochastic_network.generators_t.p - gen_t_before_optimization).values.T

    # Process results and calculate GSK
    return process_generation_difference(gen_difference, network)

def gsk_iterative_fbmc(
    network: pypsa.Network,
    config: FBMCConfig = FBMCConfig(),
    uncertain_carriers: List[str] = ['offshore-wind', 'onshore-wind'],
    num_iterations: int = 100,
    max_gsk_iterations: int = 5,
    gen_variation_std_dev: float = 0.1,
    load_variation_std_dev: float = 0.1,
    initial_gsk_method: str = "CURRENT_GENERATION"
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Calculate GSK using an iterative FBMC approach for multiple iterations.
    
    This method:
    1. Starts with an initial GSK (e.g. based on current generation)
    2. For each GSK iteration:
       a. Creates variations of the base case with uncertainty in generation and load
       b. Uses FBMC (with the current GSK) to determine generation allocation
       c. Uses these responses to calculate a new GSK
    3. Repeats until convergence or max iterations reached
    
    Parameters
    ----------
    network : pypsa.Network
        The network to calculate GSKs for
    config : FBMCConfig
        Configuration object with FBMC parameters
    uncertain_carriers : list
        List of generation technologies with uncertainty (e.g. wind, solar)
    num_iterations : int
        Number of Monte Carlo iterations per GSK iteration
    max_gsk_iterations : int
        Maximum number of GSK refinement iterations
    gen_variation_std_dev : float
        Standard deviation for generator variation as fraction of nominal power
    load_variation_std_dev : float
        Standard deviation for load variation as fraction of nominal power
    initial_gsk_method : str
        Method to use for initial GSK calculation
        
    Returns
    -------
    Dict[pd.Timestamp, pd.DataFrame]
        Dictionary of DataFrames containing GSK values for each bus and zone, one per snapshot
    """
    # Validate inputs
    if num_iterations < 1:
        raise ValueError("Number of iterations must be at least 1")
    if gen_variation_std_dev < 0 or load_variation_std_dev < 0:
        raise ValueError("Standard deviations must be non-negative")
    if max_gsk_iterations < 1:
        raise ValueError("Number of GSK iterations must be at least 1")

    # Get initial GSK
    current_gsk = _get_initial_gsk(network, initial_gsk_method)
    
    # Prepare tracking arrays for GSK iterations
    all_gsks = [current_gsk]
    
    # Main GSK iteration loop
    for gsk_iteration in range(max_gsk_iterations):
        print(f"Starting GSK iteration {gsk_iteration + 1}/{max_gsk_iterations}")
        
        # Collect uncertain elements
        uncertain_gens, uncertain_loads = get_uncertain_elements(network, uncertain_carriers)
        
        # Initialize storage for generation differences in this GSK iteration
        gen_difference_data = initialize_gen_difference(network, num_iterations)
        
        # Monte Carlo iterations using current GSK
        for i in range(num_iterations):
            # Create a fresh copy for this iteration
            perturbed_network = network.copy(snapshots=network.snapshots)
            
            # Apply uncertainty to generators and loads
            introduce_variation_to_network(
                perturbed_network,
                uncertain_gens,
                uncertain_loads,
                gen_variation_std_dev,
                load_variation_std_dev
            )
            
            # Run FBMC with current GSK to get generation allocation
            fbmc_results = _run_fbmc_with_gsk(perturbed_network, current_gsk, config)
            
            # Calculate generation differences from the FBMC results
            gen_difference_data[i,:,:] = _calculate_fbmc_gen_difference(network, fbmc_results)
            
        # Calculate new GSK from Monte Carlo results
        new_gsk = process_generation_difference(gen_difference_data, network)
        
        # Check for convergence
        if _check_gsk_convergence(new_gsk, current_gsk):
            print(f"GSK converged after {gsk_iteration + 1} iterations")
            break
            
        # Update current GSK for next iteration
        current_gsk = new_gsk
        all_gsks.append(current_gsk)
    
    # Return final GSK
    return current_gsk

def _get_initial_gsk(network: pypsa.Network, method: str) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Get initial GSK using specified method."""

    print(f"Calculating initial GSK using {method} method")
    if method == GSKMethod.CURRENT_GENERATION:
        return gsk_current_generation(network.generators, network.generators_t.p, network.buses)
    elif method == GSKMethod.ADJUSTABLE_CAP:
        gsk = gsk_adjustable_cap(network.generators, network.buses)
        # Convert to dict format for consistency
        return {ts: gsk.copy() for ts in network.snapshots}
    else:
        raise ValueError(f"Unsupported initial GSK method: {method}")

def _run_fbmc_with_gsk(
    perturbed_network: pypsa.Network, 
    current_gsk: Dict[pd.Timestamp, pd.DataFrame],
    config: FBMCConfig
) -> Dict:
    """
    Run FBMC with the current GSK and return generation allocation.
    
    This function:
    1. Creates a zonal network from the perturbed network
    2. Calculates FBMC parameters using the provided GSK
    3. Adds FBMC constraints to the zonal network
    4. Optimizes the zonal network
    5. Returns the generation allocation results
    
    Parameters
    ----------
    perturbed_network : pypsa.Network
        The perturbed network with variations in generation/load
    current_gsk : Dict[pd.Timestamp, pd.DataFrame]
        The current GSK to use for FBMC calculation
    config : FBMCConfig
        Configuration for the FBMC calculation
        
    Returns
    -------
    Dict
        Dictionary containing optimization results, including generation allocation
    """
    from ..parameters.main import calculate_fbmc_parameters
    from ..constraints import create_zonal_generation, add_fbmc_constraints, remove_original_constraints
    from ..network_conversion import nodal_to_zonal
    
    try:
        # Use a single silence_output block for the entire operation
        # to avoid duplicate print statements
        with silence_output():
            # Convert nodal network to zonal network
            zonal_network = nodal_to_zonal(perturbed_network)
            
            # Initial optimization of zonal network
            zonal_network.optimize(solver_name='gurobi')
            
            # Calculate FBMC parameters using the provided GSK (not the default GSK calculation)
            ram_cnes, z_ptdf_cnes = calculate_fbmc_parameters(
                perturbed_network,
                config=config,
                gsk=current_gsk
            )
            
            # Set up zonal model for FBMC
            zonal_network = create_zonal_generation(zonal_network)
            zonal_network = add_fbmc_constraints(zonal_network, z_ptdf_cnes, ram_cnes)
            remove_original_constraints(zonal_network)
            
            # Optimize zonal network with FBMC constraints
            zonal_network.model.solve(solver_name='gurobi')
        
        # Extract generation results from the optimization model solution
        if hasattr(zonal_network, 'model') and hasattr(zonal_network.model, 'solution'):
            # Get generator output from the solution
            opt_gens = zonal_network.model.solution['Generator-p']
            
            # Convert to DataFrame with generators as columns
            opt_gens_df = opt_gens.to_dataframe().unstack('Generator')
            
            # Clean up the MultiIndex columns
            opt_gens_df.columns = opt_gens_df.columns.droplevel(0)
            
            # Return the generation allocation results
            results = {
                'generators_t': {
                    'p': opt_gens_df
                }
            }
            return results
        else:
            print("Warning: Optimization model solution not found, using direct network values")
            return {
                'generators_t': {
                    'p': zonal_network.generators_t.p.copy()
                }
            }
        
    except Exception as e:
        print(f"Error in FBMC optimization: {e}")
        # If optimization fails, return the original generation as fallback
        return {
            'generators_t': {
                'p': perturbed_network.generators_t.p.copy()
            }
        }

def _calculate_fbmc_gen_difference(
    original_network: pypsa.Network,
    fbmc_results: Dict
) -> np.ndarray:
    """Calculate generation differences between original and FBMC results."""
    # Extract generation from FBMC results
    fbmc_gen = fbmc_results['generators_t']['p']
    
    # Calculate difference with original generation
    difference = (fbmc_gen - original_network.generators_t.p).values.T
    
    return difference

def _check_gsk_convergence(
    new_gsk: Dict[pd.Timestamp, pd.DataFrame],
    current_gsk: Dict[pd.Timestamp, pd.DataFrame],
    tolerance: float = 0.01
) -> bool:
    """Check if GSK has converged by comparing new and current values."""
    # Simple implementation: check if maximum absolute difference is below tolerance
    for ts in new_gsk:
        max_diff = abs(new_gsk[ts].values - current_gsk[ts].values).max()
        if max_diff > tolerance:
            return False
    return True


def gsk_adjustable_cap(
        generators: pd.DataFrame, 
        buses: pd.DataFrame, 
        adjustable_carriers: list[str] = ['hydropower']
        ) -> pd.DataFrame:
    """
    Calculate the Grid Supply Contribution (GSK) based on adjustable energy capacity.
    
    This method assigns GSK values proportional to adjustable generation capacity at each node,
    assuming adjustable plants are the main flexible resources that respond to changes
    in net position.
    
    If a zone has no adjustable generators, GSK values are distributed evenly
    across all generators in that zone.
    
    Parameters
    ----------
    generators : pd.DataFrame
        The DataFrame containing generator data with columns ['carrier', 'bus', 'p_nom'].
    buses : pd.DataFrame
        The DataFrame containing bus data with column ['zone_name'].
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the GSKs for each bus-zone pair.
    
    Raises
    ------
    ValueError
        If no generators are found in one or more zones.
    """
    # Constants
    ZONE_COLUMN = 'zone_name'
    BUS_COLUMN = 'bus'
    P_NOM_COLUMN = 'p_nom'
    
    # Filter generators for adjustable carriers and remove up and down regulators
    adjustable_generators = generators[generators['carrier'].isin(adjustable_carriers)]
    adjustable_generators = adjustable_generators[~(adjustable_generators.index.str.endswith('--rd_up') | adjustable_generators.index.str.endswith('--rd_dn'))]
    
    if len(adjustable_generators) == 0:
        raise ValueError("No adjustable generators found in the network.")
    
    # Assign zone names to adjustable generators based on their bus
    adjustable_zones = buses.loc[adjustable_generators[BUS_COLUMN]][ZONE_COLUMN]
    adjustable_generators = adjustable_generators.assign(zone_name=adjustable_zones.values)
        
    # Get the total adjustable capacity per zone and per bus
    total_adjustable_capacity_per_zone = adjustable_generators.groupby(ZONE_COLUMN)[P_NOM_COLUMN].sum()
    adjustable_capacity_per_node = adjustable_generators.groupby([BUS_COLUMN, ZONE_COLUMN])[P_NOM_COLUMN].sum().to_frame()
    
    # Identify zones with zero adjustable capacity
    zero_capacity_zones = total_adjustable_capacity_per_zone[total_adjustable_capacity_per_zone == 0].index.tolist()
    
    # Get all zones from the network
    all_zones = buses[ZONE_COLUMN].unique()
    
    # Create an adjustable capacity matrix (initially zeros)
    adjustable_capacity_matrix = pd.DataFrame(0.0, index=all_zones, columns=buses.index)
    
    # Fill in the matrix with adjustable capacity values
    for (bus, zone), capacity in adjustable_capacity_per_node.itertuples():
        adjustable_capacity_matrix.at[zone, bus] = capacity
    
    # Initialize GSK matrix with zeros 
    gsk_matrix = pd.DataFrame(0.0, index=all_zones, columns=buses.index)
    
    # For zones with adjustable capacity, calculate GSK based on capacity
    zones_with_adjustable = total_adjustable_capacity_per_zone[total_adjustable_capacity_per_zone > 0].index
    for zone in zones_with_adjustable:
        gsk_matrix.loc[zone] = adjustable_capacity_matrix.loc[zone] / total_adjustable_capacity_per_zone[zone]
    
    # For zones with zero adjustable capacity, distribute GSK evenly across all generators in the zone
    for zone in zero_capacity_zones:
        # Create zone_generators using a safer approach to avoid alignment issues
        # First map buses to zones for all generators
        gen_buses = generators[BUS_COLUMN].values
        bus_zones = buses.loc[gen_buses, ZONE_COLUMN].values
        # Create a boolean mask for generators in this zone
        in_zone_mask = (bus_zones == zone)
        zone_generators = generators.loc[in_zone_mask]
        
        if len(zone_generators) == 0:
            raise ValueError(f"Zone {zone} has no generators at all. Cannot calculate GSK.")
        
        # Group generators by bus and count
        bus_generator_counts = zone_generators.groupby(BUS_COLUMN).size()
        
        # Assign equal weight to each bus proportional to number of generators at that bus
        for bus, count in bus_generator_counts.items():
            gsk_matrix.at[zone, bus] = count / len(zone_generators)
    
    # Find any missing zones not covered by adjustable or zero-adjustable processing
    missing_zones = set(all_zones) - set(zones_with_adjustable) - set(zero_capacity_zones)
    
    # For any remaining zones, also distribute GSK evenly across all generators
    for zone in missing_zones:
        # Use the same safer approach as above
        gen_buses = generators[BUS_COLUMN].values
        bus_zones = buses.loc[gen_buses, ZONE_COLUMN].values
        in_zone_mask = (bus_zones == zone)
        zone_generators = generators.loc[in_zone_mask]
        
        if len(zone_generators) == 0:
            raise ValueError(f"Zone {zone} has no generators at all. Cannot calculate GSK.")
        
        bus_generator_counts = zone_generators.groupby(BUS_COLUMN).size()
        for bus, count in bus_generator_counts.items():
            gsk_matrix.at[zone, bus] = count / len(zone_generators)
    
    gsk_matrix.index.name = None
    
    # Ensure that the sum of GSKs in each zone is 1
    if not np.all(np.isclose(gsk_matrix.sum(axis=1), 1)):
        raise ValueError("GSK calculation error: The sum of GSKs in every zone should be 1.")
    
    return gsk_matrix


def gsk_current_generation(generators: pd.DataFrame, generators_t_p: pd.DataFrame, buses: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Calculate the GSK based on current generation values for each snapshot.

    Parameters
    ----------
    generators : pd.DataFrame
        DataFrame containing static generator data (must include 'bus').
    generators_t_p : pd.DataFrame
        DataFrame containing time-series generator power output (p). 
        Index: snapshots, Columns: generator names.
    buses : pd.DataFrame
        DataFrame containing bus data (must include 'zone_name').

    Returns
    -------
    Dict[pd.Timestamp, pd.DataFrame]
        A dictionary where keys are snapshots and values are DataFrames 
        containing the GSKs (Index: zones, Columns: buses).
        
    Raises
    ------
    ValueError
        If generators_t_p is empty, or if mapping generators to zones fails.
    """
    ZONE_COLUMN = 'zone_name'
    BUS_COLUMN = 'bus'

    if generators_t_p.empty:
        raise ValueError("generators_t_p DataFrame cannot be empty.")
        
    if 'bus' not in generators.columns:
         raise ValueError("Generators DataFrame must include 'bus' column.")
         
    if 'zone_name' not in buses.columns:
        raise ValueError("Buses DataFrame must include 'zone_name' column.")

    # Map generators to zones and buses
    try:
        bus_to_zone = buses[ZONE_COLUMN].to_dict()
        gen_to_bus = generators[BUS_COLUMN].to_dict()
        gen_to_zone = {gen: bus_to_zone.get(gen_to_bus.get(gen)) 
                       for gen in generators.index if gen in generators_t_p.columns} # Only consider generators present in generators_t_p
    except KeyError as e:
        raise ValueError(f"Failed to map generators to zones or buses: {e}")

    all_zones = sorted(buses[ZONE_COLUMN].unique())
    all_buses = sorted(buses.index)
    gsk_matrices = {}

    for snapshot in generators_t_p.index:
        snapshot_generation = generators_t_p.loc[snapshot]
        
        # Create DataFrame for the snapshot's generation with bus and zone info
        gen_data_snapshot = pd.DataFrame({'p': snapshot_generation})
        gen_data_snapshot['bus'] = gen_data_snapshot.index.map(gen_to_bus)
        gen_data_snapshot['zone'] = gen_data_snapshot['bus'].map(bus_to_zone)
        
        # Drop generators that couldn't be mapped (e.g., if bus is missing)
        gen_data_snapshot = gen_data_snapshot.dropna(subset=['bus', 'zone'])

        # Calculate total generation per zone for this snapshot
        total_generation_per_zone = gen_data_snapshot.groupby('zone')['p'].sum()
        
        # Calculate generation per bus for this snapshot
        generation_per_bus = gen_data_snapshot.groupby('bus')['p'].sum()

        # Initialize GSK matrix for the snapshot
        gsk_matrix = pd.DataFrame(0.0, index=all_zones, columns=all_buses)

        for zone in all_zones:
            zone_total_gen = total_generation_per_zone.get(zone, 0)
            
            # Get buses belonging to this zone
            buses_in_zone = buses.index[buses[ZONE_COLUMN] == zone].tolist()

            if zone_total_gen > 1e-6: # Use a small threshold to avoid division by zero issues
                # Calculate GSK based on generation share
                for bus in buses_in_zone:
                    bus_gen = generation_per_bus.get(bus, 0)
                    if bus_gen > 0:
                         gsk_matrix.at[zone, bus] = bus_gen / zone_total_gen
            else:
                # Distribute GSK evenly among buses in the zone if total generation is zero
                if buses_in_zone:
                    gsk_matrix.loc[zone, buses_in_zone] = 1.0 / len(buses_in_zone)

        # Ensure rows sum to 1 (handle potential floating point inaccuracies)
        gsk_matrix = gsk_matrix.apply(lambda row: row / row.sum() if row.sum() > 1e-6 else row, axis=1)

        # Re-apply equal distribution for zero-sum rows if normalization failed
        for zone in all_zones:
             if gsk_matrix.loc[zone].sum() < 1e-6:
                 buses_in_zone = buses.index[buses[ZONE_COLUMN] == zone].tolist()
                 if buses_in_zone:
                     gsk_matrix.loc[zone, buses_in_zone] = 1.0 / len(buses_in_zone)

        gsk_matrix.index.name = None # Remove index name
        gsk_matrices[snapshot] = gsk_matrix

    return gsk_matrices
