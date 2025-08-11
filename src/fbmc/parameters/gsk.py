import numpy as np
import pandas as pd
import pypsa
from typing import List, Dict, Tuple, Union, Optional


from ..config import FBMCConfig
from .helpers import (
    get_uncertain_elements, 
    initialize_nodal_injection_difference,
    introduce_variation_to_network,
    calculate_gsk_per_bus,
    silence_output,
    silent_run_opf
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
    if config.gsk_method == "ADJUSTABLE_CAP":
        return gsk_adjustable_cap(network.generators, network.buses)
    elif config.gsk_method == "ITERATIVE_UNCERTAINTY":
        return gsk_iterative_uncertainty(
            network,
            uncertain_carriers=config.uncertain_carriers,
            num_scenarios=config.num_scenarios,
            gen_variation_std_dev=config.gen_variation_std_dev,
            load_variation_std_dev=config.load_variation_std_dev,
            config=config
        )
    elif config.gsk_method == "CURRENT_GENERATION":
        return gsk_current_generation(network.generators, network.generators_t.p, network.buses)
    elif config.gsk_method == "ITERATIVE_FBMC":
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
    else:
        raise ValueError(f"Unknown method: {config.gsk_method}. Supported methods are: 'ADJUSTABLE_CAP', 'ITERATIVE_UNCERTAINTY', 'CURRENT_GENERATION', 'ITERATIVE_FBMC'.")
    

def gsk_iterative_uncertainty(
    network: pypsa.Network,
    uncertain_carriers: List[str] = ['offshore-wind', 'onshore-wind'],
    num_scenarios: int = 10,
    gen_variation_std_dev: float = 0.1,
    load_variation_std_dev: float = 0.1,
    config: Optional[FBMCConfig] = None
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
    if num_scenarios < 1:
        raise ValueError("Number of iterations must be at least 1")
    if gen_variation_std_dev < 0 or load_variation_std_dev < 0:
        raise ValueError("Standard deviations must be non-negative")


    # Set up random generator from config.base_seed
    if config is not None and hasattr(config, 'base_seed') and config.base_seed is not None:
        rng = np.random.default_rng(config.base_seed)
        np.random.seed(config.base_seed)  # For legacy code if needed
    else:
        rng = np.random.default_rng()

    # Run the network once to get initial generation and load values
    try:
        with silence_output():
            network.optimize(solver_name='gurobi')
    except Exception as e:
        raise RuntimeError(f"Failed to optimize initial network state: {e}")

    # Collect the loads and gens (wind) with uncertainty
    uncertain_gens, uncertain_loads = get_uncertain_elements(network, uncertain_carriers)
    
    # Initialize the nodal injections difference array
    nodal_injections_difference = initialize_nodal_injection_difference(network, num_scenarios)

    # Save the original nodal injections
    original_nodal_injections = network.buses_t.p.copy()

    # Perform stochastic sampling
    for i in range(num_scenarios):
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

        silent_run_opf(stochastic_network)
        nodal_injections_difference[i,:,:] = (stochastic_network.buses_t.p - original_nodal_injections).values.T

    # Calculate the GSK based on the nodal injections difference
    processed_nodal_difference = calculate_gsk_per_bus(nodal_injections_difference, network)

    # Optional convergence check: compare N vs 2N scenarios
    if config is not None and getattr(config, 'enable_uncertainty_convergence_check', False):
        # Reseed to reproduce the same first N draws, so 2N includes those plus extra
        if hasattr(config, 'base_seed') and config.base_seed is not None:
            np.random.seed(config.base_seed)
        n2 = 2 * num_scenarios
        nodal_injections_difference_2n = initialize_nodal_injection_difference(network, n2)
        original_nodal_injections_2n = network.buses_t.p.copy()

        for i in range(n2):
            stochastic_network = network.copy(snapshots=network.snapshots)
            introduce_variation_to_network(
                stochastic_network,
                uncertain_gens,
                uncertain_loads,
                gen_variation_std_dev,
                load_variation_std_dev,
            )
            silent_run_opf(stochastic_network)
            nodal_injections_difference_2n[i,:,:] = (stochastic_network.buses_t.p - original_nodal_injections_2n).values.T

        gsk_2n = calculate_gsk_per_bus(nodal_injections_difference_2n, network)

        print("Uncertainty-based GSK convergence check (N vs 2N scenarios):")
        _check_gsk_convergence(gsk_2n, processed_nodal_difference, tolerance=(config.fbmc_iter_tolerance if hasattr(config, 'fbmc_iter_tolerance') else 0.01))

    # Process results and calculate GSK
    return processed_nodal_difference

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

        # Set up random generator for this scenario refinement
        if hasattr(config, 'base_seed') and config.base_seed is not None:
            rng = np.random.default_rng(config.base_seed)
            np.random.seed(config.base_seed)  # For legacy code if needed
        else:
            rng = np.random.default_rng()

        # Collect uncertain elements
        uncertain_gens, uncertain_loads = get_uncertain_elements(network, uncertain_carriers)
        
        # Initialize storage for nodal injection differences in this GSK iteration
        nodal_difference_data = initialize_nodal_injection_difference(network, num_iterations)

        original_nodal_injections = network.buses_t.p.copy()
        
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
            
            # Run FBMC with current GSK to get nodal difference
            fbmc_nodal_difference_result = _run_fbmc_with_gsk(perturbed_network, current_gsk, config)
            nodal_difference_data[i,:,:] = (fbmc_nodal_difference_result - original_nodal_injections).values.T
           
        # Calculate new GSK from Monte Carlo results
        new_gsk = calculate_gsk_per_bus(nodal_difference_data, network)
        
        # Check for convergence
        if _check_gsk_convergence(new_gsk, current_gsk, tolerance=config.fbmc_iter_tolerance):
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
    
    if method == "CURRENT_GENERATION":
        return gsk_current_generation(network.generators, network.generators_t.p, network.buses)
    elif method == "ADJUSTABLE_CAP":
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
            
            opt_loads_df = zonal_network.loads_t.p.copy()

            # Sum generator outputs by bus
            gen_inj = opt_gens_df.T.groupby(perturbed_network.generators.bus).sum().T
            
            # Sum loads by bus (loads consumed are negative injections)
            load_inj = opt_loads_df.T.groupby(perturbed_network.loads.bus).sum().T
            
            # Align to all buses
            buses = perturbed_network.buses.index
            gen_inj = gen_inj.reindex(columns=buses, fill_value=0.0)
            load_inj = load_inj.reindex(columns=buses, fill_value=0.0)
            
            # Net injection: generation minus load
            nodal_inj = gen_inj - load_inj

            return nodal_inj
        else:
            print("Warning: Optimization model solution not found, using direct network values")
            return {
                'generators_t': {
                    'p': zonal_network.buses_t.p.copy()
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

def _check_gsk_convergence(
    new_gsk: Dict[pd.Timestamp, pd.DataFrame],
    current_gsk: Dict[pd.Timestamp, pd.DataFrame],
    tolerance: float = 0.01
) -> bool:
    """Check if GSK has converged by comparing new and current values.

    Prints the maximum absolute difference per timestamp and an overall summary
    to help track convergence across iterations.
    """
    overall_max_diff = 0.0
    converged = True

    for ts in new_gsk:
        try:
            # Fast path matching existing behavior
            max_diff = float(np.max(np.abs(new_gsk[ts].values - current_gsk[ts].values)))
        except Exception:
            # Safer path with alignment in case of ordering/shape issues
            diff_df = (new_gsk[ts] - current_gsk[ts]).abs()
            max_diff = float(np.nanmax(diff_df.to_numpy()))

        overall_max_diff = max(overall_max_diff, max_diff)
        print(f"GSK convergence check: ts={ts}, max abs diff={max_diff:.6f}, tolerance={tolerance}")

        if max_diff > tolerance:
            converged = False

    print(
        f"GSK convergence summary: overall max abs diff={overall_max_diff:.6f}, "
        f"tolerance={tolerance} -> {'converged' if converged else 'not converged'}"
    )
    return converged


def gsk_adjustable_cap(generators: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
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
    ADJUSTABLE_CARRIERS = ['hydropower']
    ZONE_COLUMN = 'zone_name'
    BUS_COLUMN = 'bus'
    P_NOM_COLUMN = 'p_nom'
    
    # Filter generators for adjustable carriers and remove up and down regulators
    adjustable_generators = generators[generators['carrier'].isin(ADJUSTABLE_CARRIERS)]
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
