"""
Functions for converting between nodal and zonal network representations.
"""

import pandas as pd
import pypsa
from typing import List, Optional

from src.fbmc.parameters.helpers import silence_output


def nodal_to_zonal(
    nodal_network: pypsa.Network,
    snapshots: Optional[List] = None,
    zone_column: str = 'zone_name',
    bidirectional_links: bool = True
) -> pypsa.Network:
    """
    Convert a nodal PyPSA network to a zonal network representation.
    
    This function aggregates buses into zones, moves generators and loads to their
    respective zonal buses, and creates links between zones based on the lines
    connecting the original buses.
    
    Parameters
    ----------
    nodal_network : pypsa.Network
        The original nodal network to convert
    snapshots : list, optional
        Specific snapshots to include in the zonal network; if None, uses all snapshots
        from the nodal_network
    zone_column : str, default 'zone_name'
        The column in nodal_network.buses that contains the zone information
    bidirectional_links : bool, default True
        If True, creates separate links for each direction between zones;
        if False, creates a single bidirectional link between zones
        
    Returns
    -------
    pypsa.Network
        A new zonal network with buses representing zones and links representing
        inter-zonal transmission capacity
    
    Notes
    -----
    - The function preserves all carrier types from the original network
    - Generators and loads maintain their original IDs but are moved to zonal buses
    - Link capacity between zones is equal to the sum of line capacities connecting buses
      in different zones
    """
    # Validate inputs
    if zone_column not in nodal_network.buses.columns:
        raise ValueError(f"The column '{zone_column}' does not exist in the nodal network's buses")
    
    # Create a new zonal network
    zonal_network = pypsa.Network()
    
    # Set snapshots
    if snapshots is None:
        snapshots = nodal_network.snapshots
    zonal_network.set_snapshots(snapshots)
    
    # Add zonal buses (one for each unique zone in the nodal network)
    zones = nodal_network.buses[zone_column].unique()
    for zone in zones:
        zonal_network.add("Bus", zone, v_nom=nodal_network.buses.v_nom.mean(), zone_name=zone)
    
    # Move generators to their respective zonal buses
    for gen in nodal_network.generators.itertuples():
        # Get the zone name for this generator's bus
        bus_name = gen.bus
        zone_name = nodal_network.buses.at[bus_name, zone_column]
        
        # Add the generator to the zonal network at its zone
        gen_dict = {col: getattr(gen, col) for col in nodal_network.generators.columns 
                   if col != 'bus' and hasattr(gen, col)}
        gen_dict['bus'] = zone_name  # Use the zone name as the bus
        
        zonal_network.add("Generator", gen.Index, **gen_dict)
    
    # Copy generator time series data if it exists
    for attr_name in nodal_network.generators_t:
        if not nodal_network.generators_t[attr_name].empty:
            zonal_network.generators_t[attr_name] = nodal_network.generators_t[attr_name].copy()
    
    # Move loads to their respective zonal buses
    if not nodal_network.loads.empty:
        for load in nodal_network.loads.itertuples():
            # Get the zone name for this load's bus
            bus_name = load.bus
            zone_name = nodal_network.buses.at[bus_name, zone_column]
            
            # Add the load to the zonal network at its zone
            load_dict = {col: getattr(load, col) for col in nodal_network.loads.columns 
                        if col != 'bus' and hasattr(load, col)}
            load_dict['bus'] = zone_name  # Use the zone name as the bus
            
            zonal_network.add("Load", load.Index, **load_dict)
        
        # Copy load time series data if it exists
        for attr_name in nodal_network.loads_t:
            if not nodal_network.loads_t[attr_name].empty:
                zonal_network.loads_t[attr_name] = nodal_network.loads_t[attr_name].copy()
    
    # Calculate the total line capacity between zones and create links
    zone_connections = {}
    
    # Loop through all lines and calculate total capacity between zones
    if not nodal_network.lines.empty:
        for line in nodal_network.lines.itertuples():
            bus0_zone = nodal_network.buses.at[line.bus0, zone_column]
            bus1_zone = nodal_network.buses.at[line.bus1, zone_column]
            
            # Skip if the line connects buses in the same zone
            if bus0_zone == bus1_zone:
                continue
            
            # Ensure consistent ordering of zones for dictionary keys
            zone_pair = tuple(sorted([bus0_zone, bus1_zone]))
            
            # Add capacity to the total for this zone pair
            if zone_pair not in zone_connections:
                zone_connections[zone_pair] = 0
            zone_connections[zone_pair] += line.s_nom
    
    # Create links between zones with the total capacity
    for (zone1, zone2), capacity in zone_connections.items():
        if bidirectional_links:
            # Create bidirectional links (two one-directional links)
            zonal_network.add("Link", 
                            f"link_{zone1}_{zone2}", 
                            bus0=zone1, 
                            bus1=zone2, 
                            p_nom=capacity,
                            p_min_pu=-1 if bidirectional_links else 0)
            
            zonal_network.add("Link", 
                            f"link_{zone2}_{zone1}", 
                            bus0=zone2, 
                            bus1=zone1, 
                            p_nom=capacity,
                            p_min_pu=-1 if bidirectional_links else 0)
        else:
            # Create a single bidirectional link
            zonal_network.add("Link", 
                            f"link_{zone1}_{zone2}", 
                            bus0=zone1, 
                            bus1=zone2, 
                            p_nom=capacity,
                            p_min_pu=-1)
    
    # Copy carriers from the original network
    if hasattr(nodal_network, 'carriers') and not nodal_network.carriers.empty:
        for carrier in nodal_network.carriers.index:
            if carrier not in zonal_network.carriers.index:
                carrier_data = {col: nodal_network.carriers.at[carrier, col] 
                               for col in nodal_network.carriers.columns 
                               if carrier in nodal_network.carriers.index}
                zonal_network.add('Carrier', carrier, **carrier_data)
    
    return zonal_network


def zonal_to_nodal(
    zonal_network: pypsa.Network, 
    nodal_network: pypsa.Network,
    zone_column: str = 'zone_name'
) -> pypsa.Network:
    """
    Map the results from a zonal FBMC network optimization back to a nodal network.
    
    This function takes the generation dispatch from a zonal network's FBMC optimization 
    (Generator-p values) and maps it directly to the corresponding generators in 
    the nodal network.
    
    Parameters
    ----------
    zonal_network : pypsa.Network
        The zonal network with FBMC optimization results
    nodal_network : pypsa.Network
        The original nodal network to map results back to
    zone_column : str, default 'zone_name'
        The column in nodal_network.buses that contains the zone information
        
    Returns
    -------
    pypsa.Network
        A copy of the nodal network with generator dispatch updated according to
        the FBMC optimization results
        
    Raises
    ------
    ValueError
        If the zonal network doesn't have optimization results in model.solution
    """
    # Validate inputs
    if zone_column not in nodal_network.buses.columns:
        raise ValueError(f"The column '{zone_column}' does not exist in the nodal network's buses")
    
    # Check if zonal network has optimization results
    if not hasattr(zonal_network, 'model') or not hasattr(zonal_network.model, 'solution'):
        raise ValueError("Zonal network must have optimization results in model.solution")
        
    # Check if Generator-p exists in the solution
    if 'Generator-p' not in zonal_network.model.solution:
        raise ValueError("Zonal network solution must contain 'Generator-p' results")
    
    # Create a copy of the nodal network
    result_network = nodal_network.copy(snapshots=zonal_network.snapshots)

    # Optimise the nodal network such that it has a linopy model and a solution
    with silence_output():
        result_network.optimize(solver_name="gurobi")

    # Get the generator production from the FBMC market clearing
    # Handle xarray DataArray format in model.solution
    generator_output_xarray = zonal_network.model.solution['Generator-p']
    
    # Convert xarray to dataframe and reshape
    generator_output_df = generator_output_xarray.to_dataframe().unstack('Generator')
    
    # Remove the extra level
    generator_output_df.columns = generator_output_df.columns.droplevel(0)
    
    # Map the values to the nodal network
    for snapshot in zonal_network.snapshots:
        for gen_id in result_network.generators.index:
            if gen_id in generator_output_df.columns:
                result_network.generators_t.p.at[snapshot, gen_id] = generator_output_df.at[snapshot, gen_id]
    
    # Add the loads from the zonal network to the nodal network. We can use loads_t from zonal_network

    for snapshot in zonal_network.snapshots:
        for load_id in result_network.loads.index:
            if load_id in zonal_network.loads.index:
                result_network.loads_t.p.at[snapshot, load_id] = zonal_network.loads_t.p.at[snapshot, load_id]
                result_network.loads_t.p_set.at[snapshot, load_id] = zonal_network.loads_t.p_set.at[snapshot, load_id]

    return result_network