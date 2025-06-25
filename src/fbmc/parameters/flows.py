import pandas as pd
import pypsa


def calculate_flow_reliability_margin(line_capacities: pd.Series, reliability_margin_factor: float = 0.1) -> pd.Series:
    """Calculate Flow Reliability Margin (FRM) for transmission lines.
    
    Args:
        line_capacities: Thermal capacity limits in MW
        reliability_margin_factor: Safety factor (0-1, default 0.1)
        
    Returns:
        Flow Reliability Margin values in MW
    """
    if not 0 <= reliability_margin_factor <= 1:
        raise ValueError("Reliability margin factor must be between 0 and 1")
    
    if (line_capacities < 0).any():
        raise ValueError("Line capacities must be positive")

    return line_capacities * reliability_margin_factor

def get_base_flows(basecase: pypsa.Network) -> pd.DataFrame:
    """Get the base case power flows from transformers, links and lines.
    Assumes there are no transformers, links or lines with the same name."""
    return pd.concat([
        basecase.transformers_t.p0.T, 
        basecase.links_t.p0.T, 
        basecase.lines_t.p0.T
    ])

def get_net_positions(buses: pd.DataFrame, buses_t: pd.DataFrame, zones: pd.Index) -> pd.DataFrame:
    """Calculate net positions for each zone based on bus power values.
    
    Args:
        buses: DataFrame containing bus data with zone_name column
        buses_t: DataFrame containing time series bus power values
        zones: Index of zone names to calculate positions for
        
    Returns:
        DataFrame with net positions per zone
    """
    net_positions = pd.DataFrame(0.0, index=buses_t.p.index, columns=zones)
    
    for bus in buses_t.p.columns:
        zone = buses.loc[bus, 'zone_name']
        if zone in net_positions.columns:
            net_positions.loc[:, zone] += buses_t.p[bus]
            
    return net_positions

def calculate_reference_flow(base_flows: pd.DataFrame, 
                           zonal_ptdf: pd.DataFrame,
                           zonal_net_positions: pd.DataFrame) -> pd.DataFrame:
    """Calculate reference flows using base flows, PTDF and net positions."""
    # Align base flows with zonal PTDF indices
    base_flows_aligned = base_flows.loc[zonal_ptdf.index]
    return base_flows_aligned - zonal_ptdf.dot(zonal_net_positions.T)

def calculate_ram(network: pypsa.Network,
                 zonal_ptdf: pd.DataFrame,
                 min_ram: float = 0.0,
                 reliability_margin_factor: float = 0.1) -> pd.DataFrame:
    """
    Calculate the Remaining Available Margin (RAM) for a given power network.
    
    Args:
        network: PyPSA network containing the initial state
        zonal_ptdf: The zonal Power Transfer Distribution Factors matrix
        min_ram: Minimum RAM value as fraction of capacity (default 0.0)
        reliability_margin_factor: Safety factor for reliability margin (default 0.1)
    
    Returns:
        DataFrame containing RAM values for each branch
    """
    # Get base state
    base_flows = get_base_flows(network)
    branch_capacity = network.branches().s_nom
    branch_capacity.index = branch_capacity.index.droplevel(0) # Drop MultiIndex
    
    # Calculate components
    frm = calculate_flow_reliability_margin(branch_capacity, 
                                          reliability_margin_factor=reliability_margin_factor)
    net_positions = get_net_positions(network.buses, network.buses_t, zonal_ptdf.columns)
    f_reference = calculate_reference_flow(base_flows, zonal_ptdf, net_positions)
    
    # Calculate final RAM values
    partial_ram = branch_capacity - frm
    partial_ram = partial_ram.loc[zonal_ptdf.index]
    ram = pd.DataFrame(partial_ram.to_numpy().reshape(-1,1) - f_reference.to_numpy(),
                      index=zonal_ptdf.index, 
                      columns=f_reference.columns)
    
    if min_ram > 0:
        ram = ram.clip(lower=min_ram*branch_capacity)
        
    return ram
