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

# def calculate_ram(network: pypsa.Network,
#                  zonal_ptdf: pd.DataFrame,
#                  min_ram: float = 0.0,
#                  reliability_margin_factor: float = 0.1,
#                  add_zptdf_np_term: bool = True,
#                  ) -> pd.DataFrame:
#     """
#     Calculate the Remaining Available Margin (RAM) for a given power network.
    
#     Args:
#         network: PyPSA network containing the initial state
#         zonal_ptdf: The zonal Power Transfer Distribution Factors matrix
#         min_ram: Minimum RAM value as fraction of capacity (default 0.0)
#         reliability_margin_factor: Safety factor for reliability margin (default 0.1)
    
#     Returns:
#         DataFrame containing RAM values for each branch
#     """
#     if network.transformers.index.isin(network.lines.index).any():
#         raise ValueError("Transformers and lines cannot have the same names")
#     if not network.links.empty:
#         raise Warning("Links are not fully supported.")
#     # Get base state
#     base_flows = get_base_flows(network)
#     branch_capacity = network.branches().s_nom
#     branch_capacity.index = branch_capacity.index.droplevel(0) # Drop MultiIndex
    
#     # Calculate components
#     frm = calculate_flow_reliability_margin(branch_capacity, 
#                                           reliability_margin_factor=reliability_margin_factor)
#     net_positions = get_net_positions(network.buses, network.buses_t, zonal_ptdf.columns)

#     # Calculate final RAM values
#     partial_ram = branch_capacity - frm
#     partial_ram = partial_ram.loc[zonal_ptdf.index]


#     ram = (partial_ram - base_flows.loc[zonal_ptdf.index].T).T

#     # Optionally add the zonal PTDF term
#     if add_zptdf_np_term:
#         zptdf_term = zonal_ptdf @ net_positions.T  # matrix multiplication with pandas
#         ram = ram.add(zptdf_term)


#     if min_ram > 0:
#         ram = ram.clip(lower=min_ram*branch_capacity)
#     return ram * 10 + 1000


def calculate_ram(network: pypsa.Network,
                  zonal_ptdf,
                  min_ram: float = 0.0,
                  reliability_margin_factor: float = 0.1,
                  add_zptdf_np_term: bool = True,
                  ) -> pd.DataFrame:
    """
    Calculate the Remaining Available Margin (RAM) for a given power network.

    Args:
        network: PyPSA network containing the initial state
        zonal_ptdf: Either a static pd.DataFrame or a dict {snapshot: pd.DataFrame}
        min_ram: Minimum RAM value as fraction of capacity (default 0.0)
        reliability_margin_factor: Safety factor for reliability margin (default 0.1)

    Returns:
        DataFrame (snapshots x branches) of RAM values
    """
    if network.transformers.index.isin(network.lines.index).any():
        raise ValueError("Transformers and lines cannot have the same names")
    if not network.links.empty:
        raise Warning("Links are not fully supported.")
    # Get base state
    base_flows = get_base_flows(network)  # shape: (branches, snapshots)
    branch_capacity = network.branches().s_nom
    branch_capacity.index = branch_capacity.index.droplevel(0)  # Drop MultiIndex

    # Calculate flow reliability margin
    frm = calculate_flow_reliability_margin(branch_capacity, 
                                            reliability_margin_factor=reliability_margin_factor)

    # Prepare output container
    snapshots = network.snapshots

    if isinstance(zonal_ptdf, dict):
        # Time-dependent zonal PTDF case
        ram_dict = {}

        for snapshot in snapshots:
            if snapshot not in zonal_ptdf:
                raise ValueError(f"Snapshot {snapshot} missing from zonal_ptdf dict.")
            
            zptdf_df = zonal_ptdf[snapshot]
            partial_ram = branch_capacity - frm
            partial_ram = partial_ram.loc[zptdf_df.index]

            net_positions = get_net_positions(network.buses, network.buses_t, zptdf_df.columns).loc[snapshot]

            ram = (partial_ram - base_flows.loc[zptdf_df.index, snapshot])
            
            if add_zptdf_np_term:
                zptdf_term = zptdf_df @ net_positions.T  # shape: (branches,)
                ram = ram + zptdf_term

            if min_ram > 0:
                ram = ram.clip(lower=min_ram * branch_capacity)

            ram_dict[snapshot] = ram

        ram_df = pd.DataFrame(ram_dict) # shape: (branches, snapshots)

    else:
        # Static zonal PTDF case
        partial_ram = branch_capacity - frm
        partial_ram = partial_ram.loc[zonal_ptdf.index]

        net_positions = get_net_positions(network.buses, network.buses_t, zonal_ptdf.columns)

        ram_df = (partial_ram - base_flows.loc[zonal_ptdf.index].T).T  # shape: (snapshots, branches)

        if add_zptdf_np_term:
            zptdf_term = zonal_ptdf @ net_positions.T  # shape: (branches, snapshots)
            ram_df = ram_df.add(zptdf_term.T)

        if min_ram > 0:
            ram_df = ram_df.clip(lower=min_ram * branch_capacity)

    return ram_df
