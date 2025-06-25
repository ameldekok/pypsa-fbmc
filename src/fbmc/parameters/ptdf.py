from typing import Tuple
import numpy as np
import pandas as pd
import pypsa


def get_network_ptdf(basecase: pypsa.Network) -> Tuple[pd.DataFrame, pypsa.SubNetwork]:
    """
    Extract PTDF matrix from the network model.
    Returns PTDF matrix and associated sub_network.
    """
    basecase.determine_network_topology()
    sub_network = basecase.sub_networks.loc['0'].obj
    sub_network.calculate_PTDF()
    
    ptdf = pd.DataFrame(
        sub_network.PTDF,
        index=sub_network.branches().index.droplevel(0), # Drop MultiIndex
        columns=sub_network.buses_o # Ordered list of buses used in all PF and PTDF calculations (slack first, then PV, then PQ)
    )

    # Reindex PTDF to match the order of buses in the basecase
    ptdf_readable = ptdf.reindex(columns=basecase.buses.index) 
    
    return ptdf_readable, sub_network

def calculate_zonal_ptdf(ptdf: pd.DataFrame, gsk: pd.DataFrame) -> pd.DataFrame:
    """
    Transform nodal PTDF to zonal PTDF using Generation Shift Keys (GSK).
    """
    if not set(ptdf.columns).issubset(set(gsk.columns)):
        raise ValueError("PTDF columns must match GSK bus names") #PTDF is based on subnetwork, GSK is based on full network
        
    #TODO: Check if the signs of the CNE are correct - if not, multiply by -1
    
    # Get the GSK for the nodes in the PTDF
    gsk_filtered = gsk.loc[:, ptdf.columns]
    
    # Calculate the zPTDF by multiplying the PTDF with the GSK
    z_ptdf_array = np.dot(ptdf.values, gsk_filtered.T)
    
    # To dataframe; index = branches, columns = zones
    z_ptdf = pd.DataFrame(z_ptdf_array, index=ptdf.index, columns=gsk_filtered.index)
    
    return z_ptdf
