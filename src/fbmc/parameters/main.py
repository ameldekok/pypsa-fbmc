import pypsa
from .cne import determine_cnes, filter_on_cne
from .flows import calculate_ram
from .gsk import calculate_gsk
from .ptdf import calculate_zonal_ptdf, get_network_ptdf

from ..config import FBMCConfig
import pandas as pd


from typing import Tuple, Dict



def calculate_fbmc_parameters(
        basecase: pypsa.Network,
        config: FBMCConfig = FBMCConfig(),
        gsk = None,
        ) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Calculate the Flow-Based Market Coupling (FBMC) parameters for a given power network basecase.

    Parameters:
        basecase (pypsa.Network): The power network basecase object containing network data such as lines, generators, and buses.
        config (FBMCConfig): Configuration object containing parameters for FBMC calculations. Defaults to FBMCConfig().

    Returns:
        Tuple[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]: 
            - ram_cnes: A DataFrame containing the Remaining Available Margin (RAM) filtered on Critical Network Elements (CNEs).
            - z_ptdf_cnes: A dictionary of DataFrames containing the zonal Power Transfer Distribution Factors (PTDF) filtered on CNEs,
              with one DataFrame per snapshot if using snapshot-based GSKs.

    Notes:
        - The function calculates the maximum absolute flow on lines, determines the CNEs, computes the Generation Shift Key (GSK),
          and calculates the PTDF and zonal PTDF.
        - The RAM is computed using a reliability margin factor and a minimum RAM threshold.
        - Both RAM and zonal PTDF are filtered based on the identified CNEs.
        - When using ITERATIVE_UNCERTAINTY GSK method, the zonal PTDF will vary by snapshot.
    """

    # Calculate the FBMC parameters
    max_absolute_flow = basecase.lines_t.p0.abs().max()
    line_capacity = basecase.lines.s_nom

    cnes = determine_cnes(
        max_absolute_flow,
        line_capacity,
        line_usage_threshold = config.line_usage_threshold
        )
    if gsk is None:
        gsk = calculate_gsk(basecase, config)
        
    ptdf, _ = get_network_ptdf(basecase)
    
    # Handle both static and snapshot-based GSKs
    if isinstance(gsk, dict):  # Snapshot-based GSKs
        z_ptdf = {}
        for snapshot, gsk_snapshot in gsk.items():
            z_ptdf[snapshot] = calculate_zonal_ptdf(ptdf, gsk_snapshot)
            
        # Calculate RAM - this remains the same as it's already snapshot-based
        ram = calculate_ram(basecase,
                          zonal_ptdf = z_ptdf[basecase.snapshots[0]],  # Use first snapshot's PTDF for RAM calculation
                          min_ram = config.min_ram, 
                          reliability_margin_factor = config.reliability_margin_factor)
        
        # Filter on CNEs for each snapshot
        ram_cnes = filter_on_cne(ram, cnes)
        z_ptdf_cnes = {snapshot: filter_on_cne(z_ptdf_snapshot, cnes) 
                       for snapshot, z_ptdf_snapshot in z_ptdf.items()}
    else:  # Static GSK
        z_ptdf = calculate_zonal_ptdf(ptdf, gsk)
        
        ram = calculate_ram(basecase,
                          zonal_ptdf = z_ptdf, 
                          min_ram = config.min_ram, 
                          reliability_margin_factor = config.reliability_margin_factor)
        
        ram_cnes = filter_on_cne(ram, cnes)
        z_ptdf_cnes = filter_on_cne(z_ptdf, cnes)

    return ram_cnes, z_ptdf_cnes