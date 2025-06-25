import linopy as lp
import pandas as pd
import xarray as xr
import numpy as np

# ---- Data Transformations ----

def convert_zPTDF_to_xarray(zPTDF_data) -> xr.DataArray:
    """
    Convert zPTDF data to an xarray DataArray.
    
    Parameters
    ----------
    zPTDF_data : pd.DataFrame or dict of pd.DataFrame
        Either a single DataFrame containing zPTDF values for all snapshots,
        or a dictionary of DataFrames with snapshots as keys.
        
    Returns
    -------
    xr.DataArray
        For snapshot-dependent zPTDF: A 3D DataArray with dimensions [snapshot, CNE, Zone]
        For static zPTDF: A 2D DataArray with dimensions [CNE, Zone]
    """
    if isinstance(zPTDF_data, dict):
        # For snapshot-dependent zPTDF (dictionary of DataFrames)
        snapshots = list(zPTDF_data.keys())
        cnes = zPTDF_data[snapshots[0]].index
        zones = zPTDF_data[snapshots[0]].columns
        
        # Create a 3D array to hold all zPTDF values
        data_array = np.zeros((len(snapshots), len(cnes), len(zones)))
        
        # Fill the array with values from each snapshot
        for i, snapshot in enumerate(snapshots):
            data_array[i, :, :] = zPTDF_data[snapshot].values
        
        # Convert to xarray DataArray with proper dimensions and coordinates
        return xr.DataArray(
            data_array,
            dims=["snapshot", "CNE", "Zone"],
            coords={
                "snapshot": snapshots,
                "CNE": cnes,
                "Zone": zones
            }
        )
    else:
        # For static zPTDF (single DataFrame)
        return xr.DataArray(
            zPTDF_data,
            dims=["CNE", "Zone"],
            coords={"CNE": zPTDF_data.index, "Zone": zPTDF_data.columns}
        )

def convert_RAM_to_xarray(RAM_df: pd.DataFrame) -> xr.DataArray:
    """
    Convert a DataFrame containing RAM values to a DataArray.
    """
    return xr.DataArray(
        RAM_df,
        dims=["CNE", "snapshot"],
        coords={"CNE": RAM_df.index, "snapshot": RAM_df.columns}
    )

# ---- Load Mapping ----

def create_load_zone_mapping(loads: pd.DataFrame) -> xr.DataArray:
    """
    Create an xarray mapping of loads to their respective zones.
    """
    load_zone_mapping = xr.DataArray(
        [loads.at[load, 'bus'] for load in loads.index],
        dims=["Load"],
        coords={"Load": loads.index}
    )
    return load_zone_mapping

def create_load_zone_mask(load_zone_mapping: xr.DataArray, zones: list) -> xr.DataArray:
    """
    Create a mask for the loads in the zones.
    """
    zone_da = xr.DataArray(zones, dims = ["Zone"], coords = {"Zone": zones})
    mask = zone_da == load_zone_mapping
    return mask

def get_zonal_loads(load_zone_mask, loads_t_pset):
    """
    Get the total loads per zone (xarray)
    """
    loads_xr = xr.DataArray(
        loads_t_pset.T,
        dims=["Load", "snapshot"],
        coords={"Load": loads_t_pset.columns, "snapshot": loads_t_pset.index}
    )
    return (loads_xr * load_zone_mask).sum(dim="Load")

# ---- Constraint Construction ----

def construct_cne_constraint(zPTDF: xr.DataArray, total_zonal_generation: lp.Variable, zonal_loads: xr.DataArray, RAM: xr.DataArray):
    """
    Create the constraint restricting the flow on CNEs by the Remaining Available Margin (RAM).
    
    This function handles both snapshot-dependent and static zPTDFs.
    
    Parameters
    ----------
    zPTDF : xr.DataArray
        Either a 2D DataArray with dimensions [CNE, Zone] for static zPTDF,
        or a 3D DataArray with dimensions [snapshot, CNE, Zone] for snapshot-dependent zPTDF.
    total_zonal_generation : lp.Variable
        Linopy variable for zonal generation with dimensions [Zone, snapshot].
    zonal_loads : xr.DataArray
        DataArray with zonal loads with dimensions [Zone, snapshot].
    RAM : xr.DataArray
        DataArray with RAM values with dimensions [CNE, snapshot].
        
    Returns
    -------
    lp.Constraint
        Constraint ensuring flows on CNEs are within the RAM.
    """
    # Check if zPTDF is snapshot-dependent
    snapshot_dependent = "snapshot" in zPTDF.dims
    
    # Get zones that are in both zPTDF and total_zonal_generation
    zones = [zone for zone in zPTDF.coords['Zone'].values if zone in total_zonal_generation.indexes['Zone']]
    
    # Get the generation and load for these zones
    internal_zonal_gen = total_zonal_generation.sel(Zone=zones)
    internal_zonal_loads = zonal_loads.sel(Zone=zones)
    
    # Calculate net position (generation minus load)
    net_position = internal_zonal_gen - internal_zonal_loads
    
    # Handle snapshot-dependent and static zPTDFs differently
    if snapshot_dependent:
        # For snapshot-dependent zPTDF
        cne_lhs_list = []
        
        # Make sure snapshots are aligned
        snapshots = [snap for snap in zPTDF.coords['snapshot'].values if snap in RAM.coords['snapshot'].values]
        
        # Calculate the LHS for each CNE
        for cne in zPTDF.CNE.values:
            # For each snapshot and CNE, calculate the flow
            flow_terms = []
            for snap in snapshots:
                # Get zPTDF for this snapshot and CNE
                ptdf_slice = zPTDF.sel(snapshot=snap, CNE=cne)
                
                # Get net position for this snapshot
                net_pos_slice = net_position.sel(snapshot=snap)
                
                # Calculate flow term
                term = (ptdf_slice * net_pos_slice).sum(dim="Zone")
                flow_terms.append(term)
            
            # Combine flow terms for all snapshots for this CNE
            cne_term = lp.merge(flow_terms, dim="snapshot")
            cne_lhs_list.append(cne_term)
        
        # Combine all CNE terms
        cne_lhs = lp.merge(cne_lhs_list, dim="CNE")
        
        # Get corresponding RAM values
        ram_subset = RAM.sel(CNE=zPTDF.CNE.values, snapshot=snapshots)
        
        # Create the constraint
        cne_constraint = cne_lhs <= ram_subset
    else:
        # For static zPTDF (original implementation)
        cne_lhs_list = []
        for cne in zPTDF.CNE.values:
            term = (zPTDF.sel(CNE=cne) * net_position).sum(dim="Zone")
            cne_lhs_list.append(term)
        
        cne_lhs = lp.merge(cne_lhs_list, dim="CNE")
        cne_constraint = cne_lhs <= RAM
    
    return cne_constraint

def construct_zonal_balance_constraint(total_zonal_generation: lp.Variable, zonal_loads: xr.DataArray):
    """
    Get the zonal balance constraint.
    """
    assert total_zonal_generation.indexes["Zone"].equals(zonal_loads.indexes["Zone"]), "Zone mismatch in zonal_gen and zonal_loads!"

    total_gen = total_zonal_generation.sum(dim="Zone")
    total_loads = zonal_loads.sum(dim="Zone")

    return total_gen == total_loads

