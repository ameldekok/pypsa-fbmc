import pandas as pd
import pypsa
import xarray as xr
import linopy as lp

def create_generator_zone_mapping(generators: pd.DataFrame) -> xr.DataArray:
    """
    Create an xarray mapping of generators to their respective zones.
    
    Args:
        generators (pd.DataFrame): DataFrame containing generator data with 'bus' column
            indicating the zone (assuming zonal network)
    
    Returns:
        xarray.DataArray: A 1D array with dimension "Generator" mapping generators to zones.
    """
    generator_zone_mapping = xr.DataArray(
        [generators.at[gen, 'bus'] for gen in generators.index],
        dims=["Generator"],
        coords={"Generator": generators.index}
    )
    return generator_zone_mapping


def create_signed_generator_mask(generator_zone_mapping: xr.DataArray, 
                               zones: list, 
                               gen_sign: xr.DataArray) -> xr.DataArray:
    """
    Create a signed mask for generators based on their zones and signs.
    
    Args:
        generator_zone_mapping (xarray.DataArray): Mapping of generators to zones.
        zones (list): List of zone names.
        gen_sign (xarray.DataArray): Array containing the sign of each generator.
    
    Returns:
        xarray.DataArray: A 2D array with dimensions ("Generator", "Zone") containing
            signed binary values indicating generator-zone assignments.
    """
    zone_da = xr.DataArray(zones, dims=["Zone"], coords={"Zone": zones})
    signed_mask = (generator_zone_mapping == zone_da) * gen_sign
    return signed_mask

def add_zonal_generation_variable(network: pypsa.Network, zones: list, snapshots: list) -> lp.Variable: 
    """
    Add zonal generation variables for a given PyPSA network.
    
    Args:
        network (pypsa.Network): The PyPSA network object containing the model.
        zones (list): List of zone names.
        snapshots (list): List of timestamps for the analysis period.
    
    Returns:
        linopy.Variable: A variable with dimensions ("snapshot", "Zone") representing 
            the zonal generation for each snapshot and zone.
    """
    
    zonal_generation = network.model.add_variables(
        name="Zone-p",
        coords={"snapshot": snapshots, "Zone": zones},  # Changed order
        dims=["snapshot", "Zone"],  # Changed order and capitalization
        )
    
    return zonal_generation

def construct_zonal_generation_constraint(total_zonal_generation: lp.Variable, generators: lp.Variable, signed_mask: xr.DataArray) -> lp.Constraint:
    """
    Create constraints to define zonal generation as the sum of signed generator outputs.
    """

    zonal_generation_constraint = total_zonal_generation == (generators * signed_mask).sum(dim="Generator")
    return zonal_generation_constraint

