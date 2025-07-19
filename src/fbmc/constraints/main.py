"""
Add the FBMC constraints to the network
"""

import pypsa
from typing import Union, Dict
import pandas as pd

from .redispatch import add_gen_up_and_down_regulators, update_objective_function
from .fbmc_constraints import construct_cne_constraint, construct_zonal_balance_constraint, convert_RAM_to_xarray, convert_zPTDF_to_xarray, create_load_zone_mapping, create_load_zone_mask, get_zonal_loads
from .zonal_generation import construct_zonal_generation_constraint, create_generator_zone_mapping, create_signed_generator_mask, add_zonal_generation_variable

def create_zonal_generation(network: pypsa.Network):
    """
    Main function to add zonal generation variables and constraints to the network.
    """
    # Add the zonal generation variable to the model
    zones = network.buses.index.to_list()
    snapshots = network.snapshots.to_list()
    zonal_generation_var = add_zonal_generation_variable(network, zones, snapshots)

    # Create mask for generation, according to zone and sign.
    generator_zone_mapping = create_generator_zone_mapping(network.generators)
    generator_sign = network.generators.sign.to_xarray()
    signed_mask = create_signed_generator_mask(generator_zone_mapping, zones, generator_sign)

    # Add the zonal generation constraint
    zonal_generation_constraint = construct_zonal_generation_constraint(
        total_zonal_generation = zonal_generation_var, 
        generators = network.model.variables["Generator-p"],
        signed_mask = signed_mask)
    network.model.add_constraints(zonal_generation_constraint, name="Zone-p_definition")

    return network

def add_fbmc_constraints(network: pypsa.Network, 
                         zPTDF_df: Union[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]],
                         RAM_df: pd.DataFrame
                         ) -> pypsa.Network:
    """
    Main function to add FBMC constraints to the network.
    
    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add constraints to.
    zPTDF_df : pd.DataFrame or dict of pd.DataFrame
        Either a single DataFrame containing zPTDF values (static GSKs),
        or a dictionary of DataFrames with snapshots as keys (snapshot-based GSKs).
    RAM_df : pd.DataFrame
        DataFrame containing RAM values.
    
    Returns
    -------
    pypsa.Network
        The network with added FBMC constraints.
    """
    # xarray conversion
    zPTDF_xr = convert_zPTDF_to_xarray(zPTDF_df)
    RAM_xr = convert_RAM_to_xarray(RAM_df)

    # Retrieve the zonal generation variable
    zonal_generation = network.model.variables["Zone-p"]

    # zonal loads
    load_zone_mapping = create_load_zone_mapping(network.loads)
    zones = network.buses.index.to_list()
    load_zone_mask = create_load_zone_mask(load_zone_mapping, zones)
    zonal_loads = get_zonal_loads(load_zone_mask, network.get_switchable_as_dense("Load", "p_set"))

    # Restrict the load on CNEs by the Remaining Available Margin (RAM)
    cne_constraint = construct_cne_constraint(zPTDF_xr, zonal_generation, zonal_loads, RAM_xr)
    network.model.add_constraints(cne_constraint, name="CNE-RAM")

    # Ensure the Net Position of all zones adds up to 0
    zonal_balance_constraint = construct_zonal_balance_constraint(zonal_generation, zonal_loads)
    network.model.add_constraints(zonal_balance_constraint, name="Zonal_balance")

    return network


def remove_original_constraints(network):
    """"
    Remove the original constraints introduced by pyPSA from the network model.
    
    Parameters
    ----------
    network : pypsa.Network
        The zonal PyPSA network from which to remove constraints.

    """

    network.model.remove_variables("Link-p")
    network.model.remove_constraints("Link-fix-p-lower")
    network.model.remove_constraints("Link-fix-p-upper")
    network.model.remove_constraints("Bus-nodal_balance")
    
    # In PowerVision, remove the bus-meshed-nodal_balance constraint as well.
    if "Bus-meshed-nodal_balance" in network.model.constraints:
        network.model.remove_constraints("Bus-meshed-nodal_balance")

def add_redispatch_constraints(rd_nodal_network):
    """
    Add the redispatch constraints to the network.
    
    Parameters
    ----------
    rd_nodal_network : pypsa.Network
        The PyPSA network after Market Clearing, to add constraints to.
    
    Returns
    -------
    pypsa.Network
        The network with added redispatch constraints.
    """

    rd_nodal_network.optimize.create_model()

    # Add up- and down- regulators.
    add_gen_up_and_down_regulators(rd_nodal_network)

    # Add redispatch constraints
    rd_nodal_network = update_objective_function(rd_nodal_network)

    return rd_nodal_network
