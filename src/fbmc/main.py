# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:01:07 2025

@author: ameldekok
"""

from matplotlib.pylab import solve
import pypsa

from . import network_conversion
from .constraints.main import add_redispatch_constraints
from .parameters import calculate_fbmc_parameters
from .constraints import create_zonal_generation, add_fbmc_constraints, remove_original_constraints
from .config import FBMCConfig

def setup_fbmc_model(basecase_nodal_network: pypsa.Network, target_zonal_network: pypsa.Network, config: FBMCConfig = FBMCConfig()) -> pypsa.Network:
    """
    Set up the FBMC model by calculating parameters and adding constraints.
    
    Parameters
    ----------
    basecase_nodal_network : pypsa.Network
        The base case nodal network to be used for FBMC.
    target_zonal_network : pypsa.Network
        The target zonal network to be used for FBMC.
    config : FBMCConfig
        Configuration object for FBMC parameters.
    
    Returns
    -------
    pypsa.Network
        The target zonal network with added FBMC constraints.
    """
    # Calculate parameters
    ram_cnes, z_ptdf_cnes = calculate_fbmc_parameters(basecase_nodal_network, config=config)
    
    # Add constraints
    target_zonal_network = create_zonal_generation(target_zonal_network)
    target_zonal_network = add_fbmc_constraints(target_zonal_network, z_ptdf_cnes, ram_cnes)
    remove_original_constraints(target_zonal_network)

    return target_zonal_network

def run_redispatch(nodal_network: pypsa.Network, zonal_network: pypsa.Network) -> pypsa.Network:
    """
    Run the redispatch process on the given network.

    Parameters
    ----------
    nodal_network : pypsa.Network
        The nodal network to be used for redispatch.
    zonal_network : pypsa.Network
        The zonal network after market clearing to be used for redispatch.

    Returns
    -------
    pypsa.Network
        The updated nodal network after redispatch.
    """
    
    # Add up and down regulation variables to the model
    rd_nodal_network = network_conversion.zonal_to_nodal(zonal_network, nodal_network)

    add_redispatch_constraints(rd_nodal_network)

    # Run the optimization
    rd_nodal_network.model.solve(solver_name="gurobi")

    
    return rd_nodal_network

def run_fbmc(nodal_network: pypsa.Network, zonal_network: pypsa.Network, config: FBMCConfig = FBMCConfig()) -> pypsa.Network:
    """
    Run the FBMC process on the given networks.

    Parameters
    ----------
    nodal_network : pypsa.Network
        The nodal network to be used for FBMC.
    zonal_network : pypsa.Network
        The zonal network to be used for FBMC.
    config : FBMCConfig
        Configuration object for FBMC parameters.

    Returns
    -------
    pypsa.Network
        The updated zonal network after FBMC.
    """
    
    # Set up the model
    zonal_network = setup_fbmc_model(nodal_network, zonal_network, config=config)

    # Run the optimization and save the results to the nodal network
    zonal_network.model.solve(solver_name="gurobi")
    nodal_network = network_conversion.zonal_to_nodal(zonal_network, nodal_network)

    # Add the redispatch constraints to the nodal network and run the optimization
    nodal_network = add_redispatch_constraints(nodal_network)
    nodal_network.model.solve(solver_name="gurobi")

    return nodal_network

if __name__ == '__main__':
    n_rd = pypsa.Network()
    z_da = pypsa.Network()
    # Get the networks from files:
    folder = "C:/Users/ameld/thesis_local/PowerVision/output_networks/FBMC"
    n_rd.import_from_netcdf(f"{folder}/nodal.nc")
    z_da.import_from_netcdf(f"{folder}/zonal.nc")
    # Calculate the FBMC parameters:
    fbmc_parameters = calculate_fbmc_parameters(n_rd)
    print(fbmc_parameters)