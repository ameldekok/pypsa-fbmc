import pandas as pd

def add_gen_up_and_down_regulators(network):
    """
    Add up and down regulation variables to the network.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add regulation variables to.

    Returns
    -------
    pypsa.Network
        The network with added up and down regulation variables.
    """
    # Get existing generators
    generators = network.generators
    
    # Check for each generator if up and down regulators already exist
    for gen_name, gen in generators.iterrows():
        up_regulator_name = f"{gen_name}_up"
        down_regulator_name = f"{gen_name}_dn"
        
        # Skip if (one of the) regulators already exist
        if up_regulator_name in network.generators.index or down_regulator_name in network.generators.index:
            continue
        
        # Get generator parameters
        bus = gen.bus
        p_nom = gen.p_nom
        marginal_cost = gen.marginal_cost
        carrier = gen.carrier
        
        # Get current generation for all snapshots
        current_gen = network.generators_t.p[gen_name] if gen_name in network.generators_t.p else 0
        
        # Add up-regulator with maximum nominal capacity
        network.add("Generator", 
                   up_regulator_name, 
                   bus=bus,
                   p_nom=p_nom,  # Maximum possible capacity
                   marginal_cost=marginal_cost,
                   carrier=carrier)
        
        # Add down-regulator with maximum nominal capacity
        network.add("Generator", 
                   down_regulator_name, 
                   bus=bus,
                   p_nom=p_nom,  # Maximum possible capacity
                   marginal_cost=-marginal_cost,  # Negative cost for down-regulation
                   sign=-1,
                   carrier=carrier)

        # For time-dependent regulation, set p_max_pu for each snapshot
        if gen_name in network.generators_t.p:
            # Calculate per-unit availability for each snapshot
            # Up-regulator: what percentage of p_nom is available to increase
            up_pu = 1 - (current_gen / p_nom)
            # Down-regulator: what percentage of p_nom is available to decrease
            down_pu = current_gen / p_nom
            
            # Set time-dependent availability using p_max_pu
            if 'p_max_pu' not in network.generators_t:
                network.generators_t['p_max_pu'] = pd.DataFrame(index=network.snapshots)
            
            network.generators_t.p_max_pu[up_regulator_name] = up_pu
            network.generators_t.p_max_pu[down_regulator_name] = down_pu
            
            # Fix the original generator's output to prevent changes
            network.generators_t.p_set.loc[:,gen_name] = network.generators_t.p.loc[:,gen_name]
    
    return network

def fix_generation(nodal_network):
    nodal_network.generators_t.p_set = nodal_network.generators_t.p
    return nodal_network
    


def get_nodal_objective(nodal_network):
    UP_REGULATOR_SUFFIX = "_up"
    DOWN_REGULATOR_SUFFIX = "_dn"
    
    up_down_regulators = [gen_name for gen_name in nodal_network.generators.index if UP_REGULATOR_SUFFIX in gen_name or DOWN_REGULATOR_SUFFIX in gen_name]
    
    obj_function = nodal_network.model.variables["Generator-p"].sum("snapshot").loc[up_down_regulators].sum()

    return obj_function

def update_objective_function(nodal_network):
    model = nodal_network.optimize.create_model()

    capture_system_cost(model)

    model.objective = get_nodal_objective(nodal_network)
    return nodal_network

def capture_system_cost(model):
    model.add_variables(name = "System-Cost",)
    model.add_constraints(lhs = model.variables["System-Cost"],
                           rhs = model.objective.expression,
                           sign = "=")

def add_redispatch_cost_variable(nodal_network):
    """
    Add a variable to the nodal network to account for redispatch costs.

    Parameters
    ----------
    nodal_network : pypsa.Network
        The PyPSA network to add the redispatch cost variable to.

    Returns
    -------
    pypsa.Network
        The network with added redispatch cost variable.
    """
    # Create a new variable for redispatch costs
    nodal_network.model.add_variable("Redispatch_cost", "cost", "snapshot")
    
    # Set the objective function to minimize redispatch costs
    nodal_network.model.objective = nodal_network.model.variables["Redispatch_cost"].sum("snapshot")
    
    return nodal_network