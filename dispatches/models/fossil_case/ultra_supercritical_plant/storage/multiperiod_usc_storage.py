
# import multiperiod object and rankine example
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
# from idaes.apps.multiperiod.examples.simple_rankine_cycle import (
#     create_model, set_inputs, initialize_model,
#     close_flowsheet_loop, add_operating_cost)

import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var)
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
import numpy as np
import copy
# from random import random
from idaes.core.util.model_statistics import degrees_of_freedom

from dispatches.models.fossil_case.ultra_supercritical_plant.storage import (
    usc_storage_nlp_mp as usc)

# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)

# method = "with_efficiency"  # options: with_efficiency and without_efficiency
# max_power_total = max_power + max_power_storage
# min_power_total = min_power + min_power_storage

# # Add number of days and hours per week
# number_days = 1
# number_hours = 24 * number_days

lx = False
if lx:
    scaling_obj = 1
    scaling_factor = 1e-3
else:
    scaling_obj = 1
    scaling_factor = 1


def create_ss_usc_model(pmin, pmax):

    # Set bounds for plant power
    min_storage_heat_duty = 10  # in MW
    max_storage_heat_duty = 150  # in MW

    max_power = 436  # in MW
    min_power = int(0.65 * max_power)  # 283 in MW
    max_power_storage = 24  # in MW
    min_power_storage = 1  # in MW

    # Load from the json file for faster initialization
    load_from_file = 'initialized_usc_storage_mlp_mp.json'

    m = pyo.ConcreteModel()
    m.rankine = usc.main(method="with_efficiency",
                         max_power=max_power,
                         load_from_file=load_from_file)

    m.rankine.fs.plant_min_power_eq = pyo.Constraint(
        expr=m.rankine.fs.plant_power_out[0] >= min_power
    )
    m.rankine.fs.plant_max_power_eq = pyo.Constraint(
        expr=m.rankine.fs.plant_power_out[0] <= max_power
    )

    # Set bounds for discharge turbine
    m.rankine.fs.es_turbine_min_power_eq = pyo.Constraint(
        expr=m.rankine.fs.es_turbine.work[0] * (-1e-6) >= min_power_storage
    )
    m.rankine.fs.es_turbine_max_power_eq = pyo.Constraint(
        expr=m.rankine.fs.es_turbine.work[0] * (-1e-6) <= max_power_storage
    )

    m.rankine.fs.hxc.heat_duty.setlb(min_storage_heat_duty * 1e6)
    m.rankine.fs.hxd.heat_duty.setlb(min_storage_heat_duty * 1e6)

    # Unfix data
    m.rankine.fs.boiler.inlet.flow_mol[0].unfix()

    # Unfix storage system data
    m.rankine.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.rankine.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.rankine.fs.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    for salt_hxd in [m.rankine.fs.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix()  # 1 DOF

    for unit in [m.rankine.fs.cooler]:
        unit.inlet.unfix()
    m.rankine.fs.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers area and salt temperatures
    # m.rankine.fs.hxc.inlet_1.flow_mol.fix(2000)
    m.rankine.fs.hxc.area.fix(1904)
    m.rankine.fs.hxd.area.fix(1095)
    m.rankine.fs.hxc.outlet_2.temperature[0].fix(831)
    m.rankine.fs.hxd.inlet_1.temperature[0].fix(831)
    m.rankine.fs.hxd.outlet_1.temperature[0].fix(513.15)

    return m


def create_mp_usc_block(pmin=None, pmax=None):
    print('>>> Creating USC model and initialization for each time period')

    max_power_total = 436 + 24
    min_power_total = int(0.65 * 436) + 1

    m = create_ss_usc_model(pmin, pmax)
    b1 = m.rankine

    # print('DOFs within mp create 1 =', degrees_of_freedom(m))

    # Add coupling variables
    b1.previous_power = Var(
        domain=NonNegativeReals,
        initialize=400,
        bounds=(min_power_total, max_power_total),
        doc="Previous period power (MW)"
        )

    inventory_max = 1e7 * scaling_factor
    b1.previous_salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, inventory_max),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, inventory_max),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    b1.previous_salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, inventory_max),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, inventory_max),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @b1.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            b1.previous_power - 60 <=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            b1.previous_power + 60 >=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            b1.salt_inventory_hot ==
            b1.previous_salt_inventory_hot
            + (3600*b1.fs.hxc.inlet_2.flow_mass[0]
               - 3600*b1.fs.hxd.inlet_1.flow_mass[0]) * scaling_factor
        )

    @b1.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            b1.salt_inventory_hot +
            b1.salt_inventory_cold == b1.fs.salt_amount * scaling_factor)

    return m


# The tank level and power output are linked between time periods
def get_usc_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.rankine.salt_inventory_hot,
             b2.rankine.previous_salt_inventory_hot),
            (b1.rankine.fs.plant_power_out[0],
             b2.rankine.previous_power)]


# The final tank level and power output must be the same as the initial
# tank level and power output state
def get_usc_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [(b1.rankine.salt_inventory_hot,
             b2.rankine.previous_salt_inventory_hot)]

# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.


def create_multiperiod_usc_model(n_time_points=4, pmin=None, pmax=None):
    """
    Create a multi-period rankine cycle object. This object contains a pyomo
    model with a block for each time instance.

    n_time_points: Number of time blocks to create
    """
    mp_rankine = MultiPeriodModel(
        n_time_points,
        lambda: create_mp_usc_block(pmin=None, pmax=None),
        get_usc_link_variable_pairs,
        # process_model_func=create_mp_usc_block(pmin=None, pmax=None),
        # linking_variable_func=get_usc_link_variable_pairs,
        # periodic_variable_func=get_rankine_periodic_variable_pairs
        )

    # OPTIONAL KEYWORD ARGUMENTS
    # time_points = np.arange(0,n_time_points)
    # data_points = [{} for i in range(n_time_points)]
    # data_kwargs = dict(zip(time_points,data_points))
    # mp_rankine.build_multi_period_model(data_kwargs);

    # If you have no arguments, you don't actually need to pass in
    # anything. NOTE: building the model will initialize each time block
    mp_rankine.build_multi_period_model()
    return mp_rankine
