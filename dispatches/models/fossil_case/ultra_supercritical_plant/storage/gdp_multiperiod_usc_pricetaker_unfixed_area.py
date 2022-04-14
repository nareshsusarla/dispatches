# import multiperiod object
import csv
import json
import copy
import os
import numpy as np

from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from idaes.apps.multiperiod.examples.simple_rankine_cycle import (
    create_model, set_inputs, initialize_model,
    close_flowsheet_loop, add_operating_cost)
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale

import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var)
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)

from dispatches.models.fossil_case.ultra_supercritical_plant.storage import (
    usc_storage_nlp_mp as usc_nlp)

new_design = False
if new_design:
    print('>>>>> Solving for new storage design')
    import usc_storage_gdp_mp_unfixed_area_new_storage_design as usc_gdp
else:
    print('>>>>> Solving for original storage design')
    import usc_storage_gdp_mp_unfixed_area as usc_gdp

# Imports for plots
from matplotlib import pyplot as plt
import matplotlib

lx = True
if lx:
    # scaling_obj = 1e-2 # 12 hrs
    if new_design:
        scaling_obj = 1e-2
    else:
        scaling_obj = 1e-1
    scaling_factor = 1
    scaling_cost = 1e-3
else:
    scaling_obj = 1
    scaling_factor = 1
print()
print('Scaling_factor:', scaling_factor)
print('Scaling cost:', scaling_cost)
print('Scaling obj:', scaling_obj)

# Add design data from .json file
if new_design:
    data_path = 'uscp_design_data_new_storage_design.json'
else:
    data_path = 'uscp_design_data.json'

with open(data_path) as design_data:
    design_data_dict = json.load(design_data)

    hxc_area = design_data_dict["hxc_area"] # in MW
    hxd_area = design_data_dict["hxd_area"] # in MW
    min_power = design_data_dict["plant_min_power"] # in MW
    max_power = design_data_dict["plant_max_power"] # in MW
    ramp_rate = design_data_dict["ramp_rate"]
    min_power_storage = design_data_dict["min_discharge_turbine_power"] # in MW
    # max_power_storage = design_data_dict["max_discharge_turbine_power"] # in MW
    max_salt_amount = design_data_dict["max_storage_salt_amount"] # in kg
    # hot_salt_temp = design_data_dict["hot_salt_temperature"] # in K
    cold_salt_temp = design_data_dict["cold_salt_temperature"] # in K
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
    min_area = design_data_dict["min_storage_area_design"] # in MW
    max_area = design_data_dict["max_storage_area_design"] # in MW
    min_salt_temp = design_data_dict["min_solar_salt_temperature"] # in K
    max_salt_temp = design_data_dict["max_solar_salt_temperature"] # in K

factor_mton = 1e-3 # factor for conversion kg to metric ton
tank_min = 1 * scaling_factor * factor_mton # in mton
tank_max = max_salt_amount * scaling_factor * factor_mton # in mton

# Add number of days and hours per week
hours_per_day = 12
number_days = 1
number_hours = hours_per_day * number_days
n_weeks = 1 # needed for plots
n_weeks_to_plot = 1

# Add options to model
deact_arcs_after_init = True # needed for GDP model
save_results = True # Saves results in a .csv file for each master iteration
method = "with_efficiency" # adds boiler and cycle efficiencies
tank_scenario = "hot_empty" # Initial state of salt tank:
                            # hot_empty, hot_full, hot_half_full

# Add lower and upper bounds for power and heat duty based on the
# multiperiod model to solve
min_power_storage = design_data_dict["min_discharge_turbine_power"] # in MW
min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
path_init_file = design_data_dict["init_file_path"]

# max_power_total = max_power + max_power_storage
# min_power_total = min_power + min_power_storage
max_power_total = 700 # random high value
min_power_total = min_power
load_init_file = False
load_from_file = path_init_file

# Select lmp source data and scaling factor according to that
use_rts_data = False
if use_rts_data:
    use_mod_rts_data = False
else:
    use_mod_rts_data = True

if use_rts_data:
    print('>> Using RTS lmp data')
    with open('rts_results_all_prices_base_case.npy', 'rb') as f:
        dispatch = np.load(f)
        price = np.load(f)
    lmp = price[0:number_hours].tolist()
elif use_mod_rts_data:
    # price = [22.9684, 21.1168, 0, 0, 200, 200]
    # price = [
    #     22.9684, 21.1168, 20.4, 20.419,
    #     0, 0, 0, 0,
    #     200, 200, 200, 200,
    # ]
    price = [
        22.9684, 21.1168, 20.4, 20.419,
        # 20.419, 21.2877, 23.07, 25,
        # 18.4634, 0, 0, 0,
        0, 0, 0, 0,
        # 19.0342, 23.07, 200, 200,
        200, 200, 200, 200,
    ]
    if len(price) < hours_per_day:
        print()
        print('**ERROR: I need more LMP data!')
        raise Exception
    lmp = price
else:
    print('>> Using NREL lmp data')
    price = np.load("nrel_scenario_average_hourly.npy")
# print(lmp)


def create_ss_model():

    m = pyo.ConcreteModel()
    m.usc = usc_gdp.main(method=method,
                         max_power=max_power,
                         load_init_file=load_init_file,
                         path_init_file=path_init_file,
                         deact_arcs_after_init=deact_arcs_after_init)

    # storage_work = m.usc.fs.discharge_turbine_work
    storage_work = m.usc.fs.discharge_mode_disjunct.es_turbine.work[0]
    charge_mode = m.usc.fs.charge_mode_disjunct
    discharge_mode = m.usc.fs.discharge_mode_disjunct

    m.usc.fs.hx_pump_work.unfix()
    m.usc.fs.discharge_turbine_work.unfix()

    if not deact_arcs_after_init:
        m.usc.fs.turbine[3].inlet.unfix()
        m.usc.fs.fwh[8].inlet_2.unfix()

    # Set bounds for plant power
    m.usc.fs.plant_min_power_eq = pyo.Constraint(
        expr=m.usc.fs.plant_power_out[0] >= min_power
    )
    m.usc.fs.plant_max_power_eq = pyo.Constraint(
        expr=m.usc.fs.plant_power_out[0] <= max_power
    )

    # Unfix data
    m.usc.fs.boiler.inlet.flow_mol[0].unfix()

    # Set lower bound in charge heat exchanger
    m.usc.fs.discharge_mode_disjunct.storage_lower_bound_eq = pyo.Constraint(
        expr=discharge_mode.hxd.heat_duty[0] * 1e-6 >= min_storage_heat_duty
    )
    m.usc.fs.charge_mode_disjunct.storage_lower_bound_eq = pyo.Constraint(
        expr=charge_mode.hxc.heat_duty[0] * 1e-6 >= min_storage_heat_duty
    )

    # Unfix storage system data
    m.usc.fs.charge_mode_disjunct.ess_charge_split.split_fraction[0, "to_hxc"].unfix()
    m.usc.fs.discharge_mode_disjunct.ess_discharge_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [charge_mode.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    for salt_hxd in [discharge_mode.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix()  # 1 DOF

    if not new_design:
        for unit in [charge_mode.cooler]:
            unit.inlet.unfix()
            m.usc.fs.charge_mode_disjunct.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers area and salt temperatures
    # charge_mode.hxc.area.fix(hxc_area)
    # charge_mode.hxc.outlet_2.temperature[0].fix(hot_salt_temp)
    # discharge_mode.hxd.area.fix(hxd_area)
    # discharge_mode.hxd.inlet_1.temperature[0].fix(hot_salt_temp)
    discharge_mode.hxd.outlet_1.temperature[0].fix(cold_salt_temp)

    return m


def create_mp_block():
    print('>>> Creating USC model and initialization for each time period')
    m = create_ss_model()
    b1 = m.usc

    # print('DOFs within mp create 1 =', degrees_of_freedom(m))

    # Add coupling variables
    b1.previous_power = Var(
        domain=NonNegativeReals,
        initialize=400,
        bounds=(min_power, max_power_total),
        doc="Previous period power (MW)"
        )

    max_inventory = 1e7 * scaling_factor * factor_mton # in mton

    b1.previous_salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, max_inventory),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    b1.previous_salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, max_inventory),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @b1.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            b1.previous_power - ramp_rate <=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            b1.previous_power + ramp_rate >=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            1e-3 * b1.salt_inventory_hot == (
                b1.previous_salt_inventory_hot
                + (3600 * b1.fs.salt_storage) * scaling_factor * factor_mton # in mton
            ) * 1e-3
        )

    @b1.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            1e-3 * b1.fs.salt_amount * scaling_factor == (
                b1.salt_inventory_hot
                + b1.salt_inventory_cold
            ) * 1e-3
        )

    # Scale variables and constraints
    iscale.set_scaling_factor(b1.fs.operating_cost, 1e-3)
    iscale.set_scaling_factor(b1.fs.plant_fixed_operating_cost, 1e-3)
    iscale.set_scaling_factor(b1.fs.plant_variable_operating_cost, 1e-3)
    iscale.set_scaling_factor(b1.fs.plant_capital_cost, 1e-3)

    iscale.set_scaling_factor(b1.fs.salt_amount, 1e-3)
    iscale.set_scaling_factor(b1.salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(b1.salt_inventory_cold, 1e-3)
    iscale.set_scaling_factor(b1.previous_salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(b1.previous_salt_inventory_cold, 1e-3)
    iscale.set_scaling_factor(b1.fs.constraint_salt_inventory_hot, 1e-3)

    iscale.set_scaling_factor(b1.fs.charge_mode_disjunct.capital_cost, 1e-3)
    iscale.set_scaling_factor(b1.fs.discharge_mode_disjunct.capital_cost, 1e-3)
    iscale.set_scaling_factor(b1.fs.storage_capital_cost, 1e-3)

    return m


# The tank level and power output are linked between time periods
def get_rankine_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [
        (b1.usc.salt_inventory_hot, b2.usc.previous_salt_inventory_hot),
        (b1.usc.fs.plant_power_out[0], b2.usc.previous_power)
    ]

# The final tank level and power output must be the same as the
# initial tank level and power output state
def get_rankine_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """

    return [(b1.usc.salt_inventory_hot, b2.usc.previous_salt_inventory_hot)]


# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.
n_time_points = 1 * number_hours  # hours in a week
mp_usc = MultiPeriodModel(
    n_time_points=n_time_points,
    process_model_func=create_mp_block,
    linking_variable_func=get_rankine_link_variable_pairs,
    periodic_variable_func=get_rankine_periodic_variable_pairs
)

# If you have no arguments, you don't actually need to pass in
# anything. NOTE: building the model will initialize each time block
mp_usc.build_multi_period_model()

# Retrieve pyomo model and active process blocks (i.e. time blocks)
m = mp_usc.pyomo_model
blks = mp_usc.get_active_process_blocks()

# Add lmp market data for each block
count = 0
for blk in blks:
    blk.revenue = pyo.Var(initialize=1e3,
                          bounds=(0, 1e6),
                          doc="Revenue in $/h")
    blk.revenue_eq = pyo.Constraint(
        expr=1e-3 * blk.revenue == lmp[count] * blk.usc.fs.net_power * scaling_cost * 1e-3)
    blk.operating_cost = pyo.Var(initialize=1e4,
                                 bounds=(0, 1e6),
                                 doc="Total operating cost in $/h")
    blk.operating_cost_eq = pyo.Constraint(
        expr=(
            1e-3 * blk.operating_cost == (
                (blk.usc.fs.operating_cost
                 + blk.usc.fs.plant_fixed_operating_cost
                 + blk.usc.fs.plant_variable_operating_cost) / (365 * 24)
            ) * scaling_cost * 1e-3
        )
    )
    blk.capital_cost = pyo.Var(initialize=1e4,
                               bounds=(0, 1e6),
                               doc="Storage capital cost in $/h")
    blk.capital_cost_eq = pyo.Constraint(
        expr=(
            1e-3 * blk.capital_cost == \
            (blk.usc.fs.storage_capital_cost / (365 * 24)) * scaling_cost * 1e-3
        )
    )
    blk.cost = pyo.Expression(expr=-(blk.revenue - blk.operating_cost - blk.capital_cost))
    count += 1

m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]) * scaling_obj)

# Initial state for linking variables: power and salt tank. Different
# scenarios are included for salt tank
blks[0].usc.previous_power.fix(400)

# Fix previous salt tank level based on salt scenario
if tank_scenario == "hot_empty":
    blks[0].usc.previous_salt_inventory_hot.fix(tank_min)
    blks[0].usc.previous_salt_inventory_cold.fix(tank_max-tank_min)
elif tank_scenario == "hot_half_full":
    blks[0].usc.previous_salt_inventory_hot.fix(tank_max/2)
    blks[0].usc.previous_salt_inventory_cold.fix(tank_max/2)
elif tank_scenario == "hot_full":
    blks[0].usc.previous_salt_inventory_hot.fix(tank_max-tank_min)
    blks[0].usc.previous_salt_inventory_cold.fix(tank_min)
else:
    print("Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full")



# Add constraint to save calculate charge and discharge area in a
# global variable
m.hours_set = RangeSet(0, number_hours - 1)
m.hours_set2 = RangeSet(0, number_hours - 2)

@m.Constraint(m.hours_set2)
def constraint_charge_previous_area(m, h):
    return m.blocks[h + 1].process.usc.fs.charge_area == m.blocks[h].process.usc.fs.charge_area

@m.Constraint(m.hours_set2)
def constraint_discharge_previous_area(m, h):
    return m.blocks[h + 1].process.usc.fs.discharge_area == m.blocks[h].process.usc.fs.discharge_area

@m.Constraint(m.hours_set)
def constraint_discharge_hot_salt_temperature(m, h):
    return m.blocks[h].process.usc.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0] == m.blocks[h].process.usc.fs.hot_salt_temp


discharge_min_salt = 379 # in mton, 8MW min es turbine
min_hot_salt = 2000
@m.Constraint(m.hours_set)
def _constraint_no_discharge_with_min_hot_tank(m, h):
    if h <= 2:
        b = min_hot_salt
    else:
        b = discharge_min_salt
    return (m.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var * b) <= blks[h].usc.previous_salt_inventory_hot


# Add a minimum number charge, discharge, and no storage operation modes
@m.Constraint(m.hours_set)
def _constraint_min_charge(m, h):
        return sum(m.blocks[h].process.usc.fs.charge_mode_disjunct.binary_indicator_var for h in m.hours_set) >= 1

@m.Constraint(m.hours_set)
def _constraint_min_discharge(m, h):
        return sum(m.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var for h in m.hours_set) >= 1
# @m.Constraint()
# def _logic_constraint2_min_discharge(m):
#     return sum(m.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var for h in m.hours_set) >= sum(m.blocks[h].process.usc.fs.charge_mode_disjunct.binary_indicator_var for h in m.hours_set) + 1

@m.Constraint(m.hours_set)
def _constraint_min_no_storage(m, h):
        return sum(m.blocks[h].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var for h in m.hours_set) >= 1


if tank_scenario == "hot_empty":
    # Add logical constraint to help reduce the alternatives to explore
    # when periodic behavior is expected
    @m.Constraint()
    def _logic_constraint_no_discharge_time0(m):
        return m.blocks[0].process.usc.fs.discharge_mode_disjunct.binary_indicator_var == 0
    @m.Constraint()
    def _logic_constraint_no_charge_at_timen(m):
        return (m.blocks[0].process.usc.fs.charge_mode_disjunct.binary_indicator_var
                + m.blocks[number_hours - 1].process.usc.fs.charge_mode_disjunct.binary_indicator_var) <= 1
    @m.Constraint()
    def _logic_constraint_no_storage_time0_no_charge_at_timen(m):
        return (m.blocks[0].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var
                + m.blocks[number_hours - 1].process.usc.fs.charge_mode_disjunct.binary_indicator_var) <= 1
elif tank_scenario == "hot_full":
    @m.Constraint()
    def _logic_constraint_no_discharge_at_timen(m):
        return (m.blocks[0].process.usc.fs.discharge_mode_disjunct.binary_indicator_var
                + m.blocks[number_hours - 1].process.usc.fs.discharge_mode_disjunct.binary_indicator_var) <= 1

# Declare arrays for plots
hours = np.arange(n_time_points*n_weeks_to_plot)
lmp_array = np.asarray(lmp[0:n_time_points])
color = ['r', 'b', 'tab:green', 'k', 'tab:orange']

# Create directory to save results
def _mkdir(dir):
    try:
        os.mkdir(dir)
        print('Directory {} created'.format(dir))
    except:
        print('Directory {} not created'.format(dir))
        pass

_mkdir('results')
_mkdir('results/gdp_mp_unfixed_area_{}h'.format(number_hours))


def print_model(mdl, mdl_data, csvfile):
    
    mdl.disjunction1_selection = {}
    hot_tank_level_iter = []
    cold_tank_level_iter = []

    print('       ___________________________________________')
    print('        Schedule')
    print('         Obj ($): {:.4f}'.format((value(mdl.obj) / scaling_cost) / scaling_obj))

    for blk in mdl.blocks:
        if mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
            print('         Period {}: Charge (HXC: {:.0f} MW, {:.0f} m2)'.format(
                blk,
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6,
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.area)))
        if mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
            print('         Period {}: Discharge (HXD: {:.0f} MW, {:.0f} m2)'.format(
                blk,
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6,
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.area)))
        if mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var.value == 1:
            print('         Period {}: No storage'.format(blk))

    print()
    for blk in mdl.blocks:
        print('       Time period {} '.format(blk+1))
        print('        Charge: {}'.format(
            mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.binary_indicator_var.value))
        print('        Discharge: {}'.format(
            mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.binary_indicator_var.value))
        print('        No storage: {}'.format(
            mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var.value))
        if mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[mdl_data.master_iteration] = 'Charge selected'
            print('         HXC area (m2): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.area)))
            print('         HXC Duty (MW): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6))
            print('         HXC salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.inlet_2.temperature[0]),
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.outlet_2.temperature[0])))
            print('         Salt flow HXC (kg/s): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.outlet_2.flow_mass[0])))
            print('         HXC steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].temperature),
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.side_1.properties_out[0].temperature)))
            print('         Steam flow HXC (mol/s): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.outlet_1.flow_mol[0])))
            if not new_design:
                print('         Cooling heat duty (MW): {:.4f}'.format(
                    value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.cooler.heat_duty[0]) * 1e-6))
        elif mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[mdl_data.master_iteration] = 'Discharge selected'
            print('         HXD area (m2): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.area)))
            print('         HXD Duty (MW): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6))
            print('         HXD salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0]),
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.outlet_1.temperature[0])))
            print('         Salt flow HXD (kg/s): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.outlet_1.flow_mass[0])))
            print('         HXD steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].temperature),
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.side_2.properties_out[0].temperature)))
            print('         Steam flow HXD (mol/s): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.outlet_2.flow_mol[0])))
            print('         ES turbine work (MW): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.es_turbine.work_mechanical[0]) * -1e-6))
        elif mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[mdl_data.master_iteration] = 'No_storage selected'
            print('         Salt flow (kg/s): {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.salt_storage)))
        else:
            print('        No other operation mode is available!')
        print('        Net power: {:.4f}'.format(value(mdl.blocks[blk].process.usc.fs.net_power)))
        print('        Discharge turbine work (MW): {:.4f}'.format(
            value(mdl.blocks[blk].process.usc.fs.discharge_turbine_work)))
        if not new_design:
            print('        Cooler heat duty: {:.4f}'.format(
                value(mdl.blocks[blk].process.usc.fs.cooler_heat_duty)))
        print('        Efficiencies (%): boiler: {:.4f}, cycle: {:.4f}'.format(
            value(mdl.blocks[blk].process.usc.fs.boiler_eff) *100,
            value(mdl.blocks[blk].process.usc.fs.cycle_efficiency) * 100))
        print('        Boiler heat duty: {:.4f}'.format(
            value(mdl.blocks[blk].process.usc.fs.boiler.heat_duty[0]) * 1e-6))
        print('        Boiler flow mol (mol/s): {:.4f}'.format(
            value(mdl.blocks[blk].process.usc.fs.boiler.outlet.flow_mol[0])))
        print('        Salt to storage (kg/s) [mton]: {:.4f} [{:.4f}]'.format(
            value(mdl.blocks[blk].process.usc.fs.salt_storage),
            value(mdl.blocks[blk].process.usc.fs.salt_storage) * 3600 * factor_mton))
        print('        Hot salt inventory (mton): {:.4f}, previous: {:.4f}'.format(
            value(mdl.blocks[blk].process.usc.salt_inventory_hot) / scaling_factor,
            value(mdl.blocks[blk].process.usc.previous_salt_inventory_hot) / scaling_factor))
        print('        Makeup water flow (mol/s): {:.4f}'.format(
            value(mdl.blocks[blk].process.usc.fs.condenser_mix.makeup.flow_mol[0])))
        print('        Total op cost ($/h): {:.4f}'.format((value(mdl.blocks[blk].process.operating_cost) / scaling_cost)))
        print('        Total cap cost ($/h): {:.4f}'.format(
            (value(mdl.blocks[blk].process.capital_cost) / scaling_cost)))
        print('        Revenue (M$/year): {:.4f}'.format((value(mdl.blocks[blk].process.revenue) / scaling_cost)))
        print()

        mdl.objective_value = {}
        mdl.boiler_heat_duty_value = {}
        mdl.discharge_turbine_work_value = {}
        mdl.hxc_area_value = {}
        mdl.hxd_area_value = {}
        mdl.hot_salt_temp_value = {}
        mdl.objective_value[mdl_data.master_iteration] = (value(mdl.obj) / scaling_cost) / scaling_obj
        mdl.iterations = mdl_data.master_iteration
        mdl.period = blk
        mdl.boiler_heat_duty_value[mdl_data.master_iteration] = value(mdl.blocks[blk].process.usc.fs.boiler.heat_duty[0]) * 1e-6
        mdl.discharge_turbine_work_value[mdl_data.master_iteration] = value(mdl.blocks[blk].process.usc.fs.discharge_turbine_work)
        mdl.hxc_area_value[mdl_data.master_iteration] = value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.area)
        mdl.hxd_area_value[mdl_data.master_iteration] = value(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.hxd.area)
        mdl.hot_salt_temp_value[mdl_data.master_iteration] = value(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.hxc.outlet_2.temperature[0])

        if save_results:
            writer = csv.writer(csvfile)
            writer.writerow(
                (mdl_data.master_iteration,
                 mdl.period,
                 mdl.disjunction1_selection[mdl_data.master_iteration],
                 mdl.boiler_heat_duty_value[mdl_data.master_iteration],
                 mdl.discharge_turbine_work_value[mdl_data.master_iteration],
                 mdl.hxc_area_value[mdl_data.master_iteration],
                 mdl.hxd_area_value[mdl_data.master_iteration],
                 mdl.hot_salt_temp_value[mdl_data.master_iteration],
                 mdl.objective_value[mdl_data.master_iteration])
            )
            csvfile.flush()

    print('        Obj (M$/year): {:.4f}'.format((value(mdl.obj) / scaling_cost) / scaling_obj))
    print('       ___________________________________________')

    hot_tank_level_iter.append(
        [(pyo.value(mdl.blocks[i].process.usc.salt_inventory_hot) / scaling_factor) # in mton
         for i in range(n_time_points)])
    cold_tank_level_iter.append(
        [(pyo.value(mdl.blocks[i].process.usc.salt_inventory_cold) / scaling_factor) # in mton
         for i in range(n_time_points)])

    # Plot results
    hot_tank_array = np.asarray(hot_tank_level_iter[0:n_weeks_to_plot]).flatten()
    cold_tank_array = np.asarray(cold_tank_level_iter[0:n_weeks_to_plot]).flatten()

    # Convert array to list to include hot tank level at time zero
    lmp_list = [0] + lmp_array.tolist()
    hot_tank_array0 = (
        value(mdl.blocks[0].process.usc.previous_salt_inventory_hot) / scaling_factor)
    cold_tank_array0 = (
        value(mdl.blocks[0].process.usc.previous_salt_inventory_cold) / scaling_factor)
    hours_list = hours.tolist() + [number_hours]
    hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
    cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()

    font = {'size':16}
    plt.rc('font', **font)
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Tank Level (metric ton)',
                   color=color[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.axhline((tank_max / scaling_factor),
                ls=':', lw=1.75,
                color=color[4])
    plt.text(number_hours / 2 - 1.5,
             (tank_max/scaling_factor) + 100, 'max salt',
             color=color[4])
    ax1.step(hours_list, hot_tank_list,
             marker='^', ms=4, label='Hot Salt',
             lw=1, color=color[0])
    ax1.step(hours_list, cold_tank_list,
             marker='v', ms=4, label='Cold Salt',
             lw=1, color=color[1])
    ax1.legend(loc="center right", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=2))

    ax2 = ax1.twinx()
    ax2.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax2.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', lw=1,
             color=color[2])
    ax2.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig(
        'results/gdp_mp_unfixed_area_{}h/salt_tank_level_master_iter{}.png'.
        format(number_hours, mdl_data.master_iteration))
    plt.close(fig1)


    fig2, ax3 = plt.subplots(figsize=(12, 8))

    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Salt Tank Level (metric ton)',
                   color=color[3])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.axhline((tank_max / scaling_factor),
                ls=':', lw=1.75,
                color=color[4])
    plt.text(number_hours / 2 - 1.5,
             (tank_max/scaling_factor) + 100, 'max salt',
             color=color[4])
    ax3.plot(hours_list, hot_tank_list,
             marker='^', ms=4, label='Hot Salt',
             lw=1, color=color[0])
    ax3.plot(hours_list, cold_tank_list,
             marker='v', ms=4, label='Cold Salt',
             lw=1, color=color[1])
    ax3.legend(loc="center right", frameon=False)
    ax3.tick_params(axis='y')
    ax3.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=2))

    ax4 = ax3.twinx()
    ax4.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax4.plot([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', lw=1,
             color=color[2])
    ax4.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig(
        'results/gdp_mp_unfixed_area_{}h/salt_tank_level_nostep_master_iter{}.png'.
        format(hours_per_day, mdl_data.master_iteration))
    plt.close(fig2)

    log_close_to_bounds(mdl)
    log_infeasible_constraints(mdl)


def create_csv_header():
    csvfile = open('results/gdp_mp_unfixed_area_{}h/results_subnlps_master_iterations.csv'.format(number_hours),
                   'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(
        ('Iteration', 'TimePeriod(hr)', 'OperationMode',
         'BoilerHeatDuty(MW)', 'DischargeWork(MW)', 'HXCArea',
         'HXDArea', 'SaltHotTemp', 'Obj($/hr)')
    )
    return csvfile


# Select solver
csvfile = create_csv_header()

opt = pyo.SolverFactory('gdpopt')
opt.CONFIG.strategy = 'RIC'
opt.CONFIG.OA_penalty_factor = 1e4
opt.CONFIG.max_slack = 1e4
# opt.CONFIG.call_after_subproblem_solve = print_model
opt.CONFIG.call_after_subproblem_solve = (lambda a, b: print_model(a, b, csvfile))
opt.CONFIG.mip_solver = 'gurobi_direct'
opt.CONFIG.nlp_solver = 'ipopt'
opt.CONFIG.tee = True
opt.CONFIG.init_strategy = "no_init"
opt.CONFIG.time_limit = "28000"

# Initializing disjuncts
print()
print()
print('>>Initializing disjuncts')
for k in range(number_hours):
    if k < (hours_per_day / 3):
        print('  **Setting block {} to no storage'.format(k))
        blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
        blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
        blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(1)
    elif k < (2 * (hours_per_day / 3)):
        print('  **Setting block {} to charge'.format(k))
        blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(1)
        blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
        blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
    elif k < hours_per_day:
        print('  **Setting block {} to discharge'.format(k))
        blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
        blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(1)
        blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)

# Solve problem
hot_tank_level = []
cold_tank_level = []
net_power = []
hxc_duty = []
hxd_duty = []
discharge_work = []
boiler_heat_duty = []
for week in range(n_weeks):
    print()
    print(">> Solving for week {}: {} hours of operation in {} day(s) ".
          format(week + 1, number_hours, number_days))
    results = opt.solve(m,
                        tee=True,
                        nlp_solver_args=dict(
                            tee=True,
                            symbolic_solver_labels=True,
                            options={
                                "linear_solver": "ma27",
                                "max_iter": 150,
                                "halt_on_ampl_error": "yes"
                            }))

    hot_tank_level.append(
        [(pyo.value(blks[i].usc.salt_inventory_hot) / scaling_factor) # in mton
         for i in range(n_time_points)])
    cold_tank_level.append(
        [(pyo.value(blks[i].usc.salt_inventory_cold) / scaling_factor) # in mton
         for i in range(n_time_points)])
    net_power.append(
        [pyo.value(blks[i].usc.fs.net_power)
         for i in range(n_time_points)])
    boiler_heat_duty.append([pyo.value(blks[i].usc.fs.boiler.heat_duty[0]) * 1e-6
                             for i in range(n_time_points)])
    discharge_work.append([pyo.value(blks[i].usc.fs.discharge_turbine_work)
                           for i in range(n_time_points)])
    hxc_duty.append(
        [pyo.value(blks[i].usc.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6
         for i in range(n_time_points)])
    hxd_duty.append(
        [pyo.value(blks[i].usc.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6
         for i in range(n_time_points)])

print(results)
csvfile.close()

# Print and plot results
c = 0
print('Objective: {:.4f}'.format((value(m.obj) / scaling_cost) / scaling_obj))
for blk in blks:
    print()
    print('Period {}'.format(c+1))
    storage_work = blks[c].usc.fs.discharge_turbine_work
    charge_mode = blks[c].usc.fs.charge_mode_disjunct
    discharge_mode = blks[c].usc.fs.discharge_mode_disjunct
    perc = 100
    factor = 1

    print(' Charge mode: {}'.format(
        blks[c].usc.fs.charge_mode_disjunct.binary_indicator_var.value))
    print(' Discharge mode: {}'.format(
        blks[c].usc.fs.discharge_mode_disjunct.binary_indicator_var.value))
    print(' No storage mode: {}'.format(
        blks[c].usc.fs.no_storage_mode_disjunct.binary_indicator_var.value))
    if blks[c].usc.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
        print('  HXC area (m2): {:.4f}'.format(
            value(charge_mode.hxc.area)))
        print('  HXC Duty (MW): {:.4f}'.format(
            value(charge_mode.hxc.heat_duty[0]) * 1e-6))
        print('  HXC salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
            value(charge_mode.hxc.inlet_2.temperature[0]),
            value(charge_mode.hxc.outlet_2.temperature[0])))
        print('  HXC steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
            value(charge_mode.hxc.side_1.properties_in[0].temperature),
            value(charge_mode.hxc.side_1.properties_out[0].temperature)))
        print('  HXC salt flow (kg/s) [mton/h]: {:.4f} [{:.4f}]'.format(
            value(charge_mode.hxc.outlet_2.flow_mass[0]),
            value(charge_mode.hxc.outlet_2.flow_mass[0]) * 3600 * factor_mton))
        print('  HXC steam flow (mol/s): {:.4f}'.format(
            value(charge_mode.hxc.outlet_1.flow_mol[0])))
        print('  HXC Delta T (K): in: {:.4f}, out: {:.4f}'.format(
            value(charge_mode.hxc.delta_temperature_in[0]),
            value(charge_mode.hxc.delta_temperature_out[0])))
    elif blks[c].usc.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
        print('  HXD area (m2): {:.4f}'.format(
            value(discharge_mode.hxd.area)))
        print('  HXD Duty (MW): {:.4f}'.format(
            value(discharge_mode.hxd.heat_duty[0]) * 1e-6))
        print('  HXD salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
            value(discharge_mode.hxd.inlet_1.temperature[0]),
            value(discharge_mode.hxd.outlet_1.temperature[0])))
        print('  HXD steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
            value(discharge_mode.hxd.side_2.properties_in[0].temperature),
            value(discharge_mode.hxd.side_2.properties_out[0].temperature)))
        print('  HXD salt flow (kg/s) [mton/h]: {:.4f} [{:.4f}]'.format(
            value(discharge_mode.hxd.outlet_1.flow_mass[0]),
            value(discharge_mode.hxd.outlet_1.flow_mass[0]) * 3600 * factor_mton))
        print('  HXD steam flow (mol/s): {:.4f}'.format(
            value(discharge_mode.hxd.outlet_2.flow_mol[0])))
        print('  HXD Delta T (K): in: {:.4f}, out: {:.4f}'.format(
            value(discharge_mode.hxd.delta_temperature_in[0]),
            value(discharge_mode.hxd.delta_temperature_out[0])))
        print('  ES turbine work (MW): {:.4f}'.format(
            value(discharge_mode.es_turbine.work_mechanical[0]) * -1e-6))
    elif blks[c].usc.fs.no_storage_mode_disjunct.binary_indicator_var.value == 1:
        print('  **Note: no storage heat exchangers exist, so the units have the init values ')
        print('  HXC area (m2): {:.4f}'.format(
            value(charge_mode.hxc.area)))
        print('  HXC Duty (MW): {:.4f}'.format(
            value(charge_mode.hxc.heat_duty[0]) * 1e-6))
        print('  HXC salt flow (kg/s): {:.4f} '.format(
            value(charge_mode.hxc.outlet_2.flow_mass[0])))
        print('  HXD area (m2): {:.4f}'.format(
            value(discharge_mode.hxd.area)))
        print('  HXD Duty (MW): {:.4f}'.format(
            value(discharge_mode.hxd.heat_duty[0]) * 1e-6))
        print('  HXD salt flow (kg/s): {:.4f}'.format(
            value(discharge_mode.hxd.outlet_1.flow_mass[0])))
    else:
        print('  No other operation modes!')

    print(' Net power: {:.4f}'.format(
        value(blks[c].usc.fs.net_power)))
    print(' Plant Power Out: {:.4f}'.format(
        value(blks[c].usc.fs.plant_power_out[0])))
    print(' Discharge turbine work (MW): {:.4f}'.format(
        value(storage_work) * factor))
    print(' Cost ($): {:.4f}'.format(value(blks[c].cost) / scaling_cost))
    print(' Revenue ($): {:.4f}'.format(value(blks[c].revenue) / scaling_cost))
    print(' Operating cost ($): {:.4f}'.format(value(blks[c].operating_cost) / scaling_cost))
    print(' Specific Operating cost ($/MWh): {:.4f}'.format(
        (value(blks[c].operating_cost) / scaling_cost) / value(blks[c].usc.fs.net_power)))
    print(' Efficiencies (%): boiler: {:.4f}, cycle: {:.4f}'.format(
        value(blks[c].usc.fs.boiler_eff) * 100,
        value(blks[c].usc.fs.cycle_efficiency) * perc))
    print(' Boiler heat duty: {:.4f}'.format(
        value(blks[c].usc.fs.boiler.heat_duty[0]) * 1e-6))
    print(' Boiler flow mol (mol/s): {:.4f}'.format(
        value(blks[c].usc.fs.boiler.outlet.flow_mol[0])))
    print(' Hot salt inventory (mton): previous: {:.4f}, current: {:.4f}'.format(
        value(blks[c].usc.previous_salt_inventory_hot) / scaling_factor,
        value(blks[c].usc.salt_inventory_hot) / scaling_factor))
    print(' Cold salt inventory (mton): previous: {:.4f}, current: {:.4f}'.format(
        value(blks[c].usc.previous_salt_inventory_cold) / scaling_factor,
        value(blks[c].usc.salt_inventory_cold) / scaling_factor))

    c += 1
print(results)

# Plot results
# hours = np.arange(n_time_points*n_weeks_to_plot)
# lmp_array = np.asarray(lmp[0:n_time_points])
hot_tank_array = np.asarray(hot_tank_level[0:n_weeks_to_plot]).flatten()
cold_tank_array = np.asarray(cold_tank_level[0:n_weeks_to_plot]).flatten()

# Convert array to list to include hot tank level at time zero
lmp_list = [0] + lmp_array.tolist()
hot_tank_array0 = (value(blks[0].usc.previous_salt_inventory_hot) / scaling_factor)
cold_tank_array0 = (value(blks[0].usc.previous_salt_inventory_cold) / scaling_factor)
hours_list = hours.tolist() + [number_hours]
hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()

font = {'size':16}
plt.rc('font', **font)
fig3, ax1 = plt.subplots(figsize=(12, 8))

color = ['r', 'b', 'tab:green', 'k', 'tab:orange']
ax1.set_xlabel('Time Period (hr)')
ax1.set_ylabel('Salt Tank Level (metric ton)',
               color=color[3])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(linestyle=':', which='both',
         color='gray', alpha=0.30)
plt.axhline((tank_max / scaling_factor), ls=':', lw=1.75,
            color=color[4])
plt.text(number_hours / 2 - 1.5, (tank_max / scaling_factor) + 100, 'max salt',
         color=color[4])
ax1.step(# [x + 1 for x in hours], hot_tank_array,
    hours_list, hot_tank_list,
    marker='^', ms=4, label='Hot Salt',
    lw=1, color=color[0])
ax1.step(# [x + 1 for x in hours], hot_tank_array,
    hours_list, cold_tank_list,
    marker='v', ms=4, label='Cold Salt',
    lw=1, color=color[1])
ax1.legend(loc="center right", frameon=False)
ax1.tick_params(axis='y')#,
                # labelcolor=color[3])
ax1.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=2))

ax2 = ax1.twinx()
ax2.set_ylabel('LMP ($/MWh)',
               color=color[2])
ax2.step([x + 1 for x in hours], lmp_array,
         marker='o', ms=3, alpha=0.5,
         ls='-', lw=1,
         color=color[2])
ax2.tick_params(axis='y',
                labelcolor=color[2])
plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_salt_tank_level.png'.format(number_hours))


font = {'size':18}
plt.rc('font', **font)

power_array = np.asarray(net_power[0:n_weeks_to_plot]).flatten()
# Convert array to list to include net power at time zero
power_array0 = value(blks[0].usc.previous_power)
power_list = [power_array0] + power_array.tolist()

fig4, ax3 = plt.subplots(figsize=(12, 8))
ax3.set_xlabel('Time Period (hr)')
ax3.set_ylabel('Net Power Output (MW)',
               color=color[1])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(linestyle=':', which='both',
         color='gray', alpha=0.30)
plt.text(number_hours / 2 - 3, max_power - 5.5, 'max plant power',
         color=color[4])
# plt.text(number_hours / 2 - 2.8, max_power_total + 1, 'max net power',
#          color=color[4])
plt.axhline(max_power, ls='-.', lw=1.75,
            color=color[4])
# plt.axhline(max_power_total, ls=':', lw=1.75,
#             color=color[4])
ax3.step(hours_list, power_list,
         marker='o', ms=4,
         lw=1, color=color[1])
ax3.tick_params(axis='y',
                labelcolor=color[1])
ax3.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=2))

ax4 = ax3.twinx()
ax4.set_ylabel('LMP ($/MWh)',
               color=color[2])
ax4.step([x + 1 for x in hours], lmp_array,
         marker='o', ms=3, alpha=0.5,
         ls='-', lw=1,
         color=color[2])
ax4.tick_params(axis='y',
                labelcolor=color[2])
plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_power.png'.format(number_hours))


zero_point = True
hxc_array = np.asarray(hxc_duty[0:n_weeks_to_plot]).flatten()
hxd_array = np.asarray(hxd_duty[0:n_weeks_to_plot]).flatten()
hxc_duty0 = 0 # zero since the plant is not operating
hxc_duty_list = [hxc_duty0] + hxc_array.tolist()
hxd_duty0 = 0 # zero since the plant is not operating
hxd_duty_list = [hxd_duty0] + hxd_array.tolist()

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.set_xlabel('Time Period (hr)')
ax5.set_ylabel('Storage Heat Duty (MW)',
               color=color[3])
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.grid(linestyle=':', which='both',
         color='gray', alpha=0.30)
# plt.text(number_hours / 2 - 2.2, max_storage_heat_duty + 1, 'max storage',
#          color=color[4])
plt.text(number_hours / 2 - 2, min_storage_heat_duty - 6.5, 'min storage',
         color=color[4])
# plt.axhline(max_storage_heat_duty, ls=':', lw=1.75,
#             color=color[4])
plt.axhline(min_storage_heat_duty, ls=':', lw=1.75,
            color=color[4])
if zero_point:
    ax5.step(hours_list, hxc_duty_list,
             marker='^', ms=4, label='Charge',
             color=color[0])
    ax5.step(hours_list, hxd_duty_list,
             marker='v', ms=4, label='Discharge',
             color=color[1])
else:
    ax5.step([x + 1 for x in hours], hxc_array,
             marker='^', ms=4, lw=1,
             label='Charge',
             color=color[0])
    ax5.step([x + 1 for x in hours], hxd_array,
             marker='v', ms=4, lw=1,
             label='Discharge',
             color=color[1])
ax5.legend(loc="center right", frameon=False)
ax5.tick_params(axis='y',
                labelcolor=color[3])
ax5.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=2))

ax6 = ax5.twinx()
ax6.set_ylabel('LMP ($/MWh)',
               color=color[2])
ax6.step([x + 1 for x in hours], lmp_array,
         marker='o', ms=3, alpha=0.5,
         ls='-', color=color[2])
ax6.tick_params(axis='y',
                labelcolor=color[2])
plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_hxduty.png'.format(number_hours))

zero_point = True
boiler_heat_duty_array = np.asarray(boiler_heat_duty[0:n_weeks_to_plot]).flatten()
boiler_heat_duty0 = 0 # zero since the plant is not operating
boiler_heat_duty_list = [boiler_heat_duty0] + boiler_heat_duty_array.tolist()
discharge_work_array = np.asarray(discharge_work[0:n_weeks_to_plot]).flatten()
discharge_work0 = 0 # zero since the plant is not operating
discharge_work_list = [discharge_work0] + discharge_work_array.tolist()

fig6, ax7 = plt.subplots(figsize=(12, 8))
ax7.set_xlabel('Time Period (hr)')
ax7.set_ylabel('(MW)',
               color=color[3])
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)
ax7.grid(linestyle=':', which='both',
         color='gray', alpha=0.30)
if zero_point:
    ax7.step(hours_list, boiler_heat_duty_list,
             marker='^', ms=4, label='Boiler Heat Duty',
             color=color[0])
    ax7.step(hours_list, discharge_work_list,
             marker='^', ms=4, label='Discharge Work',
             color=color[1])
else:
    ax7.step([x + 1 for x in hours], boiler_heat_duty_array,
             marker='^', ms=4, lw=1,
             label='Boiler Heat Duty',
             color=color[0])
    ax7.step([x + 1 for x in hours], discharge_work_list,
             marker='^', ms=4, label='Discharge Work',
             color=color[1])
ax7.legend(loc="center right", frameon=False)
ax7.tick_params(axis='y',
                labelcolor=color[3])
ax7.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=2))

ax8 = ax7.twinx()
ax8.set_ylabel('LMP ($/MWh)',
               color=color[2])
ax8.step([x + 1 for x in hours], lmp_array,
         marker='o', ms=3, alpha=0.5,
         ls='-', color=color[2])
ax8.tick_params(axis='y',
                labelcolor=color[2])
plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_boilerduty.png'.format(number_hours))


plt.show()
