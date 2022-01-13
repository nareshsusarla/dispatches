
# import multiperiod object and rankine example
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from idaes.apps.multiperiod.examples.simple_rankine_cycle import (
    create_model, set_inputs, initialize_model,
    close_flowsheet_loop, add_operating_cost)

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

method = "with_efficiency" # options: with_efficiency and without_efficiency
max_power = 450 # in MW

def create_ss_rankine_model():
    # power_max = 436 # in MW
    min_power = int(0.65 * max_power) # 283 in MW
    # boiler_heat_max = 918e6  # in W
    # boiler_heat_min = 586e6  # in W
    max_power_storage = 100 # in MW
    min_power_storage = 1 # in MW

    m = pyo.ConcreteModel()
    m.rankine = usc.main(method=method, max_power=max_power)

    # Set bounds for plant power
    # m.rankine.fs.plant_power_out[0].setlb(None)
    # m.rankine.fs.plant_power_out[0].setub(None)
    # m.rankine.fs.plant_min_power_eq = pyo.Constraint(
    #     expr=m.rankine.fs.plant_power_out[0] >= min_power
    # )
    # m.rankine.fs.plant_max_power_eq = pyo.Constraint(
    #     expr=m.rankine.fs.plant_power_out[0] <= max_power
    # )
    # m.rankine.fs.plant_min_power_eq = pyo.Constraint(
    #     expr=m.rankine.fs.net_power >= min_power
    # )
    # m.rankine.fs.plant_max_power_eq = pyo.Constraint(
    #     expr=m.rankine.fs.net_power <= max_power
    # )

    # Set bounds for discharge turbine
    # m.rankine.fs.es_turbine_min_power_eq = pyo.Constraint(
    #     expr=m.rankine.fs.es_turbine.work[0] * (-1e-6) >= min_power_storage
    # )
    # m.rankine.fs.es_turbine_max_power_eq = pyo.Constraint(
    #     expr=m.rankine.fs.es_turbine.work[0] * (-1e-6) <= max_power_storage
    # )

    # No bounds on boiler heat duty
    # m.rankine.fs.boiler.heat_duty[0].setlb(boiler_heat_min)
    # m.rankine.fs.boiler.heat_duty[0].setub(boiler_heat_max)

    # Unfix data
    m.rankine.fs.boiler.inlet.flow_mol[0].unfix()  # normally fixed
    m.rankine.fs.boiler.inlet.flow_mol[0].setlb(1)
    m.rankine.fs.boiler.inlet.flow_mol[0].setub(None)
    m.rankine.fs.boiler.outlet.flow_mol[0].setlb(1)
    m.rankine.fs.boiler.outlet.flow_mol[0].setub(None)
    m.rankine.fs.plant_power_out[0].setlb(min_power)
    m.rankine.fs.plant_power_out[0].setub(max_power)
    m.rankine.fs.es_turbine.work[0].setlb(max_power_storage * (-1e-6))
    m.rankine.fs.es_turbine.work[0].setub(min_power_storage * (-1e-6))

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
    # m.rankine.fs.salt_hot_temperature = 831
    m.rankine.fs.hxc.area.fix(1904)  # 1904
    m.rankine.fs.hxd.area.fix(1095)  # 1095
    m.rankine.fs.hxc.outlet_2.temperature[0].fix(831)
    m.rankine.fs.hxd.inlet_1.temperature[0].fix(831)
    m.rankine.fs.hxd.outlet_1.temperature[0].fix(513.15)

    return m


# with open('rts_results_all_prices.npy', 'rb') as f:
#     dispatch = np.load(f)
#     price = np.load(f)

# plt.figure(figsize=(12, 8))
# prices_used = copy.copy(price)
# prices_used[prices_used > 200] = 200
# x = list(range(0, len(prices_used)))
# plt.bar(x, (prices_used))
# plt.xlabel("Hour")
# plt.ylabel("LMP $/MWh")

# weekly_prices = prices_used.reshape(52, 168)
# plt.figure(figsize=(12, 8))
# for week in [0, 15, 25, 35, 45, 51]:
#     plt.plot(weekly_prices[week])
# plt.title("6 Representative Weeks")
# plt.xlabel("Hour")
# plt.ylabel("LMP $/MWh")

# plt.figure(figsize=(12, 8))
# for week in range(0, 52):
#     plt.plot(weekly_prices[week], color="blue", alpha=0.1)
# plt.title("52 Representative Weeks")
# plt.xlabel("Hour")
# plt.ylabel("LMP $/MWh")

# turbine_ramp_rate = 100
# battery_ramp_rate = 50
def create_mp_rankine_block():
    m = create_ss_rankine_model()
    b1 = m.rankine
    #  DOF = 1
    print('DOFs within mp create 1 =', degrees_of_freedom(m))
    # Add coupling variable (next_power_output)
    b1.previous_power = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=400,
        bounds=(100, 450),
        # bounds=(0, 6739292),
        doc="Previous period power (MW)"
        )

    b1.previous_salt_inventory_hot = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1e7),
        # bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_hot = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, 1e7),
        # bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    b1.previous_salt_inventory_cold = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1e7),
        # bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_cold = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, 1e7),
        # bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @b1.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            b1.previous_power - 50 <=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            b1.previous_power + 50 >=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            b1.salt_inventory_hot ==
            b1.previous_salt_inventory_hot
            + 3600*b1.fs.hxc.inlet_2.flow_mass[0]
            - 3600*b1.fs.hxd.inlet_1.flow_mass[0])

    @b1.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            b1.salt_inventory_hot +
            b1.salt_inventory_cold == b1.fs.salt_amount)
    print('DOFs after mp create =', degrees_of_freedom(m))

    # @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    # def constraint_salt_inventory_cold(b):
    #     return (
    #         b1.salt_inventory_cold ==
    #         b1.previous_salt_inventory_cold
    #         - 3600*b1.fs.hxc.inlet_2.flow_mass[0]
    #         + 3600*b1.fs.hxd.inlet_1.flow_mass[0])


    # @b1.fs.Constraint(doc="Maximum previous salt inventory at any time")
    # def constraint_salt_previous_inventory(b):
    #     return (
    #         b1.previous_salt_inventory_hot +
    #         b1.previous_salt_inventory_cold == b1.fs.salt_amount)
    # print('DOFs after mp create =', degrees_of_freedom(m))
    # raise Exception()
    return m

# the power output and battery state are linked between time periods


def get_rankine_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.rankine.salt_inventory_hot,
             b2.rankine.previous_salt_inventory_hot),
            (b1.rankine.fs.plant_power_out[0],
             b2.rankine.previous_power)]

# the final power output and battery state must be the same
# as the intial power output and battery state


def get_rankine_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [(b1.rankine.salt_inventory_hot,
             b2.rankine.previous_salt_inventory_hot)]#,
            # (b1.rankine.fs.plant_power_out[0],
            #  b2.rankine.previous_power)]


number_hours = 8
n_time_points = 1*number_hours  # hours in a week

# create the multiperiod model object
mp_rankine = MultiPeriodModel(
    n_time_points=n_time_points,
    process_model_func=create_mp_rankine_block,
    linking_variable_func=get_rankine_link_variable_pairs,
    periodic_variable_func=get_rankine_periodic_variable_pairs
    )

# you can pass arguments to your `process_model_func`
# for each time period using a dict of dicts as shown here.
# In this case, it is setting up empty dictionaries for each time period.

# OPTIONAL KEYWORD ARGUMENTS
# time_points = np.arange(0,n_time_points)
# data_points = [{} for i in range(n_time_points)]
# data_kwargs = dict(zip(time_points,data_points))
# mp_rankine.build_multi_period_model(data_kwargs);

# if you have no arguments, you don't actually need to pass in anything.
mp_rankine.build_multi_period_model()
# NOTE: building the model will initialize each time block

# retrieve pyomo model and active process blocks (i.e. time blocks)
m = mp_rankine.pyomo_model
blks = mp_rankine.get_active_process_blocks()

# power = [310, 325, 420, 400]
# lmp = [15, 19, 21, 25]
lmp = [
    15, 19, 21, 25,
    12, 16, 22, 20]#,
#     15, 19, 21, 25,
#     12, 16, 22, 20,
#     15, 19, 21, 25,
#     12, 16, 22, 20
# ]
# lmp = [22.4929, 21.8439, 23.4379, 23.4379, 23.4379, 21.6473, 21.6473]

count = 0
# m.hot_level = Param(default=1, mutable=False)
# m.cold_level = Param(default=6739292 - m.hotlevel, mutable=False)
# add market data for each block
for blk in blks:
    blk_rankine = blk.rankine
    blk.lmp_signal = Param(default=0, mutable=True)
    blk.revenue = lmp[count]*blk.rankine.fs.net_power
    # blk.revenue = blk.lmp_signal*blk_rankine.fs.plant_power_out[0]
    blk.operating_cost = pyo.Expression(
        expr=(
            (blk_rankine.fs.operating_cost
             + blk_rankine.fs.plant_fixed_operating_cost
             + blk_rankine.fs.plant_variable_operating_cost) / (365 * 24)
        )
    )
    blk.cost = pyo.Expression(expr=-(blk.revenue - blk.operating_cost))
    # blk.fix_power = pyo.Constraint(
    #     expr=power[count] == blk.rankine.fs.net_power
    # )
    # blk.fix_power = pyo.Constraint(
    #     expr=blk.dispatch == (
    #         blk.rankine.fs.plant_power_out[0]
    #         + (-1e-6) * blk.rankine.fs.es_turbine.work_mechanical[0]
    #     )
    # )
    count += 1

scaling_obj = 1e-3 # for 8 hours analysis
# scaling_obj = 1e-4 # for 24 hours analysis
m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]) * scaling_obj)
blks[0].rankine.previous_salt_inventory_hot.fix(1)
blks[0].rankine.previous_salt_inventory_cold.fix(6739291)
blks[0].rankine.previous_power.fix(300)

n_weeks = 1
opt = pyo.SolverFactory('ipopt')
hot_tank_level = []
net_power = []

for week in range(n_weeks):
    print("Solving for week: ", week)
    # for (i, blk) in enumerate(blks):
    #     blk.lmp_signal = weekly_prices[week][i]
    opt.solve(m, tee=True)
    hot_tank_level.append(
        [pyo.value(blks[i].rankine.salt_inventory_hot)
         for i in range(n_time_points)])
    net_power.append(
        [pyo.value(blks[i].rankine.fs.net_power)
         for i in range(n_time_points)])
log_close_to_bounds(m)
log_infeasible_constraints(m)

c = 0
for blk in blks:
    print()
    print('Period {}'.format(c+1))
    print(' Net power: {:.4f}'.format(
        value(blks[c].rankine.fs.net_power)))
    print(' Plant Power Out: {:.4f}'.format(
        value(blks[c].rankine.fs.plant_power_out[0])))
    print(' ES Turbine Power: {:.4f}'.format(
        value(blks[c].rankine.fs.es_turbine.work_mechanical[0])*(-1e-6)))
    print(' Revenue ($): {:.4f}'.format(value(blks[c].revenue)))
    print(' Operating cost ($): {:.4f}'.format(value(blks[c].operating_cost)))
    print(' Specific Operating cost ($/MWh): {:.4f}'.format(
        value(blks[c].operating_cost) / value(blks[c].rankine.fs.net_power)))
    print(' Cycle efficiency (%): {:.4f}'.format(
        value(blks[c].rankine.fs.cycle_efficiency)))
    print(' Boiler efficiency (%): {:.4f}'.format(
        value(blks[c].rankine.fs.boiler_eff) * 100))
    print(' Boiler heat duty: {:.4f}'.format(
        value(blks[c].rankine.fs.boiler.heat_duty[0]) * 1e-6))
    print(' Boiler flow mol (mol/s): {:.4f}'.format(
        value(blks[c].rankine.fs.boiler.outlet.flow_mol[0])))
    print(' Previous salt inventory: {:.4f}'.format(
        value(blks[c].rankine.previous_salt_inventory_hot)))
    print(' Salt from HXC (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxc.outlet_2.flow_mass[0]) * 3600))
    print(' Salt from HXD (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxd.outlet_1.flow_mass[0]) * 3600))
    print(' HXC Duty (MW): {:.4f}'.format(
        value(blks[c].rankine.fs.hxc.heat_duty[0]) * 1e-6))
    print(' HXD Duty (MW): {:.4f}'.format(
        value(blks[c].rankine.fs.hxd.heat_duty[0]) * 1e-6))
    print(' Split fraction to HXC: {:.4f}'.format(
        value(blks[c].rankine.fs.ess_hp_split.split_fraction[0, "to_hxc"])))
    print(' Split fraction to HXD: {:.4f}'.format(
        value(blks[c].rankine.fs.ess_bfp_split.split_fraction[0, "to_hxd"])))
    print(' Salt flow HXC (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxc.outlet_2.flow_mass[0])))
    print(' Salt flow HXD (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxd.outlet_1.flow_mass[0])))
    print(' Steam flow HXC (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxc.outlet_1.flow_mol[0])))
    print(' Steam flow HXD (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxd.outlet_2.flow_mol[0])))
    print(' Delta T in HXC (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxc.delta_temperature_in[0])))
    print(' Delta T out HXC (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxc.delta_temperature_out[0])))
    print(' Delta T in HXD (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxd.delta_temperature_in[0])))
    print(' Delta T out HXD (kg): {:.4f}'.format(
        value(blks[c].rankine.fs.hxd.delta_temperature_out[0])))
    c += 1

n_weeks_to_plot = 1
hours = np.arange(n_time_points*n_weeks_to_plot)
# lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
lmp_array = np.asarray(lmp[0:n_time_points])
hot_tank_array = np.asarray(hot_tank_level[0:n_weeks_to_plot]).flatten()

# Convert array to list to include hot tank level at time zero
lmp_list = [0] + lmp_array.tolist()
hot_tank_array0 = value(blks[0].rankine.previous_salt_inventory_hot)
hours_list = hours.tolist() + [number_hours]
hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()

font = {'size':16}
plt.rc('font', **font)
fig1, ax1 = plt.subplots(figsize=(12, 8))

color = ['r', 'b', 'tab:green']
ax1.set_xlabel('Time Period (hr)')
ax1.set_ylabel('Hot Tank Level (kg)',
               color=color[0])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(linestyle=':', which='both',
         color='gray', alpha=0.30)
ax1.step(# [x + 1 for x in hours], hot_tank_array,
    hours_list, hot_tank_list,
    marker='o', ms=8,
    color=color[0])
ax1.tick_params(axis='y',
                labelcolor=color[0])
ax1.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=1))

ax2 = ax1.twinx()
ax2.set_ylabel('LMP ($/MWh)',
               color=color[1])
ax2.step(# [x + 1 for x in hours], lmp_array,
    hours_list, lmp_list,
    marker='o', ms=7, alpha=0.75,
    ls=':', color=color[1])
ax2.tick_params(axis='y',
                labelcolor=color[1])
plt.savefig('multiperiod_usc_storage_new_hot_tank_level.png')


power_array = np.asarray(net_power[0:n_weeks_to_plot]).flatten()
# Convert array to list to include net power at time zero
power_array0 = 0 # zero since the plant is not operating
power_list = [power_array0] + power_array.tolist()

fig2, ax3 = plt.subplots(figsize=(12, 8))
ax3.set_xlabel('Time Period (hr)')
ax3.set_ylabel('Power Output (MW)',
               color=color[2])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(linestyle=':', which='both',
         color='gray', alpha=0.30)
ax3.step(# [x + 1 for x in hours], power_array,
    hours_list, power_list,
    marker='o', ms=8,
    color=color[2])
ax3.tick_params(axis='y',
                labelcolor=color[2])
ax3.set_xticks(np.arange(0, n_time_points*n_weeks_to_plot + 1, step=1))

ax4 = ax3.twinx()
ax4.set_ylabel('LMP ($/MWh)',
               color=color[1])
ax4.step(# [x + 1 for x in hours], lmp_array,
    hours_list, lmp_list,
    marker='o', ms=7, alpha=0.75,
    ls=':', color=color[1]
)
ax4.tick_params(axis='y',
                labelcolor=color[1])
plt.savefig('multiperiod_usc_storage_new_power.png')

plt.show()
