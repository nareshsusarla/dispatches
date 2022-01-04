
# import multiperiod object and rankine example
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from idaes.apps.multiperiod.examples.simple_rankine_cycle import (
    create_model, set_inputs, initialize_model,
    close_flowsheet_loop, add_operating_cost)

import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var)
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


def create_ss_rankine_model():
    p_lower_bound = 100  # MW
    p_upper_bound = 450  # MW

    m = pyo.ConcreteModel()
    m.rankine = usc.main()
    # set bounds for net cycle power output
    m.rankine.fs.plant_power_out[0].unfix()
    m.rankine.fs.eq_min_power = pyo.Constraint(
        expr=m.rankine.fs.plant_power_out[0] >= p_lower_bound)

    m.rankine.fs.eq_max_power = pyo.Constraint(
        expr=m.rankine.fs.plant_power_out[0] <= p_upper_bound)

    m.rankine.fs.boiler.inlet.flow_mol[0].unfix()  # normally fixed
    # m.rankine.fs.boiler.inlet.flow_mol[0].setlb(1)
    m.rankine.fs.boiler.inlet.flow_mol[0].setlb(11804)
    m.rankine.fs.boiler.inlet.flow_mol[0].setub(17854)

    # Unfix all data
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
    m.rankine.fs.hxc.area.fix(1904)
    m.rankine.fs.hxd.area.fix(1095)
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
# plt.savefig("lmp_vs_hour.pdf")

# weekly_prices = prices_used.reshape(52, 168)
# plt.figure(figsize=(12, 8))
# for week in [0, 15, 25, 35, 45, 51]:
#     plt.plot(weekly_prices[week])
# plt.title("6 Representative Weeks")
# plt.xlabel("Hour")
# plt.ylabel("LMP $/MWh")
# plt.savefig("lmp_vs_hour_6week.pdf")

# plt.figure(figsize=(12, 8))
# for week in range(0, 52):
#     plt.plot(weekly_prices[week], color="blue", alpha=0.1)
# plt.title("52 Representative Weeks")
# plt.xlabel("Hour")
# plt.ylabel("LMP $/MWh")
# plt.savefig("lmp_vs_hour_52week.pdf")

# turbine_ramp_rate = 100
# battery_ramp_rate = 50
def create_mp_rankine_block():
    m = create_ss_rankine_model()
    b1 = m.rankine
    #  DOF = 1
    print('DOFs within mp create 1 =', degrees_of_freedom(m))
    # Add coupling variable (next_power_output)
    b1.previous_salt_inventory_hot = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_hot = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    b1.previous_salt_inventory_cold = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_cold = Var(
        # b1.fs.time,
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, 1e12),
        # bounds=(0, 6739292),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            b1.salt_inventory_hot ==
            b1.previous_salt_inventory_hot
            + 3600*b1.fs.hxc.inlet_2.flow_mass[0]
            - 3600*b1.fs.hxd.inlet_1.flow_mass[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_cold(b):
        return (
            b1.salt_inventory_cold ==
            b1.previous_salt_inventory_cold
            - 3600*b1.fs.hxc.inlet_2.flow_mass[0]
            + 3600*b1.fs.hxd.inlet_1.flow_mass[0])

    @b1.fs.Constraint(doc="Maximum previous salt inventory at any time")
    def constraint_salt_previous_inventory(b):
        return (
            b1.previous_salt_inventory_hot +
            b1.previous_salt_inventory_cold == b1.fs.salt_amount)
    print('DOFs after mp create =', degrees_of_freedom(m))
    # raise Exception()
    return m

# the power output and battery state are linked between time periods


def get_rankine_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.rankine.salt_inventory_hot,
             b2.rankine.previous_salt_inventory_hot)]

# the final power output and battery state must be the same
# as the intial power output and battery state


def get_rankine_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    return [(b1.rankine.salt_inventory_hot,
             b2.rankine.previous_salt_inventory_hot)]


n_time_points = 1*4  # hours in a week

# create the multiperiod model object
mp_rankine = MultiPeriodModel(
    n_time_points=n_time_points,
    process_model_func=create_mp_rankine_block,
    linking_variable_func=get_rankine_link_variable_pairs,
    periodic_variable_func=get_rankine_periodic_variable_pairs)

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

power = [325, 325, 410, 410]
lmp = [22.4929, 21.8439, 23.4379, 23.4379]
# lmp = [22.4929, 21.8439, 23.4379, 23.4379, 23.4379, 21.6473, 21.6473]

count = 0
# add market data for each block
for blk in blks:
    blk_rankine = blk.rankine
    blk.lmp_signal = pyo.Param(default=0, mutable=True)
    blk.revenue = lmp[count]*blk_rankine.fs.plant_power_out[0]
    # blk.revenue = blk.lmp_signal*blk_rankine.fs.plant_power_out[0]
    blk.operating_cost = pyo.Expression(expr=(
        (blk_rankine.fs.operating_cost
         + blk_rankine.fs.plant_fixed_operating_cost
         + blk_rankine.fs.plant_variable_operating_cost) / (365 * 24)))
    blk.cost = pyo.Expression(expr=-(blk.revenue - blk.operating_cost))
    blk.fix_power = pyo.Constraint(
        expr=power[count] == (
            blk.rankine.fs.plant_power_out[0]
            + (-1e-6) * blk.rankine.fs.es_turbine.work_mechanical[0]
        )
    )
    # blk.fix_power = pyo.Constraint(
    #     expr=blk.dispatch == (
    #         blk.rankine.fs.plant_power_out[0]
    #         + (-1e-6) * blk.rankine.fs.es_turbine.work_mechanical[0]
    #     )
    # )
    count += 1

m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]))
blks[0].rankine.previous_salt_inventory_hot.fix(1)

n_weeks = 1
opt = pyo.SolverFactory('ipopt')
tank_level = []
net_power = []

for week in range(n_weeks):
    print("Solving for week: ", week)
    # for (i, blk) in enumerate(blks):
    #     blk.lmp_signal = weekly_prices[week][i]
    opt.solve(m, tee=True)
    tank_level.append(
        [pyo.value(blks[i].rankine.salt_inventory_hot)
         for i in range(n_time_points)])
    net_power.append(
        [pyo.value(blks[i].rankine.fs.plant_power_out[0])
         for i in range(n_time_points)])
c = 0
for blk in blks:
    print()
    print('Block {}'.format(c))
    print('Previous salt inventory', value(blks[c].rankine.previous_salt_inventory_hot))
    print('Salt from HXC (kg)', value(blks[c].rankine.fs.hxc.outlet_2.flow_mass[0]) * 3600)
    print('Salt from HXD (kg)', value(blks[c].rankine.fs.hxd.outlet_1.flow_mass[0]) * 3600)
    print('ESS HP split', value(blks[c].rankine.fs.ess_hp_split.split_fraction[0, "to_hxc"]))
    print('ESS BFP split', value(blks[c].rankine.fs.ess_bfp_split.split_fraction[0, "to_hxd"]))
    c += 1

n_weeks_to_plot = 1
hours = np.arange(n_time_points*n_weeks_to_plot)
# lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
lmp_array = np.asarray(lmp)
tank_array = np.asarray(tank_level[0:n_weeks_to_plot]).flatten()
fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:green'
ax1.set_xlabel('Hour')
ax1.set_ylabel('Hot Tank Level [kg]', color=color)
ax1.step(hours, tank_array, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('LMP [$/MWh]', color=color)
ax2.plot(hours, lmp_array, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.savefig("salt_level_lmp_vs_hour.pdf")
plt.show()

# n_weeks_to_plot = 1
# hours = np.arange(n_time_points*n_weeks_to_plot)
# lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
# power_array = np.asarray(net_power[0:n_weeks_to_plot]).flatten()

# fig, ax1 = plt.subplots(figsize=(12, 8))

# color = 'tab:red'
# ax1.set_xlabel('Hour')
# ax1.set_ylabel('Power Output [MW]', color=color)
# ax1.step(hours, power_array, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('LMP [$/MWh]', color=color)
# ax2.plot(hours, lmp_array, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
