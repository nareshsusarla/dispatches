
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

import ultra_supercritical_powerplant_analysis as usc


# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)

method = "with_efficiency" # options: with and without efficiency

def create_ss_rankine_model():
    power_max = 436 # in MW
    power_min = int(0.65 * power_max) # 283 MW
    # power_min = 100 # random value, in MW
    boiler_heat_max = 918e6  # in W
    boiler_heat_min = 586e6  # in W

    m = pyo.ConcreteModel()
    m.rankine = usc.build_plant_model(method=method)

    m.rankine.fs.plant_power_out[0].setlb(None)
    m.rankine.fs.plant_power_out[0].setub(None)
    m.rankine.fs.net_power = Expression(
        expr=m.rankine.fs.plant_power_out[0]
    )
    m.rankine.fs.eq_min_power = pyo.Constraint(
        expr=m.rankine.fs.net_power >= power_min)

    m.rankine.fs.eq_max_power = pyo.Constraint(
        expr=m.rankine.fs.net_power <= power_max)

    # Note: boiler flow is now unfixed in usc.bluid_plant_model in
    # function usc_{without/with}_boiler_efficiency. The boiler flow
    # bounds are also included there
    # m.rankine.fs.boiler.inlet.flow_mol[0].unfix()  # normally fixed
    # m.rankine.fs.boiler.inlet.flow_mol[:].setlb(1)
    # m.rankine.fs.boiler.inlet.flow_mol[:].setub(None)
    # m.rankine.fs.boiler.outlet.flow_mol[:].setlb(1)
    # m.rankine.fs.boiler.outlet.flow_mol[:].setub(None)

    # m.rankine.fs.boiler.heat_duty[0].setlb(boiler_heat_min)
    # m.rankine.fs.boiler.heat_duty[0].setub(boiler_heat_max)

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

def create_mp_rankine_block():
    m = create_ss_rankine_model()
    b1 = m.rankine
    #  DOF = 1
    print('DOFs within mp create 1 =', degrees_of_freedom(m))

    b1.previous_power = Var(
        domain=NonNegativeReals,
        initialize=400,
        bounds=(100, 450),
        doc="Previous period power (MW)"
        )

    return m

# the power output and battery state are linked between time periods
def get_rankine_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.rankine.fs.plant_power_out[0],
             b2.rankine.previous_power)]

# the final power output and battery state must be the same
# as the intial power output and battery state


def get_rankine_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [(b1.rankine.fs.plant_power_out[0],
             b2.rankine.previous_power)]


n_time_points = 1*4  # hours in a week

# create the multiperiod model object
mp_rankine = MultiPeriodModel(
    n_time_points=n_time_points,
    process_model_func=create_mp_rankine_block,
    linking_variable_func=get_rankine_link_variable_pairs,
    # periodic_variable_func=get_rankine_periodic_variable_pairs
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

power = [310, 325, 420, 400] # in MW
lmp = [4, 10, 50, 100] # in $/MWh

count = 0
# add market data for each block
for blk in blks:
    blk_rankine = blk.rankine
    blk.lmp_signal = pyo.Param(default=0, mutable=True)
    blk.revenue = lmp[count]*blk.rankine.fs.net_power
    blk.operating_cost = pyo.Expression(
        expr=blk_rankine.fs.operating_cost * blk.rankine.fs.net_power
    )
    blk.cost = pyo.Expression(expr=-(blk.revenue - blk.operating_cost))
    # blk.fix_power = pyo.Constraint(
    #     expr=power[count] == blk.rankine.fs.net_power
    # )
    count += 1

m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]))
blks[0].rankine.previous_power.fix(400)

n_weeks = 1
opt = pyo.SolverFactory('ipopt')
hot_tank_level = []
net_power = []

for week in range(n_weeks):
    print("Solving for week: ", week)
    # for (i, blk) in enumerate(blks):
    #     blk.lmp_signal = weekly_prices[week][i]
    opt.solve(m, tee=True, options={"max_iter": 250})
    net_power.append(
        [pyo.value(blks[i].rankine.fs.net_power)
         for i in range(n_time_points)])
log_close_to_bounds(m)
log_infeasible_constraints(m)

c = 0
for blk in blks:
    print()
    print('Period {}'.format(c+1))
    print(' Net power (MW): {} (dummy previous power: {:.4f})'.format(
        value(blks[c].rankine.fs.net_power),
        value(blks[c].rankine.previous_power)))
    print(' Revenue: {:.4f}'.format(
        value(blks[c].revenue)))
    print(' Operating Cost: {:.4f}'.format(
        value(blks[c].operating_cost)))
    print(' Boiler heat duty: {:.4f}'.format(
        value(blks[c].rankine.fs.boiler.heat_duty[0]) * 1e-6))
    print(' Boiler flow mol (mol/s): {:.4f}'.format(
        value(blks[c].rankine.fs.boiler.outlet.flow_mol[0])))
    print(' Cycle efficiency (%): {:.4f}'.format(
        value(blks[c].rankine.fs.cycle_efficiency)))
    if method == "with_efficiency":
        print(' Boiler efficiency (%): {:.4f}'.format(
            value(blks[c].rankine.fs.boiler_eff) * 100))
    c += 1

n_weeks_to_plot = 1
hours = np.arange(n_time_points*n_weeks_to_plot)
# lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
lmp_array = np.asarray(lmp)
net_power_array = np.asarray(net_power[0:n_weeks_to_plot]).flatten()

color = ['tab:green', 'b', 'r']
plt.rcParams['lines.linewidth'] = 2
font = {'size':16}
plt.rc('font', **font)

fig2, ax1 = plt.subplots(figsize=(10, 5))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(linestyle=':', which='both',
         color='#696969', alpha=0.20)
ax1.set_xlabel('Time Period (hr)')
ax1.set_ylabel('Net Power [MW]', color=color[0])
ax1.plot([x + 1 for x in hours], net_power_array,
         marker='.', ms=10,
         ls='-', lw=1,
         color=color[0])
ax1.tick_params(axis='y', labelcolor=color[0])
ax1.set_xticks(np.arange(1, n_time_points*n_weeks_to_plot + 1, step=1))

ax2 = ax1.twinx()
ax2.set_ylabel('LMP [$/MWh]',
               color=color[1])
ax2.plot([x + 1 for x in hours], lmp_array,
         marker='o', ls='-', lw=1,
         color=color[1])
ax2.tick_params(axis='y', labelcolor=color[1])
# plt.savefig('multiperiod_solo_net_power_lmp_vs_hours.png')
plt.show()
