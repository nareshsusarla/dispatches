#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################

"""This script uses the multiperiod model for the simulatenous design
and operation of an integrated ultra-supercritical power plant with
energy storage and performs market analysis using the pricetaker
assumption. The electricity prices, LMP (locational marginal prices),
are assumed constant. The prices used in this study are either
obtained from a synthetic database or from NREL data.

"""

__author__ = "Soraya Rawlings and Naresh Susarla"


# Import Python libraries
import numpy as np
import json

# Import Pyomo objects
import pyomo.environ as pyo
from pyomo.environ import (Objective, Expression, value, maximize)
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)

# Import multiperiod model
from nlp_multiperiod_usc_pricetaker_unfixed_area import create_nlp_multiperiod_usc_model

# Import IDAES libraries
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers.get_solver import get_solver

# Import objects for plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
font = {'size':16}
plt.rc('axes', titlesize=24)
plt.rc('font', **font)


def _get_lmp(hours_per_day=None, nhours=None):

    # Select lmp source data and scaling factor according to that
    use_rts_data = False
    use_mod_rts_data = True
    if use_rts_data:
        print('>>>>>> Using RTS LMP data')
        with open('rts_results_all_prices_base_case.npy', 'rb') as f:
            dispatch = np.load(f)
            price = np.load(f)
        lmp = price[0:nhours].tolist()
    elif use_mod_rts_data:
        print('>>>>>> Using given LMP data')
        price = [
            22.9684, 21.1168, 20.4, 20.419,
            # 20.419, 21.2877, 23.07, 25,
            # 18.4634, 0, 0, 0,
            0, 0, 0, 0,
            # 19.0342, 23.07, 200, 200,
            200, 200, 200, 200, #1
            # 0, 0, 0, 0,
            # 20.419, 21.2877, 23.07, 25,
            # 18.4634, 0, 0, 0,
            # 22.9684, 21.1168, 20.4, 20.419,
            # 200, 200, 200, 200,
            # 19.0342, 23.07, 200, 200, #2
            # 18.4634, 0, 0, 0,
            # 0, 0, 0, 0,
            # 19.0342, 23.07, 200, 200,
            # 200, 200, 200, 200,
            # 0, 0, 0, 0,
            # 22.9684, 21.1168, 20.4, 20.419, #3
            # 20.419, 21.2877, 23.07, 25,
            # 18.4634, 0, 0, 0,
            # 0, 0, 0, 0,
            # 19.0342, 23.07, 200, 200,
            # 200, 200, 200, 200,
            # 22.9684, 21.1168, 20.4, 20.419,#4
            # 0, 0, 0, 0,
            # 19.0342, 23.07, 200, 200,
            # 200, 200, 200, 200,
            # 0, 0, 0, 0,
            # 20.419, 21.2877, 23.07, 25,
            # 18.4634, 0, 0, 0,#5
            # 200, 200, 200, 200,
            # 200, 200, 200, 200,
            # 0, 0, 0, 0,
            # 22.9684, 21.1168, 20.4, 20.419,
            # 20.419, 21.2877, 23.07, 25,
            # 18.4634, 0, 0, 0,#6
            # 0, 0, 0, 0,
            # 19.0342, 23.07, 200, 200,
            # 200, 200, 200, 200,
            # 0, 0, 0, 0,
            # 22.9684, 21.1168, 20.4, 20.419,
            # 20.419, 21.2877, 23.07, 25#7
        ]

        lmp = price
        if len(price) < hours_per_day:
            print()
            print('**ERROR: I need more LMP data!')
            raise Exception
    else:
        print('>>>>>> Using NREL LMP data')
        price = np.load("nrel_scenario_average_hourly.npy")

    return lmp


def run_pricetaker_analysis(hours_per_day=None,
                            nhours=None,
                            ndays=None,
                            nweeks=None,
                            n_time_points=None,
                            pmin=None,
                            tank_status=None,
                            factor_mton=None):

    # Get LMP data
    lmp = _get_lmp(hours_per_day=hours_per_day, nhours=nhours)

    # Create the multiperiod model object. You can pass arguments to your
    # "process_model_func" for each time period using a dict of dicts as
    # shown here.  In this case, it is setting up empty dictionaries for
    # each time period.
    nlp_multiperiod_usc = create_nlp_multiperiod_usc_model(
        n_time_points=n_time_points,
        pmin=pmin,
        pmax=None
    )

    # Retrieve pyomo model and active process blocks (i.e. time blocks)
    m = nlp_multiperiod_usc.pyomo_model
    blks = nlp_multiperiod_usc.get_active_process_blocks()

    # Add lmp market data for each block
    count = 0
    for blk in blks:
        # Add expression to calculate revenue in $ per hour.
        blk.revenue = pyo.Expression(
            expr=lmp[count] * blk.usc.fs.net_power
        )

        # # Declare expression to calculate the total costs in
        # # the plant, including operating and capital costs of storage
        # # and power plant in $ per hour.
        # blk.total_cost = pyo.Expression(
        #     expr=(blk.usc.fs.fuel_cost +
        #           blk.usc.fs.plant_fixed_operating_cost +
        #           blk.usc.fs.plant_variable_operating_cost +
        #           blk.usc.fs.storage_capital_cost)
        # )

        # # Declare an expression to calculate the total profit. All the
        # # costs are in $ per hour.
        # blk.profit = pyo.Expression(
        #     expr=(blk.revenue -
        #           blk.total_cost) * scaling_cost
        # )
        count += 1

    # m.obj = pyo.Objective(
    #     expr=sum([blk.profit for blk in blks]) * scaling_obj,
    #     sense=maximize
    # )
    m.obj = pyo.Objective(
        expr=sum(
            [blk.revenue -
             (blk.usc.fs.fuel_cost +
              blk.usc.fs.plant_fixed_operating_cost +
              blk.usc.fs.plant_variable_operating_cost +
              blk.usc.fs.storage_capital_cost)
             for blk in blks]
        ) * scaling_obj,
        sense=maximize
    )

    # Initial state for linking variables: power and salt
    # tank. Different tank scenarios are included for the Solar salt
    # tank levels and the previous tank level of the tank is based on
    # that.
    tank_min = 1 * factor_mton
    tank_max = max_salt_amount
    if tank_status == "hot_empty":
        # blks[0].usc.previous_salt_inventory_hot.fix(tank_min)
        # blks[0].usc.previous_salt_inventory_cold.fix(tank_max - tank_min)
        blks[0].usc.previous_salt_inventory_hot.fix(1103053.48 * factor_mton)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_max - 1103053.48 * factor_mton)
    elif tank_status == "hot_half_full":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_max / 2)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_max / 2)
    elif tank_status == "hot_full":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_max - tank_min)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_min)
    else:
        print("Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full")

    # blks[0].usc.previous_power.fix(400)
    blks[0].usc.previous_power.fix(447.66)


    # Declare the solver and a set of lists to save the results
    opt = pyo.SolverFactory('ipopt',
                            options={"max_iter": 300})

    hot_tank_level = []
    cold_tank_level = []
    net_power = []
    hxc_duty = []
    hxd_duty = []
    boiler_heat_duty = []
    discharge_work = []
    tank_max_list = []
    for week in range(nweeks):
        print()
        print(">>>>>> Solving for week {}: {} hours of operation in {} day(s) ".format(
            week + 1, nhours, ndays))

        results = opt.solve(m, tee=True)

        # Save results in lists
        tank_max_list.append(value(tank_max))
        boiler_heat_duty.append([pyo.value(blks[i].usc.fs.boiler.heat_duty[0]) * 1e-6
                                 for i in range(n_time_points)]) # in MW
        discharge_work.append([pyo.value(blks[i].usc.fs.es_turbine.work[0]) * (-1e-6)
                               for i in range(n_time_points)]) # in MW
        hot_tank_level.append([(pyo.value(blks[i].usc.salt_inventory_hot)) # in mton
                               for i in range(n_time_points)])
        cold_tank_level.append([(pyo.value(blks[i].usc.salt_inventory_cold))# in mton
                                for i in range(n_time_points)])
        net_power.append([pyo.value(blks[i].usc.fs.net_power)
                          for i in range(n_time_points)])
        hxc_duty.append([pyo.value(blks[i].usc.fs.hxc.heat_duty[0]) * 1e-6 # in MW
                         for i in range(n_time_points)])
        hxd_duty.append([pyo.value(blks[i].usc.fs.hxd.heat_duty[0]) * 1e-6 # in MW
                         for i in range(n_time_points)])

        log_close_to_bounds(m)
        log_infeasible_constraints(m)

    return (m, blks, lmp, net_power, results, tank_max, tank_max_list, hot_tank_level,
            cold_tank_level, hxc_duty, hxd_duty, boiler_heat_duty, discharge_work)


def print_results(m, blks, results):
    c = 0
    print('Objective: {:.4f}'.format(value(m.obj) / scaling_obj))
    for blk in blks:
        print()
        print('Period {}'.format(c+1))
        print(' Net power: {:.4f}'.format(
            value(blks[c].usc.fs.net_power)))
        print(' Plant Power Out (MW): {:.4f}'.format(
            value(blks[c].usc.fs.plant_power_out[0])))
        print(' Coal heat duty (MW): {:.4f}'.format(
            value(blks[c].usc.fs.coal_heat_duty)))
        print(' ES Turbine Power (MW): {:.4f}'.format(
            value(blks[c].usc.fs.es_turbine.work_mechanical[0])*(-1e-6)))
        print(' Storage capital cost ($/h): {:.4f}'.format(
            value(blks[c].usc.fs.storage_capital_cost)))
        # print(' Profit ($/h): {:.4f}'.format(value(blks[c].profit)))
        print(' Revenue ($/h): {:.4f}'.format(value(blks[c].revenue)))
        # print(' Total operating cost ($/h): {:.4f}'.format(
        #     value(blks[c].total_cost)))
        print(' Storage cost ($/h): {:.4f}'.format(
            value(blks[c].usc.fs.storage_capital_cost)))
        print(' Plant fuel cost ($/h): {:.4f}'.format(
            value(blks[c].usc.fs.fuel_cost)))
        print(' Plant fixed operating cost ($/h): {:.4f}'.format(
            value(blks[c].usc.fs.plant_fixed_operating_cost)))
        print(' Plant variable operating cost ($/h): {:.4f}'.format(
            value(blks[c].usc.fs.plant_variable_operating_cost)))
        print(' Boiler/cycle efficiency (%): {:.4f}/{:.4f}'.format(
            value(blks[c].usc.fs.boiler_efficiency) * 100,
            value(blks[c].usc.fs.boiler_efficiency) * 100))
        print(' Boiler heat duty: {:.4f}'.format(
            value(blks[c].usc.fs.plant_heat_duty[0])))
        print(' Boiler flow mol (mol/s): {:.4f}'.format(
            value(blks[c].usc.fs.boiler.outlet.flow_mol[0])))
        print(' Previous hot salt inventory (mton): {:.4f}'.format(
            (value(blks[c].usc.previous_salt_inventory_hot))))
        print(' Hot salt inventory (mton): {:.4f}'.format(
            (value(blks[c].usc.salt_inventory_hot))))
        print(' Hot salt from HXC (mton): {:.4f}'.format(
            value(blks[c].usc.fs.hxc.tube_outlet.flow_mass[0]) * 3600))
        print(' Hot salt into HXD (mton): {:.4f}'.format(
            value(blks[c].usc.fs.hxd.shell_inlet.flow_mass[0]) * 3600))
        print(' Cold salt into HXC (mton): {:.4f}'.format(
            value(blks[c].usc.fs.hxc.tube_inlet.flow_mass[0]) * 3600))
        print(' Cold salt from HXD (mton): {:.4f}'.format(
            value(blks[c].usc.fs.hxd.shell_outlet.flow_mass[0]) * 3600))
        print(' HXC area (m2): {:.4f}'.format(
            value(blks[c].usc.fs.hxc.area)))
        print(' HXD area (m2): {:.4f}'.format(
            value(blks[c].usc.fs.hxd.area)))
        print(' HXC Duty (MW): {:.4f}'.format(
            value(blks[c].usc.fs.hxc.heat_duty[0]) * 1e-6))
        print(' HXD Duty (MW): {:.4f}'.format(
            value(blks[c].usc.fs.hxd.heat_duty[0]) * 1e-6))
        print(' Split fraction to HXC: {:.4f}'.format(
            value(blks[c].usc.fs.ess_hp_split.split_fraction[0, "to_hxc"])))
        print(' Split fraction to HXD: {:.4f}'.format(
            value(blks[c].usc.fs.ess_bfp_split.split_fraction[0, "to_hxd"])))
        print(' Salt flow HXC (kg/s): {:.4f}'.format(
            value(blks[c].usc.fs.hxc.tube_outlet.flow_mass[0])))
        print(' Salt flow HXD (kg/s): {:.4f}'.format(
            value(blks[c].usc.fs.hxd.shell_outlet.flow_mass[0])))
        print(' Steam flow HXC (mol/s): {:.4f}'.format(
            value(blks[c].usc.fs.hxc.shell_outlet.flow_mol[0])))
        print(' Steam flow HXD (mol/s): {:.4f}'.format(
            value(blks[c].usc.fs.hxd.tube_outlet.flow_mol[0])))
        print(' HXC salt inlet/outlet temperature (K): {:.4f}/{:.4f}'.format(
            value(blks[c].usc.fs.hxc.tube_inlet.temperature[0]),
            value(blks[c].usc.fs.hxc.tube_outlet.temperature[0])))
        print(' HXD salt inlet/outlet temperature (K): {:.4f}/{:.4f}'.format(
            value(blks[c].usc.fs.hxd.shell_inlet.temperature[0]),
            value(blks[c].usc.fs.hxd.shell_outlet.temperature[0])))
        print(' HXC delta temperature in/out (K): {:.4f}/{:.4f}'.format(
            value(blks[c].usc.fs.hxc.delta_temperature_in[0]),
            value(blks[c].usc.fs.hxc.delta_temperature_out[0])))
        print(' HXD delta temperature in/out (K): {:.4f}/{:.4f}'.format(
            value(blks[c].usc.fs.hxd.delta_temperature_in[0]),
            value(blks[c].usc.fs.hxd.delta_temperature_out[0])))
        c += 1

    print(results)

def plot_results(m,
                 blks,
                 lmp,
                 ndays=None,
                 nweeks=None,
                 n_time_points=None,
                 net_power=None,
                 tank_max=None,
                 tank_max_list=None,
                 hot_tank_level=None,
                 cold_tank_level=None,
                 hxc_duty=None,
                 hxd_duty=None,
                 boiler_heat_duty=None,
                 discharge_work=None,
                 pmax_total=None):


    # List of colors to be used in the plots
    c = ['darkred', 'midnightblue', 'tab:green', 'k', 'gray']

    # Save and convert array to list to include values at time zero
    # for all the data that is going to be plotted
    hours = np.arange(n_time_points * nweeks)
    lmp_array = np.asarray(lmp[0:n_time_points])
    hot_tank_array = np.asarray(hot_tank_level[0:nweeks]).flatten()
    cold_tank_array = np.asarray(cold_tank_level[0:nweeks]).flatten()
    hot_tank_array0 = value(blks[0].usc.previous_salt_inventory_hot)
    cold_tank_array0 = value(blks[0].usc.previous_salt_inventory_cold)
    hours_list = hours.tolist() + [nhours]
    hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
    cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()
    hxc_array = np.asarray(hxc_duty[0:nweeks]).flatten()
    hxd_array = np.asarray(hxd_duty[0:nweeks]).flatten()
    hxc_duty_list = [0] + hxc_array.tolist()
    hxd_duty_list = [0] + hxd_array.tolist()
    boiler_heat_duty_array = np.asarray(boiler_heat_duty[0:nweeks]).flatten()
    boiler_heat_duty_list = [0] + boiler_heat_duty_array.tolist()
    power_array = np.asarray(net_power[0:nweeks]).flatten()
    power_array0 = value(blks[0].usc.previous_power)
    power_list = [power_array0] + power_array.tolist()
    discharge_work_array = np.asarray(discharge_work[0:nweeks]).flatten()
    discharge_work_list = [0] + discharge_work_array.tolist()

    # Plot salt inventory profiles
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xticks(range(0, n_time_points+1, 1))
    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Amount (metric ton)', color=c[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    # ax1.set_ylim((0, 7000))
    plt.axhline(tank_max, ls=':', lw=1.5, color=c[4])
    ax1.step(hours_list, hot_tank_list, marker='o', ms=8, lw=1.5, color=c[0], alpha=0.85,
             label='Hot Tank')
    ax1.fill_between(hours_list, hot_tank_list, step="pre", color=c[0], alpha=0.35)
    ax1.step(hours_list, cold_tank_list, marker='o', ms=8, lw=1.5, color=c[1], alpha=0.65,
             label='Cold Tank')
    # ax1.fill_between(hours_list, cold_tank_list, tank_max, step="pre", color=c[1], alpha=0.15)
    # ax1.fill_between(hours_list, hot_tank_list, tank_max, step="pre", color=c[1], alpha=0.25,
    #                  label='Cold Tank')
    ax1.fill_between(hours_list, cold_tank_list, step="pre", color=c[1], alpha=0.1)
    ax1.legend(loc="center left", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points * nweeks + 1, step=1))
    ax2 = ax1.twinx()
    ax2.set_ylim((-25, 225))
    ax2.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax2.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax2.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/nlp_mp_unfixed_area/salt_tank_level_{}hrs.png'.format(
        hours_per_day * ndays))

    # Plot boiler and charge and discharge heat exchangers heat duty
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Heat Duty (MW)', color=c[3])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both', color='gray', alpha=0.40)
    ax3.set_ylim((-25, 825))
    ax3.step(hours_list, boiler_heat_duty_list, marker='o', ms=8, color=c[3], ls='-', lw=1.5, alpha=0.85,
             label='Boiler')
    ax3.fill_between(hours_list, boiler_heat_duty_list, step="pre", color=c[3], alpha=0.15)
    plt.axhline(max_storage_heat_duty, ls=':', lw=1.5, color=c[4])
    plt.axhline(min_storage_heat_duty, ls=':', lw=1.5, color=c[4])
    ax3.step(hours_list, hxc_duty_list, marker='o', ms=8, color=c[0], alpha=0.75,
             label='Charge')
    ax3.fill_between(hours_list, hxc_duty_list, step="pre", color=c[0], alpha=0.25)
    ax3.step(hours_list, hxd_duty_list, marker='o', ms=8, color=c[1], alpha=0.75,
             label='Discharge')
    ax3.fill_between(hours_list, hxd_duty_list, step="pre", color=c[1], alpha=0.25)
    ax3.tick_params(axis='y', labelcolor=c[3])
    ax3.legend(loc="center left", frameon=False)
    ax3.tick_params(axis='y')
    ax3.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=1))
    ax4 = ax3.twinx()
    ax4.set_ylim((-25, 225))
    ax4.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax4.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax4.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/nlp_mp_unfixed_area/heat_duty_{}hrs.png'.format(hours_per_day * ndays))

    # Plot net power and discharge power profiles
    fig3, ax5 = plt.subplots(figsize=(12, 8))
    ax5.set_xlabel('Time Period (hr)')
    ax5.set_ylabel('Power Output (MW)', color='midnightblue')
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    plt.axhline(pmax, ls=':', lw=1.5, color=c[4])
    plt.axhline(pmax_total, ls=':', lw=1.5, color=c[4])
    ax5.step(hours_list, power_list, marker='o', ms=8, lw=1.5, color=c[3], alpha=0.85,
             label='Plant Net Power')
    ax5.fill_between(hours_list, power_list, step="pre", color=c[3], alpha=0.15)
    ax5.step(hours_list, discharge_work_list, marker='o', ms=8, color=c[1], alpha=0.75,
             label='Discharge Turbine')
    ax5.fill_between(hours_list, discharge_work_list, step="pre", color=c[1], alpha=0.15)
    ax5.tick_params(axis='y', labelcolor=c[1])
    ax5.legend(loc="center left", frameon=False)
    ax5.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=1))
    ax6 = ax5.twinx()
    ax6.set_ylim((-25, 225))
    ax6.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax6.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5,
             color=c[2])
    ax6.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/nlp_mp_unfixed_area/power_{}hrs.png'.format(hours_per_day * ndays))

    plt.show()

def _mkdir(dir):
    """Create directory to save results

    """

    try:
        os.mkdir(dir)
        print('Directory {} created'.format(dir))
    except:
        print('Directory {} not created because it already exists!'.format(dir))
        pass


if __name__ == '__main__':
    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # If new_design is set to True, use GDP design for charge and
    # discharge heat exchanger from 4-12 disjunctions model. When set
    # to False, use the GDP design from 4-5 disjunctions
    # model. **Note** When changing this, make sure to change it in
    # the gdp_multiperiod... python script too.
    new_design = True

    lx = True
    if lx:
        if new_design:
            # scaling_obj = 1e-2
            # scaling_cost = 1e-3
            scaling_obj = 1e-3
            scaling_cost = 1
        else:
            # Old design
            scaling_obj = 1e-3 # 24 hrs
            # scaling_obj = 1e-2 # < 24 hrs
            # scaling_obj = 1e-4 # > 24 hrs
            scaling_cost = 1
    else:
        scaling_obj = 1
        scaling_cost = 1

    print()
    print('scaling_obj:', scaling_obj)
    print('scaling_cost:', scaling_cost)

    # Add design data from .json file
    if new_design:
        data_path = 'uscp_design_data_new_storage_design.json'
    else:
        data_path = 'uscp_design_data.json'

    with open(data_path) as design_data:
        design_data_dict = json.load(design_data)

    factor_mton = design_data_dict["factor_mton"]
    max_salt_amount = design_data_dict["max_salt_amount"] * factor_mton # in mton
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    # min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
    min_storage_heat_duty = 10
    pmax = design_data_dict["plant_max_power"]
    pmin = design_data_dict["plant_min_power"]
    pmin_storage = design_data_dict["min_discharge_turbine_power"]
    pmax_storage = design_data_dict["max_discharge_turbine_power"]
    pmax_total = pmax + pmax_storage
    pmin_total = pmin + pmin_storage

    hours_per_day = 12
    ndays = 1
    nhours = hours_per_day * ndays
    nweeks = 1

    # Add number of hours per week
    n_time_points = nweeks * nhours

    tank_status = "hot_empty"

    # Create a directory to save the results for each NLP sbproblem
    # and plots
    _mkdir('results')
    _mkdir('results/nlp_mp_unfixed_area_{}h'.format(nhours))


    (m,
     blks,
     lmp,
     net_power,
     results,
     tank_max,
     tank_max_list,
     hot_tank_level,
     cold_tank_level,
     hxc_duty,
     hxd_duty,
     boiler_heat_duty,
     discharge_work) = run_pricetaker_analysis(hours_per_day=hours_per_day,
                                               nhours=nhours,
                                               ndays=ndays,
                                               nweeks=nweeks,
                                               n_time_points=n_time_points,
                                               pmin=pmin,
                                               tank_status=tank_status,
                                               factor_mton=factor_mton)

    print_results(m,
                  blks,
                  results)

    plot_results(m,
                 blks,
                 lmp,
                 ndays=ndays,
                 nweeks=nweeks,
                 n_time_points=n_time_points,
                 hot_tank_level=hot_tank_level,
                 cold_tank_level=cold_tank_level,
                 net_power=net_power,
                 hxc_duty=hxc_duty,
                 hxd_duty=hxd_duty,
                 tank_max=tank_max,
                 tank_max_list=tank_max_list,
                 boiler_heat_duty=boiler_heat_duty,
                 discharge_work=discharge_work,
                 pmax_total=pmax_total)
