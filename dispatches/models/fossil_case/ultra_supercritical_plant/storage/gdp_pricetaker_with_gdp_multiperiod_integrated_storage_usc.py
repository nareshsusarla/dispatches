##############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
##############################################################################

"""This script uses the multiperiod model in the GDP integrated
ultra-supercritical power plant model with energy storage and performs
market analysis using the pricetaker assumption. The electricity
prices or LMP (locational marginal prices) are assumed to not
change. The prices used in this study are obtained from a synthetic
database.

"""

__author__ = "Soraya Rawlings"

import csv
import json
import os
import copy
import numpy as np
import logging

import pyomo.environ as pyo
from pyomo.environ import (Constraint, Expression,
                           Var, Objective,
                           SolverFactory,
                           value, RangeSet)
from pyomo.contrib.fbbt.fbbt import _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)

from idaes.core.util import get_solver

from gdp_multiperiod_usc_pricetaker_unfixed_area import create_gdp_multiperiod_usc_model

# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)


def _get_lmp(hours_per_day=None, nhours=None):

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
        lmp = price[0:nhours].tolist()
    elif use_mod_rts_data:
        price = [22.9684, 0, 0, 200]
        # price = [
        #     22.9684, 21.1168, 20.4, 20.419,
        #     # 20.419, 21.2877, 23.07, 25,
        #     # 18.4634, 0, 0, 0,
        #     0, 0, 0, 0,
        #     # 19.0342, 23.07, 200, 200,
        #     200, 200, 200, 200,
        # ]

        if len(price) < hours_per_day:
            print()
            print('**ERROR: I need more LMP data!')
            raise Exception
        lmp = price
    else:
        print('>> Using NREL lmp data')
        price = np.load("nrel_scenario_average_hourly.npy")
        # print(lmp)

    return lmp


def print_model(mdl,
                mdl_data,
                csvfile,
                lmp=None,
                nweeks=None,
                nhours=None,
                n_time_points=None):

    mdl.disjunction1_selection = {}
    hot_tank_level_iter = []
    cold_tank_level_iter = []

    print('       ___________________________________________')
    print('        Schedule')
    print('         Obj ($): {:.4f}'.format(
        (value(mdl.obj) / scaling_cost) / scaling_obj))

    for blk in mdl.blocks:
        blk_process_charge = mdl.blocks[blk].process.usc.fs.charge_mode_disjunct
        blk_process_discharge = mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct
        blk_process_no_storage = mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct
        if blk_process_charge.binary_indicator_var.value == 1:
            print('         Period {}: Charge (HXC: {:.0f} MW, {:.0f} m2)'.format(
                blk,
                value(blk_process_charge.hxc.heat_duty[0]) * 1e-6,
                value(blk_process_charge.hxc.area)))
        if blk_process_discharge.binary_indicator_var.value == 1:
            print('         Period {}: Discharge (HXD: {:.0f} MW, {:.0f} m2)'.format(
                blk,
                value(blk_process_discharge.hxd.heat_duty[0]) * 1e-6,
                value(blk_process_discharge.hxd.area)))
        if blk_process_no_storage.binary_indicator_var.value == 1:
            print('         Period {}: No storage'.format(blk))

    print()
    for blk in mdl.blocks:
        blk_process = mdl.blocks[blk].process.usc
        blk_process_charge = mdl.blocks[blk].process.usc.fs.charge_mode_disjunct
        blk_process_discharge = mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct
        blk_process_no_storage = mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct
        print('       Time period {} '.format(blk+1))
        print('        Charge: {}'.format(
            blk_process_charge.binary_indicator_var.value))
        print('        Discharge: {}'.format(
            blk_process_discharge.binary_indicator_var.value))
        print('        No storage: {}'.format(
            blk_process_no_storage.binary_indicator_var.value))
        if blk_process_charge.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[mdl_data.master_iteration] = 'Charge'
            print('         HXC area (m2): {:.4f}'.format(
                value(blk_process_charge.hxc.area)))
            print('         HXC Duty (MW): {:.4f}'.format(
                value(blk_process_charge.hxc.heat_duty[0]) * 1e-6))
            print('         HXC salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_charge.hxc.inlet_2.temperature[0]),
                value(blk_process_charge.hxc.outlet_2.temperature[0])))
            print('         Salt flow HXC (kg/s): {:.4f}'.format(
                value(blk_process_charge.hxc.outlet_2.flow_mass[0])))
            print('         HXC steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_charge.hxc.side_1.properties_in[0].temperature),
                value(blk_process_charge.hxc.side_1.properties_out[0].temperature)
            ))
            print('         Steam flow HXC (mol/s): {:.4f}'.format(
                value(blk_process_charge.hxc.outlet_1.flow_mol[0])))
            if not new_design:
                print('         Cooling heat duty (MW): {:.4f}'.format(
                    value(blk_process_charge.cooler.heat_duty[0]) * 1e-6))
        elif blk_process_discharge.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[mdl_data.master_iteration] = 'Discharge'
            print('         HXD area (m2): {:.4f}'.format(
                value(blk_process_discharge.hxd.area)))
            print('         HXD Duty (MW): {:.4f}'.format(
                value(blk_process_discharge.hxd.heat_duty[0]) * 1e-6))
            print('         HXD salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_discharge.hxd.inlet_1.temperature[0]),
                value(blk_process_discharge.hxd.outlet_1.temperature[0])))
            print('         Salt flow HXD (kg/s): {:.4f}'.format(
                value(blk_process_discharge.hxd.outlet_1.flow_mass[0])))
            print('         HXD steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_discharge.hxd.side_2.properties_in[0].temperature),
                value(blk_process_discharge.hxd.side_2.properties_out[0].temperature)
            ))
            print('         Steam flow HXD (mol/s): {:.4f}'.format(
                value(blk_process_discharge.hxd.outlet_2.flow_mol[0])))
            print('         ES turbine work (MW): {:.4f}'.format(
                value(blk_process_discharge.es_turbine.work_mechanical[0]) * -1e-6))
        elif blk_process_no_storage.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[mdl_data.master_iteration] = 'No_storage'
            print('         Salt flow (kg/s): {:.4f}'.format(value(blk_process.fs.salt_storage)))
        else:
            print('        No other operation mode is available!')
        print('        Net power: {:.4f}'.format(value(blk_process.fs.net_power)))
        print('        Discharge turbine work (MW): {:.4f}'.format(
            value(blk_process.fs.discharge_turbine_work)))
        if not new_design:
            print('        Cooler heat duty: {:.4f}'.format(value(blk_process.fs.cooler_heat_duty)))
        print('        Efficiencies (%): boiler: {:.4f}, cycle: {:.4f}'.format(
            value(blk_process.fs.boiler_eff) * 100,
            value(blk_process.fs.cycle_efficiency) * 100))
        print('        Boiler heat duty: {:.4f}'.format(
            value(blk_process.fs.boiler.heat_duty[0]) * 1e-6))
        print('        Boiler flow mol (mol/s): {:.4f}'.format(
            value(blk_process.fs.boiler.outlet.flow_mol[0])))
        print('        Salt to storage (kg/s) [mton]: {:.4f} [{:.4f}]'.format(
            value(blk_process.fs.salt_storage),
            value(blk_process.fs.salt_storage) * 3600 * factor_mton))
        print('        Hot salt inventory (mton): {:.4f}, previous: {:.4f}'.format(
            value(blk_process.salt_inventory_hot),
            value(blk_process.previous_salt_inventory_hot)))
        print('        Makeup water flow (mol/s): {:.4f}'.format(
            value(blk_process.fs.condenser_mix.makeup.flow_mol[0])))
        print('        Total op cost ($/h): {:.4f}'.format(
            value(mdl.blocks[blk].process.operating_cost) / scaling_cost))
        print('        Total cap cost ($/h): {:.4f}'.format(
            value(mdl.blocks[blk].process.capital_cost) / scaling_cost))
        print('        Revenue (M$/year): {:.4f}'.format(
            value(mdl.blocks[blk].process.revenue) / scaling_cost))
        print()

        # Save data for each NLP subproblem and plot results
        mdl.objective_val = {}
        mdl.boiler_heat_duty_val = {}
        mdl.discharge_turbine_work_val = {}
        mdl.hxc_area_val = {}
        mdl.hxd_area_val = {}
        mdl.hot_salt_temp_val = {}
        m_iter = mdl_data.master_iteration
        mdl.objective_val[m_iter] = (
            (value(mdl.obj) / scaling_cost) / scaling_obj
        )
        mdl.iterations = m_iter
        mdl.period = blk
        mdl.boiler_heat_duty_val[m_iter] = 1e-6 * value(blk_process.fs.boiler.heat_duty[0])
        mdl.discharge_turbine_work_val[m_iter] = value(blk_process.fs.discharge_turbine_work)
        mdl.hxc_area_val[m_iter] = value(blk_process_charge.hxc.area)
        mdl.hxd_area_val[m_iter] = value(blk_process_discharge.hxd.area)
        mdl.hot_salt_temp_val[m_iter] = value(blk_process_charge.hxc.outlet_2.temperature[0])

        if save_results:
            writer = csv.writer(csvfile)
            writer.writerow(
                (m_iter,
                 mdl.period,
                 mdl.disjunction1_selection[m_iter],
                 mdl.boiler_heat_duty_val[m_iter],
                 mdl.discharge_turbine_work_val[m_iter],
                 mdl.hxc_area_val[m_iter],
                 mdl.hxd_area_val[m_iter],
                 mdl.hot_salt_temp_val[m_iter],
                 mdl.objective_val[m_iter])
            )
            csvfile.flush()

    print('        Obj (M$/year): {:.4f}'.format((value(mdl.obj) / scaling_cost) / scaling_obj))
    print('       ___________________________________________')

    hot_tank_level_iter.append(
        [(pyo.value(mdl.blocks[i].process.usc.salt_inventory_hot)) # in mton
         for i in range(n_time_points)])
    cold_tank_level_iter.append(
        [(pyo.value(mdl.blocks[i].process.usc.salt_inventory_cold)) # in mton
         for i in range(n_time_points)])

    # Plot results
    hours = np.arange(n_time_points * nweeks)
    lmp_array = np.asarray(lmp[0:n_time_points])
    color = ['r', 'b', 'tab:green', 'k', 'tab:orange']
    hot_tank_array = np.asarray(hot_tank_level_iter[0:nweeks]).flatten()
    cold_tank_array = np.asarray(cold_tank_level_iter[0:nweeks]).flatten()

    # Convert array to list to include hot tank level at time zero
    hot_tank_array0 = (
        value(mdl.blocks[0].process.usc.previous_salt_inventory_hot))
    cold_tank_array0 = (
        value(mdl.blocks[0].process.usc.previous_salt_inventory_cold))
    hours_list = hours.tolist() + [nhours]
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
    plt.axhline(tank_max,
                ls=':', lw=1.75,
                color=color[4])
    plt.text(nhours / 2 - 1.5,
             tank_max + 100, 'max salt',
             color=color[4])
    ax1.step(hours_list, hot_tank_list,
             marker='^', ms=4, label='Hot Salt',
             lw=1, color=color[0])
    ax1.step(hours_list, cold_tank_list,
             marker='v', ms=4, label='Cold Salt',
             lw=1, color=color[1])
    ax1.legend(loc="center right", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points * nweeks + 1, step=2))

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
        format(nhours, mdl_data.master_iteration))
    plt.close(fig1)


    fig2, ax3 = plt.subplots(figsize=(12, 8))

    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Salt Tank Level (metric ton)',
                   color=color[3])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.axhline(tank_max,
                ls=':', lw=1.75,
                color=color[4])
    plt.text(nhours / 2 - 1.5,
             tank_max + 100, 'max salt',
             color=color[4])
    ax3.plot(hours_list, hot_tank_list,
             marker='^', ms=4, label='Hot Salt',
             lw=1, color=color[0])
    ax3.plot(hours_list, cold_tank_list,
             marker='v', ms=4, label='Cold Salt',
             lw=1, color=color[1])
    ax3.legend(loc="center right", frameon=False)
    ax3.tick_params(axis='y')
    ax3.set_xticks(np.arange(0, n_time_points * nweeks + 1, step=2))

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


def create_csv_header(nhours):

    csvfile = open('results/gdp_mp_unfixed_area_{}h/results_subnlps_master_iter.csv'.format(nhours),
                   'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(
        ('Iteration',
         'TimePeriod(hr)',
         'OperationMode',
         'BoilerHeatDuty(MW)',
         'DischargeWork(MW)',
         'HXCArea',
         'HXDArea',
         'SaltHotTemp',
         'Obj($/hr)')
    )
    return csvfile


def run_pricetaker_analysis(hours_per_day=None,
                            nhours=None,
                            ndays=None,
                            nweeks=None,
                            n_time_points=None,
                            pmin=None,
                            tank_status=None,
                            tank_min=None,
                            tank_max=None):

    # Get LMP data
    lmp = _get_lmp(hours_per_day=hours_per_day, nhours=nhours)

    # Create the multiperiod model object. You can pass arguments to
    # the "process_model_func" for each time period using a dict of
    # dicts as shown here.  In this case, it is setting up empty
    # dictionaries for each time period.
    gdp_multiperiod_usc = create_gdp_multiperiod_usc_model(
        n_time_points=n_time_points,
        pmin=pmin,
        pmax=None
    )

    # Retrieve pyomo model and active process blocks
    m = gdp_multiperiod_usc.pyomo_model
    blks = gdp_multiperiod_usc.get_active_process_blocks()

    ##################################################################
    # Add logical constraints
    ##################################################################
    m.hours_set = RangeSet(0, nhours - 1)
    m.hours_set2 = RangeSet(0, nhours - 2)

    # Add constraint to save calculate charge and discharge area in a
    # global variable
    @m.Constraint(m.hours_set2)
    def constraint_charge_previous_area(m, h):
        return (
            m.blocks[h + 1].process.usc.fs.charge_area ==
            m.blocks[h].process.usc.fs.charge_area
        )

    @m.Constraint(m.hours_set2)
    def constraint_discharge_previous_area(m, h):
        return (
            m.blocks[h + 1].process.usc.fs.discharge_area ==
            m.blocks[h].process.usc.fs.discharge_area
        )

    @m.Constraint(m.hours_set)
    def constraint_discharge_hot_salt_temperature(m, h):
        return (
            m.blocks[h].process.usc.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0] ==
            m.blocks[h].process.usc.fs.hot_salt_temp
        )


    discharge_min_salt = 379 # in mton, 8MW min es turbine
    min_hot_salt = 2000
    @m.Constraint(m.hours_set)
    def _constraint_no_discharge_with_min_hot_tank(m, h):
        if h <= 2:
            b = min_hot_salt
        else:
            b = discharge_min_salt
        return (
            (m.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var * b) <=
            blks[h].usc.previous_salt_inventory_hot
        )


    # Add a minimum number charge, discharge, and no storage operation modes
    @m.Constraint(m.hours_set)
    def _constraint_min_charge(m, h):
        return sum(m.blocks[h].process.usc.fs.charge_mode_disjunct.binary_indicator_var
                   for h in m.hours_set) >= 1

    @m.Constraint(m.hours_set)
    def _constraint_min_discharge(m, h):
        return sum(m.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var
                   for h in m.hours_set) >= 1
    # @m.Constraint()
    # def _logic_constraint2_min_discharge(m):
    #     return sum(m.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var for h in m.hours_set) >= sum(m.blocks[h].process.usc.fs.charge_mode_disjunct.binary_indicator_var for h in m.hours_set) + 1

    @m.Constraint(m.hours_set)
    def _constraint_min_no_storage(m, h):
        return sum(m.blocks[h].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var
                   for h in m.hours_set) >= 1


    if tank_status == "hot_empty":
        # Add logical constraint to help reduce the alternatives to explore
        # when periodic behavior is expected
        @m.Constraint()
        def _logic_constraint_no_discharge_time0(m):
            return m.blocks[0].process.usc.fs.discharge_mode_disjunct.binary_indicator_var == 0
        @m.Constraint()
        def _logic_constraint_no_charge_at_timen(m):
            return (
                (m.blocks[0].process.usc.fs.charge_mode_disjunct.binary_indicator_var
                 + m.blocks[nhours - 1].process.usc.fs.charge_mode_disjunct.binary_indicator_var) <= 1
            )
        @m.Constraint()
        def _logic_constraint_no_storage_time0_no_charge_at_timen(m):
            return (
                (m.blocks[0].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var
                 + m.blocks[nhours - 1].process.usc.fs.charge_mode_disjunct.binary_indicator_var) <= 1
            )
    elif tank_status == "hot_full":
        @m.Constraint()
        def _logic_constraint_no_discharge_at_timen(m):
            return (
                (m.blocks[0].process.usc.fs.discharge_mode_disjunct.binary_indicator_var
                 + m.blocks[nhours - 1].process.usc.fs.discharge_mode_disjunct.binary_indicator_var) <= 1
            )

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
    # tank scenarios are included for the moletn salt tank levels and the previous tank level of the tank is based on that.
    blks[0].usc.previous_power.fix(400)


    if tank_status == "hot_empty":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_min)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_max-tank_min)
    elif tank_status == "hot_half_full":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_max/2)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_max/2)
    elif tank_status == "hot_full":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_max-tank_min)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_min)
    else:
        print("Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full")

    # Select solver
    csvfile = create_csv_header(nhours=nhours)

    opt = pyo.SolverFactory('gdpopt')
    opt.CONFIG.strategy = 'RIC'
    opt.CONFIG.OA_penalty_factor = 1e4
    opt.CONFIG.max_slack = 1e4
    # opt.CONFIG.call_after_subproblem_solve = print_model
    opt.CONFIG.call_after_subproblem_solve = (lambda a, b: print_model(
        a, b, csvfile, nweeks=nweeks, nhours=nhours, lmp=lmp, n_time_points=n_time_points))
    opt.CONFIG.mip_solver = 'gurobi_direct'
    opt.CONFIG.nlp_solver = 'ipopt'
    opt.CONFIG.tee = True
    opt.CONFIG.init_strategy = "no_init"
    opt.CONFIG.time_limit = "28000"
    opt.CONFIG.subproblem_presolve = True
    _prop_bnds_root_to_leaf_map[ExternalFunctionExpression] = lambda x, y, z: None

    # Initializing disjuncts
    print()
    print()
    print('>>Initializing disjuncts')
    for k in range(nhours):
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
    net_power = []
    hot_tank_level = []
    cold_tank_level = []
    hxc_duty = []
    hxd_duty = []
    boiler_heat_duty = []
    discharge_work = []
    for week in range(nweeks):
        print()
        print(">> Solving for week {}: {} hours of operation in {} day(s) ".
              format(week + 1, nhours, ndays))
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
            [pyo.value(blks[i].usc.salt_inventory_hot) # in mton
             for i in range(n_time_points)])
        cold_tank_level.append(
            [pyo.value(blks[i].usc.salt_inventory_cold) # in mton
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

    return (m,
            blks,
            lmp,
            net_power,
            results,
            hot_tank_level,
            cold_tank_level,
            hxc_duty,
            hxd_duty,
            boiler_heat_duty,
            discharge_work)

def print_results(m, blks, results):
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
            value(blks[c].usc.previous_salt_inventory_hot),
            value(blks[c].usc.salt_inventory_hot)))
        print(' Cold salt inventory (mton): previous: {:.4f}, current: {:.4f}'.format(
            value(blks[c].usc.previous_salt_inventory_cold),
            value(blks[c].usc.salt_inventory_cold)))
        c += 1
    # print(results)

def plot_results(m,
                 blks,
                 lmp,
                 ndays=None,
                 nweeks=None,
                 n_time_points=None,
                 net_power=None,
                 tank_max=None,
                 hot_tank_level=None,
                 cold_tank_level=None,
                 hxc_duty=None,
                 hxd_duty=None,
                 boiler_heat_duty=None,
                 discharge_work=None):


    hours = np.arange(n_time_points * nweeks)
    lmp_array = np.asarray(lmp[0:n_time_points])
    hot_tank_array = np.asarray(hot_tank_level[0:nweeks]).flatten()
    cold_tank_array = np.asarray(cold_tank_level[0:nweeks]).flatten()
    color = ['r', 'b', 'tab:green', 'k', 'tab:orange']

    # Plot molten salt tank levels for each period. First, convert
    # array to list to include hot tank level at initial period zero.
    hot_tank_array0 = value(blks[0].usc.previous_salt_inventory_hot)
    cold_tank_array0 = value(blks[0].usc.previous_salt_inventory_cold)
    hours_list = hours.tolist() + [nhours]
    hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
    cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()

    font = {'size':16}
    plt.rc('font', **font)
    fig3, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Tank Level (metric ton)',
                   color=color[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.axhline(tank_max, ls=':', lw=1.75,
                color=color[4])
    plt.text(nhours / 2 - 1.5, tank_max + 100, 'max salt',
             color=color[4])
    ax1.step(hours_list, hot_tank_list,
             marker='^', ms=4, label='Hot Salt',
             lw=1, color=color[0])
    ax1.step(hours_list, cold_tank_list,
             marker='v', ms=4, label='Cold Salt',
             lw=1, color=color[1])
    ax1.legend(loc="center right", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=2))

    ax2 = ax1.twinx()
    ax2.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax2.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', lw=1,
             color=color[2])
    ax2.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_salt_tank_level.png'.format(nhours))

    # Plot power production for each period. First, convert power array
    # to list to include the pwoer value at initial period zero.
    font = {'size':18}
    plt.rc('font', **font)
    power_array = np.asarray(net_power[0:nweeks]).flatten()
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
    plt.text(nhours / 2 - 3, max_power - 5.5, 'max plant power',
             color=color[4])
    plt.axhline(max_power, ls='-.', lw=1.75,
                color=color[4])
    ax3.step(hours_list, power_list,
             marker='o', ms=4,
             lw=1, color=color[1])
    ax3.tick_params(axis='y',
                    labelcolor=color[1])
    ax3.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=2))

    ax4 = ax3.twinx()
    ax4.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax4.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', lw=1,
             color=color[2])
    ax4.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_power.png'.format(nhours))

    # Plot charge and discharge heat exchangers heat duty values for
    # each time period. First, convert array to list to include the
    # value at period zero, which for this analysis is zero since the
    # plant is not operating.
    zero_point = True
    hxc_array = np.asarray(hxc_duty[0:nweeks]).flatten()
    hxd_array = np.asarray(hxd_duty[0:nweeks]).flatten()
    hxc_duty0 = 0
    hxc_duty_list = [hxc_duty0] + hxc_array.tolist()
    hxd_duty0 = 0
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
    plt.text(nhours / 2 - 2, min_storage_heat_duty - 6.5, 'min storage',
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
        ax5.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=2))

    ax6 = ax5.twinx()
    ax6.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax6.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', color=color[2])
    ax6.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_hxduty.png'.format(nhours))

    # Plot boiler heat duty values and discharge work for each time
    # period. First, convert arrays to lists to include the value at
    # period zero, which for this analysis is zero for both since the
    # plant is not operating.
    zero_point = True
    boiler_heat_duty_array = np.asarray(boiler_heat_duty[0:nweeks]).flatten()
    boiler_heat_duty0 = 0
    boiler_heat_duty_list = [boiler_heat_duty0] + boiler_heat_duty_array.tolist()
    discharge_work_array = np.asarray(discharge_work[0:nweeks]).flatten()
    discharge_work0 = 0
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
    ax7.set_xticks(np.arange(0, n_time_points * nweeks + 1, step=2))

    ax8 = ax7.twinx()
    ax8.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax8.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', color=color[2])
    ax8.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/optimal_boilerduty.png'.format(nhours))

    # plt.show()


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

    # Save results in a .csv file for each master iteration
    save_results = True

    # Use GDP design for charge and discharge heat exchanger from 4-12
    # disjunctions model when True. If False, use the GDP design from 4-5
    # disjunctions model.
    new_design = False

    lx = True
    if lx:
        # scaling_obj = 1e-2 # 12 hrs
        if new_design:
            scaling_obj = 1e-2
        else:
            scaling_obj = 1e-1
            scaling_cost = 1e-3
    else:
        scaling_obj = 1
    print()
    print('Scaling cost:', scaling_cost)
    print('Scaling obj:', scaling_obj)

    # Add design data from .json file
    if new_design:
        data_path = 'uscp_design_data_new_storage_design.json'
    else:
        data_path = 'uscp_design_data.json'

    with open(data_path) as design_data:
        design_data_dict = json.load(design_data)

    max_salt_amount = design_data_dict["max_salt_amount"] # in kg
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
    factor_mton = design_data_dict["factor_mton"] # factor for conversion kg to metric ton
    max_power = design_data_dict["plant_max_power"] # in MW
    pmin = design_data_dict["plant_min_power"] # in MW

    hours_per_day = 4
    ndays = 1
    nhours = hours_per_day * ndays
    nweeks = 1

    # Add number of hours per week
    n_time_points = nweeks * nhours

    tank_status = "hot_empty"
    tank_min = 1 * factor_mton # in mton
    tank_max = max_salt_amount * factor_mton # in mton

    # Create a directory to save the results for each NLP sbproblem
    # and plots
    _mkdir('results')
    _mkdir('results/gdp_mp_unfixed_area_{}h'.format(nhours))

    (m,
     blks,
     lmp,
     net_power,
     results,
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
                                               tank_min=tank_min,
                                               tank_max=tank_max)

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
                 boiler_heat_duty=boiler_heat_duty,
                 discharge_work=discharge_work)
