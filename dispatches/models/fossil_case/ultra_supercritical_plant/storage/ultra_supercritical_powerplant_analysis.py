#################################################################################
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
#################################################################################

"""

This is a simple model for an ultra-supercritical pulverized coal power plant
based on a flowsheet presented in Ref [1]: 1999 USDOE Report #DOE/FE-0400.
This model uses some of the simpler unit models from the power generation
unit model library and some of the parameters in the model,
such as feed water heater areas, overall heat
transfer coefficients, and turbine efficiencies at multiple stages
have all been estimated for a total power out of 437 MW.
Additional assumptions are as follows:
(1) The flowsheet and main steam conditions, i. e. pressure & temperature
are adopted from the aforementioned DOE report
(2) Heater unit models are used to model main steam boiler, reheater,
and condenser.
(3) Multi-stage turbines are modeled as multiple lumped single
stage turbines

updated (10/07/2021)
"""

__author__ = "Naresh Susarla & E S Rawlings"

import os
import logging

import numpy as np

# Import Pyomo libraries
from pyomo.environ import (ConcreteModel, RangeSet, TransformationFactory,
                           Constraint, Expression, Param, Var, Reals, value)
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.util import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (
    HeatExchanger,
    MomentumMixingType,
    Heater,
)
from idaes.power_generation.unit_models.helm import (
    HelmMixer,
    HelmIsentropicCompressor,
    HelmTurbineStage,
    HelmSplitter
)
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.misc import svg_tag

# Import Property Packages (IAPWS95 for Water/Steam)
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
from idaes.generic_models.properties import iapws95
from pyomo.util.infeasible import (log_infeasible_constraints,
                                    log_close_to_bounds)
logging.basicConfig(level=logging.INFO)
# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)


def usc_without_boiler_efficiency(m, solver):

    #   Solving the flowsheet and check result
    # unfix boiler flow and remove bounds
    m.fs.boiler.inlet.flow_mol.unfix()
    m.fs.boiler.inlet.flow_mol.setlb(1)
    m.fs.boiler.inlet.flow_mol.setub(None)
    m.fs.boiler.outlet.flow_mol.setlb(1)
    m.fs.boiler.outlet.flow_mol.setub(None)
    ###########################################################################

    # Main flowsheet operation data
    m.fs.num_of_years = 30
    m.CE_index = 607.5  # Chemical engineering cost index for 2019
    m.fs.coal_price = Param(
        initialize=2.11e-9,
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.operating_hours = Param(
        initialize=365 * 3600 * 24,
        doc="Number of operating hours per year")
    m.fs.fuel_cost = Var(
        initialize=1000000,
        bounds=(0, 1e15),
        doc="Fuel cost per year")  # add units

    # Modified to remove q_baseline, this now is the fuel cost (if no cooler)
    def fuel_cost_rule(b):
        return m.fs.fuel_cost == (
            m.fs.operating_hours * m.fs.coal_price *
            (m.fs.plant_heat_duty[0] * 1e6))
    m.fs.fuel_cost_eq = Constraint(rule=fuel_cost_rule)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.fuel_cost,
        m.fs.fuel_cost_eq)

    m.fs.plant_fixed_operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $/yr")
    m.fs.plant_variable_operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $/yr")

    # Add function to calculate fixed and variable operating costs in
    # the plant. Equations from "USC Cost function.pptx" sent by
    # Naresh
    def op_fixed_plant_cost_rule(b):
        return m.fs.plant_fixed_operating_cost == (
            (16657.5 * m.fs.plant_power_out[0]  # in MW
              + 6109833.3) /
            m.fs.num_of_years
        ) * (m.CE_index / 575.4)  # annualized, in $/y
    m.fs.op_fixed_plant_cost_eq = Constraint(
        rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return m.fs.plant_variable_operating_cost == (
            31754.7 * m.fs.plant_power_out[0]  # in MW
        ) * (m.CE_index / 575.4)  # in $/yr
    m.fs.op_variable_plant_cost_eq = Constraint(
        rule=op_variable_plant_cost_rule)

    m.fs.operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e15),
        doc="Operating cost in $/MWh")

    # Plant operating cost in $/MWh
    def operating_cost_rule(b):
        return (
            m.fs.operating_cost * m.fs.plant_power_out[0] ==
            (m.fs.fuel_cost
             + m.fs.plant_fixed_operating_cost
             + m.fs.plant_variable_operating_cost
             ) / (365 * 24))
    m.fs.operating_cost_eq = Constraint(
        rule=operating_cost_rule)

    m.fs.cycle_efficiency = Expression(
        expr=m.fs.plant_power_out[0] /
        m.fs.plant_heat_duty[0] * 100
    )
    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.plant_fixed_operating_cost,
        m.fs.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.plant_variable_operating_cost,
        m.fs.op_variable_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.operating_cost,
        m.fs.operating_cost_eq)
    # --------
    return m


def usc_with_boiler_efficiency(m, solver):

    #   Solving the flowsheet and check result
    # unfix boiler flow and remove bounds
    m.fs.boiler.inlet.flow_mol.unfix()
    m.fs.boiler.inlet.flow_mol.setlb(1)
    m.fs.boiler.inlet.flow_mol.setub(None)
    m.fs.boiler.outlet.flow_mol.setlb(1)
    m.fs.boiler.outlet.flow_mol.setub(None)
    ###########################################################################
    m.fs.net_power_max = 436
    m.fs.boiler_eff = Expression(
        expr=0.2143*(m.fs.plant_power_out[0]/m.fs.net_power_max)
        + 0.7357
    )

    m.fs.coal_heat_duty = Var(
        initialize=1000000,
        bounds=(0, 1e15),
        doc="Coal heat duty supplied to boiler (MW)")

    def coal_heat_duty_rule(b):
        return m.fs.coal_heat_duty * m.fs.boiler_eff == (
            m.fs.plant_heat_duty[0])
    m.fs.coal_heat_duty_eq = Constraint(rule=coal_heat_duty_rule)

    # Main flowsheet operation data
    m.fs.num_of_years = 30
    m.CE_index = 607.5  # Chemical engineering cost index for 2019
    m.fs.coal_price = Param(
        initialize=2.11e-9,
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.operating_hours = Param(
        initialize=365 * 3600 * 24,
        doc="Number of operating hours per year")
    m.fs.fuel_cost = Var(
        initialize=1000000,
        bounds=(0, 1e15),
        doc="Fuel cost per year")  # add units

    # Modified to remove q_baseline, this now is the fuel cost (if no cooler)
    def fuel_cost_rule(b):
        return m.fs.fuel_cost == (
            m.fs.operating_hours * m.fs.coal_price *
            (m.fs.coal_heat_duty * 1e6))
    m.fs.fuel_cost_eq = Constraint(rule=fuel_cost_rule)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.fuel_cost,
        m.fs.fuel_cost_eq)

    m.fs.plant_fixed_operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $/yr")
    m.fs.plant_variable_operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $/yr")

    # Add function to calculate fixed and variable operating costs in
    # the plant. Equations from "USC Cost function.pptx" sent by
    # Naresh
    def op_fixed_plant_cost_rule(b):
        return m.fs.plant_fixed_operating_cost == (
            (16657.5 * m.fs.plant_power_out[0]  # in MW
              + 6109833.3) /
            m.fs.num_of_years
        ) * (m.CE_index / 575.4)  # annualized, in $/y
    m.fs.op_fixed_plant_cost_eq = Constraint(
        rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return m.fs.plant_variable_operating_cost == (
            31754.7 * m.fs.plant_power_out[0]  # in MW
        ) * (m.CE_index / 575.4)  # in $/yr
    m.fs.op_variable_plant_cost_eq = Constraint(
        rule=op_variable_plant_cost_rule)

    m.fs.operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e15),
        doc="Operating cost in $/MWh")

    # Plant operating cost in $/MWh
    def operating_cost_rule(b):
        return (
            m.fs.operating_cost * m.fs.plant_power_out[0] ==
            (m.fs.fuel_cost
             + m.fs.plant_fixed_operating_cost
             + m.fs.plant_variable_operating_cost
             ) / (365 * 24))
    m.fs.operating_cost_eq = Constraint(
        rule=operating_cost_rule)

    m.fs.cycle_efficiency = Expression(
        expr=m.fs.plant_power_out[0] /
        m.fs.coal_heat_duty * 100
    )
    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.plant_fixed_operating_cost,
        m.fs.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.plant_variable_operating_cost,
        m.fs.op_variable_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.operating_cost,
        m.fs.operating_cost_eq)
    # --------
    return m


def build_plant_model(method=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = usc.build_plant_model()
    usc.initialize(m)

    if method == "with_efficiency":
        m = usc_with_boiler_efficiency(m)
    else:
        m = usc_with_boiler_efficiency(m)

    return m


def model_analysis(m):
    cf_list = []
    opex_list = []
    cycle_eff = []

    m.capacity_factor = Param(initialize=1, mutable=True)
    capacity_factor_list = [1, 0.9, 0.8, 0.7, .65]

    for cf in capacity_factor_list:
        m.fs.plant_power_out[0].fix(cf*436)
        solver.solve(m, tee=True, symbolic_solver_labels=True)
        cf_list.append(cf*100)
        opex_list.append(value(m.fs.operating_cost))
        cycle_eff.append(value(m.fs.cycle_efficiency))
        print('Plant Power (MW) =', value(m.fs.plant_power_out[0]))
        print('Plant Heat Duty (MW) =', value(m.fs.plant_heat_duty[0]))
        print('Plant Operating cost ($/MWh) =', value(m.fs.operating_cost))

    cf_array = np.asarray(cf_list)
    opex_array = np.asarray(opex_list)
    cycle_array = np.asarray(cycle_eff)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:green'
    ax1.set_xlabel('Capacity Factor (%)')
    ax1.set_ylabel('Operating Cost ($/MWh)', color=color)
    ax1.plot(cf_array, opex_array, marker='o', markersize=10, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'b'
    ax2 = ax1.twinx()
    # ax1.set_xlabel('Capacity Factor (%)')
    ax2.set_ylabel('Cycle Efficiency (%)', color=color)
    ax2.plot(cf_array, cycle_array, marker='o', markersize=10, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes"
    }
    solver = get_solver("ipopt", optarg)

    # Build ultra supercriticla power plant model for analysis
    method = "with_efficiency"
    m = build_plant_model(method="with_efficiency")

    model_analysis(m)
