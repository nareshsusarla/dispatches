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

"""This is a GDP model for the conceptual design of an ultra
supercritical coal-fired power plant based on a flowsheet presented in
1999 USDOE Report #DOE/FE-0400

This model uses some of the simpler unit models from the power
generation unit model library.

Some of the parameters in the model such as feed water heater areas,
overall heat transfer coefficient, turbine efficiencies at multiple
stages have all been estimated for a total power out of 437 MW.

Additional main assumptions are as follows:
(1) The flowsheet and main steam conditions, i. e. pressure &
    temperature are adopted from the aforementioned DOE report
(2) Heater unit models are used to model main steam boiler, reheater,
    and condenser.
(3) Multi-stage turbines are modeled as multiple lumped single stage
    turbines

updated (04/14/2022)
"""

__author__ = "Soraya Rawlings and Naresh Susarla"

# Import Python libraries
from math import pi
import logging

# Import Pyomo libraries
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var, SolverFactory)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)

# Import IDAES libraries
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.util import get_solver
from idaes.core.util import model_serializer as ms
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (HeatExchanger,
                                              MomentumMixingType,
                                              Heater)
from idaes.generic_models.unit_models import PressureChanger
from idaes.power_generation.unit_models.helm import (
    HelmMixer,
    HelmTurbineStage,
    HelmSplitter
)
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
from idaes.generic_models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
import idaes.core.util.unit_costing as icost
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import ultra supercritical power plant model
# from dispatches.models.fossil_case.ultra_supercritical_plant import (
#     ultra_supercritical_powerplant_mixcon as usc)
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

import solarsalt_properties

from pyomo.network.plugins import expand_arcs

from IPython import embed
logging.basicConfig(level=logging.INFO)


import json

# Open json file
with open('uscp_design_data.json') as design_data:
    design_data_dict = json.load(design_data)

    hxc_area = design_data_dict["hxc_area"] # in MW
    hxd_area = design_data_dict["hxd_area"] # in MW
    min_power = design_data_dict["plant_min_power"] # in MW
    max_power = design_data_dict["plant_max_power"] # in MW
    ramp_rate = design_data_dict["ramp_rate"]
    min_power_storage = design_data_dict["min_discharge_turbine_power"] # in MW
    max_power_storage = design_data_dict["max_discharge_turbine_power"] # in MW
    hot_salt_temp = design_data_dict["hot_salt_temperature"] # in K
    cold_salt_temp = design_data_dict["cold_salt_temperature"] # in K
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW

scaling_obj = 1e-2

# For old design data
# scaling_obj = 1e-2 # hot_empty tank scenario
# scaling_obj = 1e-3 # hot_full tank scenario

max_salt_amount = design_data_dict["max_storage_salt_amount"] * 1e-3 # in mton

def create_gdp_model(m,
                     method=None,
                     max_power=None,
                     deact_arcs_after_init=None):
    """Create flowsheet and add unit models.
    """

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()

    ###########################################################################
    #  Add data
    ###########################################################################

    # Global data
    m.max_salt_flow = 500  # in kg/s

    # Chemical engineering cost index for 2019
    m.CE_index = 607.5

    #  Operating hours
    m.number_hours_per_day = 24
    m.number_of_years = 30

    # Data in flowsheet
    m.fs.hours_per_day = Var(
        initialize=m.number_hours_per_day,
        bounds=(0, 24),
        doc='Estimated number of hours of charging per day'
    )
    # Fix number of hours
    m.fs.hours_per_day.fix(m.number_hours_per_day)

    # Define number of years over which the capital cost is annualized
    m.fs.num_of_years = Param(
        initialize=m.number_of_years,
        doc='Number of years for capital cost annualization')

    # Design of Storage Heat Exchanger: Shell-n-tube counter-flow heat
    # exchanger design parameters. Data to compute overall heat
    # transfer coefficient for the charge heat exchanger using the
    # Sieder-Tate Correlation. Parameters for tube diameter and
    # thickness assumed from the data in (2017) He et al., Energy
    # Procedia 105, 980-985
    m.fs.data_storage_hx = {
        'tube_thickness': 0.004,
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    m.fs.tube_thickness = Param(
        initialize=m.fs.data_storage_hx['tube_thickness'],
        doc='Tube thickness [m]')
    m.fs.tube_inner_dia = Param(
        initialize=m.fs.data_storage_hx['tube_inner_dia'],
        doc='Tube inner diameter [m]')
    m.fs.tube_outer_dia = Param(
        initialize=m.fs.data_storage_hx['tube_outer_dia'],
        doc='Tube outer diameter [m]')
    m.fs.k_steel = Param(
        initialize=m.fs.data_storage_hx['k_steel'],
        doc='Thermal conductivity of steel [W/mK]')
    m.fs.n_tubes = Param(
        initialize=m.fs.data_storage_hx['number_tubes'],
        doc='Number of tubes')
    m.fs.shell_inner_dia = Param(
        initialize=m.fs.data_storage_hx['shell_inner_dia'],
        doc='Shell inner diameter [m]')
    m.fs.tube_cs_area = Expression(
        expr=(pi / 4) *
        (m.fs.tube_inner_dia ** 2),
        doc="Tube cross sectional area")
    m.fs.tube_out_area = Expression(
        expr=(pi / 4) *
        (m.fs.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]")
    m.fs.shell_eff_area = Expression(
        expr=(
            (pi / 4) *
            (m.fs.shell_inner_dia ** 2) -
            m.fs.n_tubes *
            m.fs.tube_out_area),
        doc="Effective shell cross sectional area [m2]")

    m.fs.tube_dia_ratio = (m.fs.tube_outer_dia / m.fs.tube_inner_dia)
    m.fs.log_tube_dia_ratio = log(m.fs.tube_dia_ratio)

    # Data for main flowsheet operation. The q baseline_charge
    # corresponds to heat duty of a plant with no storage and
    # producing 400 MW power
    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'q_baseline_charge': 838565942.4732262,
        'solar_salt_price': 0.49,
        'hitec_salt_price': 0.93,
        'thermal_oil_price': 6.72,  # $/kg
    }
    m.fs.coal_price = Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.cooling_price = Param(
        initialize=m.data_cost['cooling_price'],
        doc='Cost of chilled water for cooler from Sieder et al. $/J')
    m.fs.q_baseline = Param(
        initialize=m.data_cost['q_baseline_charge'],
        doc='Boiler duty in Wth @ 699MW for baseline plant with no storage')
    m.fs.solar_salt_price = Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')

    # Data for salt pump
    m.data_salt_pump = {
        'FT': 1.5,
        'FM': 2.0,
        'head': 3.281*5,
        'motor_FT': 1,
        'nm': 1
    }
    m.fs.spump_FT = Param(
        initialize=m.data_salt_pump['FT'],
        doc='Pump Type Factor for vertical split case')
    m.fs.spump_FM = Param(
        initialize=m.data_salt_pump['FM'],
        doc='Pump Material Factor Stainless Steel')
    m.fs.spump_head = Param(
        initialize=m.data_salt_pump['head'],
        doc='Pump Head 5m in Ft.')
    m.fs.spump_motorFT = Param(
        initialize=m.data_salt_pump['motor_FT'],
        doc='Motor Shaft Type Factor')
    m.fs.spump_nm = Param(
        initialize=m.data_salt_pump['nm'],
        doc='Motor Shaft Type Factor')

    # Data for salt storage tank
    m.data_storage_tank = {
        'material_price': 3.5,
        'insulation_price': 235,
        'foundation_price': 1210,
        'LbyD': 0.325,
        'tank_thickness': 0.039,
        'material_density': 7800
    }
    m.fs.material_cost = Param(
        initialize=m.data_storage_tank['material_price'],
        doc='$/kg of SS316 material')
    m.fs.insulation_cost = Param(
        initialize=m.data_storage_tank['insulation_price'],
        doc='$/m2')
    m.fs.foundation_cost = Param(
        initialize=m.data_storage_tank['foundation_price'],
        doc='$/m2')
    m.fs.material_density = Param(
        initialize=m.data_storage_tank['material_density'],
        doc='Kg/m3')
    m.fs.l_by_d = Param(
        initialize=m.data_storage_tank['LbyD'],
        doc='L by D assumption for computing storage tank dimensions')
    m.fs.tank_thickness = Param(
        initialize=m.data_storage_tank['tank_thickness'],
        doc='Storage tank thickness assumed based on reference'
    )
    m.fs.hxc_salt_design_flow = Param(
        initialize=312,
        doc='Design flow of salt through hxc')
    m.fs.hxc_salt_design_density = Param(
        initialize=1937.36,
        doc='Design density of salt through hxc')

    m.fs.hxd_salt_design_flow = Param(
        initialize=362.2,
        doc='Design flow of salt through hxd')
    m.fs.hxd_salt_design_density = Param(
        initialize=1721.12,
        doc='Design density of salt through hxd')


    ###########################################################################
    # Add global variables
    ###########################################################################

    # m.fs.salt_amount = Var(
    #     initialize=max_salt_amount,
    #     doc="Solar salt purchase cost in $"
    # )
    # m.fs.salt_amount.fix()
    m.fs.salt_amount = Param(
        initialize=max_salt_amount,
        doc="Solar salt amount in mton"
    )

    m.fs.salt_storage = Var(
        bounds=(-m.max_salt_flow, m.max_salt_flow),
        initialize=1,
        doc="Hot solar salt amount for storage in kg/s"
    )
    m.fs.cooler_heat_duty = Var(
        bounds=(0, 1e3),
        initialize=1,
        doc="Cooler heat duty in MW"
    )
    m.fs.hx_pump_work = Var(
        bounds=(0, 1e3),
        initialize=1,
        doc="Pump work in charge mode in MW"
    )
    m.fs.discharge_turbine_work = Var(
        bounds=(0, 1e3),
        initialize=1,
        doc="Discharge turbine work in MW"
    )
    m.fs.energy_loss = Param(
        initialize=1.5,
        doc="Discharge energy loss in MW"
    )

    ###########################################################################
    # Add disjunction
    ###########################################################################

    m.fs.discharge_mode_disjunct = Disjunct(rule=discharge_mode_disjunct_equations)
    m.fs.charge_mode_disjunct = Disjunct(rule=charge_mode_disjunct_equations)
    m.fs.no_storage_mode_disjunct = Disjunct(rule=no_storage_mode_disjunct_equations)

    ###########################################################################
    # Add constraints
    ###########################################################################

    if deact_arcs_after_init:
        print('**^^**Arcs connecting reheater 1 to turbine 3 and BFP to FWH8 are deactivated after initialization')
    else:
        print('**^^**Arcs connecting reheater 1 to turbine 3 and BFP to FWH8 are deactivated in create_gdp_model, before initialization')
        _deactivate_arcs(m)

    _make_constraints(m, method=method, max_power=max_power)
    return m


def _make_constraints(m, method=None, max_power=None):

    m.fs.production_cons.deactivate()
    @m.fs.Constraint(m.fs.time)
    def production_cons_with_storage(b, t):
        return (
            (-1 * sum(b.turbine[p].work_mechanical[t]
                      for p in m.set_turbine)
             - b.hx_pump_work * 1e6 # in W
            ) ==
            b.plant_power_out[t] * 1e6 * (pyunits.W/pyunits.MW)
        )

    m.fs.net_power = Expression(
        expr=(m.fs.plant_power_out[0]
              + m.fs.discharge_turbine_work)
    )

    m.fs.boiler_eff = Var(initialize=0.9,
                          bounds=(0, 1),
                          doc="Boiler efficiency")
    m.fs.boiler_efficiency_eq = Constraint(
        expr=m.fs.boiler_eff == (
            0.2143 * (m.fs.net_power / max_power)
            + 0.7357
        ),
        doc="Boiler efficiency in fraction"
    )
    m.fs.coal_heat_duty = Var(
        initialize=1000,
        bounds=(0, 1e5),
        doc="Coal heat duty supplied to boiler (MW)")

    if method == "with_efficiency":
        m.fs.coal_heat_duty_eq = Constraint(
            expr=m.fs.coal_heat_duty * m.fs.boiler_eff ==
            m.fs.plant_heat_duty[0]
        )
    else:
        m.fs.coal_heat_duty_eq = Constraint(
            expr=m.fs.coal_heat_duty == m.fs.plant_heat_duty[0]
        )

    m.fs.cycle_efficiency = Var(initialize=0.4,
                                bounds=(0, 1),
                                doc="Cycle efficiency")
    m.fs.cycle_efficiency_eq = Constraint(
        expr=m.fs.cycle_efficiency * m.fs.coal_heat_duty == m.fs.net_power,
        doc="Cycle efficiency in %"
    )


def add_disjunction(m):
    """Add storage fluid selection and steam source disjunctions to the
    model
    """

    m.fs.operation_mode_disjunction = Disjunction(
        expr=[m.fs.no_storage_mode_disjunct,
              m.fs.charge_mode_disjunct,
              m.fs.discharge_mode_disjunct])

    # Expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs)

    return m


def no_storage_mode_disjunct_equations(disj):
    m = disj.model()

    # Connect cycle
    m.fs.no_storage_mode_disjunct.rh1_to_turb3 = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from reheater 1 to turbine 3"
    )
    m.fs.no_storage_mode_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    # Set global constraints that depend on the charge and discharge
    # operation modes to zero since the units do not exist in no
    # storage mode
    m.fs.no_storage_mode_disjunct.eq_salt_amount_in_storage = Constraint(
        expr=m.fs.salt_storage == 0
    )
    m.fs.no_storage_mode_disjunct.eq_cooler_heat_duty = Constraint(
        expr=m.fs.cooler_heat_duty == 0
    )
    m.fs.no_storage_mode_disjunct.eq_hx_pump_work = Constraint(
        expr=m.fs.hx_pump_work == 0
    )
    m.fs.no_storage_mode_disjunct.eq_discharge_turbine_work = Constraint(
        expr=m.fs.discharge_turbine_work == 0
    )


def charge_mode_disjunct_equations(disj):
    m = disj.model()

    # Declare units for the charge storage system. A splitter to
    # divert some steam from high pressure inlet and intermediate
    # pressure inlet to charge the storage heat exchanger. To ensure
    # the outlet of charge heat exchanger is a subcooled liquid before
    # mixing it with the plant, a cooler is added after the heat
    # exchanger. A pump, if needed, is used to increase the pressure of
    # the water to allow mixing it at a desired location within the
    # plant
    m.fs.charge_mode_disjunct.ess_hp_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_hxc", "to_turbine"],
        }
    )

    m.fs.charge_mode_disjunct.hxc = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.prop_water
            },
            "tube": {
                "property_package": m.fs.solar_salt_properties
            }
        }
    )

    m.fs.charge_mode_disjunct.cooler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    m.fs.charge_mode_disjunct.hx_pump = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.pump,
        }
    )

    m.fs.charge_mode_disjunct.recycle_mixer = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["from_bfw_out", "from_hx_pump"],
            "property_package": m.fs.prop_water,
        }
    )

    # Add constraint for the minimum pressure in recycle mixer
    m.fs.charge_mode_disjunct.recyclemixer_pressure_constraint = Constraint(
        expr=m.fs.charge_mode_disjunct.recycle_mixer.from_bfw_out_state[0].pressure ==
        m.fs.charge_mode_disjunct.recycle_mixer.mixed_state[0].pressure,
        doc="Recycle mixer outlet pressure equal to min inlet pressure")

    # Add pump pressure constraint
    m.fs.charge_mode_disjunct.constraint_hxpump_presout = Constraint(
        expr=m.fs.charge_mode_disjunct.hx_pump.outlet.pressure[0] >=
        m.main_steam_pressure * 1.1231
        # expr=m.fs.charge_mode_disjunct.hx_pump.outlet.pressure[0] ==
        # m.main_steam_pressure * 1.1231
    )

    # Add cooler outlet temperature constraint
    m.fs.charge_mode_disjunct.constraint_cooler_enth2 = Constraint(
        expr=(
            m.fs.charge_mode_disjunct.cooler.control_volume.properties_out[0].temperature <=
            (m.fs.charge_mode_disjunct.cooler.control_volume.properties_out[0].temperature_sat - 5)
        ),
        doc="Cooler outlet temperature to be subcooled"
    )

    # Equations to calculate the charge heat exchanger overall heat
    # transfer coefficient
    m.fs.charge_mode_disjunct.hxc.salt_reynolds_number = Expression(
        expr=(
            (m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0] *
             m.fs.tube_outer_dia) /
            (m.fs.shell_eff_area *
             m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number")
    m.fs.charge_mode_disjunct.hxc.salt_prandtl_number = Expression(
        expr=(
            m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].cp_specific_heat["Liq"] *
            m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] /
            m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number")
    m.fs.charge_mode_disjunct.hxc.salt_prandtl_wall = Expression(
        expr=(
            m.fs.charge_mode_disjunct.hxc.side_2.properties_out[0].cp_specific_heat["Liq"] *
            m.fs.charge_mode_disjunct.hxc.side_2.properties_out[0].dynamic_viscosity["Liq"] /
            m.fs.charge_mode_disjunct.hxc.side_2.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number at wall")
    m.fs.charge_mode_disjunct.hxc.salt_nusselt_number = Expression(
        expr=(
            0.35 *
            (m.fs.charge_mode_disjunct.hxc.salt_reynolds_number**0.6) *
            (m.fs.charge_mode_disjunct.hxc.salt_prandtl_number**0.4) *
            ((m.fs.charge_mode_disjunct.hxc.salt_prandtl_number /
              m.fs.charge_mode_disjunct.hxc.salt_prandtl_wall) ** 0.25) *
            (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")
    m.fs.charge_mode_disjunct.hxc.steam_reynolds_number = Expression(
        expr=(
            m.fs.charge_mode_disjunct.hxc.inlet_1.flow_mol[0] *
            m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].mw *
            m.fs.tube_inner_dia /
            (m.fs.tube_cs_area *
             m.fs.n_tubes *
             m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")
    m.fs.charge_mode_disjunct.hxc.steam_prandtl_number = Expression(
        expr=(
            (m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].cp_mol /
             m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].mw) *
            m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
            m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")
    m.fs.charge_mode_disjunct.hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 *
            (m.fs.charge_mode_disjunct.hxc.steam_reynolds_number**0.8) *
            (m.fs.charge_mode_disjunct.hxc.steam_prandtl_number**(0.33)) *
            ((m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
              m.fs.charge_mode_disjunct.hxc.side_1.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of charge heat exchanger
    m.fs.charge_mode_disjunct.hxc.h_salt = Expression(
        expr=(
            m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].thermal_conductivity["Liq"] *
            m.fs.charge_mode_disjunct.hxc.salt_nusselt_number /
            m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    m.fs.charge_mode_disjunct.hxc.h_steam = Expression(
        expr=(
            m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].therm_cond_phase["Vap"] *
            m.fs.charge_mode_disjunct.hxc.steam_nusselt_number /
            m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")

    # Rewrite overall heat transfer coefficient constraint to avoid
    # denominators
    m.fs.charge_mode_disjunct.constraint_hxc_ohtc = Constraint(
        expr=(
            m.fs.charge_mode_disjunct.hxc.overall_heat_transfer_coefficient[0] *
            (2 * m.fs.k_steel *
             m.fs.charge_mode_disjunct.hxc.h_steam +
             m.fs.tube_outer_dia *
             m.fs.log_tube_dia_ratio *
             m.fs.charge_mode_disjunct.hxc.h_salt *
             m.fs.charge_mode_disjunct.hxc.h_steam +
             m.fs.tube_dia_ratio *
             m.fs.charge_mode_disjunct.hxc.h_salt *
             2 * m.fs.k_steel)
        ) == (2 * m.fs.k_steel *
              m.fs.charge_mode_disjunct.hxc.h_salt *
              m.fs.charge_mode_disjunct.hxc.h_steam)
    )

    # Declare arcs to connect storage charge system to the plant
    m.fs.charge_mode_disjunct.rh1_to_esshp = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.charge_mode_disjunct.ess_hp_split.inlet,
        doc="Connection from reheater 1 to HP splitter"
    )
    m.fs.charge_mode_disjunct.esshp_to_turb3 = Arc(
        source=m.fs.charge_mode_disjunct.ess_hp_split.to_turbine,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from HP splitter to turbine 3"
    )
    m.fs.charge_mode_disjunct.esshp_to_hxc = Arc(
        source=m.fs.charge_mode_disjunct.ess_hp_split.to_hxc,
        destination=m.fs.charge_mode_disjunct.hxc.inlet_1,
        doc="Connection from HP splitter to HXC inlet 1"
    )
    m.fs.charge_mode_disjunct.hxc_to_cooler = Arc(
        source=m.fs.charge_mode_disjunct.hxc.outlet_1,
        destination=m.fs.charge_mode_disjunct.cooler.inlet,
        doc="Connection from cooler to solar charge heat exchanger"
    )
    m.fs.charge_mode_disjunct.cooler_to_hxpump = Arc(
        source=m.fs.charge_mode_disjunct.cooler.outlet,
        destination=m.fs.charge_mode_disjunct.hx_pump.inlet,
        doc="Connection from cooler to HX pump"
    )

    # Declare arcs to connect the recycle mixer to the plant
    m.fs.charge_mode_disjunct.bfp_to_recyclemix = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.charge_mode_disjunct.recycle_mixer.from_bfw_out,
        doc="Connection from BFP outlet to recycle mixer"
    )
    m.fs.charge_mode_disjunct.hxpump_to_recyclemix = Arc(
        source=m.fs.charge_mode_disjunct.hx_pump.outlet,
        destination=m.fs.charge_mode_disjunct.recycle_mixer.from_hx_pump,
        doc="Connection from HX pump to recycle mixer"
    )
    m.fs.charge_mode_disjunct.recyclemix_to_fwh8 = Arc(
        source=m.fs.charge_mode_disjunct.recycle_mixer.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from recycle mixer outlet to FWH8"
    )


    # Declare constraints to save global variables
    m.fs.charge_mode_disjunct.eq_salt_amount_in_charge_storage = Constraint(
        expr=m.fs.salt_storage == m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0]
    )
    m.fs.charge_mode_disjunct.eq_cooler_heat_duty = Constraint(
        expr=m.fs.cooler_heat_duty == (-1e-6) * m.fs.charge_mode_disjunct.cooler.heat_duty[0]
    )
    m.fs.charge_mode_disjunct.eq_hx_pump_work = Constraint(
        expr=m.fs.hx_pump_work == (1e-6) * m.fs.charge_mode_disjunct.hx_pump.control_volume.work[0]
    )
    m.fs.charge_mode_disjunct.eq_discharge_turbine_work = Constraint(
        expr=m.fs.discharge_turbine_work == 0
    )

    # m.fs.charge_mode_disjunct.eq_charge_heat_duty = Constraint(
    #     expr=m.fs.charge_mode_disjunct.hxc.heat_duty[0] * (1e-6) <= max_storage_heat_duty
    # )


def discharge_mode_disjunct_equations(disj):
    m = disj.model()

    m.fs.discharge_mode_disjunct.ess_bfp_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_hxd", "to_fwh8"],
        }
    )

    # Add discharge heat exchanger
    m.fs.discharge_mode_disjunct.hxd = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.solar_salt_properties
            },
            "tube": {
                "property_package": m.fs.prop_water
            }
        }
    )

    # Discharge heat exchanger salt and steam side constraints to
    # calculate Reynolds number, Prandtl number, and Nusselt number
    m.fs.discharge_mode_disjunct.hxd.salt_reynolds_number = Expression(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0]
            * m.fs.tube_outer_dia
            / (m.fs.shell_eff_area
               * m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.discharge_mode_disjunct.hxd.salt_prandtl_number = Expression(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].cp_specific_heat["Liq"]
            * m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].dynamic_viscosity["Liq"]
            / m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    # Assuming that the wall conditions are same as those at the outlet
    m.fs.discharge_mode_disjunct.hxd.salt_prandtl_wall = Expression(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.side_1.properties_out[0].cp_specific_heat["Liq"]
            * m.fs.discharge_mode_disjunct.hxd.side_1.properties_out[0].dynamic_viscosity["Liq"]
            / m.fs.discharge_mode_disjunct.hxd.side_1.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Wall Salt Prandtl Number"
    )
    m.fs.discharge_mode_disjunct.hxd.salt_nusselt_number = Expression(
        expr=(
            0.35 * (m.fs.discharge_mode_disjunct.hxd.salt_reynolds_number**0.6)
            * (m.fs.discharge_mode_disjunct.hxd.salt_prandtl_number**0.4)
            * ((m.fs.discharge_mode_disjunct.hxd.salt_prandtl_number
                / m.fs.discharge_mode_disjunct.hxd.salt_prandtl_wall)**0.25)
            * (2**0.2)
        ),
        doc="Solar Salt Nusslet Number from 2019, App Ener (233-234), 126"
    )
    m.fs.discharge_mode_disjunct.hxd.steam_reynolds_number = Expression(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.inlet_2.flow_mol[0]
            * m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].mw
            * m.fs.tube_inner_dia
            / (m.fs.tube_cs_area
               * m.fs.n_tubes
               * m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.discharge_mode_disjunct.hxd.steam_prandtl_number = Expression(
        expr=(
            (m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].cp_mol
             / m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].mw)
            * m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].visc_d_phase["Liq"]
            / m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.discharge_mode_disjunct.hxd.steam_nusselt_number = Expression(
        expr=(
            0.023 * (m.fs.discharge_mode_disjunct.hxd.steam_reynolds_number ** 0.8)
            * (m.fs.discharge_mode_disjunct.hxd.steam_prandtl_number ** (0.33))
            * ((m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].visc_d_phase["Liq"]
                / m.fs.discharge_mode_disjunct.hxd.side_2.properties_out[0].visc_d_phase["Vap"]
                ) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Discharge heat exchanger salt and steam side heat transfer
    # coefficients
    m.fs.discharge_mode_disjunct.hxd.h_salt = Expression(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].thermal_conductivity["Liq"]
            * m.fs.discharge_mode_disjunct.hxd.salt_nusselt_number / m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    m.fs.discharge_mode_disjunct.hxd.h_steam = Expression(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].therm_cond_phase["Liq"]
            * m.fs.discharge_mode_disjunct.hxd.steam_nusselt_number / m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    m.fs.discharge_mode_disjunct.constraint_hxd_ohtc = Constraint(
        expr=(
            m.fs.discharge_mode_disjunct.hxd.overall_heat_transfer_coefficient[0] *
            (2 * m.fs.k_steel *
             m.fs.discharge_mode_disjunct.hxd.h_steam +
             m.fs.tube_outer_dia *
             m.fs.log_tube_dia_ratio *
             m.fs.discharge_mode_disjunct.hxd.h_salt *
             m.fs.discharge_mode_disjunct.hxd.h_steam +
             m.fs.tube_dia_ratio *
             m.fs.discharge_mode_disjunct.hxd.h_salt *
             2 * m.fs.k_steel)
        ) == (2 * m.fs.k_steel *
              m.fs.discharge_mode_disjunct.hxd.h_salt *
              m.fs.discharge_mode_disjunct.hxd.h_steam),
        doc="Overall heat transfer coefficient for hxd"
    )

    m.fs.discharge_mode_disjunct.es_turbine = HelmTurbineStage(
        default={
            "property_package": m.fs.prop_water,
        }
    )

    # Reconnect reheater 1 to turbine 3 since the arc was disconnected
    # in the global model
    m.fs.discharge_mode_disjunct.rh1_to_turb3 = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.turbine[3].inlet
    )

    # Declare arcs to connect discharge heat exchanger to plant
    m.fs.discharge_mode_disjunct.bfp_to_essbfp = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.discharge_mode_disjunct.ess_bfp_split.inlet,
        doc="Connection from BFP outlet to BFP splitter"
    )
    m.fs.discharge_mode_disjunct.essbfp_to_fwh8 = Arc(
        source=m.fs.discharge_mode_disjunct.ess_bfp_split.to_fwh8,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP splitter to FWH8"
    )
    m.fs.discharge_mode_disjunct.essbfp_to_hxd = Arc(
        source=m.fs.discharge_mode_disjunct.ess_bfp_split.to_hxd,
        destination=m.fs.discharge_mode_disjunct.hxd.inlet_2,
        doc="Connection from BFP splitter to discharge heat exchanger"
    )
    m.fs.discharge_mode_disjunct.hxd_to_esturbine = Arc(
        source=m.fs.discharge_mode_disjunct.hxd.outlet_2,
        destination=m.fs.discharge_mode_disjunct.es_turbine.inlet,
        doc="Connection from discharge heat exchanger to ES turbine"
    )


    # Save the amount of salt used in the discharge heat exchanger
    m.fs.discharge_mode_disjunct.eq_salt_amount_in_discharge_storage = Constraint(
        expr=m.fs.salt_storage == -m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0]
    )

    # Fix HX pump work and cooler heat duty to zero since they do not
    # exist during discharge mode and the value is saved in a global variable
    m.fs.discharge_mode_disjunct.eq_cooler_heat_duty = Constraint(
        expr=m.fs.cooler_heat_duty == 0
    )
    m.fs.discharge_mode_disjunct.eq_hx_pump_work = Constraint(
        expr=m.fs.hx_pump_work == 0
    )
    m.fs.discharge_mode_disjunct.eq_discharge_turbine_work = Constraint(
        expr=m.fs.discharge_turbine_work == (
            (-1e-6) * m.fs.discharge_mode_disjunct.es_turbine.work[0])
    )

    m.fs.discharge_mode_disjunct.eq_discharge_heat_duty = Constraint(
        expr=((1e-6) * m.fs.discharge_mode_disjunct.hxd.heat_duty[0]
              + m.fs.energy_loss) <= max_storage_heat_duty
    )


def _deactivate_arcs(m):
    """Deactivate arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the charge heat exchanger
    for arc_s in [m.fs.rh1_to_turb3,
                  m.fs.bfp_to_fwh8]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()


def set_model_input(m):
    """Define model inputs and fixed variables or parameter values
    """

    # All the parameter values in this block, unless otherwise stated
    # explicitly, are either assumed or estimated for a total power
    # out of 437 MW

    # These inputs will also fix all necessary inputs to the model
    # i.e. the degrees of freedom = 0

    ###########################################################################
    #  Charge Heat Exchanger section                                          #
    ###########################################################################
    # Add heat exchanger area from supercritical plant model_input. For
    # conceptual design optimization, area is unfixed and optimized
    m.fs.charge_mode_disjunct.hxc.area.fix(2500)  # m2
    m.fs.discharge_mode_disjunct.hxd.area.fix(2000)  # m2

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass.fix(140)   # kg/s
    m.fs.charge_mode_disjunct.hxc.inlet_2.temperature.fix(513.15)  # K
    m.fs.charge_mode_disjunct.hxc.inlet_2.pressure.fix(101325)  # Pa

    m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass.fix(250)  # 250
    m.fs.discharge_mode_disjunct.hxd.inlet_1.temperature.fix(853.15)
    m.fs.discharge_mode_disjunct.hxd.inlet_1.pressure.fix(101325)

    # Cooler outlet enthalpy is fixed during model build to ensure the
    # inlet to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler
    # is constrained in the model
    m.fs.charge_mode_disjunct.cooler.outlet.enth_mol[0].fix(10000)
    m.fs.charge_mode_disjunct.cooler.deltaP[0].fix(0)

    # HX pump efficiecncy assumption
    m.fs.charge_mode_disjunct.hx_pump.efficiency_pump.fix(0.80)
    # m.fs.charge.hx_pump.outlet.pressure[0].fix(
    # m.main_steam_pressure * 1.1231)

    m.fs.discharge_mode_disjunct.es_turbine.ratioP.fix(0.0286)
    m.fs.discharge_mode_disjunct.es_turbine.efficiency_isentropic.fix(0.5)
    ###########################################################################
    #  ESS VHP and HP splitters                                               #
    ###########################################################################
    # The model is built for a fixed flow of steam through the
    # charger.  This flow of steam to the charger is unfixed and
    # determine during design optimization
    m.fs.charge_mode_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.1)
    m.fs.discharge_mode_disjunct.ess_bfp_split.split_fraction[0, "to_hxd"].fix(0.1)  # 0.1

    # Fix global variables
    # m.fs.hx_pump_work.fix(0)
    # m.fs.discharge_turbine_work.fix(0)


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

    """

    # Include scaling factors for solar, hitec, and thermal oil charge
    # heat exchangers
    for fluid in [m.fs.charge_mode_disjunct.hxc, m.fs.discharge_mode_disjunct.hxd]:
        iscale.set_scaling_factor(fluid.area, 1e-2)
        iscale.set_scaling_factor(
            fluid.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(fluid.shell.heat, 1e-6)
        iscale.set_scaling_factor(fluid.tube.heat, 1e-6)

    iscale.set_scaling_factor(m.fs.charge_mode_disjunct.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.discharge_mode_disjunct.es_turbine.control_volume.work, 1e-6)

    for k in [m.fs.charge_mode_disjunct.cooler]:
        iscale.set_scaling_factor(k.control_volume.heat, 1e-6)

def set_scaling_var(m):
    iscale.set_scaling_factor(m.fs.operating_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.plant_fixed_operating_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.plant_variable_operating_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.plant_capital_cost, 1e-6)

    iscale.set_scaling_factor(m.fs.salt_amount, 1e-6)
    iscale.set_scaling_factor(m.fs.salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(m.fs.salt_inventory_cold, 1e-3)
    iscale.set_scaling_factor(m.fs.previous_salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(m.fs.previous_salt_inventory_cold, 1e-3)

    iscale.set_scaling_factor(m.fs.constraint_salt_inventory_hot, 1e-3)

    # iscale.set_scaling_factor(m.fs.cooler_heat_duty, 1e-6)
    # iscale.set_scaling_factor(m.fs.hx_pump_work, 1e-6)
    # iscale.set_scaling_factor(m.fs.discharge_turbine_work, 1e-6)


def initialize(m,
               solver=None,
               deact_arcs_after_init=None,
               outlvl=idaeslog.NOTSET,
               optarg={"tol": 1e-8, "max_iter": 300}):
    """Initialize the units included in the charge model
    """
    print()
    print('>> Start initialization of charge units in ultra-supercritical plant')
    print('   {} DOFs before initialization'.format(degrees_of_freedom(m)))

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    # Include scaling factors
    iscale.calculate_scaling_factors(m)

    # Initialize all units in charge mode operation
    propagate_state(m.fs.charge_mode_disjunct.rh1_to_esshp)
    m.fs.charge_mode_disjunct.ess_hp_split.initialize(outlvl=outlvl,
                                                      optarg=solver.options)
    propagate_state(m.fs.charge_mode_disjunct.esshp_to_hxc)
    m.fs.charge_mode_disjunct.hxc.initialize(outlvl=outlvl,
                                             optarg=solver.options)

    if not deact_arcs_after_init:
        # Reinitialize and fix turbine 3 inlet since the arcs is
        # disconnected
        propagate_state(m.fs.charge_mode_disjunct.esshp_to_turb3)
        m.fs.turbine[3].inlet.fix()
        m.fs.turbine[3].initialize(outlvl=outlvl,
                                   optarg=solver.options)

    propagate_state(m.fs.charge_mode_disjunct.hxc_to_cooler)
    m.fs.charge_mode_disjunct.cooler.initialize(outlvl=outlvl,
                                                optarg=solver.options)
    propagate_state(m.fs.charge_mode_disjunct.cooler_to_hxpump)
    m.fs.charge_mode_disjunct.hx_pump.initialize(outlvl=outlvl,
                                                 optarg=solver.options)

    # Fix value of global variable
    m.fs.hx_pump_work.fix(
        m.fs.charge_mode_disjunct.hx_pump.control_volume.work[0].value * 1e-6)

    propagate_state(m.fs.charge_mode_disjunct.bfp_to_recyclemix)
    propagate_state(m.fs.charge_mode_disjunct.hxpump_to_recyclemix)
    m.fs.charge_mode_disjunct.recycle_mixer.initialize(outlvl=outlvl)

    # Initialize all units in discharge mode operation
    propagate_state(m.fs.discharge_mode_disjunct.bfp_to_essbfp)
    m.fs.discharge_mode_disjunct.ess_bfp_split.initialize(outlvl=outlvl,
                                                          optarg=solver.options)
    propagate_state(m.fs.discharge_mode_disjunct.essbfp_to_hxd)
    m.fs.discharge_mode_disjunct.hxd.initialize(outlvl=outlvl,
                                                optarg=solver.options)
    propagate_state(m.fs.discharge_mode_disjunct.hxd_to_esturbine)
    m.fs.discharge_mode_disjunct.es_turbine.initialize(outlvl=outlvl,
                                                       optarg=solver.options)
    # Fix value of global variable
    m.fs.discharge_turbine_work.fix(
        m.fs.discharge_mode_disjunct.es_turbine.work[0].value * (-1e-6))

    if not deact_arcs_after_init:
        # Reinitialize FWH8 using bfp outlet
        m.fs.fwh[8].fwh_vfrac_constraint.deactivate()
        m.fs.fwh[8].inlet_2.flow_mol.fix(m.fs.bfp.outlet.flow_mol[0])
        m.fs.fwh[8].inlet_2.enth_mol.fix(m.fs.bfp.outlet.enth_mol[0])
        m.fs.fwh[8].inlet_2.pressure.fix(m.fs.bfp.outlet.pressure[0])
        m.fs.fwh[8].initialize(outlvl=outlvl,
                               optarg=solver.options)
        m.fs.fwh[8].fwh_vfrac_constraint.activate()

    print('   {} DOFs before initialization solution'.format(degrees_of_freedom(m)))
    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)

    print("   **Solver termination for Charge Model Initialization:",
          res.solver.termination_condition)
    print('   {} DOFs after initialization solution'.format(degrees_of_freedom(m)))
    print('>> End initialization of charge units in ultra-supercritical plant')
    print()


def build_costing(m):
    """ Add cost correlations for the storage design analysis. This
    function is used to estimate the capital and operatig cost of
    integrating an energy storage system. It contains cost
    correlations to estimate the capital cost of charge heat
    exchanger, salt storage tank, molten salt pump, and salt
    inventory. Note that it does not compute the cost of the whole
    power plant.

    """

    # All the computed capital costs are annualized. The operating
    # cost is for 1 year. In addition, operating savings in terms of
    # annual coal cost are estimated based on the differential
    # reduction of coal consumption as compared to ramped baseline
    # power plant. Unless other wise stated, the cost correlations
    # used here (except IDAES costing method) are taken from 2nd
    # Edition, Product & Process Design Principles, Seider et al.

    # Fix number of tanks needed to store the storage fluid
    m.fs.no_of_tanks = Var(
        initialize=1,
        bounds=(1, 3),
        doc='No of Tank units to use cost correlations')
    m.fs.no_of_tanks.fix()

    # Fixed for now
    m.fs.storage_capital_cost = Param(
        initialize=0.407655e6,
        doc="Annualized capital cost for solar salt in $/yr")

    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.operating_hours = Expression(
        expr=365 * 3600 * m.fs.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Operating cost in $/yr")

    # Modified to remove q_baseline, this now is the fuel cost (if no cooler)
    def op_cost_rule(b):
        return m.fs.operating_cost == (
            m.fs.operating_hours * m.fs.coal_price *
            (m.fs.coal_heat_duty * 1e6)
            - (m.fs.cooling_price * m.fs.operating_hours *
               m.fs.cooler_heat_duty)
        )
    m.fs.op_cost_eq = Constraint(rule=op_cost_rule)

    ###########################################################################
    #  Annual capital and operating cost for full plant
    ###########################################################################
    # Capital cost for power plant
    m.fs.plant_capital_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Annualized capital cost for the plant in $")
    m.fs.plant_fixed_operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $/yr")
    m.fs.plant_variable_operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $/yr")

    # Add function to calculate the plant capital cost. Equations from
    # "USC Cost function.pptx" sent by Naresh
    def plant_cap_cost_rule(b):
        return m.fs.plant_capital_cost == (
            ((2688973 * m.fs.plant_power_out[0]  # in MW
              + 618968072) /
             m.fs.num_of_years
            ) * (m.CE_index / 575.4)
        )
    m.fs.plant_cap_cost_eq = Constraint(rule=plant_cap_cost_rule)

    # Add function to calculate fixed and variable operating costs in
    # the plant. Equations from "USC Cost function.pptx" sent by
    # Naresh
    def op_fixed_plant_cost_rule(b):
        return m.fs.plant_fixed_operating_cost == (
            ((16657.5 * m.fs.plant_power_out[0]  # in MW
              + 6109833.3) /
             m.fs.num_of_years
            ) * (m.CE_index / 575.4)  # annualized, in $/y
        )
    m.fs.op_fixed_plant_cost_eq = Constraint(
        rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return m.fs.plant_variable_operating_cost == (
            (31754.7 * m.fs.plant_power_out[0]  # in MW
            ) * (m.CE_index / 575.4)  # in $/yr
        )
    m.fs.op_variable_plant_cost_eq = Constraint(
        rule=op_variable_plant_cost_rule)

    return m


def initialize_with_costing(m):

    optarg = {
        "tol": 1e-8,
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)
    print()
    print('>> Start initialization of costing correlations')

    m.fs.operating_cost.fix(1)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.operating_cost,
        m.fs.op_cost_eq)

    # Initialize capital cost of power plant
    calculate_variable_from_constraint(
        m.fs.plant_capital_cost,
        m.fs.plant_cap_cost_eq)

    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.plant_fixed_operating_cost,
        m.fs.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.plant_variable_operating_cost,
        m.fs.op_variable_plant_cost_eq)

    print('   {} DOFs before cost initialization solution'.format(degrees_of_freedom(m)))
    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)
    print("   **Solver termination in cost initialization: ",
          res.solver.termination_condition)
    print('>> End initialization of costing correlations')
    print()


def calculate_bounds(m):
    m.fs.temperature_degrees = 5

    # Calculate bounds for solar salt from properties expressions
    m.fs.solar_salt_temperature_max = 853.15 + m.fs.temperature_degrees # in K
    m.fs.solar_salt_temperature_min = 513.15 - m.fs.temperature_degrees # in K
    # Note: min/max interchanged because at max temperature we obtain the min value
    m.fs.solar_salt_enthalpy_mass_max = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.solar_salt_temperature_max - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value * 0.5 * \
           (m.fs.solar_salt_temperature_max - 273.15)**2)
    )
    m.fs.solar_salt_enthalpy_mass_min = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.solar_salt_temperature_min - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value * 0.5 * \
           (m.fs.solar_salt_temperature_min - 273.15)**2)
    )

    m.fs.salt_enthalpy_mass_max = m.fs.solar_salt_enthalpy_mass_max
    m.fs.salt_enthalpy_mass_min = m.fs.solar_salt_enthalpy_mass_min

    print('   **Calculate bounds for solar salt')
    print('     Mass enthalpy max: {: >4.4f}, min: {: >4.4f}'.format(
        m.fs.solar_salt_enthalpy_mass_max, m.fs.solar_salt_enthalpy_mass_min))


def add_bounds(m):
    """Add bounds to units in charge model

    """

    calculate_bounds(m)

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W
    m.flow_max = m.main_flow * 1.5 # in mol/s
    m.flow_min = 11804 # in mol/s
    m.heat_duty_max = max_storage_heat_duty * 1e6  # in MW
    m.factor = 2
    m.flow_max_storage = 0.2 * m.flow_max
    m.flow_min_storage = 1e-3

    # Turbines
    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e10)
        m.fs.turbine[k].work.setub(0)

    # Booster
    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s

    # Turbine splitters flow
    for k in m.set_turbine_splitter:
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setub(m.flow_max)

    # Add bounds to all units in charge mode
    for unit_in_charge in [m.fs.charge_mode_disjunct]:
        # Charge heat exchanger (HXC)
        unit_in_charge.hxc.inlet_1.flow_mol.setlb(m.flow_min_storage)
        unit_in_charge.hxc.inlet_1.flow_mol.setub(m.flow_max_storage)
        unit_in_charge.hxc.inlet_2.flow_mass.setlb(m.flow_min_storage)
        unit_in_charge.hxc.inlet_2.flow_mass.setub(m.max_salt_flow)
        unit_in_charge.hxc.outlet_1.flow_mol.setlb(m.flow_min_storage)
        unit_in_charge.hxc.outlet_1.flow_mol.setub(m.flow_max_storage)
        unit_in_charge.hxc.outlet_2.flow_mass.setlb(m.flow_min_storage)
        unit_in_charge.hxc.outlet_2.flow_mass.setub(m.max_salt_flow)
        unit_in_charge.hxc.inlet_2.pressure.setlb(101320)
        unit_in_charge.hxc.inlet_2.pressure.setub(101330)
        unit_in_charge.hxc.outlet_2.pressure.setlb(101320)
        unit_in_charge.hxc.outlet_2.pressure.setub(101330)
        unit_in_charge.hxc.heat_duty.setlb(0)
        unit_in_charge.hxc.heat_duty.setub(m.heat_duty_max)
        unit_in_charge.hxc.shell.heat.setlb(-m.heat_duty_max)
        unit_in_charge.hxc.shell.heat.setub(0)
        unit_in_charge.hxc.tube.heat.setlb(0)
        unit_in_charge.hxc.tube.heat.setub(m.heat_duty_max)
        unit_in_charge.hxc.tube.properties_in[:].enthalpy_mass.setlb(
            m.fs.salt_enthalpy_mass_min / m.factor)
        unit_in_charge.hxc.tube.properties_in[:].enthalpy_mass.setub(
            m.fs.salt_enthalpy_mass_max * m.factor)
        unit_in_charge.hxc.tube.properties_out[:].enthalpy_mass.setlb(
            m.fs.salt_enthalpy_mass_min / m.factor)
        unit_in_charge.hxc.tube.properties_out[:].enthalpy_mass.setub(
            m.fs.salt_enthalpy_mass_max * m.factor)
        unit_in_charge.hxc.overall_heat_transfer_coefficient.setlb(0)
        unit_in_charge.hxc.overall_heat_transfer_coefficient.setub(10000)
        unit_in_charge.hxc.area.setlb(0)
        unit_in_charge.hxc.area.setub(6000)
        unit_in_charge.hxc.delta_temperature_in.setlb(9)
        unit_in_charge.hxc.delta_temperature_out.setlb(5)
        unit_in_charge.hxc.delta_temperature_in.setub(82)
        unit_in_charge.hxc.delta_temperature_out.setub(81)

        # HX pump and Cooler
        unit_in_charge.cooler.heat_duty.setlb(-1e12)
        unit_in_charge.cooler.heat_duty.setub(0)
        for unit_k in [unit_in_charge.hx_pump,
                       unit_in_charge.cooler]:
            unit_k.inlet.flow_mol.setlb(0)
            unit_k.inlet.flow_mol.setub(m.flow_max_storage)
            unit_k.outlet.flow_mol.setlb(0)
            unit_k.outlet.flow_mol.setub(m.flow_max_storage)
            unit_k.deltaP.setlb(0)
            unit_k.deltaP.setub(1e10)
        unit_in_charge.hx_pump.work_mechanical[0].setlb(0)
        unit_in_charge.hx_pump.work_mechanical[0].setub(1e8)
        unit_in_charge.hx_pump.ratioP.setlb(0)
        unit_in_charge.hx_pump.ratioP.setub(100)
        unit_in_charge.hx_pump.work_fluid[0].setlb(0)
        unit_in_charge.hx_pump.work_fluid[0].setub(1e8)
        unit_in_charge.hx_pump.efficiency_pump[0].setlb(0)
        unit_in_charge.hx_pump.efficiency_pump[0].setub(1)

        # HP splitter
        unit_in_charge.ess_hp_split.to_hxc.flow_mol[:].setlb(0)
        unit_in_charge.ess_hp_split.to_hxc.flow_mol[:].setub(m.flow_max_storage)
        unit_in_charge.ess_hp_split.to_turbine.flow_mol[:].setlb(0)
        unit_in_charge.ess_hp_split.to_turbine.flow_mol[:].setub(m.flow_max)
        unit_in_charge.ess_hp_split.split_fraction[0.0, "to_hxc"].setlb(0)
        unit_in_charge.ess_hp_split.split_fraction[0.0, "to_hxc"].setub(1)
        unit_in_charge.ess_hp_split.split_fraction[0.0, "to_turbine"].setlb(0)
        unit_in_charge.ess_hp_split.split_fraction[0.0, "to_turbine"].setub(1)
        unit_in_charge.ess_hp_split.inlet.flow_mol[:].setlb(0)
        unit_in_charge.ess_hp_split.inlet.flow_mol[:].setub(m.flow_max)

        # Recycle mixer
        unit_in_charge.recycle_mixer.from_bfw_out.flow_mol.setlb(0)
        unit_in_charge.recycle_mixer.from_bfw_out.flow_mol.setub(m.flow_max)
        unit_in_charge.recycle_mixer.from_hx_pump.flow_mol.setlb(0)
        unit_in_charge.recycle_mixer.from_hx_pump.flow_mol.setub(m.flow_max_storage)
        unit_in_charge.recycle_mixer.outlet.flow_mol.setlb(0)
        unit_in_charge.recycle_mixer.outlet.flow_mol.setub(m.flow_max)


    # Add bounds to all units in discharge mode
    for unit_in_discharge in [m.fs.discharge_mode_disjunct]:
        # Discharge heat exchanger (HXD)
        unit_in_discharge.hxd.inlet_1.flow_mass.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.inlet_1.flow_mass.setub(m.max_salt_flow)
        unit_in_discharge.hxd.inlet_2.flow_mol.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.inlet_2.flow_mol.setub(m.flow_max_storage)
        unit_in_discharge.hxd.outlet_1.flow_mass.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.outlet_1.flow_mass.setub(m.max_salt_flow)
        unit_in_discharge.hxd.outlet_2.flow_mol.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.outlet_2.flow_mol.setub(m.flow_max_storage)
        unit_in_discharge.hxd.inlet_1.pressure.setlb(101320)
        unit_in_discharge.hxd.inlet_1.pressure.setub(101330)
        unit_in_discharge.hxd.outlet_1.pressure.setlb(101320)
        unit_in_discharge.hxd.outlet_1.pressure.setub(101330)
        unit_in_discharge.hxd.heat_duty.setlb(0)
        unit_in_discharge.hxd.heat_duty.setub(m.heat_duty_max)
        unit_in_discharge.hxd.tube.heat.setlb(0)
        unit_in_discharge.hxd.tube.heat.setub(m.heat_duty_max)
        unit_in_discharge.hxd.shell.heat.setlb(-m.heat_duty_max)
        unit_in_discharge.hxd.shell.heat.setub(0)
        unit_in_discharge.hxd.shell.properties_in[:].enthalpy_mass.setlb(
            m.fs.salt_enthalpy_mass_min / m.factor)
        unit_in_discharge.hxd.shell.properties_in[:].enthalpy_mass.setub(
            m.fs.salt_enthalpy_mass_max * m.factor)
        unit_in_discharge.hxd.shell.properties_out[:].enthalpy_mass.setlb(
            m.fs.salt_enthalpy_mass_min / m.factor)
        unit_in_discharge.hxd.shell.properties_out[:].enthalpy_mass.setub(
            m.fs.salt_enthalpy_mass_max * m.factor)
        unit_in_discharge.hxd.overall_heat_transfer_coefficient.setlb(0)
        unit_in_discharge.hxd.overall_heat_transfer_coefficient.setub(10000)
        unit_in_discharge.hxd.area.setlb(0)
        unit_in_discharge.hxd.area.setub(6000)
        unit_in_discharge.hxd.delta_temperature_in.setlb(5)
        unit_in_discharge.hxd.delta_temperature_out.setlb(10)
        unit_in_discharge.hxd.delta_temperature_in.setub(300)
        unit_in_discharge.hxd.delta_temperature_out.setub(300)

        # BFP splitter
        unit_in_discharge.ess_bfp_split.inlet.flow_mol[:].setlb(0)
        unit_in_discharge.ess_bfp_split.inlet.flow_mol[:].setub(m.flow_max)
        unit_in_discharge.ess_bfp_split.to_hxd.flow_mol[:].setlb(0)
        unit_in_discharge.ess_bfp_split.to_hxd.flow_mol[:].setub(m.flow_max_storage)
        unit_in_discharge.ess_bfp_split.to_fwh8.flow_mol[:].setlb(0)
        unit_in_discharge.ess_bfp_split.to_fwh8.flow_mol[:].setub(m.flow_max)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_hxd"].setlb(0)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_hxd"].setub(1)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_fwh8"].setlb(0)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_fwh8"].setub(1)

        # ES Turbine
        unit_in_discharge.es_turbine.inlet.flow_mol[:].setlb(0)
        unit_in_discharge.es_turbine.inlet.flow_mol[:].setub(m.flow_max_storage)
        unit_in_discharge.es_turbine.outlet.flow_mol[:].setlb(0)
        unit_in_discharge.es_turbine.outlet.flow_mol[:].setub(m.flow_max_storage)
        unit_in_discharge.es_turbine.deltaP.setlb(-1e10)
        unit_in_discharge.es_turbine.deltaP.setub(1e10)
        unit_in_discharge.es_turbine.work.setlb(-1e12)
        unit_in_discharge.es_turbine.work.setub(0)
        unit_in_discharge.es_turbine.efficiency_isentropic.setlb(0)
        unit_in_discharge.es_turbine.efficiency_isentropic.setub(1)
        unit_in_discharge.es_turbine.ratioP.setlb(0)
        unit_in_discharge.es_turbine.ratioP.setub(100)
        unit_in_discharge.es_turbine.efficiency_mech.setlb(0)
        unit_in_discharge.es_turbine.efficiency_mech.setub(1)
        unit_in_discharge.es_turbine.shaft_speed.setlb(0)
        unit_in_discharge.es_turbine.shaft_speed.setub(1000)



def main(method=None,
         max_power=None,
         load_init_file=None,
         path_init_file=None,
         deact_arcs_after_init=None):

    if load_init_file:
        # Build ultra-supercritical plant model and initialize it
        m = usc.build_plant_model()
        print()
        print('>> Start initialization of ultra-supercritical plant base model')
        usc.initialize(m)
        print('>> End initialization of ultra-supercritical plant base model')
        print()

        # Create a flowsheet, add properties, unit models, and arcs
        m = create_gdp_model(m,
                             method=method,
                             max_power=max_power,
                             deact_arcs_after_init=deact_arcs_after_init)

        # Set required inputs to the model to have a square problem for
        # initialization
        set_model_input(m)

        # Add scaling factors
        set_scaling_factors(m)

        # Initialize the model with a sequential initialization and custom
        # routines
        initialize(m, deact_arcs_after_init=deact_arcs_after_init)

        # Add cost correlations
        m = build_costing(m)

        # Initialize using .json file (with bounds)
        print('>>>>>>>>>> ***Initializing model using .json file: {}'.format(path_init_file))
        ms.from_json(m, fname=path_init_file)

    else:
        # Build ultra-supercritical plant model and initialize it
        m = usc.build_plant_model()
        print()
        print('>> Start initialization of ultra-supercritical plant base model')
        usc.initialize(m)
        print('>> End initialization of ultra-supercritical plant base model')

        # Create a flowsheet, add properties, unit models, and arcs
        m = create_gdp_model(m,
                             method=method,
                             max_power=max_power,
                             deact_arcs_after_init=deact_arcs_after_init)

        # Set required inputs to the model to have a square problem for
        # initialization
        set_model_input(m)

        # Add scaling factors
        set_scaling_factors(m)

        # Initialize the model with a sequential initialization and custom
        # routines
        initialize(m, deact_arcs_after_init=deact_arcs_after_init)

        # Add cost correlations
        m = build_costing(m)
        # print('DOF after costing: ', degrees_of_freedom(m))

        # Initialize costing
        initialize_with_costing(m)

        # Calculate and store initialization file
        print()
        print('>> Create and store initialization file in {}'.format(path_init_file))
        ms.to_json(m, fname=path_init_file)

    # Add bounds
    add_bounds(m)

    # Add disjunctions
    add_disjunction(m)

    if deact_arcs_after_init:
        # Deactivate arcs
        _deactivate_arcs(m)

    return m


def print_results(m, results):

    m.fs.condenser_mix.makeup.display()

    print('================================')
    print()
    print("***************** Optimization Results ******************")
    print('Revenue ($/h): {:.4f}'.format(
        value(m.fs.revenue)))
    print('Hot Previous Salt Inventory (kg): {:.4f}'.format(
        value(m.fs.previous_salt_inventory_hot)))
    print('Hot Salt Inventory (kg): {:.4f}'.format(
        value(m.fs.salt_inventory_hot)))
    print('Cold Previous Salt Inventory (kg): {:.4f}'.format(
        value(m.fs.previous_salt_inventory_cold)))
    print('Cold Salt Inventory (kg): {:.4f}'.format(
        value(m.fs.salt_inventory_cold)))
    print('Salt Amount (kg): {:.4f}'.format(
        value(m.fs.salt_amount)))
    print('Salt to storage (kg): {:.4f}'.format(
        value(m.fs.salt_storage)))
    print()
    print("***************** Costing Results ******************")
    print('Obj (M$/year): {:.4f}'.format(value(m.obj) / scaling_obj))
    print('Plant capital cost (M$/y): {:.4f}'.format(
        value(m.fs.plant_capital_cost) * 1e-6))
    print('Plant fixed operating costs (M$/y): {:.4f}'.format(
        value(m.fs.plant_fixed_operating_cost) * 1e-6))
    print('Plant variable operating costs (M$/y): {:.4f}'.format(
        value(m.fs.plant_variable_operating_cost) * 1e-6))
    print('Operating Cost (Fuel) ($/h): {:.4f}'.format(
        value(m.fs.operating_cost)/(365*24)))
    print('Storage Capital Cost ($/h): {:.4f}'.format(
        value(m.fs.storage_capital_cost)/(365*24)))
    print('')
    print("***************** Power Plant Operation ******************")
    print('')
    print('Net Power (MW): {:.4f}'.format(
        value(m.fs.net_power)))
    print('Plant Power (MW): {:.4f}'.format(
        value(m.fs.plant_power_out[0])))
    print('Discharge turbine power (MW) [ES turbine Power]: {:.4f} [{:.4f}]'.format(
        value(m.fs.discharge_turbine_work),
        value(m.fs.discharge_mode_disjunct.es_turbine.work_mechanical[0]) * (-1e-6)))
    print('HX pump work (MW): {:.4f}'.format(
        value(m.fs.hx_pump_work) * 1e-6))
        # value(m.fs.hx_pump.control_volume.work[0]) * 1e-6))
    print('Boiler feed water flow (mol/s): {:.4f}'.format(
        value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler duty (MW_th): {:.4f}'.format(
        value((m.fs.boiler.heat_duty[0]
               + m.fs.reheater[1].heat_duty[0]
               + m.fs.reheater[2].heat_duty[0])
              * 1e-6)))
    print('Cooling duty (MW_th): {:.4f}'.format(
        value(m.fs.cooler_heat_duty) * 1e-6))
    print('Makeup water flow: {:.4f}'.format(
        value(m.fs.condenser_mix.makeup.flow_mol[0])))
    print()
    print('Boiler efficiency (%): {:.4f}'.format(
        value(m.fs.boiler_eff) * 100))
    print('Cycle efficiency (%): {:.4f}'.format(
        value(m.fs.cycle_efficiency) * 100))
    print()
    if m.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
        print("***************** Charge Heat Exchanger (HXC) ******************")
        print('HXC area (m2): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.area)))
        print('HXC Salt flow (kg/s): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0])))
        print('HXC Salt temperature in (K): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.inlet_2.temperature[0])))
        print('HXC Salt temperature out (K): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.outlet_2.temperature[0])))
        print('HXC Steam flow to storage (mol/s): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.inlet_1.flow_mol[0])))
        print('HXC Water temperature in (K): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].temperature)))
        print('HXC Steam temperature out (K): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.side_1.properties_out[0].temperature)))
        print('HXC Delta temperature at inlet (K): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.delta_temperature_in[0])))
        print('HXC Delta temperature at outlet (K): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.delta_temperature_out[0])))
        print('Cooling duty (MW_th): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.cooler.heat_duty[0]) * 1e-6))
        print('')
    elif m.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
        print("*************** Discharge Heat Exchanger (HXD) ****************")
        print('')
        print('HXD area (m2): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.area)))
        print('HXD Salt flow (kg/s): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0])))
        print('HXD Salt temperature in (K): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0])))
        print('HXD Salt temperature out (K): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.outlet_1.temperature[0])))
        print('HXD Steam flow to storage (mol/s): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.inlet_2.flow_mol[0])))
        print('HXD Water temperature in (K): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].temperature)))
        print('HXD Steam temperature out (K): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.side_2.properties_out[0].temperature)))
        print('HXD Delta temperature at inlet (K): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.delta_temperature_in[0])))
        print('HXD Delta temperature at outlet (K): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.delta_temperature_out[0])))
        print('ES Turbine work (MW): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.es_turbine.work[0]) * -1e-6))
        print('')

    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')


def run_nlps(m,
             solver=None,
             operation_mode=None):
    """This function fixes the indicator variables of the disjuncts so to
    solve NLP problems

    """
    
    print()
    print('>>> You are solving an NLP problem by fixing the operation disjuncts!')
    if operation_mode == "charge":
        print('           ** Solving for charge mode')
        m.fs.charge_mode_disjunct.binary_indicator_var.fix(1)
        m.fs.discharge_mode_disjunct.binary_indicator_var.fix(0)
        m.fs.no_storage_mode_disjunct.binary_indicator_var.fix(0)
    elif operation_mode == "discharge":
        print('           ** Solving for discharge mode')
        m.fs.charge_mode_disjunct.binary_indicator_var.fix(0)
        m.fs.discharge_mode_disjunct.binary_indicator_var.fix(1)
        m.fs.no_storage_mode_disjunct.binary_indicator_var.fix(0)
    elif operation_mode == "no_storage":
        print('           ** Solving for no storage mode')
        m.fs.charge_mode_disjunct.binary_indicator_var.fix(0)
        m.fs.discharge_mode_disjunct.binary_indicator_var.fix(0)
        m.fs.no_storage_mode_disjunct.binary_indicator_var.fix(1)
    else:
        print('<(x.x)> Unrecognized operation mode! Try charge, discharge, or no_storage')
    print()
    print()
    print('>>> You are solving NLP model with fixed operation mode using GDPopt')
    print('    {} DOFs before solving NLP model '.format(degrees_of_freedom(m)))

    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    print("The degrees of freedom after gdp transformation ",
          degrees_of_freedom(m))

    results = solver.solve(
        m,
        tee=True,
        symbolic_solver_labels=True,
        options={
            "linear_solver": "ma27",
            "max_iter": 100
        }
    )

    log_close_to_bounds(m)
    log_infeasible_constraints(m)

    print_results(m, results)

    return m, results

def run_gdp(m):

    print('>>> You are solving GDP model using GDPopt')
    print('    {} DOFs before solving GDP model '.format(degrees_of_freedom(m)))

        # Solve the design optimization model

    opt = SolverFactory('gdpopt')
    opt.CONFIG.strategy = 'RIC'
    opt.CONFIG.OA_penalty_factor = 1e4
    opt.CONFIG.max_slack = 1e4
    opt.CONFIG.call_after_subproblem_solve = print_model
    opt.CONFIG.mip_solver = 'gurobi_direct'
    opt.CONFIG.nlp_solver = 'ipopt'
    opt.CONFIG.tee = True
    opt.CONFIG.init_strategy = "no_init"
    opt.CONFIG.time_limit = "2400"

    results = opt.solve(
        m,
        tee=True,
        nlp_solver_args=dict(
            tee=True,
            symbolic_solver_labels=True,
            options={
                "linear_solver": "ma27",
                "max_iter": 100
            }
        )
    )

    print_results(m, results)
    # print_reports(m)

    return results


def print_reports(m):

    print('')
    for unit_k in [m.fs.boiler, m.fs.reheater[1],
                   m.fs.reheater[2],
                   m.fs.bfp, m.fs.bfpt,
                   m.fs.booster,
                   m.fs.condenser_mix,
                   m.fs.charge.hxc]:
        unit_k.display()

    for k in RangeSet(11):
        m.fs.turbine[k].report()
    for k in RangeSet(11):
        m.fs.turbine[k].display()
    for j in RangeSet(9):
        m.fs.fwh[j].report()
    for j in m.set_fwh_mixer:
        m.fs.fwh_mixer[j].display()


def print_model(nlp_model, nlp_data):

    print('       ___________________________________________')
    if nlp_model.fs.charge_mode_disjunct.indicator_var.value == 1:
        print('        Charge mode is selected')
        print('         HXC heat duty (MW): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6))
        print('         HXC Salt flow (kg/s): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0])))
        print('         HXC Salt temperature in (K): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.inlet_2.temperature[0])))
        print('         HXC Salt temperature out (K): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.outlet_2.temperature[0])))
        print('         HXC Delta temperature at inlet (K): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.delta_temperature_in[0])))
        print('         HXC Delta temperature at outlet (K): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.delta_temperature_out[0])))
    elif nlp_model.fs.discharge_mode_disjunct.indicator_var.value == 1:
        print('        Discharge mode is selected')
        print('         HXD heat duty (MW): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6))
        print('         HXD Salt flow (kg/s): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0])))
        print('         HXD Salt temperature in (K): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0])))
        print('         HXD Salt temperature out (K): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.outlet_1.temperature[0])))
        print('         HXD Delta temperature at inlet (K): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.delta_temperature_in[0])))
        print('         HXD Delta temperature at outlet (K): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.delta_temperature_out[0])))
        print('         ES turbine work (MW): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.es_turbine.work_mechanical[0]) * 1e-6))
    elif nlp_model.fs.no_storage_mode_disjunct.indicator_var.value == 1:
        print('        No storage mode is selected')
    else:
        print('        No other operation alternative!')

    print()
    print('        Obj (M$/year): {:.4f}'.format(value(nlp_model.obj) / scaling_obj))
    print('        Net Power (MW): {:.4f}'.format(value(nlp_model.fs.net_power)))
    print('        Plant Power (MW): {:.4f}'.format(value(nlp_model.fs.plant_power_out[0])))
    print('        Discharge turbine work (MW): {:.4f}'.format(
        value(nlp_model.fs.discharge_turbine_work)))
    print('        HX pump work (MW): {:.4f}'.format(
        value(nlp_model.fs.hx_pump_work)))
    print('        Cooling duty (MW_th): {:.4f}'.format(
        value(nlp_model.fs.cooler_heat_duty)))
    print('        Boiler efficiency (%): {:.4f}'.format(value(nlp_model.fs.boiler_eff) * 100))
    print('        Cycle efficiency (%): {:.4f}'.format(value(nlp_model.fs.cycle_efficiency) * 100))
    print('        Hot Previous Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.previous_salt_inventory_hot)))
    print('        Cold Previous Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.previous_salt_inventory_cold)))
    print('        Hot salt to storage (kg/s) [mton]: {:.4f} [{:.4f}]'.format(
        value(nlp_model.fs.salt_storage),
        value(nlp_model.fs.salt_storage) * 3600 * 1e-3))
    print('        Hot Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.salt_inventory_hot)))
    print('        Cold Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.salt_inventory_cold)))

    print('       ___________________________________________')

    log_close_to_bounds(nlp_model)
    # log_infeasible_constraints(nlp_model)


def model_analysis(m,
                   solver=None,
                   power=None,
                   max_power=None,
                   tank_scenario=None,
                   fix_power=None,
                   operation_mode=None,
                   method=None,
                   deact_arcs_after_init=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    if fix_power:
        m.fs.power_demand_eq = Constraint(
            expr=m.fs.net_power == power
        )
    else:
        m.fs.plant_power_min = Constraint(
            expr=m.fs.plant_power_out[0] >= min_power
        )
        m.fs.plant_power_max = Constraint(
            expr=m.fs.plant_power_out[0] <= max_power
        )
        m.fs.storage_power_min = Constraint(
            expr=m.fs.discharge_mode_disjunct.es_turbine.work[0] * (-1e-6) >= min_power_storage
        )

    # Fix/unfix boiler data
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s

    # Unfix data fixed during initialization
    m.fs.charge_mode_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.discharge_mode_disjunct.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()

    # Unfix global variables
    m.fs.hx_pump_work.unfix()
    m.fs.discharge_turbine_work.unfix()
    m.fs.operating_cost.unfix()
    if not deact_arcs_after_init:
        m.fs.turbine[3].inlet.unfix()
        m.fs.fwh[8].inlet_2.unfix()

    for salt_hxc in [m.fs.charge_mode_disjunct.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()
        salt_hxc.area.unfix()

    for salt_hxd in [m.fs.discharge_mode_disjunct.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()
        salt_hxd.area.unfix()

    for unit in [m.fs.charge_mode_disjunct.cooler]:
        unit.inlet.unfix()
    m.fs.charge_mode_disjunct.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers design
    m.fs.charge_mode_disjunct.hxc.area.fix(hxc_area) # in m2
    m.fs.charge_mode_disjunct.hxc.outlet_2.temperature[0].fix(hot_salt_temp)
    m.fs.discharge_mode_disjunct.hxd.area.fix(hxd_area) # in m2
    m.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0].fix(hot_salt_temp)
    m.fs.discharge_mode_disjunct.hxd.outlet_1.temperature[0].fix(cold_salt_temp)


    # Add salt inventory variables
    max_inventory = 1e7 * 1e-3 # in mton
    m.fs.previous_salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the beginning of the hour (or time period) in mton"
        )
    m.fs.salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the end of the hour (or time period) in mton"
        )
    m.fs.previous_salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the beginning of the hour (or time period) in mton"
        )
    m.fs.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the end of the hour (or time period) in mton"
        )

    @m.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return b.salt_inventory_hot == (
            b.previous_salt_inventory_hot
            + (3600 * m.fs.salt_storage) * 1e-3 # in mton
        )

    @m.fs.Constraint(doc="Maximum previous salt inventory at any time")
    def constraint_salt_previous_inventory(b):
        return b.salt_amount == (
            b.salt_inventory_hot
            + b.salt_inventory_cold
        )

    # Fix the previous salt inventory based on the tank scenario
    min_tank = 1 * 1e-3 # in mton
    max_tank = max_salt_amount - min_tank # in mton
    if tank_scenario == "hot_empty":
        m.fs.previous_salt_inventory_hot.fix(min_tank)
        m.fs.previous_salt_inventory_cold.fix(max_tank)
    elif tank_scenario == "hot_half_full":
        m.fs.previous_salt_inventory_hot.fix(max_tank / 2)
        m.fs.previous_salt_inventory_cold.fix(max_tank / 2)
    elif tank_scenario == "hot_full":
        m.fs.previous_salt_inventory_hot.fix(max_tank)
        m.fs.previous_salt_inventory_cold.fix(min_tank)
    else:
        print('Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full')

    # Add LMP data
    m.fs.lmp = Var(
        m.fs.time,
        domain=Reals,
        initialize=80,
        doc="Hourly LMP in $/MWh"
        )

    # Fix LMP data according to the case we want to solve. When
    # solving GDP model, a random value is selected
    if operation_mode == "charge":
        m_chg.fs.lmp[0].fix(22.9684)
    elif operation_mode == "discharge":
        m_chg.fs.lmp[0].fix(200)
    elif operation_mode == "no_storage":
        m_chg.fs.lmp[0].fix(50)
    else:
        m_chg.fs.lmp[0].fix(22.9684)
        print('   **Use fixed LMP signal value of {} $/MWh'.format(
            value(m.fs.lmp[0])))
        print()


    m.fs.revenue = Expression(
        expr=(m.fs.lmp[0] * m.fs.net_power),
        doc="Revenue function in $/h assuming 1 hr operation"
    )

    set_scaling_var(m)

    # Objective function: total costs
    m.obj = Objective(
        expr=(
            m.fs.revenue
            - ((m.fs.operating_cost
                + m.fs.plant_fixed_operating_cost
                + m.fs.plant_variable_operating_cost) / (365 * 24))
            # - ((m.fs.storage_capital_cost
            #     + m.fs.plant_capital_cost)/ (365 * 24))
        ) * scaling_obj,
        sense=maximize
    )

    if operation_mode is not None:
        # Solve NLP problem with fix operation mode disjunct
        run_nlps(m,
                 solver=solver,
                 operation_mode=operation_mode)
    else:
        # Solve using GDPopt
        run_gdp(m)


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # How to run this model:
    #   load_init_file: Set to True if you wish to initialize using a .json file and
    #                   indicate the path of the .json file in path_init_file [WIP]
    #   fix_power:      Select True if you want to fix the power output of the plant.
    #                   If True, then provide the power value in power_demand
    #   method:         Select between "with_efficiency" or "without_efficiency"
    #   tank_scenario:  Select the initial value for the salt tank levels:
    #                   "hot_empty", "hot_full", "hot_half_full" (hot refers to hot salt)
    #   operation_mode: Select None if you want to solve the GDP formulation (GDPopt solver)
    #                   If you wish to solve for one mode, select an operation mode and the
    #                   respective NLP problem is solved. The modes are: "charge", "discharge",
    #                   or "no_storage"
    #  deact_arcs_after_init: Set to True if you wish to deactivate the arcs that are
    #                         connecting reheater 1 to turbine 3 and bfp to FWH8 after
    #                         initialization. If False, the arcs are deactivated in
    #                         create_gdp_model and turbine 3 and FWH8 inlets are fixed during
    #                         initialization.

    power_demand = 400
    load_init_file = False
    path_init_file = 'initialized_usc_storage_gdp_mp.json'
    fix_power = False
    method = "with_efficiency"
    tank_scenario = "hot_empty"
    operation_mode = None
    deact_arcs_after_init = True # when False, cost initialization takes about 20 sec more

    m_chg = main(method=method,
                 max_power=max_power,
                 load_init_file=load_init_file,
                 path_init_file=path_init_file,
                 deact_arcs_after_init=deact_arcs_after_init)

    m = model_analysis(m_chg,
                       solver,
                       power=power_demand,
                       max_power=max_power,
                       tank_scenario=tank_scenario,
                       fix_power=fix_power,
                       operation_mode=operation_mode,
                       method=method,
                       deact_arcs_after_init=deact_arcs_after_init)
