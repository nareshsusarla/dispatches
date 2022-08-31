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
    and condenser.  (3) Multi-stage turbines are modeled as multiple
    lumped single stage turbines

updated (12/20/2021)
"""

__author__ = "Naresh Susarla and Soraya Rawlings"

# Import Python libraries
from math import pi
import logging
# Import Pyomo libraries
# import os
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var, SolverFactory)
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.infeasible import (log_infeasible_constraints,
                                    log_close_to_bounds)

# Import IDAES libraries
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (HeatExchanger,
                                              MomentumMixingType,
                                              Heater)
import idaes.core.util.unit_costing as icost
from pyomo.gdp import Disjunct, Disjunction

# Import IDAES Libraries
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
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import ultra supercritical power plant model
# from dispatches.models.fossil_case.ultra_supercritical_plant import (
#     ultra_supercritical_powerplant_mixcon as usc)
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
import solarsalt_properties

from pyomo.network.plugins import expand_arcs

from IPython import embed
logging.basicConfig(level=logging.INFO)


def create_charge_model(m):
    """Create flowsheet and add unit models.
    """

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()


    ###########################################################################
    #  Data
    ###########################################################################

    # Chemical engineering cost index for 2019
    m.CE_index = 607.5

    #  Operating hours
    m.number_hours_per_day = 6
    m.number_of_years = 30

    m.fs.hours_per_day = Var(
        initialize=m.number_hours_per_day,
        bounds=(0, 12),
        doc='Estimated number of hours of charging per day'
    )
    # Fix number of hours of discharging to 6
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

    ###########################################################################
    # Add global variables
    ###########################################################################

    m.fs.previous_salt_inventory = Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1e12),
        doc="Salt inventory at the beginning of the hour (or time period), kg"
        )
    m.fs.salt_inventory = Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1e12),
        doc="Salt inventory at the end of the hour (or time period), kg"
    )
    m.fs.salt_amount = Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=100,
        bounds=(0, 1e12),
        doc=""
    )
    m.fs.cooler_heat_duty = Var(
        m.fs.time,
        bounds=(-1e12, 0),
        initialize=1
    )
    m.fs.hx_pump_work = Var(
        m.fs.time,
        bounds=(0, 1e12),
        initialize=1
    )
    m.fs.discharge_turbine_work = Var(
        m.fs.time,
        bounds=(-1e12, 0),
        initialize=1
    )
    ###########################################################################
    # Add disjunction
    ###########################################################################

    m.fs.no_storage_mode_disjunct = Disjunct(rule=no_storage_mode_disjunct_equations)
    m.fs.charge_mode_disjunct = Disjunct(rule=charge_mode_disjunct_equations)
    m.fs.discharge_mode_disjunct = Disjunct(rule=discharge_mode_disjunct_equations)

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    # _deactivate_arcs(m)

    return m


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

    # Close plant cycle
    m.fs.no_storage_mode_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2
    )
    m.fs.no_storage_mode_disjunct.rh1_to_turb3 = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.turbine[3].inlet
    )

    # Add global constraints
    m.fs.no_storage_mode_disjunct.eq_salt_amount_in_storage = Constraint(
        expr=m.fs.salt_amount[0] == 0
    )
    m.fs.no_storage_mode_disjunct.eq_cooler_heat_duty = Constraint(
        expr=m.fs.cooler_heat_duty[0] == 0
    )
    m.fs.no_storage_mode_disjunct.eq_hx_pump_work = Constraint(
        expr=m.fs.hx_pump_work[0] == 0
    )
    m.fs.no_storage_mode_disjunct.eq_discharge_turbine_work = Constraint(
        expr=m.fs.discharge_turbine_work[0] == 0
    )

    # Add an initial value for salt inventory
    m.fs.no_storage_mode_disjunct.prev_salt_inventory_constraint = Constraint(
        expr=m.fs.previous_salt_inventory[0] == 0
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
    @m.fs.charge_mode_disjunct.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        return (
            m.fs.charge_mode_disjunct.hxc.overall_heat_transfer_coefficient[t] *
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

    # Fix the charge heat exchanger heat duty
    m.fs.charge_mode_disjunct.hxc_heat_duty = Constraint(
        expr=m.fs.charge_mode_disjunct.hxc.heat_duty[0] == 150*1e6
    )

    # Declare constraints to save global variables
    m.fs.charge_mode_disjunct.eq_salt_amount_in_storage = Constraint(
        expr=m.fs.salt_amount[0] == (
            m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0] * 3600
        )
    )
    m.fs.charge_mode_disjunct.eq_hx_pump_work = Constraint(
        expr=m.fs.hx_pump_work[0] == m.fs.charge_mode_disjunct.hx_pump.control_volume.work[0]
    )
    m.fs.charge_mode_disjunct.eq_discharge_turbine_work = Constraint(
        expr=m.fs.discharge_turbine_work[0] == 0
    )
    m.fs.charge_mode_disjunct.prev_salt_inventory_constraint = Constraint(
        expr=m.fs.previous_salt_inventory[0] == 1
    )


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

    @m.fs.discharge_mode_disjunct.hxd.Constraint(m.fs.time,
                                                 doc="Overall heat transfer coefficient for hxd")
    def constraint_hxd_ohtc(b, t):
        return (
            m.fs.discharge_mode_disjunct.hxd.overall_heat_transfer_coefficient[t] *
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
              m.fs.discharge_mode_disjunct.hxd.h_steam)

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

    # Fix discharge heat duty (in W)
    m.fs.discharge_mode_disjunct.hxd_heat_duty = Constraint(
        expr=m.fs.discharge_mode_disjunct.hxd.heat_duty[0] == 148.5*1e6
    )

    # Add an initial amount of salt to the discharge heat exchanger
    m.fs.discharge_mode_disjunct.prev_salt_inventory_constraint = Constraint(
        expr=m.fs.previous_salt_inventory[0] == 6500000
        # expr=m.fs.previous_salt_inventory[0] == 8500000
    )

    # Save the amount of salt used in the discharge heat exchanger
    m.fs.discharge_mode_disjunct.eq_salt_amount_in_storage = Constraint(
        expr=m.fs.salt_amount[0] == m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0] *
        3600
    )

    # Fix HX pump work and cooler heat duty to zero since they do not
    # exist in this mode and the value is saved in a global variable
    m.fs.discharge_mode_disjunct.eq_cooler_heat_duty = Constraint(
        expr=m.fs.cooler_heat_duty[0] == 0
    )
    m.fs.discharge_mode_disjunct.eq_hx_pump_work = Constraint(
        expr=m.fs.hx_pump_work[0] == 0
    )

    m.fs.discharge_mode_disjunct.eq_discharge_turbine_work = Constraint(
        expr=m.fs.discharge_turbine_work[0] == m.fs.discharge_mode_disjunct.es_turbine.work[0]
    )


def _deactivate_arcs(m):
    """Deactivate arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the charge heat exchanger
    for arc_s in [m.fs.bfp_to_fwh8,
                  m.fs.rh1_to_turb3]:
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
    m.fs.discharge_mode_disjunct.es_turbine.efficiency_isentropic.fix(0.8)
    ###########################################################################
    #  ESS VHP and HP splitters                                               #
    ###########################################################################
    # The model is built for a fixed flow of steam through the
    # charger.  This flow of steam to the charger is unfixed and
    # determine during design optimization
    m.fs.charge_mode_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.1)
    m.fs.discharge_mode_disjunct.ess_bfp_split.split_fraction[0, "to_hxd"].fix(0.1)  # 0.1


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


def initialize(m, solver=None, outlvl=idaeslog.NOTSET,
               optarg={"tol": 1e-8, "max_iter": 300}):
    """Initialize the units included in the charge model
    """

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
    propagate_state(m.fs.charge_mode_disjunct.hxc_to_cooler)
    m.fs.charge_mode_disjunct.cooler.initialize(outlvl=outlvl,
                                                optarg=solver.options)
    propagate_state(m.fs.charge_mode_disjunct.cooler_to_hxpump)
    m.fs.charge_mode_disjunct.hx_pump.initialize(outlvl=outlvl,
                                                 optarg=solver.options)
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

    print('DOFs before init solution =', degrees_of_freedom(m))
    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)

    print("Charge Model Initialization = ",
          res.solver.termination_condition)
    print("***************   Charge Model Initialized   ********************")


def build_costing(m, solver=None, optarg={"tol": 1e-8, "max_iter": 300}):
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


    #  Solar salt charge and discharge heat exchangers costing is
    # estimated using the IDAES costing method with default options,
    # i.e. a U-tube heat exchanger, stainless steel material, and a
    # tube length of 12ft. Refer to costing documentation to change
    # any of the default options. Purchase cost of heat exchanger has
    # to be annualized when used
    for salt_hxc in [m.fs.charge_mode_disjunct.hxc,
                     m.fs.discharge_mode_disjunct.hxd]:
        salt_hxc.get_costing()
        salt_hxc.costing.CE_index = m.CE_index
        # Initialize Solar heat exchanger costing correlations
        icost.initialize(salt_hxc.costing)

    ###########################################################################
    #  Capital cost for charge heat exchanger and related variables
    ###########################################################################

    # Salt inventory for charge operation mode
    m.fs.charge_mode_disjunct.salt_amount = Expression(
        expr=(m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0] *
              m.fs.hours_per_day * 3600),
        doc="Total Solar salt inventory flow in kg per s"
    )
    m.fs.charge_mode_disjunct.salt_purchase_cost = Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Solar salt purchase cost in $"
    )
    def charge_solar_salt_purchase_cost_rule(b):
        return (m.fs.charge_mode_disjunct.salt_purchase_cost *
                m.fs.num_of_years == (
                    m.fs.charge_mode_disjunct.salt_amount *
                    m.fs.solar_salt_price))
    m.fs.charge_mode_disjunct.salt_purchase_cost_eq = Constraint(
        rule=charge_solar_salt_purchase_cost_rule)

    # Initialize charge and discharge solar cost correlations
    calculate_variable_from_constraint(
        m.fs.charge_mode_disjunct.salt_purchase_cost,
        m.fs.charge_mode_disjunct.salt_purchase_cost_eq)

    # --------------------------------------------
    #  Water pump
    # --------------------------------------------
    # The water pump is added as hx_pump before sending the feed water
    # to the discharge heat exchanger in order to increase its
    # pressure.  The outlet pressure of hx_pump is set based on its
    # return point in the flowsheet Purchase cost of hx_pump has to be
    # annualized when used
    m.fs.charge_mode_disjunct.hx_pump.get_costing(
        Mat_factor="stain_steel",
        mover_type="compressor",
        compressor_type="centrifugal",
        driver_mover_type="electrical_motor",
        pump_type="centrifugal",
        pump_type_factor='1.4',
        pump_motor_type_factor='open'
        )
    m.fs.charge_mode_disjunct.hx_pump.costing.CE_index = m.CE_index

    # Initialize HX pump cost correlation
    icost.initialize(m.fs.charge_mode_disjunct.hx_pump.costing)

    # --------------------------------------------
    #  Salt-pump costing
    # --------------------------------------------
    # The salt pump is not explicitly modeled. Thus, the IDAES cost
    # method is not used for this equipment at this time.  The primary
    # purpose of the salt pump is to move molten salt and not to
    # change the pressure. Thus the pressure head is computed assuming
    # that the salt is moved on an average of 5m linear distance.

    #  Charge solar salt-pump costing
    m.fs.charge_mode_disjunct.spump_Qgpm = Expression(
        expr=(
            m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].flow_mass *
            264.17 * 60 /
            (m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].density["Liq"])
        ),
        doc="Conversion of solar salt flow mass to vol flow [gal per min]"
    )
    m.fs.charge_mode_disjunct.dens_lbft3 = Expression(
        expr=(m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].density["Liq"]
              * 0.062428),
        doc="pump size factor"
    )  # density in lb per ft3
    m.fs.charge_mode_disjunct.spump_sf = Expression(
        expr=(m.fs.charge_mode_disjunct.spump_Qgpm
              * (m.fs.spump_head ** 0.5)),
        doc="Pump size factor"
    )
    m.fs.charge_mode_disjunct.pump_CP = Expression(
        expr=(
            m.fs.spump_FT * m.fs.spump_FM *
            exp(9.2951
                - 0.6019 * log(m.fs.charge_mode_disjunct.spump_sf)
                + 0.0519 * ((log(m.fs.charge_mode_disjunct.spump_sf))**2))
        ),
        doc="Salt pump base (purchase) cost in $"
    )

    # Costing motor
    m.fs.charge_mode_disjunct.spump_np = Expression(
        expr=(
            -0.316
            + 0.24015 * log(m.fs.charge_mode_disjunct.spump_Qgpm)
            - 0.01199 * ((log(m.fs.charge_mode_disjunct.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump horse power"
    )
    m.fs.charge_mode_disjunct.motor_pc = Expression(
        expr=(
            (m.fs.charge_mode_disjunct.spump_Qgpm *
             m.fs.spump_head *
             m.fs.charge_mode_disjunct.dens_lbft3) /
            (33000 * m.fs.charge_mode_disjunct.spump_np *
             m.fs.spump_nm)
        ),
        doc="Motor power consumption in horsepower"
    )
    # Motor purchase cost
    m.fs.charge_mode_disjunct.log_motor_pc = log(m.fs.charge_mode_disjunct.motor_pc)
    m.fs.charge_mode_disjunct.motor_CP = Expression(
        expr=(
            m.fs.spump_motorFT *
            exp(
                5.4866
                + 0.13141 * m.fs.charge_mode_disjunct.log_motor_pc
                + 0.053255 * (m.fs.charge_mode_disjunct.log_motor_pc**2)
                + 0.028628 * (m.fs.charge_mode_disjunct.log_motor_pc**3)
                - 0.0035549 * (m.fs.charge_mode_disjunct.log_motor_pc**4)
            )
        ),
        doc="Salt Pump's Motor Base Cost in $"
    )
    # Pump and motor purchase cost pump
    m.fs.charge_mode_disjunct.spump_purchase_cost = Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Salt pump and motor purchase cost in $"
    )
    def charge_solar_spump_purchase_cost_rule(b):
        return (
            m.fs.charge_mode_disjunct.spump_purchase_cost *
            m.fs.num_of_years == (m.fs.charge_mode_disjunct.pump_CP
                                  + m.fs.charge_mode_disjunct.motor_CP) *
            (m.CE_index / 394)  # used to be multiplied by 2 to include discharge but was changed by esrawli to 1
        )
    m.fs.charge_mode_disjunct.spump_purchase_cost_eq = Constraint(
        rule=charge_solar_spump_purchase_cost_rule)

    calculate_variable_from_constraint(
        m.fs.charge_mode_disjunct.spump_purchase_cost,
        m.fs.charge_mode_disjunct.spump_purchase_cost_eq)

    # --------------------------------------------
    #  Solar salt storage tank costing: vertical vessel
    # --------------------------------------------
    # Tank size and dimension computation
    m.fs.charge_mode_disjunct.tank_volume = Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank w/20% excess capacity")
    m.fs.charge_mode_disjunct.tank_surf_area = Var(
        initialize=1000,
        bounds=(1, 6000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge_mode_disjunct.tank_diameter = Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge_mode_disjunct.tank_height = Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank [m]")

    # Computing tank volume - jfr: editing to include 20% margin
    def charge_solar_tank_volume_rule(b):
        return (
            m.fs.charge_mode_disjunct.tank_volume *
            m.fs.charge_mode_disjunct.hxc.side_2.properties_in[0].density["Liq"] ==
            m.fs.charge_mode_disjunct.salt_amount * 1.10
        )
    m.fs.charge_mode_disjunct.tank_volume_eq = Constraint(
        rule=charge_solar_tank_volume_rule)

    # Compute surface area of tank: surf area of sides + top surf area
    # base area is accounted in foundation costs
    def charge_solar_tank_surf_area_rule(b):
        return (
            m.fs.charge_mode_disjunct.tank_surf_area == (
                pi * m.fs.charge_mode_disjunct.tank_diameter *
                m.fs.charge_mode_disjunct.tank_height)
            + (pi * m.fs.charge_mode_disjunct.tank_diameter**2) / 4
        )
    m.fs.charge_mode_disjunct.tank_surf_area_eq = Constraint(
        rule=charge_solar_tank_surf_area_rule)

    # Computing diameter for an assumed L by D
    def charge_solar_tank_diameter_rule(b):
        return (
            m.fs.charge_mode_disjunct.tank_diameter == (
                (4 * (m.fs.charge_mode_disjunct.tank_volume /
                      m.fs.no_of_tanks) /
                 (m.fs.l_by_d * pi)) ** (1 / 3))
        )
    m.fs.charge_mode_disjunct.tank_diameter_eq = Constraint(
        rule=charge_solar_tank_diameter_rule)

    # Computing height of tank
    def charge_solar_tank_height_rule(b):
        return m.fs.charge_mode_disjunct.tank_height == (
            m.fs.l_by_d * m.fs.charge_mode_disjunct.tank_diameter)
    m.fs.charge_mode_disjunct.tank_height_eq = Constraint(
        rule=charge_solar_tank_height_rule)

    # Initialize tanks design correlations
    calculate_variable_from_constraint(
        m.fs.charge_mode_disjunct.tank_volume,
        m.fs.charge_mode_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.charge_mode_disjunct.tank_diameter,
        m.fs.charge_mode_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.charge_mode_disjunct.tank_height,
        m.fs.charge_mode_disjunct.tank_height_eq)

    # A dummy pyomo block for salt storage tank is declared for costing
    # The diameter and length for this tank is assumed
    # based on a number of tank (see the above for m.fs.no_of_tanks)
    # Costing for each vessel designed above
    # m.fs.charge.salt_tank = pyo.Block()
    m.fs.charge_mode_disjunct.solar_costing = Block()

    m.fs.charge_mode_disjunct.solar_costing.tank_material_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge_mode_disjunct.solar_costing.tank_insulation_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge_mode_disjunct.solar_costing.tank_foundation_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    def charge_rule_tank_material_cost(b):
        return m.fs.charge_mode_disjunct.solar_costing.tank_material_cost == (
            m.fs.material_cost *
            m.fs.material_density *
            m.fs.charge_mode_disjunct.tank_surf_area *
            m.fs.tank_thickness
        )
    m.fs.charge_mode_disjunct.solar_costing.eq_tank_material_cost = \
        Constraint(rule=charge_rule_tank_material_cost)

    def charge_rule_tank_insulation_cost(b):
        return (
            m.fs.charge_mode_disjunct.solar_costing.tank_insulation_cost == (
                m.fs.insulation_cost *
                m.fs.charge_mode_disjunct.tank_surf_area))
    m.fs.charge_mode_disjunct.solar_costing.eq_tank_insulation_cost = \
        Constraint(rule=charge_rule_tank_insulation_cost)

    def charge_rule_tank_foundation_cost(b):
        return (
            m.fs.charge_mode_disjunct.solar_costing.tank_foundation_cost == (
                m.fs.foundation_cost *
                pi * m.fs.charge_mode_disjunct.tank_diameter**2 / 4))
    m.fs.charge_mode_disjunct.solar_costing.eq_tank_foundation_cost = \
        Constraint(rule=charge_rule_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.charge_mode_disjunct.solar_costing.total_tank_cost = Expression(
        expr=m.fs.charge_mode_disjunct.solar_costing.tank_material_cost
        + m.fs.charge_mode_disjunct.solar_costing.tank_foundation_cost
        + m.fs.charge_mode_disjunct.solar_costing.tank_insulation_cost
    )

    # --------------------------------------------
    # Total annualized capital cost for solar salt
    # --------------------------------------------
    # Capital cost var at flowsheet level to handle the salt capital
    # cost depending on the salt selected.
    m.fs.charge_mode_disjunct.capital_cost = Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Annualized capital cost for solar salt")

    # Annualize capital cost for the solar salt
    def charge_solar_cap_cost_rule(b):
        return m.fs.charge_mode_disjunct.capital_cost == (
            m.fs.charge_mode_disjunct.salt_purchase_cost
            + m.fs.charge_mode_disjunct.spump_purchase_cost
            + (m.fs.charge_mode_disjunct.hxc.costing.purchase_cost
               + m.fs.charge_mode_disjunct.hx_pump.costing.purchase_cost
               + m.fs.no_of_tanks *
               m.fs.charge_mode_disjunct.solar_costing.total_tank_cost)
            / m.fs.num_of_years
        )
    m.fs.charge_mode_disjunct.cap_cost_eq = Constraint(
        rule=charge_solar_cap_cost_rule)


    ###########################################################################
    #  Capital cost for discharge heat exchanger and related variables
    ###########################################################################
    # Salt inventory for discharge operation mode
    m.fs.discharge_mode_disjunct.salt_amount = Expression(
        expr=(m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0] *
              m.fs.hours_per_day * 3600),
        doc="Total Solar salt inventory flow in kg per s"
    )
    m.fs.discharge_mode_disjunct.salt_purchase_cost = Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Solar salt purchase cost in $"
    )
    def discharge_solar_salt_purchase_cost_rule(b):
        return (m.fs.discharge_mode_disjunct.salt_purchase_cost *
                m.fs.num_of_years == (
                    m.fs.discharge_mode_disjunct.salt_amount *
                    m.fs.solar_salt_price))
    m.fs.discharge_mode_disjunct.salt_purchase_cost_eq = Constraint(
        rule=discharge_solar_salt_purchase_cost_rule)

    # Initialize discharge solar cost correlations
    calculate_variable_from_constraint(
        m.fs.discharge_mode_disjunct.salt_purchase_cost,
        m.fs.discharge_mode_disjunct.salt_purchase_cost_eq)

    # --------------------------------------------
    #  Salt-pump costing
    # --------------------------------------------
    # The salt pump is not explicitly modeled. Thus, the IDAES cost
    # method is not used for this equipment at this time.  The primary
    # purpose of the salt pump is to move molten salt and not to
    # change the pressure. Thus the pressure head is computed assuming
    # that the salt is moved on an average of 5m linear distance.

    #  Charge solar salt-pump costing
    m.fs.discharge_mode_disjunct.spump_Qgpm = Expression(
        expr=(
            (m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].flow_mass) *
            264.17 * 60 /
            (m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].density["Liq"])
            ),
        doc="Conversion of solar salt flow mass to vol flow [gal per min]"
    )
    m.fs.discharge_mode_disjunct.dens_lbft3 = Expression(
        expr=(
            (m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].density["Liq"])
            * 0.062428),
        doc="pump size factor"
    )  # density in lb per ft3
    m.fs.discharge_mode_disjunct.spump_sf = Expression(
        expr=(m.fs.discharge_mode_disjunct.spump_Qgpm
              * (m.fs.spump_head ** 0.5)),
        doc="Pump size factor"
    )
    m.fs.discharge_mode_disjunct.pump_CP = Expression(
        expr=(
            m.fs.spump_FT * m.fs.spump_FM *
            exp(
                9.2951
                - 0.6019 * log(m.fs.discharge_mode_disjunct.spump_sf)
                + 0.0519 * ((log(m.fs.discharge_mode_disjunct.spump_sf))**2)
            )
        ),
        doc="Salt pump base (purchase) cost in $"
    )

    # Costing motor
    m.fs.discharge_mode_disjunct.spump_np = Expression(
        expr=(
            -0.316
            + 0.24015 * log(m.fs.discharge_mode_disjunct.spump_Qgpm)
            - 0.01199 * ((log(m.fs.discharge_mode_disjunct.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump horse power"
    )
    m.fs.discharge_mode_disjunct.motor_pc = Expression(
        expr=(
            (m.fs.discharge_mode_disjunct.spump_Qgpm *
             m.fs.spump_head *
             m.fs.discharge_mode_disjunct.dens_lbft3) /
            (33000 * m.fs.discharge_mode_disjunct.spump_np *
             m.fs.spump_nm)
        ),
        doc="Motor power consumption in horsepower"
    )

    # Motor purchase cost
    m.fs.discharge_mode_disjunct.log_motor_pc = log(m.fs.discharge_mode_disjunct.motor_pc)
    m.fs.discharge_mode_disjunct.motor_CP = Expression(
        expr=(
            m.fs.spump_motorFT *
            exp(
                5.4866
                + 0.13141 * m.fs.discharge_mode_disjunct.log_motor_pc
                + 0.053255 * (m.fs.discharge_mode_disjunct.log_motor_pc**2)
                + 0.028628 * (m.fs.discharge_mode_disjunct.log_motor_pc**3)
                - 0.0035549 * (m.fs.discharge_mode_disjunct.log_motor_pc**4)
            )
        ),
        doc="Salt Pump's Motor Base Cost in $"
    )

    # Pump and motor purchase cost pump
    m.fs.discharge_mode_disjunct.spump_purchase_cost = Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Salt pump and motor purchase cost in $"
    )
    def discharge_solar_spump_purchase_cost_rule(b):
        return (
            m.fs.discharge_mode_disjunct.spump_purchase_cost *
            m.fs.num_of_years == (m.fs.discharge_mode_disjunct.pump_CP
                                  + m.fs.discharge_mode_disjunct.motor_CP) *
            (m.CE_index / 394)  # used to be multiplied by 2 to include discharge but was changed by esrawli to 1
        )
    m.fs.discharge_mode_disjunct.spump_purchase_cost_eq = Constraint(
        rule=discharge_solar_spump_purchase_cost_rule)

    # Initialize charge and discharge cost correlation
    calculate_variable_from_constraint(
        m.fs.discharge_mode_disjunct.spump_purchase_cost,
        m.fs.discharge_mode_disjunct.spump_purchase_cost_eq)

    # --------------------------------------------
    #  Solar salt storage tank costing: vertical vessel
    # --------------------------------------------
    # Discharge tank size and dimension computation
    m.fs.discharge_mode_disjunct.tank_volume = Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank w/20% excess capacity")
    m.fs.discharge_mode_disjunct.tank_surf_area = Var(
        initialize=1000,
        bounds=(1, 6000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.discharge_mode_disjunct.tank_diameter = Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.discharge_mode_disjunct.tank_height = Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank [m]")

    # Compute tank volume
    def discharge_solar_tank_volume_rule(b):
        return (
            m.fs.discharge_mode_disjunct.tank_volume *
            m.fs.discharge_mode_disjunct.hxd.side_1.properties_in[0].density["Liq"] ==
            m.fs.discharge_mode_disjunct.salt_amount * 1.10
        )
    m.fs.discharge_mode_disjunct.tank_volume_eq = Constraint(
        rule=discharge_solar_tank_volume_rule)

    # Compute surface area of tank: surf area of sides + top surf area
    # base area is accounted in foundation costs
    def discharge_solar_tank_surf_area_rule(b):
        return (
            m.fs.discharge_mode_disjunct.tank_surf_area == (
                pi * m.fs.discharge_mode_disjunct.tank_diameter *
                m.fs.discharge_mode_disjunct.tank_height)
            + (pi * m.fs.discharge_mode_disjunct.tank_diameter**2) / 4
        )
    m.fs.discharge_mode_disjunct.tank_surf_area_eq = Constraint(
        rule=discharge_solar_tank_surf_area_rule)

    # Computing diameter for an assumed L by D
    def discharge_solar_tank_diameter_rule(b):
        return (
            m.fs.discharge_mode_disjunct.tank_diameter == (
                (4 * (m.fs.discharge_mode_disjunct.tank_volume /
                      m.fs.no_of_tanks) /
                 (m.fs.l_by_d * pi)) ** (1 / 3))
        )
    m.fs.discharge_mode_disjunct.tank_diameter_eq = Constraint(
        rule=discharge_solar_tank_diameter_rule)

    # Computing height of tank
    def discharge_solar_tank_height_rule(b):
        return m.fs.discharge_mode_disjunct.tank_height == (
            m.fs.l_by_d * m.fs.discharge_mode_disjunct.tank_diameter)
    m.fs.discharge_mode_disjunct.tank_height_eq = Constraint(
        rule=discharge_solar_tank_height_rule)

    calculate_variable_from_constraint(
        m.fs.discharge_mode_disjunct.tank_volume,
        m.fs.discharge_mode_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.discharge_mode_disjunct.tank_diameter,
        m.fs.discharge_mode_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.discharge_mode_disjunct.tank_height,
        m.fs.discharge_mode_disjunct.tank_height_eq)

    # A dummy pyomo block for salt storage tank is declared for costing
    # The diameter and length for this tank is assumed
    # based on a number of tank (see the above for m.fs.no_of_tanks)
    # Costing for each vessel designed above
    m.fs.discharge_mode_disjunct.solar_costing = Block()

    m.fs.discharge_mode_disjunct.solar_costing.tank_material_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.discharge_mode_disjunct.solar_costing.tank_insulation_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.discharge_mode_disjunct.solar_costing.tank_foundation_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def discharge_rule_tank_material_cost(b):
        return m.fs.discharge_mode_disjunct.solar_costing.tank_material_cost == (
            m.fs.material_cost *
            m.fs.material_density *
            m.fs.discharge_mode_disjunct.tank_surf_area *
            m.fs.tank_thickness
        )
    m.fs.discharge_mode_disjunct.solar_costing.eq_tank_material_cost = \
        Constraint(rule=discharge_rule_tank_material_cost)

    def discharge_rule_tank_insulation_cost(b):
        return (
            m.fs.discharge_mode_disjunct.solar_costing.tank_insulation_cost == (
                m.fs.insulation_cost *
                m.fs.discharge_mode_disjunct.tank_surf_area))

    m.fs.discharge_mode_disjunct.solar_costing.eq_tank_insulation_cost = \
        Constraint(rule=discharge_rule_tank_insulation_cost)

    def discharge_rule_tank_foundation_cost(b):
        return (
            m.fs.discharge_mode_disjunct.solar_costing.tank_foundation_cost == (
                m.fs.foundation_cost *
                pi * m.fs.discharge_mode_disjunct.tank_diameter**2 / 4))
    m.fs.discharge_mode_disjunct.solar_costing.eq_tank_foundation_cost = \
        Constraint(rule=discharge_rule_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.discharge_mode_disjunct.solar_costing.total_tank_cost = Expression(
        expr=m.fs.discharge_mode_disjunct.solar_costing.tank_material_cost
        + m.fs.discharge_mode_disjunct.solar_costing.tank_foundation_cost
        + m.fs.discharge_mode_disjunct.solar_costing.tank_insulation_cost
    )

    # --------------------------------------------
    # Total annualized capital cost for solar salt
    # --------------------------------------------
    # Capital cost var at flowsheet level to handle the salt capital
    # cost depending on the salt selected.
    m.fs.discharge_mode_disjunct.capital_cost = Var(
        initialize=1000000,
        bounds=(0, 1e10),
        doc="Annualized capital cost for solar salt")

    # Annualize capital cost for the solar salt
    def discharge_solar_cap_cost_rule(b):
        return m.fs.discharge_mode_disjunct.capital_cost == (
            m.fs.discharge_mode_disjunct.salt_purchase_cost
            + m.fs.discharge_mode_disjunct.spump_purchase_cost
            + (m.fs.discharge_mode_disjunct.hxd.costing.purchase_cost
               + m.fs.no_of_tanks *
               m.fs.discharge_mode_disjunct.solar_costing.total_tank_cost)
            / m.fs.num_of_years
        )
    m.fs.discharge_mode_disjunct.cap_cost_eq = Constraint(
        rule=discharge_solar_cap_cost_rule)


    ###########################################################################
    #  Total annual capital cost
    ###########################################################################
    # Capital cost var at global level to handle the salt from the
    # selected operation mode
    m.fs.storage_capital_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Annualized capital cost for solar salt")

    m.fs.no_storage_mode_disjunct.capital_cost_eq_constraint = Constraint(
        expr=m.fs.storage_capital_cost == 0
    )
    m.fs.charge_mode_disjunct.capital_cost_eq_constraint = Constraint(
        expr=m.fs.storage_capital_cost == m.fs.charge_mode_disjunct.capital_cost
    )
    m.fs.discharge_mode_disjunct.capital_cost_eq_constraint = Constraint(
        expr=m.fs.storage_capital_cost == m.fs.discharge_mode_disjunct.capital_cost
    )

    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.operating_hours = Expression(
        expr=365 * 3600 * m.fs.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.operating_cost = Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Operating cost")  # add units

    # Modified to remove q_baseline, this now is the fuel cost (if no cooler)
    def op_cost_rule(b):
        return m.fs.operating_cost == (
            m.fs.operating_hours * m.fs.coal_price *
            (m.fs.plant_heat_duty[0] * 1e6)
            - (m.fs.cooling_price * m.fs.operating_hours *
               m.fs.cooler_heat_duty[0])
        )
    m.fs.op_cost_eq = Constraint(rule=op_cost_rule)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.operating_cost,
        m.fs.op_cost_eq)

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
            (2688973 * m.fs.plant_power_out[0]  # in MW
             + 618968072) /
            m.fs.num_of_years
        ) * (m.CE_index / 575.4)
    m.fs.plant_cap_cost_eq = Constraint(rule=plant_cap_cost_rule)

    # Initialize capital cost of power plant
    calculate_variable_from_constraint(
        m.fs.plant_capital_cost,
        m.fs.plant_cap_cost_eq)

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

    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.plant_fixed_operating_cost,
        m.fs.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.plant_variable_operating_cost,
        m.fs.op_variable_plant_cost_eq)

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)
    print("Cost Initialization = ",
          res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
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
    print()
    print('************ Calculate bounds for solar salt')
    print('Enthalpy_mass max: {: >4.4f}'.format(
        m.fs.solar_salt_enthalpy_mass_max))
    print('Enthalpy_mass min: {: >4.4f}'.format(
        m.fs.solar_salt_enthalpy_mass_min))
    print()


def add_bounds(m):
    """Add bounds to units in charge model

    """

    calculate_bounds(m)

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W

    m.flow_max = m.main_flow * 3  # in mol/s
    # m.salt_flow_max = 500  # in kg/s
    m.salt_flow_max = 1000  # in kg/s
    m.fs.heat_duty_max = 200e6  # in MW
    m.factor = 2

    # Add bounds to global variables and units
    m.fs.plant_power_out[0].setlb(300)
    m.fs.plant_power_out[0].setub(700)

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
        unit_in_charge.hxc.inlet_1.flow_mol.setlb(0)
        unit_in_charge.hxc.inlet_1.flow_mol.setub(0.2 * m.flow_max)
        unit_in_charge.hxc.inlet_2.flow_mass.setlb(0)
        unit_in_charge.hxc.inlet_2.flow_mass.setub(m.salt_flow_max)
        unit_in_charge.hxc.outlet_1.flow_mol.setlb(0)
        unit_in_charge.hxc.outlet_1.flow_mol.setub(0.2 * m.flow_max)
        unit_in_charge.hxc.outlet_2.flow_mass.setlb(0)
        unit_in_charge.hxc.outlet_2.flow_mass.setub(m.salt_flow_max)
        unit_in_charge.hxc.inlet_2.pressure.setlb(101320)
        unit_in_charge.hxc.inlet_2.pressure.setub(101330)
        unit_in_charge.hxc.outlet_2.pressure.setlb(101320)
        unit_in_charge.hxc.outlet_2.pressure.setub(101330)
        unit_in_charge.hxc.heat_duty.setlb(0)
        unit_in_charge.hxc.heat_duty.setub(m.fs.heat_duty_max)
        unit_in_charge.hxc.shell.heat.setlb(-m.fs.heat_duty_max)
        unit_in_charge.hxc.shell.heat.setub(0)
        unit_in_charge.hxc.tube.heat.setlb(0)
        unit_in_charge.hxc.tube.heat.setub(m.fs.heat_duty_max)
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
        unit_in_charge.hxc.costing.pressure_factor.setlb(0)
        unit_in_charge.hxc.costing.pressure_factor.setub(1000)
        unit_in_charge.hxc.costing.purchase_cost.setlb(0)
        unit_in_charge.hxc.costing.purchase_cost.setub(1e7)
        unit_in_charge.hxc.costing.base_cost_per_unit.setlb(0)
        unit_in_charge.hxc.costing.base_cost_per_unit.setub(1e6)
        unit_in_charge.hxc.costing.material_factor.setlb(0)
        unit_in_charge.hxc.costing.material_factor.setub(10)
        unit_in_charge.hxc.delta_temperature_in.setlb(10)
        unit_in_charge.hxc.delta_temperature_out.setlb(10)
        unit_in_charge.hxc.delta_temperature_in.setub(80.5)
        unit_in_charge.hxc.delta_temperature_out.setub(81)

        # HX pump and Cooler
        unit_in_charge.cooler.heat_duty.setlb(-1e12)
        unit_in_charge.cooler.heat_duty.setub(0)
        for unit_k in [unit_in_charge.hx_pump,
                       unit_in_charge.cooler]:
            unit_k.inlet.flow_mol.setlb(0)
            unit_k.inlet.flow_mol.setub(0.2*m.flow_max)
            unit_k.outlet.flow_mol.setlb(0)
            unit_k.outlet.flow_mol.setub(0.2*m.flow_max)
            unit_k.deltaP.setlb(0)
            unit_k.deltaP.setub(1e10)
        unit_in_charge.hx_pump.work_mechanical[0].setlb(0)
        unit_in_charge.hx_pump.work_mechanical[0].setub(1e7)
        unit_in_charge.hx_pump.ratioP.setlb(0)
        unit_in_charge.hx_pump.ratioP.setub(100)
        unit_in_charge.hx_pump.work_fluid[0].setlb(0)
        unit_in_charge.hx_pump.work_fluid[0].setub(1e7)
        unit_in_charge.hx_pump.efficiency_pump[0].setlb(0)
        unit_in_charge.hx_pump.efficiency_pump[0].setub(1)
        unit_in_charge.hx_pump.costing.base_cost_per_unit.setlb(0)
        unit_in_charge.hx_pump.costing.base_cost_per_unit.setub(1e8)
        unit_in_charge.hx_pump.costing.purchase_cost.setlb(0)
        unit_in_charge.hx_pump.costing.purchase_cost.setub(1e8)
        unit_in_charge.hx_pump.costing.pump_head.setlb(0)
        unit_in_charge.hx_pump.costing.pump_head.setub(1e9)
        unit_in_charge.hx_pump.costing.size_factor.setlb(0)
        unit_in_charge.hx_pump.costing.size_factor.setub(1e9)
        unit_in_charge.hx_pump.costing.motor_base_cost_per_unit.setlb(0)
        unit_in_charge.hx_pump.costing.motor_base_cost_per_unit.setub(1e8)
        unit_in_charge.hx_pump.costing.pump_purchase_cost.setlb(0)
        unit_in_charge.hx_pump.costing.pump_purchase_cost.setub(1e8)
        unit_in_charge.hx_pump.costing.motor_purchase_cost.setlb(0)
        unit_in_charge.hx_pump.costing.motor_purchase_cost.setub(1e8)

        # HP splitter
        unit_in_charge.ess_hp_split.to_hxc.flow_mol[:].setlb(0)
        unit_in_charge.ess_hp_split.to_hxc.flow_mol[:].setub(0.2 * m.flow_max)
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
        unit_in_charge.recycle_mixer.from_hx_pump.flow_mol.setub(0.2 * m.flow_max)
        unit_in_charge.recycle_mixer.outlet.flow_mol.setlb(0)
        unit_in_charge.recycle_mixer.outlet.flow_mol.setub(m.flow_max)


    # Add bounds to all units in discharge mode
    for unit_in_discharge in [m.fs.discharge_mode_disjunct]:
        # Discharge heat exchanger (HXD)
        unit_in_discharge.hxd.inlet_1.flow_mass.setlb(0)
        unit_in_discharge.hxd.inlet_1.flow_mass.setub(m.salt_flow_max)
        unit_in_discharge.hxd.inlet_2.flow_mol.setlb(0)
        unit_in_discharge.hxd.inlet_2.flow_mol.setub(0.2 * m.flow_max)
        unit_in_discharge.hxd.outlet_1.flow_mass.setlb(0)
        unit_in_discharge.hxd.outlet_1.flow_mass.setub(m.salt_flow_max)
        unit_in_discharge.hxd.outlet_2.flow_mol.setlb(0)
        unit_in_discharge.hxd.outlet_2.flow_mol.setub(0.2 * m.flow_max)
        unit_in_discharge.hxd.inlet_1.pressure.setlb(101320)
        unit_in_discharge.hxd.inlet_1.pressure.setub(101330)
        unit_in_discharge.hxd.outlet_1.pressure.setlb(101320)
        unit_in_discharge.hxd.outlet_1.pressure.setub(101330)
        unit_in_discharge.hxd.heat_duty.setlb(0)
        unit_in_discharge.hxd.heat_duty.setub(m.fs.heat_duty_max)
        unit_in_discharge.hxd.tube.heat.setlb(0)
        unit_in_discharge.hxd.tube.heat.setub(m.fs.heat_duty_max)
        unit_in_discharge.hxd.shell.heat.setlb(-m.fs.heat_duty_max)
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
        unit_in_discharge.hxd.costing.pressure_factor.setlb(0)
        unit_in_discharge.hxd.costing.pressure_factor.setub(1000)
        unit_in_discharge.hxd.costing.purchase_cost.setlb(0)
        unit_in_discharge.hxd.costing.purchase_cost.setub(1e7)
        unit_in_discharge.hxd.costing.base_cost_per_unit.setlb(0)
        unit_in_discharge.hxd.costing.base_cost_per_unit.setub(1e6)
        unit_in_discharge.hxd.costing.material_factor.setlb(0)
        unit_in_discharge.hxd.costing.material_factor.setub(10)
        unit_in_discharge.hxd.delta_temperature_in.setlb(10)
        unit_in_discharge.hxd.delta_temperature_out.setlb(10)
        unit_in_discharge.hxd.delta_temperature_in.setub(300)
        unit_in_discharge.hxd.delta_temperature_out.setub(300)

        # BFP splitter
        unit_in_discharge.ess_bfp_split.to_hxd.flow_mol[:].setlb(0)
        unit_in_discharge.ess_bfp_split.to_hxd.flow_mol[:].setub(0.2 * m.flow_max)
        unit_in_discharge.ess_bfp_split.to_fwh8.flow_mol[:].setlb(0)
        unit_in_discharge.ess_bfp_split.to_fwh8.flow_mol[:].setub(m.flow_max)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_hxd"].setlb(0)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_hxd"].setub(1)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_fwh8"].setlb(0)
        unit_in_discharge.ess_bfp_split.split_fraction[0.0, "to_fwh8"].setub(1)
        unit_in_discharge.ess_bfp_split.inlet.flow_mol[:].setlb(0)
        unit_in_discharge.ess_bfp_split.inlet.flow_mol[:].setub(m.flow_max)

        # ES Turbine
        unit_in_discharge.es_turbine.inlet.flow_mol[:].setlb(0)
        unit_in_discharge.es_turbine.inlet.flow_mol[:].setub(0.2 * m.flow_max)
        unit_in_discharge.es_turbine.outlet.flow_mol[:].setlb(0)
        unit_in_discharge.es_turbine.outlet.flow_mol[:].setub(0.2 * m.flow_max)
        unit_in_discharge.es_turbine.deltaP.setlb(-1e10)
        unit_in_discharge.es_turbine.deltaP.setub(1e10)
        unit_in_discharge.es_turbine.work.setlb(-1e8)
        unit_in_discharge.es_turbine.work.setub(0)
        unit_in_discharge.es_turbine.efficiency_isentropic.setlb(0)
        unit_in_discharge.es_turbine.efficiency_isentropic.setub(1)
        unit_in_discharge.es_turbine.ratioP.setlb(0)
        unit_in_discharge.es_turbine.ratioP.setub(100)
        unit_in_discharge.es_turbine.efficiency_mech.setlb(0)
        unit_in_discharge.es_turbine.efficiency_mech.setub(1)
        unit_in_discharge.es_turbine.shaft_speed.setlb(0)
        unit_in_discharge.es_turbine.shaft_speed.setub(1000)


    return m


def main(m_usc):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_charge_model(m_usc)

    # Give all the required inputs to the model
    set_model_input(m)
    # print('DOF after build: ', degrees_of_freedom(m))
    # Add scaling factor
    set_scaling_factors(m)

    # Initialize the model with a sequential initialization and custom
    # routines
    print('DOF before initialization: ', degrees_of_freedom(m))
    initialize(m)
    print('DOF after initialization: ', degrees_of_freedom(m))

    # Add cost correlations
    build_costing(m, solver=solver)
    # print('DOF after costing: ', degrees_of_freedom(m))
    # raise Exception()

    # Add bounds
    add_bounds(m)

    # Add disjunctions
    add_disjunction(m)

    # print('DOF after bounds: ', degrees_of_freedom(m))

    return m, solver


def print_results(m, results):

    print('================================')
    print('')
    print('')
    print("***************** Optimization Results ******************")
    print('Revenue ($/h): {:.6f}'.format(
        value(m.fs.revenue)))
    print('Previous Salt Inventory (kg): {:.6f}'.format(
        value(m.fs.previous_salt_inventory[0])))
    print('Salt Inventory (kg): {:.6f}'.format(
        value(m.fs.salt_inventory[0])))
    print('Salt Amount (kg): {:.6f}'.format(
        value(m.fs.salt_amount[0])))
    print('Cooling duty (MW_th): {:.6f}'.format(
        value(m.fs.cooler_heat_duty[0]) * 1e-6))
    print('HX pump work (MW): {:.6f}'.format(
        value(m.fs.hx_pump_work[0]) * 1e-6))
    print('')
    print('')
    print("***************** Costing Results ******************")
    print('Obj (M$/year): {:.6f}'.format(value(m.obj)))
    print('Plant capital cost (M$/y): {:.6f}'.format(
        value(m.fs.plant_capital_cost) * 1e-6))
    print('Plant fixed operating costs (M$/y): {:.6f}'.format(
        value(m.fs.plant_fixed_operating_cost) * 1e-6))
    print('Plant variable operating costs (M$/y): {:.6f}'.format(
        value(m.fs.plant_variable_operating_cost) * 1e-6))
    print('Operating Cost (Fuel) ($/h): {:.6f}'.format(
        value(m.fs.operating_cost)/(365*24)))
    print('Storage Capital Cost ($/h): {:.6f}'.format(
        value(m.fs.storage_capital_cost)/(365*24)))
    print('')
    print('')
    print("***************** Power Plant Operation ******************")
    print('')
    print('Plant Power (MW): {:.6f}'.format(
        value(m.fs.plant_power_out[0])))
    print('Boiler feed water flow (mol/s): {:.6f}'.format(
        value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler duty (MW_th): {:.6f}'.format(
        value((m.fs.boiler.heat_duty[0]
               + m.fs.reheater[1].heat_duty[0]
               + m.fs.reheater[2].heat_duty[0])
              * 1e-6)))
    print('Makeup water flow: {:.6f}'.format(
        value(m.fs.condenser_mix.makeup.flow_mol[0])))
    print()
    print()
    if m.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
        print("***************** Charge Heat Exchanger (HXC) ******************")
        print('HXC area (m2): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.area)))
        print('HXC cost ($/y): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.costing.purchase_cost / 15)))
        print('HXC Salt flow (kg/s): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.inlet_2.flow_mass[0])))
        print('HXC Salt temperature in (K): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.inlet_2.temperature[0])))
        print('HXC Salt temperature out (K): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.outlet_2.temperature[0])))
        print('HXC Steam flow to storage (mol/s): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.inlet_1.flow_mol[0])))
        print('HXC Water temperature in (K): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.side_1.properties_in[0].temperature)))
        print('HXC Steam temperature out (K): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.side_1.properties_out[0].temperature)))
        print('HXC Delta temperature at inlet (K): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.delta_temperature_in[0])))
        print('HXC Delta temperature at outlet (K): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.delta_temperature_out[0])))
        print('Cooling duty (MW_th): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.cooler.heat_duty[0]) * -1e-6))
        print('Salt storage tank volume in m3: {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.tank_volume)))
        print('Salt cost ($/y): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.salt_purchase_cost)))
        print('Tank cost ($/y): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.solar_costing.total_tank_cost / 15)))
        print('Salt pump cost ($/y): {:.6f}'.format(
            value(m.fs.charge_mode_disjunct.spump_purchase_cost)))
        print('')
    elif m.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
        print("*************** Discharge Heat Exchanger (HXD) ****************")
        print('')
        print('HXD area (m2): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.area)))
        print('HXD cost ($/y): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.costing.purchase_cost / 15)))
        print('HXD Salt flow (kg/s): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.inlet_1.flow_mass[0])))
        print('HXD Salt temperature in (K): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.inlet_1.temperature[0])))
        print('HXD Salt temperature out (K): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.outlet_1.temperature[0])))
        print('HXD Steam flow to storage (mol/s): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.inlet_2.flow_mol[0])))
        print('HXD Water temperature in (K): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.side_2.properties_in[0].temperature)))
        print('HXD Steam temperature out (K): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.side_2.properties_out[0].temperature)))
        print('HXD Delta temperature at inlet (K): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.delta_temperature_in[0])))
        print('HXD Delta temperature at outlet (K): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.delta_temperature_out[0])))
        print('ES Turbine work (MW): {:.6f}'.format(
            value(m.fs.discharge_mode_disjunct.es_turbine.work[0]) * -1e-6))
        print('')

    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')


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
        print('        Disjunction 1: Charge mode is selected')
        nlp_model.fs.charge_mode_disjunct.hxc.report()
        # nlp_model.fs.es_turbine.display()
        # nlp_model.fs.cooler.display()
        # nlp_model.fs.hx_pump.display()
    elif nlp_model.fs.discharge_mode_disjunct.indicator_var.value == 1:
        print('        Disjunction 1: Discharge mode is selected')
        nlp_model.fs.discharge_mode_disjunct.hxd.report()
    elif nlp_model.fs.no_storage_mode_disjunct.indicator_var.value == 1:
        print('        Disjunction 1: No storage mode is selected')
    else:
        print('        No other operation alternative!')
    print('       ___________________________________________')

    log_close_to_bounds(nlp_model)
    log_infeasible_constraints(nlp_model)


def run_gdp(m, cycle=None):

    if cycle == "charge":
        m.fs.charge_mode_disjunct.indicator_var.fix(True)
        m.fs.discharge_mode_disjunct.indicator_var.fix(False)
        m.fs.no_storage_mode_disjunct.indicator_var.fix(False)
    elif cycle == "discharge":
        m.fs.charge_mode_disjunct.indicator_var.fix(False)
        m.fs.discharge_mode_disjunct.indicator_var.fix(True)
        m.fs.no_storage_mode_disjunct.indicator_var.fix(False)
    elif cycle == "no_storage":
        m.fs.charge_mode_disjunct.indicator_var.fix(False)
        m.fs.discharge_mode_disjunct.indicator_var.fix(False)
        m.fs.no_storage_mode_disjunct.indicator_var.fix(True)
    else:
        print('**^^** Unrecognized operation mode! Try charge or discharge')

    print('DOF before solve = ', degrees_of_freedom(m))

    # Solve the design optimization model

    opt = SolverFactory('gdpopt')
    # opt.CONFIG.strategy = 'RIC'  # LOA is an option
    opt.CONFIG.strategy = 'LOA'
    opt.CONFIG.OA_penalty_factor = 1e4
    opt.CONFIG.max_slack = 1e4
    opt.CONFIG.call_after_subproblem_solve = print_model
    # opt.CONFIG.mip_solver = 'glpk'
    # opt.CONFIG.mip_solver = 'cbc'
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
                "max_iter": 200
            }
        )
    )

    print_results(m, results)
    # print_reports(m)

    return results


def model_analysis(m, solver, cycle=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    # Fix variables in the flowsheet
    m.fs.plant_power_out.fix(400)
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)

    # Unfix variables fixed in model input and during initialization
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s

    # Unfix data
    m.fs.charge_mode_disjunct.hxc.area.unfix()
    m.fs.discharge_mode_disjunct.hxd.area.unfix()
    m.fs.charge_mode_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.discharge_mode_disjunct.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.fs.charge_mode_disjunct.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix() # 1 DOF

    for salt_hxd in [m.fs.discharge_mode_disjunct.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix() # 1 DOF

    for unit in [m.fs.charge_mode_disjunct.cooler]:
        unit.inlet.unfix()
    m.fs.charge_mode_disjunct.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Add salt inventory constraint
    @m.fs.Constraint(m.fs.time,
                     doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory(b, t):
        return (
            b.salt_inventory[t] ==
            b.previous_salt_inventory[t]
            + m.fs.charge_mode_disjunct.binary_indicator_var * m.fs.salt_amount[0]
            - m.fs.discharge_mode_disjunct.binary_indicator_var * m.fs.salt_amount[0]
        )

    #-------- modified by esrawli
    # Add constraints for discharge mode
    # @m.fs.Constraint(m.fs.time,
    #                   doc="Maximum salt inventory at any time")
    # def constraint_salt_max_inventory1(b, t):
    #     return (
    #         b.salt_inventory[t] <= b.salt_amount[0])

    # @m.fs.Constraint(m.fs.time,
    #                   doc="Maximum previous salt inventory at any time")
    # def constraint_salt_max_inventory2(b, t):
    #     return (
    #         b.previous_salt_inventory[t] <= b.salt_amount[0])
    #--------

    # Deactivate arcs
    _deactivate_arcs(m)

    m.fs.revenue = Expression(
        expr=(m.fs.lmp[0] *
              m.fs.plant_power_out[0]),
        doc="Revenue function in $/h assuming 1 hr operation"
    )

    m.fs.production_cons.deactivate()
    @m.fs.Constraint(m.fs.time)
    def production_cons_with_storage(b, t):
        return (
            (-1 * sum(m.fs.turbine[p].work_mechanical[t]
                      for p in m.set_turbine)
             - m.fs.hx_pump_work[0]
             # + ((-1) * m.fs.discharge_turbine_work[0]) # added by esrawli to account for es turbine work
            ) ==
            m.fs.plant_power_out[t] * 1e6 * (pyunits.W/pyunits.MW)
        )

    # Objective function: total costs
    m.obj = Objective(
        expr=(
            m.fs.revenue
            - (
                (m.fs.operating_cost
                 + m.fs.plant_fixed_operating_cost
                 + m.fs.plant_variable_operating_cost
                ) / (365 * 24))
            - (
                (m.fs.storage_capital_cost
                 + m.fs.plant_capital_cost)/ (365 * 24))
        ),
        sense=maximize
    )

    # Solve using GDPopt
    run_gdp(m, cycle=cycle)


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    m_usc = usc.build_plant_model()
    usc.initialize(m_usc)
    # m_ready = disconnect_arcs(m_usc)
    # m = build_model(m_ready,
    #                 scenario=i)
    m_chg, solver = main(m_usc)


    m_chg.fs.lmp = Var(
        m_chg.fs.time,
        domain=Reals,
        initialize=80,
        doc="Hourly LMP in $/MWh"
        )

    operation_mode = "charge"
    if operation_mode == "charge":
        m_chg.fs.lmp[0].fix(80)
    elif operation_mode == "discharge":
        m_chg.fs.lmp[0].fix(120)
    elif operation_mode == "no_storage":
        m_chg.fs.lmp[0].fix(100)
    else:
        print('**^^** Unrecognized operation mode! Try charge or discharge')
    m = model_analysis(m_chg,
                       solver,
                       cycle=operation_mode)
