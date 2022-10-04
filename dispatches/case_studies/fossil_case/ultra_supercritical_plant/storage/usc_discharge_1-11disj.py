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

updated (08/12/2021)
"""

__author__ = "Naresh Susarla and Soraya Rawlings"

# Import Python libraries
from math import pi
import logging
import os
import csv

# Import Pyomo libraries
from pyomo.environ import (Block, Param, Constraint, Objective,
                           TransformationFactory, SolverFactory,
                           Expression, value, log, exp, Var)
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.gdp import Disjunct, Disjunction
from pyomo.contrib.fbbt.fbbt import  _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression

# Import IDAES libraries
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core import UnitModelCostingBlock
from idaes.core.util import model_serializer as ms
from idaes.models.unit_models import (HeatExchanger,
                                      MomentumMixingType,
                                      Heater)
from idaes.models.unit_models import (Mixer,
                                      PressureChanger)
from idaes.models_extra.power_generation.unit_models.helm import (HelmMixer,
                                                                  HelmTurbineStage,
                                                                  HelmSplitter)
from idaes.models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback, HeatExchangerFlowPattern)
from idaes.models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.misc import svg_tag
from idaes.models.costing.SSLW import (SSLWCosting,
                                       SSLWCostingData)
from idaes.core.util.exceptions import ConfigurationError

# Import ultra supercritical power plant model
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
# Import properties package for Solar salt
from dispatches.properties import solarsalt_properties

from pyomo.network.plugins import expand_arcs

from IPython import embed

logging.basicConfig(level=logging.INFO)


def create_discharge_model(m):
    """Create flowsheet and add unit models.
    """

    # Create a block to add charge storage model
    m.fs.discharge = Block()

    add_data(m)

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()

    ###########################################################################
    #  Add vhp and hp splitters                                               #
    ###########################################################################
    # Declared to divert some steam from high pressure inlet and
    # intermediate pressure inlet to charge the storage heat exchanger
    m.fs.discharge.es_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_fwh", "to_hxd"]
        }
    )
    m.fs.discharge.hxd = HeatExchanger(
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

    m.fs.discharge.es_turbine = HelmTurbineStage(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    ###########################################################################
    #  Declare disjuncts
    ###########################################################################
    # Disjunction 1 for the sink of discharge HX consists of 2 disjuncts:
    #   1. ccs_sink_disjunct ======> steam from hxd is used in ccs
    #   2. plant_sink_disjunct ======> steam from hxd is used in the turbines

    m.fs.discharge.condpump_source_disjunct = Disjunct(
        rule=condpump_source_disjunct_equations)
    m.fs.discharge.fwh1_source_disjunct = Disjunct(
        rule=fwh1_source_disjunct_equations)
    m.fs.discharge.fwh2_source_disjunct = Disjunct(
        rule=fwh2_source_disjunct_equations)
    m.fs.discharge.fwh3_source_disjunct = Disjunct(
        rule=fwh3_source_disjunct_equations)
    m.fs.discharge.fwh4_source_disjunct = Disjunct(
        rule=fwh4_source_disjunct_equations)
    m.fs.discharge.fwh5_source_disjunct = Disjunct(
        rule=fwh5_source_disjunct_equations)
    m.fs.discharge.booster_source_disjunct = Disjunct(
        rule=booster_source_disjunct_equations)
    m.fs.discharge.bfp_source_disjunct = Disjunct(
        rule=bfp_source_disjunct_equations)
    m.fs.discharge.fwh6_source_disjunct = Disjunct(
        rule=fwh6_source_disjunct_equations)
    m.fs.discharge.fwh8_source_disjunct = Disjunct(
        rule=fwh8_source_disjunct_equations)
    m.fs.discharge.fwh9_source_disjunct = Disjunct(
        rule=fwh9_source_disjunct_equations)

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m)
    _solar_salt_ohtc_calculation(m)
    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.discharge)
    return m


def add_data(m):
    """Add data to the model
    """

    # Add Chemical Engineering cost index for 2019
    m.CE_index = 607.5

    # Add operating hours
    m.fs.discharge.hours_per_day = pyo.Param(
        initialize=6,
        doc='Estimated number of hours of charging per day'
    )

    # Define number of years over which the capital cost is annualized
    m.fs.discharge.num_of_years = Param(
        initialize=30,
        doc='Number of years for capital cost annualization')

    # Add data to compute overall heat transfer coefficient for the
    # Solar salt storage heat exchanger using the Sieder-Tate
    # correlation. Parameters for tube diameter and thickness assumed
    # from the data in (2017) He et al., Energy Procedia 105, 980-985
    m.fs.discharge.data_hxd_solar = {
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    # Data to compute overall heat transfer coefficient for the charge
    # heat exchanger using the Sieder-Tate Correlation. Parameters for
    # tube diameter and thickness assumed from the data in (2017) He
    # et al., Energy Procedia 105, 980-985
    m.fs.discharge.hxd_tube_inner_dia = Param(
        initialize=m.fs.discharge.data_hxd_solar['tube_inner_dia'],
        doc='Tube inner diameter [m]')
    m.fs.discharge.hxd_tube_outer_dia = Param(
        initialize=m.fs.discharge.data_hxd_solar['tube_outer_dia'],
        doc='Tube outer diameter [m]')
    m.fs.discharge.hxd_k_steel = Param(
        initialize=m.fs.discharge.data_hxd_solar['k_steel'],
        doc='Thermal conductivity of steel [W/mK]')
    m.fs.discharge.hxd_n_tubes = Param(
        initialize=m.fs.discharge.data_hxd_solar['number_tubes'],
        doc='Number of tubes')
    m.fs.discharge.hxd_shell_inner_dia = Param(
        initialize=m.fs.discharge.data_hxd_solar['shell_inner_dia'],
        doc='Shell inner diameter [m]')

    # Add fuel cost data
    m.data_cost = {
        'coal_price': 2.11e-9,
    }
    m.fs.discharge.coal_price = Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')

    # Add parameters to calculate the Solar salt pump costing. Since
    # the unit is not explicitly modeled, the IDAES cost method is not
    # used for this equipment.  The primary purpose of the salt pump
    # is to move the molten salt without changing the pressure. Thus,
    # the pressure head is computed assuming that the salt is moved on
    # an average of 5m linear distance.
    m.data_salt_pump = {
        'FT': 1.5,
        'FM': 2.0,
        'head': 3.281*5,
        'motor_FT': 1,
        'nm': 1
    }
    m.fs.discharge.spump_FT = pyo.Param(
        initialize=m.data_salt_pump['FT'],
        doc='Pump Type Factor for vertical split case')
    m.fs.discharge.spump_FM = pyo.Param(
        initialize=m.data_salt_pump['FM'],
        doc='Pump Material Factor Stainless Steel')
    m.fs.discharge.spump_head = pyo.Param(
        initialize=m.data_salt_pump['head'],
        doc='Pump Head 5m in Ft.')
    m.fs.discharge.spump_motorFT = pyo.Param(
        initialize=m.data_salt_pump['motor_FT'],
        doc='Motor Shaft Type Factor')
    m.fs.discharge.spump_nm = pyo.Param(
        initialize=m.data_salt_pump['nm'],
        doc='Motor Shaft Type Factor')


def _solar_salt_ohtc_calculation(m):
    """Block of equations for computing heat_transfer coefficient
    """

    # Calculate Reynolds, Prandtl, and Nusselt number for the salt and
    # steam side of discharge heat exchanger.
    m.fs.discharge.hxd.tube_cs_area = Expression(
        expr=(pi / 4) *
        (m.fs.discharge.hxd_tube_inner_dia ** 2),
        doc="Tube cross sectional area")
    m.fs.discharge.hxd.tube_out_area = Expression(
        expr=(pi / 4) *
        (m.fs.discharge.hxd_tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness in m2")
    m.fs.discharge.hxd.shell_eff_area = Expression(
        expr=(
            (pi / 4) *
            (m.fs.discharge.hxd_shell_inner_dia ** 2) -
            m.fs.discharge.hxd_n_tubes *
            m.fs.discharge.hxd.tube_out_area),
        doc="Effective shell cross sectional area in m2")

    # Calculate Reynolds number for the salt
    m.fs.discharge.hxd.salt_reynolds_number = Expression(
        expr=(
            (m.fs.discharge.hxd.inlet_1.flow_mass[0] *
             m.fs.discharge.hxd_tube_outer_dia) /
            (m.fs.discharge.hxd.shell_eff_area *
             m.fs.discharge.hxd.side_1.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number")

    # Calculate Prandtl number for the salt
    m.fs.discharge.hxd.salt_prandtl_number = Expression(
        expr=(
            m.fs.discharge.hxd.side_1.properties_in[0].cp_mass["Liq"] *
            m.fs.discharge.hxd.side_1.properties_in[0].visc_d_phase["Liq"] /
            m.fs.discharge.hxd.side_1.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number")

    # Calculate Prandtl Wall number for the salt
    m.fs.discharge.hxd.salt_prandtl_wall = Expression(
        expr=(
            m.fs.discharge.hxd.side_1.properties_out[0].cp_mass["Liq"] *
            m.fs.discharge.hxd.side_1.properties_out[0].visc_d_phase["Liq"] /
            m.fs.discharge.hxd.side_1.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number at wall")

    # Calculate Nusselt number for the salt
    m.fs.discharge.hxd.salt_nusselt_number = Expression(
        expr=(
            0.35 *
            (m.fs.discharge.hxd.salt_reynolds_number**0.6) *
            (m.fs.discharge.hxd.salt_prandtl_number**0.4) *
            ((m.fs.discharge.hxd.salt_prandtl_number /
              m.fs.discharge.hxd.salt_prandtl_wall) ** 0.25) *
            (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")

    # Calculate Reynolds number for the steam
    m.fs.discharge.hxd.steam_reynolds_number = Expression(
        expr=(
            m.fs.discharge.hxd.inlet_2.flow_mol[0] *
            m.fs.discharge.hxd.side_2.properties_in[0].mw *
            m.fs.discharge.hxd_tube_inner_dia /
            (m.fs.discharge.hxd.tube_cs_area *
             m.fs.discharge.hxd_n_tubes *
             m.fs.discharge.hxd.side_2.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")

    # Calculate Reynolds number for the steam
    m.fs.discharge.hxd.steam_prandtl_number = Expression(
        expr=(
            (m.fs.discharge.hxd.side_2.properties_in[0].cp_mol /
             m.fs.discharge.hxd.side_2.properties_in[0].mw) *
            m.fs.discharge.hxd.side_2.properties_in[0].visc_d_phase["Vap"] /
            m.fs.discharge.hxd.side_2.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")

    # Calculate Reynolds number for the steam
    m.fs.discharge.hxd.steam_nusselt_number = Expression(
        expr=(
            0.023 *
            (m.fs.discharge.hxd.steam_reynolds_number**0.8) *
            (m.fs.discharge.hxd.steam_prandtl_number**(0.33)) *
            ((m.fs.discharge.hxd.side_2.properties_in[0].visc_d_phase["Vap"] /
              m.fs.discharge.hxd.side_2.properties_out[0].visc_d_phase["Liq"])** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of discharge heat exchanger
    m.fs.discharge.hxd.h_salt = Expression(
        expr=(
            m.fs.discharge.hxd.side_1.properties_in[0].therm_cond_phase["Liq"] *
            m.fs.discharge.hxd.salt_nusselt_number /
            m.fs.discharge.hxd_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    m.fs.discharge.hxd.h_steam = Expression(
        expr=(
            m.fs.discharge.hxd.side_2.properties_in[0].therm_cond_phase["Vap"] *
            m.fs.discharge.hxd.steam_nusselt_number /
            m.fs.discharge.hxd_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")


    # Calculate overall heat transfer coefficient for Solar salt heat
    # exchanger
    m.fs.discharge.hxd.tube_dia_ratio = (m.fs.discharge.hxd_tube_outer_dia /
                                         m.fs.discharge.hxd_tube_inner_dia)
    m.fs.discharge.hxd.log_tube_dia_ratio = log(m.fs.discharge.hxd.tube_dia_ratio)
    @m.fs.discharge.hxd.Constraint(m.fs.time)
    def constraint_hxd_ohtc(b, t):
        return (
            b.overall_heat_transfer_coefficient[t] *
            (2 *
             m.fs.discharge.hxd_k_steel *
             b.h_steam +
             m.fs.discharge.hxd_tube_outer_dia *
             b.log_tube_dia_ratio *
             b.h_salt *
             b.h_steam +
             b.tube_dia_ratio *
             b.h_salt *
             2 * m.fs.discharge.hxd_k_steel)
        ) == (2 * m.fs.discharge.hxd_k_steel *
              b.h_salt *
              b.h_steam)


def _make_constraints(m):
    """Create arcs"""

    @m.fs.discharge.es_turbine.Constraint(
        m.fs.time,
        doc="Turbine outlet should be saturated steam")
    def constraint_esturbine_temperature_out(b, t):
        return (
            b.control_volume.properties_out[t].temperature ==
            b.control_volume.properties_out[t].temperature_sat
        )

    # Adding a constraint to limit the flow of salt in discharge heat exchanger
    # Flow upper bound is to obtained from nlp_mp.py
    # TODO: fix the amount of salt instead of bounding flow
    @m.fs.Constraint(m.fs.time)
    def hxd_salt_flow(b, t):
        return m.fs.discharge.hxd.inlet_1.flow_mass[t] <= 500


def _create_arcs(m):
    """Create arcs"""

    m.fs.discharge.essplit_to_hxd = Arc(
        source=m.fs.discharge.es_split.to_hxd,
        destination=m.fs.discharge.hxd.inlet_2,
        doc="Connection from ES splitter to HXD"
    )
    m.fs.discharge.hxd_to_esturbine = Arc(
        source=m.fs.discharge.hxd.outlet_2,
        destination=m.fs.discharge.es_turbine.inlet,
        doc="Connection from HXD to ES turbine"
    )


def disconect_arcs(m):
    """Create arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the charge heat exchanger
    for arc_s in [m.fs.condpump_to_fwh1,
                  m.fs.fwh1_to_fwh2,
                  m.fs.fwh2_to_fwh3,
                  m.fs.fwh3_to_fwh4,
                  m.fs.fwh4_to_fwh5,
                  m.fs.fwh5_to_deaerator,
                  m.fs.booster_to_fwh6,
                  m.fs.fwh6_to_fwh7,
                  m.fs.bfp_to_fwh8,
                  m.fs.fwh8_to_fwh9,
                  m.fs.fwh9_to_boiler]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()


def add_disjunction(m):
    """Add water source disjunction to the
    model
    """

    # Add disjunction 1 for ccs source steam selection
    m.fs.hxd_source_disjunction = Disjunction(
        expr=[
            m.fs.discharge.condpump_source_disjunct,
            m.fs.discharge.fwh1_source_disjunct,
            m.fs.discharge.fwh2_source_disjunct,
            m.fs.discharge.fwh3_source_disjunct,
            m.fs.discharge.fwh4_source_disjunct,
            m.fs.discharge.fwh5_source_disjunct,
            m.fs.discharge.booster_source_disjunct,
            m.fs.discharge.fwh6_source_disjunct,
            m.fs.discharge.bfp_source_disjunct,
            m.fs.discharge.fwh8_source_disjunct,
            m.fs.discharge.fwh9_source_disjunct
            ]
    )

    # Expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.discharge)

    return m


def condpump_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from Condenser Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.condpump_source_disjunct.condpump_to_essplit = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from Condenser pump to ES splitter"
    )
    m.fs.discharge.condpump_source_disjunct.essplit_to_fwh1 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from ES splitter to FWH1"
    )

    m.fs.discharge.condpump_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.condpump_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.condpump_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.condpump_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.condpump_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.condpump_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.condpump_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.condpump_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.condpump_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.condpump_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet,
        doc="Connection from FWH9 to boiler"
    )


def fwh1_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH1
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh1_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh1_to_essplit = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH1 to ES splitter"
    )
    m.fs.discharge.fwh1_source_disjunct.essplit_to_fwh2 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from ES splitter to FWH2"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh1_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster to FWH6"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh1_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh1_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet,
        doc="Connection from FWH9 to boiler"
    )


def fwh2_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH2
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh2_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh2_to_essplit = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH2 to ES splitter"
    )
    m.fs.discharge.fwh2_source_disjunct.essplit_to_fwh3 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from ES splitter to FWH3"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh2_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh2_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh2_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh3_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH3
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh3_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2
    )

    m.fs.discharge.fwh3_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh3_to_essplit = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH3 to ES splitter"
    )
    m.fs.discharge.fwh3_source_disjunct.essplit_to_fwh4 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from ES splitter to FWH4"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh3_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh3_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh3_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh4_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH4
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh4_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh4_to_essplit = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH4 to ES splitter"
    )
    m.fs.discharge.fwh4_source_disjunct.essplit_to_fwh5 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from ES splitter to FWH5"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh4_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh4_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh4_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh5_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH5
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh5_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh5_to_essplit = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH5 to ES splitter"
    )
    m.fs.discharge.fwh5_source_disjunct.essplit_to_deaerator = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from ES splitter to Deaerator"
    )

    m.fs.discharge.fwh5_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh5_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh5_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def booster_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from Booster Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.booster_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.booster_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.booster_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.booster_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.booster_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.booster_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.booster_source_disjunct.booster_to_essplit = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from Booster pump to ES splitter"
    )
    m.fs.discharge.booster_source_disjunct.essplit_to_fwh6 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from ES splitter to FWH6"
    )

    m.fs.discharge.booster_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.booster_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.booster_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.booster_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh6_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH6
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh6_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh6_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh6_to_essplit = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH6 to ES splitter"
    )
    m.fs.discharge.fwh6_source_disjunct.essplit_to_fwh7 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from ES splitter to FWH7"
    )

    m.fs.discharge.fwh6_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh6_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def bfp_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from Boiler Feed Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.bfp_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.bfp_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.bfp_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.bfp_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.bfp_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.bfp_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.bfp_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.bfp_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.bfp_source_disjunct.bfp_to_essplit = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from BFP to ES splitter"
    )
    m.fs.discharge.bfp_source_disjunct.essplit_to_fwh8 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from ES splitter to FWH8"
    )

    m.fs.discharge.bfp_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.bfp_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh8_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH8
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh8_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh8_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh8_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh8_to_essplit = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH8 to ES splitter"
    )
    m.fs.discharge.fwh8_source_disjunct.essplit_to_fwh9 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from ES splitter to FWH9"
    )

    m.fs.discharge.fwh8_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh9_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH9
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh9_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from condenser pump to FWH1"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2,
        doc="Connection from FWH1 to FWH2"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2,
        doc="Connection from FWH2 to FWH3"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2,
        doc="Connection from FWH3 to FWH4"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from FWH4 to FWH5"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh5_to_deaerator = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater,
        doc="Connection from FWH5 to deaerator"
    )

    m.fs.discharge.fwh9_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from booster pump to FWH6"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2,
        doc="Connection from FWH6 to FWH7"
    )

    m.fs.discharge.fwh9_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from BFP to FWH8"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2,
        doc="Connection from FWH8 to FWH9"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh9_to_essplit = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH9 to the ES SPlitter"
    )
    m.fs.discharge.fwh9_source_disjunct.essplit_to_boiler = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.boiler.inlet,
        doc="Connection from ES splitter to Boiler"
    )


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
    m.fs.discharge.hxd.area.fix(2000)  # m2
    # m.fs.discharge.hxd.overall_heat_transfer_coefficient.fix(1000)

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.discharge.hxd.inlet_1.flow_mass.fix(200)   # kg/s
    m.fs.discharge.hxd.inlet_1.temperature.fix(831.15)  # K
    m.fs.discharge.hxd.inlet_1.pressure.fix(101325)  # Pa

    # m.fs.discharge.hxd.inlet_2.flow_mol.fix(3854)   # kg/s
    # m.fs.discharge.hxd.inlet_2.enth_mol.fix(52232)  # K
    # m.fs.discharge.hxd.inlet_2.pressure.fix(3.4958e+07)  # Pa

    m.fs.discharge.es_split.inlet.flow_mol.fix(17854)   # kg/s
    m.fs.discharge.es_split.inlet.enth_mol.fix(52232)  # K
    m.fs.discharge.es_split.inlet.pressure.fix(3.4958e+07)  # Pa
    # m.fs.discharge.es_split.inlet.enth_mol.fix(42232)  # K
    # m.fs.discharge.es_split.inlet.pressure.fix(7e+05)  # Pa

    m.fs.discharge.es_split.split_fraction[0, "to_hxd"].fix(0.2)
    # m.fs.discharge.hxd.inlet_2.flow_mol.fix(5000)   # kg/s

    # m.fs.discharge.es_turbine.inlet.flow_mol.fix(10854)   # kg/s
    # m.fs.discharge.es_turbine.inlet.enth_mol.fix(62232)  # K
    # m.fs.discharge.es_turbine.inlet.pressure.fix(3.4958e+06)  # Pa

    # m.fs.discharge.es_turbine.ratioP.fix(0.0286)
    m.fs.discharge.es_turbine.constraint_esturbine_temperature_out.deactivate()
    m.fs.discharge.es_turbine.outlet.pressure.fix(6896)
    m.fs.discharge.es_turbine.efficiency_isentropic.fix(0.8)


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

    """

    # Include scaling factors for Solar discharge heat exchanger
    for solar_hxd in [m.fs.discharge.hxd]:
        iscale.set_scaling_factor(solar_hxd.area, 1e-2)
        iscale.set_scaling_factor(solar_hxd.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(solar_hxd.tube.heat, 1e-6)
        iscale.set_scaling_factor(solar_hxd.shell.heat, 1e-6)

    for est in [m.fs.discharge.es_turbine.control_volume]:
        iscale.set_scaling_factor(est.work, 1e-6)


def initialize(m, solver=None,
               outlvl=idaeslog.WARNING,
               optarg={"tol": 1e-8,
                       "max_iter": 300},
               fluid=None,
               source=None):
    """Initialize the units included in the discharge model
    """

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    # Include scaling factors
    iscale.calculate_scaling_factors(m)

    # Initialize splitters
    m.fs.discharge.es_split.initialize(outlvl=outlvl,
                                       optarg=solver.options)

    propagate_state(m.fs.discharge.essplit_to_hxd)
    m.fs.discharge.hxd.initialize(outlvl=outlvl,
                                  optarg=solver.options)

    propagate_state(m.fs.discharge.hxd_to_esturbine)
    # m.fs.discharge.es_turbine.inlet.fix()
    m.fs.discharge.es_turbine.initialize(outlvl=outlvl,
                                         optarg=solver.options)
    # m.fs.discharge.es_turbine.ratioP.unfix()
    m.fs.discharge.es_turbine.constraint_esturbine_temperature_out.activate()
    m.fs.discharge.es_turbine.outlet.pressure.unfix()

    # Initialize all disjuncts
    # propagate_state(m.fs.discharge.fwh2_source_disjunct.fwh2_to_essplit)
    # m.fs.discharge.fwh2_source_disjunct.es_split.initialize(outlvl=outlvl,
    #                                    optarg=solver.options)

    # Check and raise an error if the degrees of freedom are not 0
    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building the model are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m))
            )

    # Solve initialization
    res = solver.solve(m,
                       tee=False,
                       options=optarg)

    print("Discharge Model Initialization = ",
          res.solver.termination_condition)
    print("*************   Discharge Model Initialized   ******************")


def build_costing(m,
                  solver=None,
                  optarg={"tol": 1e-8,
                          "max_iter": 300}):
    """ Add cost correlations for the storage design analysis. This
    function is used to estimate the capital and operatig cost of
    integrating an energy storage system. It contains cost
    correlations to estimate the capital cost of charge heat
    exchanger, salt storage tank, molten salt pump, and salt
    inventory. Note that it does not compute the cost of the whole
    power plant.

    """

    ###########################################################################
    # Add capital cost
    # 1. Calculate discharge heat exchanger cost
    # 2. Calculate Solar salt pump purchase cost
    # 3. Calculate total capital cost of discharge system

    # Main assumptions
    # 1. Salt life is assumed to outlast the plant life
    # 2. The economic objective is to minimize total annualized cost. So, cash
    # flows, discount rate, and NPV are not included in this study.
    ###########################################################################
    # Add capital cost: 1. Calculate discharge heat exchanger cost
    ###########################################################################
    # Calculate and initialize Solar salt discharge heat exchanger
    # cost, which is estimated using the IDAES costing method with
    # default options, i.e. a U-tube heat exchanger, stainless steel
    # material, and a tube length of 12ft. Refer to costing
    # documentation to change any of the default options. The purchase
    # cost of heat exchanger has to be annualized when used
    m.fs.costing = SSLWCosting()

    m.fs.discharge.hxd.costing = UnitModelCostingBlock(
        default={
            "flowsheet_costing_block": m.fs.costing,
            "costing_method": SSLWCostingData.cost_heat_exchanger
        }
    )

    ###########################################################################
    # Add capital cost: 2. Calculate Solar salt pump purchase cost
    ###########################################################################
    # Pump for moving Solar salt is not explicity modeled. To compute
    # the capital costs for this pump the capital cost expressions are
    # added below.  All cost expressions are from the same reference
    # as the IDAES costing framework and is given below: Seider,
    # Seader, Lewin, Windagdo, 3rd Ed. John Wiley and Sons, Chapter
    # 22. Cost Accounting and Capital Cost Estimation, Section 22.2 Cost
    # Indexes and Capital Investment

    # ---------- Solar salt ----------
    # Calculate purchase cost of Solar salt pump
    m.fs.discharge.spump_Qgpm = pyo.Expression(
        expr=(m.fs.discharge.hxd.side_1.properties_in[0].flow_mass *
              (264.17 * pyo.units.gallon / pyo.units.m**3) *
              (60 * pyo.units.s / pyo.units.min) /
              (m.fs.discharge.hxd.side_1.properties_in[0].dens_mass["Liq"])),
        doc="Conversion of solar salt flow mass to vol flow in gallons per min"
    )
    m.fs.discharge.dens_lbft3 = pyo.units.convert(
        m.fs.discharge.hxd.side_1.properties_in[0].dens_mass["Liq"],
        to_units=pyo.units.pound / pyo.units.foot**3
    )
    m.fs.discharge.spump_sf = pyo.Expression(
        expr=(m.fs.discharge.spump_Qgpm * (
            m.fs.discharge.spump_head ** 0.5)),
        doc="Pump size factor"
    )

    # Expression for pump base purchase cost
    m.fs.discharge.pump_CP = pyo.Expression(
        expr=(
            m.fs.discharge.spump_FT * m.fs.discharge.spump_FM *
            exp(
                9.2951 -
                0.6019 * log(m.fs.discharge.spump_sf) +
                0.0519 * ((log(m.fs.discharge.spump_sf))**2)
            )
        ),
        doc="Base purchase cost of Solar salt pump in $"
    )

    # Expression for pump efficiency
    m.fs.discharge.spump_np = pyo.Expression(
        expr=(
            -0.316 +
            0.24015 * log(m.fs.discharge.spump_Qgpm) -
            0.01199 * ((log(m.fs.discharge.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump in horsepower"
    )
    m.fs.discharge.motor_pc = pyo.Expression(
        expr=(
            (m.fs.discharge.spump_Qgpm *
             m.fs.discharge.spump_head *
             m.fs.discharge.dens_lbft3) /
            (33000 * m.fs.discharge.spump_np *
             m.fs.discharge.spump_nm)
        ),
        doc="Power consumption of motor in horsepower"
    )

    # Defining a local variable for the log of motor's power
    # consumption This will help writing the motor's purchase cost
    # expressions conciesly
    log_motor_pc = log(m.fs.discharge.motor_pc)
    m.fs.discharge.motor_CP = pyo.Expression(
        expr=(
            m.fs.discharge.spump_motorFT *
            exp(
                5.4866 +
                0.13141 * log_motor_pc +
                0.053255 * (log_motor_pc**2) +
                0.028628 * (log_motor_pc**3) -
                0.0035549 * (log_motor_pc**4)
            )
        ),
        doc="Base cost of Solar salt pump's motor in $"
    )

    # Calculate and initialize total cost of Solar salt pump
    m.fs.discharge.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Total purchase cost of Solar salt pump in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            m.fs.discharge.spump_purchase_cost == (
                m.fs.discharge.pump_CP +
                m.fs.discharge.motor_CP) *
            (m.CE_index / 394)
        )
    m.fs.discharge.spump_purchase_cost_eq = pyo.Constraint(
        rule=solar_spump_purchase_cost_rule)

    ###########################################################################
    # Add capital cost: 3. Calculate total capital cost for discharge system
    ###########################################################################

    # Add capital cost variable at flowsheet level to handle the Solar
    # salt capital cost
    m.fs.discharge.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Annualized capital cost")

    # Calculate and initialize annualized capital cost for the Solar
    # salt discharge storage system
    def solar_cap_cost_rule(b):
        return (m.fs.discharge.capital_cost * m.fs.discharge.num_of_years ==
                m.fs.discharge.spump_purchase_cost +
                m.fs.discharge.hxd.costing.capital_cost)
    m.fs.discharge.cap_cost_eq = pyo.Constraint(
        rule=solar_cap_cost_rule)

    ###########################################################################
    #  Add operating cost
    ###########################################################################
    m.fs.discharge.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.discharge.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.discharge.operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Operating cost in $/year")

    def op_cost_rule(b):
        return m.fs.discharge.operating_cost == (
            m.fs.discharge.operating_hours *
            m.fs.discharge.coal_price *
            m.fs.plant_heat_duty[0] * 1e6
        )
    m.fs.discharge.op_cost_eq = pyo.Constraint(rule=op_cost_rule)

    return m


def initialize_with_costing(m):

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.discharge.spump_purchase_cost,
        m.fs.discharge.spump_purchase_cost_eq)

    # Initialize capital cost
    calculate_variable_from_constraint(
        m.fs.discharge.capital_cost,
        m.fs.discharge.cap_cost_eq)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.discharge.operating_cost,
        m.fs.discharge.op_cost_eq)

    # Check and raise an error if the degrees of freedom are not 0
    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building costing block are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m))
            )

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)
    print("Cost Initialization = ",
          res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print('')
    print('')


def add_bounds(m):
    """Add bounds to units in discharge model

    """

    m.flow_max = m.main_flow * 3.2        # Units in mol/s
    m.storage_flow_max = 0.2 * m.flow_max # Units in mol/s
    m.salt_flow_max = 1200                # Units in kg/s
    m.heat_duty_bound = 200e6             # Units in W

    # Charge heat exchanger section
    for hxd in [m.fs.discharge.hxd]:
        hxd.inlet_2.flow_mol.setlb(0)
        hxd.inlet_2.flow_mol.setub(m.storage_flow_max)
        hxd.inlet_1.flow_mass.setlb(0)
        hxd.inlet_1.flow_mass.setub(m.salt_flow_max)
        hxd.outlet_2.flow_mol.setlb(0)
        hxd.outlet_2.flow_mol.setub(m.storage_flow_max)
        hxd.outlet_1.flow_mass.setlb(0)
        hxd.outlet_1.flow_mass.setub(m.salt_flow_max)
        hxd.inlet_1.pressure.setlb(101320)
        hxd.inlet_1.pressure.setub(101330)
        hxd.outlet_1.pressure.setlb(101320)
        hxd.outlet_1.pressure.setub(101330)
        hxd.heat_duty.setlb(0)
        hxd.heat_duty.setub(m.heat_duty_bound)
        hxd.shell.heat.setlb(-m.heat_duty_bound)
        hxd.shell.heat.setub(0)
        hxd.tube.heat.setlb(0)
        hxd.tube.heat.setub(m.heat_duty_bound)
        hxd.shell.properties_in[0].enth_mass.setlb(0)
        hxd.shell.properties_in[0].enth_mass.setub(1.5e6)
        hxd.shell.properties_out[0].enth_mass.setlb(0)
        hxd.shell.properties_out[0].enth_mass.setub(1.5e6)
        hxd.overall_heat_transfer_coefficient.setlb(0)
        hxd.overall_heat_transfer_coefficient.setub(10000)
        hxd.area.setlb(0)
        hxd.area.setub(5000)  # TODO: Check this value
        hxd.costing.pressure_factor.setlb(0)
        hxd.costing.pressure_factor.setub(1e5)
        hxd.costing.capital_cost.setlb(0)
        hxd.costing.capital_cost.setub(1e7)
        hxd.costing.base_cost_per_unit.setlb(0)
        hxd.costing.base_cost_per_unit.setub(1e6)
        hxd.costing.material_factor.setlb(0)
        hxd.costing.material_factor.setub(10)
        hxd.delta_temperature_in.setlb(10)
        hxd.delta_temperature_out.setlb(9.4)
        hxd.delta_temperature_in.setub(299)
        hxd.delta_temperature_out.setub(499)

    # # Add bounds to cost-related terms
    # m.fs.discharge.capital_cost.setlb(0)  # no units
    # m.fs.discharge.capital_cost.setub(1e7)

    # for salt_cost in [m.fs.discharge]:
    #     # salt_cost.capital_cost.setlb(0)
    #     # salt_cost.capital_cost.setub(1e7)
    #     salt_cost.spump_purchase_cost.setlb(0)
    #     salt_cost.spump_purchase_cost.setub(1e7)

    # Add bounds needed in VHP and HP source disjuncts
    for split in [m.fs.discharge.es_split]:
        split.to_hxd.flow_mol[:].setlb(0)
        split.to_hxd.flow_mol[:].setub(m.storage_flow_max)
        split.to_fwh.flow_mol[:].setlb(0)
        split.to_fwh.flow_mol[:].setub(0.5 * m.flow_max)
        split.split_fraction[0.0, "to_hxd"].setlb(0)
        split.split_fraction[0.0, "to_hxd"].setub(1)
        split.split_fraction[0.0, "to_fwh"].setlb(0)
        split.split_fraction[0.0, "to_fwh"].setub(1)
        split.inlet.flow_mol[:].setlb(0)
        split.inlet.flow_mol[:].setub(m.flow_max)

    m.fs.plant_power_out[0].setlb(300)
    m.fs.plant_power_out[0].setub(700)

    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(0)
        unit_k.inlet.flow_mol[:].setub(m.flow_max)
        unit_k.outlet.flow_mol[:].setlb(0)
        unit_k.outlet.flow_mol[:].setub(m.flow_max)
        # unit_k.control_volume.work.setlb(0)
        # unit_k.control_volume.work.setub(1e8)
        # unit_k.control_volume.deltaP.setlb(0)
        # unit_k.control_volume.deltaP.setub(1e8)
        # unit_k.efficiency_isentropic.setlb(0)
        # unit_k.efficiency_isentropic.setub(1)
        # # unit_k.ratioP.setlb(0)
        # # unit_k.ratioP.setub(10)

    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e12)
        m.fs.turbine[k].work.setub(0)
        m.fs.turbine[k].control_volume.deltaP.setlb(-1e10)
        m.fs.turbine[k].control_volume.deltaP.setub(0)
        m.fs.turbine[k].efficiency_isentropic.setlb(0)
        m.fs.turbine[k].efficiency_isentropic.setub(1)
        # # m.fs.turbine[k].ratioP.setlb(0)
        # # m.fs.turbine[k].ratioP.setub(1)
        # # m.fs.turbine[k].efficiency_mech.setlb(0)
        # # m.fs.turbine[k].efficiency_mech.setub(1)
        # # m.fs.turbine[k].shaft_speed.setlb(0)
        # # m.fs.turbine[k].shaft_speed.setub(100)

    for unit_k in [m.fs.bfp, m.fs.cond_pump]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s
    #     unit_k.control_volume.work.setlb(0)
    #     unit_k.control_volume.work.setub(1e9)
    #     unit_k.control_volume.deltaP.setlb(0)
    #     unit_k.control_volume.deltaP.setub(1e9)
    #     unit_k.efficiency_isentropic.setlb(0)
    #     unit_k.efficiency_isentropic.setub(1)
    #     # unit_k.ratioP.setlb(0)
    #     # unit_k.ratioP.setub(1000)

    # m.fs.bfpt.work.setlb(-1e10)
    # m.fs.bfpt.work.setub(0)
    # m.fs.bfpt.control_volume.deltaP.setlb(-1e8)
    # m.fs.bfpt.control_volume.deltaP.setub(0)
    # m.fs.bfpt.efficiency_isentropic.setlb(0)
    # m.fs.bfpt.efficiency_isentropic.setub(1)
    # m.fs.bfpt.ratioP.setlb(0)
    # m.fs.bfpt.ratioP.setub(1)
    # m.fs.bfpt.efficiency_mech.setlb(0)
    # m.fs.bfpt.efficiency_mech.setub(1)
    # m.fs.bfpt.shaft_speed.setlb(0)
    # m.fs.bfpt.shaft_speed.setub(100)

    # m.fs.plant_heat_duty[0].setlb(500)
    # m.fs.plant_heat_duty[0].setub(2000)

    # m.fs.discharge.operating_cost.setlb(0)
    # m.fs.discharge.operating_cost.setub(1e9)

    # Adding bounds on turbine splitters flow
    for k in m.set_turbine_splitter:
        ts = m.fs.turbine_splitter[k]
        ts.inlet.flow_mol[:].setlb(0)
        ts.inlet.flow_mol[:].setub(m.flow_max)
        ts.outlet_1.flow_mol[:].setlb(0)
        ts.outlet_1.flow_mol[:].setub(m.flow_max)
        ts.outlet_2.flow_mol[:].setlb(0)
        ts.outlet_2.flow_mol[:].setub(m.flow_max)
    m.fs.turbine_splitter[6].outlet_3.flow_mol[:].setlb(0)
    m.fs.turbine_splitter[6].outlet_3.flow_mol[:].setub(m.flow_max)
    m.fs.turbine_splitter[6].split_fraction[0, "outlet_3"].setlb(0)
    m.fs.turbine_splitter[6].split_fraction[0, "outlet_3"].setub(1)

    # for unit_b in [m.fs.boiler, m.fs.reheater[1], m.fs.reheater[2]]:
    #     unit_b.control_volume.heat.setlb(0)
    #     unit_b.control_volume.heat.setub(1e10)
    #     unit_b.control_volume.deltaP.setlb(-1e8)
    #     unit_b.control_volume.deltaP.setub(0)

    for j in m.set_fwh:
        fwh = m.fs.fwh[j]
        fwh.delta_temperature_in.setlb(1)
        fwh.delta_temperature_in.setub(500)
        fwh.delta_temperature_out.setlb(1)
        fwh.delta_temperature_out.setub(500)
        # fwh.tube.heat[:].setlb(0)
        # fwh.tube.heat[:].setub(1e9)
        # fwh.shell.heat[:].setlb(-1e8)
        # fwh.shell.heat[:].setub(0)
        # fwh.tube.deltaP[:].setlb(-1e8)
        # fwh.tube.deltaP[:].setub(0)
        # fwh.shell.deltaP[:].setlb(-1e9)
        # fwh.shell.deltaP[:].setub(0)

    set_fwh_mixer = [1, 2, 3, 4, 6, 7, 8]
    for p in set_fwh_mixer:
        fwhmix = m.fs.fwh_mixer[p]
        fwhmix.steam.flow_mol[:].setlb(0)
        fwhmix.steam.flow_mol[:].setub(m.flow_max)
        fwhmix.drain.flow_mol[:].setlb(0)
        fwhmix.drain.flow_mol[:].setub(m.flow_max)
        fwhmix.outlet.flow_mol[:].setlb(0)
        fwhmix.outlet.flow_mol[:].setub(m.flow_max)

    m.fs.deaerator.steam.flow_mol[:].setlb(0)
    m.fs.deaerator.steam.flow_mol[:].setub(m.flow_max)
    m.fs.deaerator.drain.flow_mol[:].setlb(0)
    m.fs.deaerator.drain.flow_mol[:].setub(m.flow_max)
    m.fs.deaerator.feedwater.flow_mol[:].setlb(0)
    m.fs.deaerator.feedwater.flow_mol[:].setub(m.flow_max)
    m.fs.deaerator.mixed_state[:].flow_mol.setlb(0)
    m.fs.deaerator.mixed_state[:].flow_mol.setub(m.flow_max)

    # m.fs.condenser_mix.main.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.main.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser_mix.bfpt.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.bfpt.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser_mix.drain.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.drain.flow_mol[:].setub(m.flow_max)
    # # m.fs.condenser_mix.makeup.flow_mol[:].setlb(0)
    # # m.fs.condenser_mix.makeup.flow_mol[:].setub(m.flow_max)
    # # m.fs.condenser_mix.outlet.flow_mol[:].setlb(0)
    # # m.fs.condenser_mix.outlet.flow_mol[:].setub(m.flow_max)

    # m.fs.condenser.inlet.flow_mol[:].setlb(0)
    # m.fs.condenser.inlet.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser.outlet.flow_mol[:].setlb(0)
    # m.fs.condenser.outlet.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser.control_volume.heat.setlb(-1e10)
    # m.fs.condenser.control_volume.heat.setub(0)

    return m


def main(m_usc, source=None, load_from_file=None):

    if load_from_file is not None:
        # Create a flowsheet, add properties, unit models, and arcs
        m = create_discharge_model(m_usc)

        # Give all the required inputs to the model
        set_model_input(m)

        # Add scaling factor
        set_scaling_factors(m)

        # Initialize the model with a sequential initialization
        print('DOF before initialization: ', degrees_of_freedom(m))
        # initialize(m, source=source)
        # print('DOF after initialization: ', degrees_of_freedom(m))

        # Add cost correlations
        m = build_costing(m, solver=solver)
        print('DOF after build costing: ', degrees_of_freedom(m))

        # Initialize with bounds
        print('Loading from file')
        ms.from_json(m, fname=load_from_file)

    else:
        # Create a flowsheet, add properties, unit models, and arcs
        m = create_discharge_model(m_usc)

        # Give all the required inputs to the model
        set_model_input(m)

        # Add scaling factor
        set_scaling_factors(m)

        # Initialize the model with a sequential initialization
        print('DOF before initialization: ', degrees_of_freedom(m))
        initialize(m, source=source)
        print('DOF after initialization: ', degrees_of_freedom(m))

        # Add cost correlations
        m = build_costing(m, solver=solver)
        print('DOF after build costing: ', degrees_of_freedom(m))

        # Initialize with bounds
        initialize_with_costing(m)

        ms.to_json(m, fname='initialized_usc_discharge.json')

    # Add bounds
    add_bounds(m)

    # Add disjunctions
    disconect_arcs(m)
    add_disjunction(m)

    return m, solver


def run_nlps(m,
             solver=None,
             source=None):
    """This function fixes the indicator variables of the disjuncts so to
    solve NLP problems

    """

    # Disjunction 1 for the steam source selection
    if source == "condpump":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh1":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh2":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh3":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh4":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh5":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "booster":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh6":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "bfp":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh8":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh9":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh1_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh2_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh3_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh5_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh6_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh8_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(True)
    else:
        print('Unrecognized source unit name!')

    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    print("The degrees of freedom after gdp transformation ",
          degrees_of_freedom(m))

    # strip_bounds = pyo.TransformationFactory('contrib.strip_var_bounds')
    # strip_bounds.apply_to(m, reversible=True)

    results = solver.solve(
        m,
        tee=True,
        symbolic_solver_labels=True,
        options={
            "linear_solver": "ma27",
            "max_iter": 150,
            # "bound_push": 1e-12,
            # "mu_init": 1e-8
        }
    )
    # log_close_to_bounds(m)

    return m, results


def print_model(solver_obj, nlp_model, nlp_data, csvfile):

    m_iter = solver_obj.iteration
    nlp_model.disjunction1_selection = {}

    nlp = nlp_model.fs.discharge
    print('       ___________________________________________')
    if nlp.condpump_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'Condpump is selected'
        print('        Disjunction 1: Condensate from Condenser Pump is selected')
    elif nlp.fwh1_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH1 is selected'
        print('        Disjunction 1: Condensate from FWH1 is selected')
    elif nlp.fwh2_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH2 is selected'
        print('        Disjunction 1: Condensate from FWH2 is selected')
    elif nlp.fwh3_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH3 is selected'
        print('        Disjunction 1: Condensate from FWH3 is selected')
    elif nlp.fwh4_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH4 is selected'
        print('        Disjunction 1: Condensate from FWH4 is selected')
    elif nlp.fwh5_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH5 is selected'
        print('        Disjunction 1: Condensate from FWH5 is selected')
    elif nlp.booster_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'Boosterpump is selected'
        print('        Disjunction 1: Condensate from Booster Pump is selected')
    elif nlp.fwh6_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH6 is selected'
        print('        Disjunction 1: Condensate from FWH6 is selected')
    elif nlp.bfp_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'BFP is selected'
        print('        Disjunction 1: Condensate from Boiler Feed Pump is selected')
    elif nlp.fwh8_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH8 is selected'
        print('        Disjunction 1: Condensate from FWH8 is selected')
    elif nlp.fwh9_source_disjunct.binary_indicator_var.value == 1:
        nlp_model.disjunction1_selection[m_iter] = 'FWH9 is selected'
        print('        Disjunction 1: Condensate from FWH9 is selected')
    else:
        print('        Disjunction 1: Error')

    print('       ___________________________________________')
    print('')

    # Save results in dictionaries
    nlp_model.objective_value = {}
    nlp_model.objective_value[m_iter] = value(nlp_model.obj) / m.scaling_obj

    if True:
        writer = csv.writer(csvfile)
        writer.writerow(
            (m_iter,
             nlp_model.disjunction1_selection[m_iter],
             nlp_model.objective_value[m_iter])
        )
        csvfile.flush()

    # log_close_to_bounds(nlp_model)
    # log_infeasible_constraints(nlp_model)


def create_csv_header():
    csvfile = open('results/subnlp_master_iterations_discharge_1-11disj_results.csv',
                   'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(
        ('Iteration', 'Disjunction 1 (Source selection)', 'Obj (MW)')
    )
    return csvfile


def run_gdp(m):
    """Declare solver GDPopt and its options
    """

    csvfile = create_csv_header()

    opt = SolverFactory('gdpopt')
    # opt.CONFIG.strategy = 'RIC'  # LOA is an option
    # opt.CONFIG.OA_penalty_factor = 1e4
    # opt.CONFIG.max_slack = 1e4
    # opt.CONFIG.call_after_subproblem_solve = print_model
    # # opt.CONFIG.mip_solver = 'glpk'
    # # opt.CONFIG.mip_solver = 'cbc'
    # opt.CONFIG.mip_solver = 'gurobi_direct'
    # opt.CONFIG.nlp_solver = 'ipopt'
    # opt.CONFIG.tee = True
    # opt.CONFIG.init_strategy = "no_init"
    # opt.CONFIG.time_limit = "2400"
    # # opt.CONFIG.subproblem_presolve = False
    _prop_bnds_root_to_leaf_map[ExternalFunctionExpression] = lambda x, y, z: None

    results = opt.solve(
        m,
        tee=True,
        algorithm='RIC',
        # mip_solver='gurobi_direct',
        mip_solver='cbc',
        nlp_solver='ipopt',
        OA_penalty_factor=1e4,
        max_slack=1e4,
        init_algorithm="no_init",
        subproblem_presolve=False,
        time_limit="2400",
        iterlim=200,
        call_after_subproblem_solve=(lambda c, a, b: print_model(c, a, b, csvfile)),
        nlp_solver_args=dict(
            tee=True,
            symbolic_solver_labels=True,
            options={
                "linear_solver": "ma27",
                "max_iter": 150
            }
        )
    )

    csvfile.close()
    return results


def print_results(m, results):

    print('================================')
    print("***************** Printing Results ******************")
    print('')
    print("Disjunctions")
    for d in m.component_data_objects(ctype=Disjunct,
                                      active=True,
                                      sort=True, descend_into=True):
        if abs(d.binary_indicator_var.value - 1) < 1e-6:
            print(d.name, ' should be selected!')
    print('')
    est = m.fs.discharge.es_turbine
    hxd = m.fs.discharge.hxd
    print('Obj (M$/year): {:.6f}'.format((value(m.obj) / m.scaling_obj) * 1e-6))
    print('Discharge capital cost ($/y): {:.6f}'.format(
        pyo.value(m.fs.discharge.capital_cost) * 1e-6))
    print('Charge Operating costs ($/y): {:.6f}'.format(
        pyo.value(m.fs.discharge.operating_cost) * 1e-6))
    print('Storage Turbine Power (MW): {:.6f}'.format(
        value(est.control_volume.work[0]) * -1e-6))
    print('Plant Power (MW): {:.6f}'.format(
        value(m.fs.plant_power_out[0])))
    print('Boiler feed water flow (mol/s): {:.6f}'.format(
        value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler duty (MW_th): {:.6f}'.format(
        value((m.fs.boiler.heat_duty[0]
               + m.fs.reheater[1].heat_duty[0]
               + m.fs.reheater[2].heat_duty[0])
              * 1e-6)))
    print('')
    print('Source: Cond pump is selected!')
    print('Heat exchanger area (m2): {:.6f}'.format(
        value(hxd.area)))
    print('Heat exchanger cost ($/y): {:.6f}'.format(
        value(hxd.costing.capital_cost / 15)))
    print('Salt flow (kg/s): {:.6f}'.format(
        value(hxd.inlet_1.flow_mass[0])))
    print('Salt temperature in (K): {:.6f}'.format(
        value(hxd.inlet_1.temperature[0])))
    print('Salt temperature out (K): {:.6f}'.format(
        value(hxd.outlet_1.temperature[0])))
    print('Steam flow to storage (mol/s): {:.6f}'.format(
        value(hxd.inlet_2.flow_mol[0])))
    print('Water temperature in (K): {:.6f}'.format(
        value(hxd.side_2.properties_in[0].temperature)))
    print('Steam temperature out (K): {:.6f}'.format(
        value(hxd.side_2.properties_out[0].temperature)))
    print('HXD overall heat transfer coefficient: {:.6f}'.format(
        value(hxd.overall_heat_transfer_coefficient[0])))
    print('Delta temperature at inlet (K): {:.6f}'.format(
        value(hxd.delta_temperature_in[0])))
    print('Delta temperature at outlet (K): {:.6f}'.format(
        value(hxd.delta_temperature_out[0])))
    print('Salt pump cost ($/y): {:.6f}'.format(
        value(m.fs.discharge.spump_purchase_cost)))
    print('')
    print('Condenser Pressure in (MPa): {:.6f}'.format(
        value(m.fs.condenser.inlet.pressure[0]) * 1e-6))
    print('Storage Turbine Pressure in (MPa): {:.6f}'.format(
        value(est.inlet.pressure[0]) * 1e-6))
    print('Storage Turbine Pressure out (MPa): {:.6f}'.format(
        value(est.outlet.pressure[0]) * 1e-6))
    print('Storage Turbine Vapor Frac in (K): {:.6f}'.format(
        value(est.control_volume.properties_in[0].vapor_frac)))
    print('Storage Turbine Temperature in (K): {:.6f}'.format(
        value(est.control_volume.properties_in[0].temperature)))
    print('Storage Turbine Temperature out (K): {:.6f}'.format(
        value(est.control_volume.properties_out[0].temperature)))
    print('Storage Turbine Saturation Temperature (K): {:.6f}'.format(
        value(est.control_volume.properties_out[0].temperature_sat)))
    print('Storage Split Fraction to HXD: {:.6f}'.format(
        value(m.fs.discharge.es_split.split_fraction[0, "to_hxd"])))
    print('')
    print('Salt dens_mass: {:.6f}'.format(
        value(hxd.side_1.properties_in[0].dens_mass['Liq'])))
    print('HXC heat duty: {:.6f}'.format(
        value(hxd.heat_duty[0]) * 1e-6))
    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')
    # for v in m.component_data_objects(Var):
    #     if v.ub is None:
    #         print(v, value(v))


def print_reports(m):

    print('')
    for unit_k in [m.fs.boiler, m.fs.reheater[1],
                   m.fs.reheater[2],
                   m.fs.bfp, m.fs.bfpt,
                   m.fs.booster,
                   m.fs.condenser_mix,
                   m.fs.charge.ess_vhp_split,
                   m.fs.charge.solar_salt_disjunct.hxc,
                   m.fs.charge.hitec_salt_disjunct.hxc,
                   m.fs.charge.thermal_oil_disjunct.hxc]:
        unit_k.display()

    for k in pyo.RangeSet(11):
        m.fs.turbine[k].report()
    for k in pyo.RangeSet(11):
        m.fs.turbine[k].display()
    for j in pyo.RangeSet(9):
        m.fs.fwh[j].report()
    for j in m.set_fwh_mixer:
        m.fs.fwh_mixer[j].display()


def model_analysis(m, solver, heat_duty=None, source=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    # Fix variables in the flowsheet
    m.fs.plant_power_out.fix(400)
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)
    m.fs.discharge.hxd.heat_duty.fix(heat_duty * 1e6)  # in W

    # Unfix variables fixed in model input and during initialization
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s
    # m.fs.boiler.inlet.flow_mol.fix(m.main_flow)  # mol/s

    # m.fs.discharge.hxd.inlet_2.unfix()
    m.fs.discharge.es_split.split_fraction[0, "to_hxd"].unfix()

    m.fs.discharge.es_split.inlet.flow_mol.unfix()   # kg/s
    m.fs.discharge.es_split.inlet.enth_mol.unfix()  # K
    m.fs.discharge.es_split.inlet.pressure.unfix()  # Pa

    m.fs.discharge.hxd.inlet_1.flow_mass.unfix()
    m.fs.discharge.hxd.area.unfix()

    # Objective function: total costs
    m.scaling_obj = 1e-5
    m.obj = Objective(
        expr=(
            (m.fs.discharge.capital_cost +
             m.fs.discharge.operating_cost
            ) / (365 * 24)
        ) * m.scaling_obj
    )

    print('DOF before solution = ', degrees_of_freedom(m))

    # Solve the design optimization modelfwh4
    # results = run_nlps(m, solver=solver, source=source)

    results = run_gdp(m)

    print_results(m, results)
    # print_reports(m)

    return m


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    m_usc = usc.build_plant_model()
    usc.initialize(m_usc)

    m, solver = main(m_usc)

    # source = ["condpump", "booster",
    #           "bfp", "fwh9",
    #           "fwh8", "fwh6",
    #           "fwh5", "fwh4",
    #           "fwh3", "fwh2",
    #           "fwh1"]

    m = model_analysis(m,
                       solver,
                       heat_duty=148.5,
                       source=None)
