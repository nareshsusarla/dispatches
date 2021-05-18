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

"""
This is a simple model for an ultrasupercritical coal-fired
power plant based on a flowsheet presented in 1999 USDOE Report #DOE/FE-0400

This model uses some of the simpler unit models from the
power generation unit model library.

Some of the parameters in the model such as feed water heater areas,
overall heat transfer coefficient, turbine efficiencies at multiple stages
have all been estimated for a total power out of 437 MW.

Additional main assumptions are as follows:
    (1) The flowsheet and main steam conditions, i. e. pressure & temperature
        are adopted from the aforementioned DOE report
    (2) Heater unit models are used to model main steam boiler,
        reheater, and condenser.
    (3) Multi-stage turbines are modeled as 
        multiple lumped single stage turbines

updated (05/11/2021)
"""

__author__ = "Naresh Susarla & Edna Soraya Rawlings"

# Import Pyomo libraries
import os
from pyomo.environ import (log, Block, Param, Constraint, Objective,
                           TransformationFactory, SolverFactory,
                           Expression, value,
                           log, exp, Var)
#--------
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.gdp import Disjunct, Disjunction
from pyomo.network.plugins import expand_arcs

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.util import get_solver, copy_port_values as _set_port
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (
    HeatExchanger,
    MomentumMixingType,
    Heater,
)
import idaes.core.util.unit_costing as icost

# Import Python libraries
from math import pi
# Import IDAES Libraries
from idaes.generic_models.unit_models import (
    Mixer,
    PressureChanger
)
from idaes.generic_models.unit_models.separator import (
    Separator,
    SplittingType
)
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
from idaes.generic_models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
#--------
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
# from idaes.generic_models.properties import iapws95

# Import ultra supercritical power plant model
import ultra_supercritical_powerplant as usc

#-------- added by esrawli
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
import solarsalt_properties_new
import hitecsalt_properties 
#--------


def create_charge_model(m):
    """Create flowsheet and add unit models.
    """

    # Create a block to add charge storage model
    m.fs.charge = Block()
    # Add molten salt properties (Solar salt)
    m.fs.solar_salt_properties = solarsalt_properties_new.SolarsaltParameterBlock()
    m.fs.hitec_salt_properties = hitecsalt_properties.HitecsaltParameterBlock()

    ###########################################################################
    #  Add hp and ip splitters                                                #
    ###########################################################################
    # Defined to divert some steam from high pressure inlet and intermediate
    # pressure inlet to charge the storage heat exchanger
    m.fs.charge.ess_hp_split = Separator(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.total,
            "split_basis": SplittingType.totalFlow,
            "ideal_separation": False,
            "outlet_list": ["to_hxc", "to_hp"],
            "has_phase_equilibrium": False
        }
    )

    #-------- added by esrawli
    ###########################################################################
    #  Add a dummy heat exchanger                                  #
    ###########################################################################
    # A connector model is defined as a dummy heat exchanger
    # with Q=0 and a deltaP=0
    m.fs.charge.connector = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": False
        }
    )
    #--------

    ###########################################################################
    #  Add cooler and hx pump                                                 #
    ###########################################################################
    # To ensure the outlet of charge heat exchanger is a subcooled liquid
    # before mixing it with the plant, a cooler is added after the heat
    # exchanger
    m.fs.charge.cooler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    # A pump, if needed, is used to increase the pressure of the water to
    # allow mixing it at a desired location within the plant
    m.fs.charge.hx_pump = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.pump,
        }
    )

    ###########################################################################
    #  Add recycle mixer                                                      #
    ###########################################################################
    m.fs.charge.recycle_mixer = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["from_bfw_out", "from_hx_pump"],
            "property_package": m.fs.prop_water,
        }
    )

    #-------- added by esrawli
    ###########################################################################
    #  Declaring the disjuncts 
    ###########################################################################
    # Disjunction 1 consists of 2 disjuncts:
    #   1. solar_salt_disjunct -------> solar salt used as the storage medium
    #   1. hitec_salt_disjunct -------> hitec salt used as the storage medium 
    #
    m.fs.charge.solar_salt_disjunct = Disjunct(rule=solar_salt_disjunct_equations)
    m.fs.charge.hitec_salt_disjunct = Disjunct(rule=hitec_salt_disjunct_equations)

    # Connecting the cooler to the two different storage heat exchangers in
    # disjunction 3
    m.fs.charge.solar_salt_disjunct.hxc_to_cooler = Arc(
        source=m.fs.charge.solar_salt_disjunct.hxc.outlet_1,
        destination=m.fs.charge.cooler.inlet
    )
    m.fs.charge.hitec_salt_disjunct.hxc_to_cooler = Arc(
        source=m.fs.charge.hitec_salt_disjunct.hxc.outlet_1,
        destination=m.fs.charge.cooler.inlet
    )
    #--------

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m)
    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)
    return m

def _make_constraints(m):
    """
    Define all constraints for the charge model
    """

    # Cooler
    # The temperature at the outlet of the cooler is required to be subcooled
    # by at least 5 degrees
    @m.fs.charge.cooler.Constraint(m.fs.time)
    def constraint_cooler_enth2(b, t):
        return (
            m.fs.charge.cooler.control_volume.properties_out[t].temperature <=
            (m.fs.charge.cooler.control_volume.properties_out[t].temperature_sat - 5)
        )

    # hx pump
    # The outlet pressure of hx_pump is then fixed to be the same as
    # boiler feed pump's outlet pressure
    @m.fs.Constraint(m.fs.time)
    def constraint_hxpump_presout(b, t):
        return m.fs.charge.hx_pump.outlet.pressure[t] == \
            (m.main_steam_pressure * 1.1231)

    # Recycle mixer
    # The outlet pressure of the recycle mixer is same as
    # the outlet pressure of the boiler feed water, i.e. inlet 'bfw_out'
    @m.fs.charge.recycle_mixer.Constraint(m.fs.time)
    def recyclemixer_pressure_constraint(b, t):
        return b.from_bfw_out_state[t].pressure == b.mixed_state[t].pressure

#-------- added by esrawli
def add_disjunction(m):
    """
    This function adds the disjunction to the model
    """

    m.fs.salt_disjunction = Disjunction(
        expr=[m.fs.charge.solar_salt_disjunct,
              m.fs.charge.hitec_salt_disjunct])

    # expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)

    return m

def solar_salt_disjunct_equations(disj):
    """
    Block of equations for disjunct 1 for the selection of 
    solar salt as the storage medium in disjunction 1
    """

    m = disj.model()

    m.fs.charge.solar_salt_disjunct.hxc = HeatExchanger(
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

    # HTC calculations for solar salt heat exchanger
    # setting the scaling factor for area = 1
    # # setting the scaling factor for overall_heat_transfer_coefficient = 1
    # iscale.set_scaling_factor(m.fs.charge.solar_salt_disjunct.hxc.area, 1e-2)
    # iscale.set_scaling_factor(
    #     m.fs.charge.solar_salt_disjunct.hxc.overall_heat_transfer_coefficient, 1e-3)
    # iscale.set_scaling_factor(m.fs.charge.solar_salt_disjunct.hxc.shell.heat, 1e-6)
    # iscale.set_scaling_factor(m.fs.charge.solar_salt_disjunct.hxc.tube.heat, 1e-6)



    m.fs.charge.data_hxc_solar = {
        'tube_thickness': 0.004,
        'tube_inner_dia': 0.032,
        'tube_outer_dia':0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    # Data needed to compute overall heat transfer coefficient for the
    # charge heat exchanger using the Sieder-Tate Correlation
    # Parameters for tube diameter and thickness assumed from the data
    # in (2017) He et al., Energy Procedia 105, 980-985
    m.fs.charge.solar_salt_disjunct.tube_thickness = Param(
        initialize=m.fs.charge.data_hxc_solar['tube_thickness'],
        doc='Tube thickness [m]'
    )
    m.fs.charge.solar_salt_disjunct.hxc.tube_inner_dia = Param(
        initialize=m.fs.charge.data_hxc_solar['tube_inner_dia'],
        doc='Tube inner diameter [m]'
    )
    m.fs.charge.solar_salt_disjunct.hxc.tube_outer_dia = Param(
        initialize=m.fs.charge.data_hxc_solar['tube_outer_dia'],
        doc='Tube outer diameter [m]'
    )
    m.fs.charge.solar_salt_disjunct.hxc.k_steel = Param(
        initialize=m.fs.charge.data_hxc_solar['k_steel'],
        doc='Thermal conductivity of steel [W/mK]'
    )
    m.fs.charge.solar_salt_disjunct.hxc.n_tubes = Param(
        initialize=m.fs.charge.data_hxc_solar['number_tubes'],
        doc='Number of tubes'
    )
    m.fs.charge.solar_salt_disjunct.hxc.shell_inner_dia = Param(
        initialize=m.fs.charge.data_hxc_solar['shell_inner_dia'],
        doc='Shell inner diameter [m]'
    )

    # Salt side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.charge.solar_salt_disjunct.hxc.tube_cs_area = Expression(
        expr=(pi / 4) * \
        (m.fs.charge.solar_salt_disjunct.hxc.tube_inner_dia ** 2),
        doc="Tube cross sectional area"
    )
    m.fs.charge.solar_salt_disjunct.hxc.tube_out_area = Expression(
        expr=(pi / 4) * \
        (m.fs.charge.solar_salt_disjunct.hxc.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]"
    )
    m.fs.charge.solar_salt_disjunct.hxc.shell_eff_area = Expression(
        expr=(
            (pi / 4) * \
            (m.fs.charge.solar_salt_disjunct.hxc.shell_inner_dia ** 2)
            - m.fs.charge.solar_salt_disjunct.hxc.n_tubes
            * m.fs.charge.solar_salt_disjunct.hxc.tube_out_area),
        doc="Effective shell cross sectional area [m2]"
    )

    # Salt (shell) side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.charge.solar_salt_disjunct.hxc.salt_reynolds_number = Expression(
        expr=(
            (m.fs.charge.solar_salt_disjunct.hxc.inlet_2.flow_mass[0] * \
             m.fs.charge.solar_salt_disjunct.hxc.tube_outer_dia) / \
            (m.fs.charge.solar_salt_disjunct.hxc.shell_eff_area * \
             m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_number = Expression(
        expr=(
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].cp_specific_heat["Liq"] * \
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] / \
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_wall = Expression(
        expr=(
            (m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_out[0].cp_specific_heat["Liq"] * \
             m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_out[0].dynamic_viscosity["Liq"]) / \
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number at wall"
    )
    m.fs.charge.solar_salt_disjunct.hxc.salt_nusselt_number = Expression(
        expr=(
            0.35 * (m.fs.charge.solar_salt_disjunct.hxc.salt_reynolds_number**0.6) * \
            (m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_number**0.4) * \
            ((m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_number / \
              m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_wall)**0.25) * (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126"
    )

    # Steam side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.charge.solar_salt_disjunct.hxc.steam_reynolds_number = Expression(
        expr=(
           m.fs.charge.solar_salt_disjunct.hxc.inlet_1.flow_mol[0] * \
            m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].mw * \
            m.fs.charge.solar_salt_disjunct.hxc.tube_inner_dia / \
            (m.fs.charge.solar_salt_disjunct.hxc.tube_cs_area * \
             m.fs.charge.solar_salt_disjunct.hxc.n_tubes * \
             m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.charge.solar_salt_disjunct.hxc.steam_prandtl_number = Expression(
        expr=(
            (m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].cp_mol / \
             m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].mw) * \
            m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].visc_d_phase["Vap"] / \
            m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.charge.solar_salt_disjunct.hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 * (m.fs.charge.solar_salt_disjunct.hxc.steam_reynolds_number**0.8) * \
            (m.fs.charge.solar_salt_disjunct.hxc.steam_prandtl_number**(0.33)) * \
            ((m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].visc_d_phase["Vap"] / \
              m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Salt and steam side heat transfer coefficients
    m.fs.charge.solar_salt_disjunct.hxc.h_salt = Expression(
        expr=(
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].thermal_conductivity["Liq"] * \
            m.fs.charge.solar_salt_disjunct.hxc.salt_nusselt_number / m.fs.charge.solar_salt_disjunct.hxc.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    m.fs.charge.solar_salt_disjunct.hxc.h_steam = Expression(
        expr=(
            m.fs.charge.solar_salt_disjunct.hxc.side_1.properties_in[0].therm_cond_phase["Vap"] * \
            m.fs.charge.solar_salt_disjunct.hxc.steam_nusselt_number / \
            m.fs.charge.solar_salt_disjunct.hxc.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    # Computing overall heat transfer coefficient
    # OHTC constraint is rewritten to avoid having a denominator in the equation
    # OHTC = _________________________________1___________________________________
    #        __1__  + _Tout_dia_*_log(Tout_dia/Tin_dia)_  + __(Tout_dia/Tin_dia)__
    #        Hsalt              2 k_steel                           Hsteam
    #
    m.fs.charge.solar_salt_disjunct.hxc.tube_dia_ratio = (
        m.fs.charge.solar_salt_disjunct.hxc.tube_outer_dia / \
        m.fs.charge.solar_salt_disjunct.hxc.tube_inner_dia
    )
    m.fs.charge.solar_salt_disjunct.hxc.log_tube_dia_ratio = log(
        m.fs.charge.solar_salt_disjunct.hxc.tube_dia_ratio)    
    @m.fs.charge.solar_salt_disjunct.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        # return (
        #     m.fs.charge.hxc.overall_heat_transfer_coefficient[t]
        #     == 1 / ((1 / m.fs.charge.hxc.h_salt)
        #             + ((m.fs.charge.hxc.tube_outer_dia * \
        #                 m.fs.charge.hxc.log_tube_dia_ratio) / \
        #                 (2 * m.fs.charge.hxc.k_steel))
        #             + (m.fs.charge.hxc.tube_dia_ratio / m.fs.charge.hxc.h_steam))
        # )
        #-------- modified by esrawli: equation rewritten to avoid having denominators
        return (
            m.fs.charge.solar_salt_disjunct.hxc.overall_heat_transfer_coefficient[t] * \
            (2 * m.fs.charge.solar_salt_disjunct.hxc.k_steel * \
             m.fs.charge.solar_salt_disjunct.hxc.h_steam
              + m.fs.charge.solar_salt_disjunct.hxc.tube_outer_dia * \
             m.fs.charge.solar_salt_disjunct.hxc.log_tube_dia_ratio *\
              m.fs.charge.solar_salt_disjunct.hxc.h_salt * \
             m.fs.charge.solar_salt_disjunct.hxc.h_steam
              + m.fs.charge.solar_salt_disjunct.hxc.tube_dia_ratio * \
             m.fs.charge.solar_salt_disjunct.hxc.h_salt * \
              2 * m.fs.charge.solar_salt_disjunct.hxc.k_steel)        
        ) == (2 * m.fs.charge.solar_salt_disjunct.hxc.k_steel * \
              m.fs.charge.solar_salt_disjunct.hxc.h_salt * \
              m.fs.charge.solar_salt_disjunct.hxc.h_steam
        )

    # Arc to connect heat exchanger inlet to the connector outlet
    m.fs.charge.solar_salt_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=m.fs.charge.solar_salt_disjunct.hxc.inlet_1
    )


def hitec_salt_disjunct_equations(disj):
    """
    Block of equations for disjunct 2 for the selection of 
    hitec salt as the storage medium in disjunction 1
    """

    m = disj.model()


    # Charge heat exchanger for storage using hitec salt
    m.fs.charge.hitec_salt_disjunct.hxc = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.prop_water
            },
            "tube": {
                "property_package": m.fs.hitec_salt_properties
            }
        }
    )

    # HTC calculations for solar salt heat exchanger
    # setting the scaling factor for area = 1
    # # setting the scaling factor for overall_heat_transfer_coefficient = 1
    # iscale.set_scaling_factor(m.fs.charge.hitec_salt_disjunct.hxc.area, 1e-2)
    # iscale.set_scaling_factor(
    #     m.fs.charge.hitec_salt_disjunct.hxc.overall_heat_transfer_coefficient, 1e-3)
    # iscale.set_scaling_factor(m.fs.charge.hitec_salt_disjunct.hxc.shell.heat, 1e-6)
    # iscale.set_scaling_factor(m.fs.charge.hitec_salt_disjunct.hxc.tube.heat, 1e-6)


    m.fs.charge.data_hxc_hitec = {
        'tube_thickness': 0.004,
        'tube_inner_dia': 0.032,
        'tube_outer_dia':0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    # Computing overall heat transfer coefficient for the heat exchanger
    # using the Sieder-Tate Correlation
    # Parameters for tube diameter and thickness assumed from the data
    # in (2017) He et al., Energy Procedia 105, 980-985
    m.fs.charge.hitec_salt_disjunct.hxc.tube_thickness = Param(
        initialize=m.fs.charge.data_hxc_hitec['tube_thickness'],
        doc='Tube thickness [m]'
    )
    m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia = Param(
        initialize=m.fs.charge.data_hxc_hitec['tube_inner_dia'],
        doc='Tube inner diameter [m]'
    )
    m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia = Param(
        initialize=m.fs.charge.data_hxc_hitec['tube_outer_dia'],
        doc='Tube outer diameter [m]'
    )
    # https://www.theworldmaterial.com/thermal-conductivity-of-stainless-steel/
    m.fs.charge.hitec_salt_disjunct.hxc.k_steel = Param(
        initialize=m.fs.charge.data_hxc_hitec['k_steel'],
        doc='Thermal conductivity of steel [W/mK]'
    )
    m.fs.charge.hitec_salt_disjunct.hxc.n_tubes = Param(
        initialize=m.fs.charge.data_hxc_hitec['number_tubes'],
        doc='Number of tubes '
    )
    m.fs.charge.hitec_salt_disjunct.hxc.shell_inner_dia = Param(
        initialize=m.fs.charge.data_hxc_hitec['shell_inner_dia'],
        doc='Shell inner diameter [m]'
    )
    
    m.fs.charge.hitec_salt_disjunct.hxc.tube_cs_area = Expression(
        expr=(pi / 4) * \
        (m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia ** 2),
        doc="Tube inside cross sectional area [m2]"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.tube_out_area = Expression(
        expr=(pi / 4) * \
        (m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.shell_eff_area = Expression(
        expr=(
            (pi / 4) * \
            (m.fs.charge.hitec_salt_disjunct.hxc.shell_inner_dia ** 2)
            - m.fs.charge.hitec_salt_disjunct.hxc.n_tubes * \
            m.fs.charge.hitec_salt_disjunct.hxc.tube_out_area),
        doc="Effective shell cross sectional area [m2]"
    )
    
    # Salt side: Reynolds number, Prandtl number, and Nusselt number
    # Reynolds number
    m.fs.charge.hitec_salt_disjunct.hxc.salt_reynolds_number = Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.flow_mass[0] * \
            m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia / \
            (m.fs.charge.hitec_salt_disjunct.hxc.shell_eff_area * \
             m.fs.charge.hitec_salt_disjunct.hxc.side_2.
             properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    
    # Prandtl number
    m.fs.charge.hitec_salt_disjunct.hxc.salt_prandtl_number = Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].
            cp_specific_heat["Liq"]
            * m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].
            dynamic_viscosity["Liq"]
            / m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].
            thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    
    # Prandtl number wall
    m.fs.charge.hitec_salt_disjunct.hxc.salt_prandtl_wall = Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_out[0].
            cp_specific_heat["Liq"]
            * m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_out[0].
            dynamic_viscosity["Liq"]
            / m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_out[0].
            thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    
    # Nusselt number
    m.fs.charge.hitec_salt_disjunct.hxc.salt_nusselt_number = Expression(
        expr=(
            1.61 * ((m.fs.charge.hitec_salt_disjunct.hxc.salt_reynolds_number
                    * m.fs.charge.hitec_salt_disjunct.hxc.salt_prandtl_number
                    * 0.009)**0.63)
            * ((m.fs.charge.hitec_salt_disjunct.hxc.side_2.
                properties_in[0].dynamic_viscosity["Liq"]
                /m.fs.charge.hitec_salt_disjunct.hxc.side_2.
                properties_out[0].dynamic_viscosity["Liq"])
                **0.25)
        ),
        doc="Salt Nusslet Number from 2014, He et al, Exp Therm Fl Sci, 59, 9"
    )
    
    # Steam side: Reynolds number, Prandtl number, and Nusselt number
    # Reynolds number
    m.fs.charge.hitec_salt_disjunct.hxc.steam_reynolds_number = Expression(
        expr=(
           m.fs.charge.hitec_salt_disjunct.hxc.inlet_1.flow_mol[0]
            * m.fs.charge.hitec_salt_disjunct.hxc.side_1.properties_in[0].mw
            * m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia
            / (m.fs.charge.hitec_salt_disjunct.hxc.tube_cs_area
               * m.fs.charge.hitec_salt_disjunct.hxc.n_tubes
               * m.fs.charge.hitec_salt_disjunct.hxc.side_1.properties_in[0].
               visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    
    # Prandtl number
    m.fs.charge.hitec_salt_disjunct.hxc.steam_prandtl_number = Expression(
        expr=(
            (m.fs.charge.hitec_salt_disjunct.hxc.side_1.properties_in[0].cp_mol
             / m.fs.charge.hitec_salt_disjunct.hxc.side_1.properties_in[0].mw)
            * m.fs.charge.hitec_salt_disjunct.hxc.
            side_1.properties_in[0].visc_d_phase["Vap"]
            / m.fs.charge.hitec_salt_disjunct.hxc.side_1.properties_in[0].
            therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    
    # Nusselt number
    m.fs.charge.hitec_salt_disjunct.hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 * (m.fs.charge.hitec_salt_disjunct.hxc.steam_reynolds_number ** 0.8)
            * (m.fs.charge.hitec_salt_disjunct.hxc.steam_prandtl_number ** (0.33))
            * ((m.fs.charge.hitec_salt_disjunct.hxc.
                side_1.properties_in[0].visc_d_phase["Vap"]
                / m.fs.charge.hitec_salt_disjunct.hxc.
                side_1.properties_out[0].visc_d_phase["Liq"]
                ) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )
    
    # Salt side heat transfer coefficient
    m.fs.charge.hitec_salt_disjunct.hxc.h_salt = Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.
            side_2.properties_in[0].thermal_conductivity["Liq"]
            * m.fs.charge.hitec_salt_disjunct.hxc.
            salt_nusselt_number / m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    # Steam side heat transfer coefficient
    m.fs.charge.hitec_salt_disjunct.hxc.h_steam = Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.
            side_1.properties_in[0].therm_cond_phase["Vap"]
            * m.fs.charge.hitec_salt_disjunct.hxc.
            steam_nusselt_number / m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    m.fs.charge.hitec_salt_disjunct.hxc.tube_dia_ratio = (
        m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia / \
        m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia
    )
    m.fs.charge.hitec_salt_disjunct.hxc.log_tube_dia_ratio = log(
        m.fs.charge.hitec_salt_disjunct.hxc.tube_dia_ratio)    
    @m.fs.charge.hitec_salt_disjunct.hxc.Constraint(m.fs.time,
                                                    doc="Hitec salt charge heat exchanger\
                                                    overall heat transfer coefficient")
    def constraint_hxc_ohtc_hitec(b, t):
        # return (
        #     m.fs.charge.hitec_salt_disjunct.hxc.overall_heat_transfer_coefficient[t]
        #     == 1 / ((1 / m.fs.charge.hitec_salt_disjunct.hxc.h_salt)
        #             + ((m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia *
        #                 log(m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia /
        #                  m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia)) /
        #                (2 * m.fs.charge.hitec_salt_disjunct.hxc.k_steel))
        #             + ((m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia /
        #                  m.fs.charge.hitec_salt_disjunct.hxc.tube_inner_dia) /
        #                m.fs.charge.hitec_salt_disjunct.hxc.h_steam))
        # )
        return (
            m.fs.charge.hitec_salt_disjunct.hxc.overall_heat_transfer_coefficient[t] * \
            (2 * m.fs.charge.hitec_salt_disjunct.hxc.k_steel * \
             m.fs.charge.hitec_salt_disjunct.hxc.h_steam
              + m.fs.charge.hitec_salt_disjunct.hxc.tube_outer_dia * \
             m.fs.charge.hitec_salt_disjunct.hxc.log_tube_dia_ratio *\
              m.fs.charge.hitec_salt_disjunct.hxc.h_salt * \
             m.fs.charge.hitec_salt_disjunct.hxc.h_steam
              + m.fs.charge.hitec_salt_disjunct.hxc.tube_dia_ratio * \
             m.fs.charge.hitec_salt_disjunct.hxc.h_salt * \
              2 * m.fs.charge.hitec_salt_disjunct.hxc.k_steel)        
        ) == (2 * m.fs.charge.hitec_salt_disjunct.hxc.k_steel * \
              m.fs.charge.hitec_salt_disjunct.hxc.h_salt * \
              m.fs.charge.hitec_salt_disjunct.hxc.h_steam
        )

    # Arc to connect heat exchanger inlet to the connector outlet
    m.fs.charge.hitec_salt_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=m.fs.charge.hitec_salt_disjunct.hxc.inlet_1
    )
#--------

def _create_arcs(m):

    # boiler to turb
    for arc_s in [m.fs.boiler_to_turb1, m.fs.bfp_to_fwh8]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()
        
    # Connecting the boiler to the ess hp splitter instead of turbine 1
    m.fs.charge.boiler_to_esshp = Arc(
        source=m.fs.boiler.outlet,
        destination=m.fs.charge.ess_hp_split.inlet
    )
    m.fs.charge.esshp_to_turb1 = Arc(
        source=m.fs.charge.ess_hp_split.to_hp,
        destination=m.fs.turbine[1].inlet
    )

    # Connections to integrate storage
    #-------- modified by esrawli
    # m.fs.charge.esshp_to_hxc = Arc(
    #     source=m.fs.charge.ess_hp_split.to_hxc,
    #     destination=m.fs.charge.hxc.inlet_1
    # )
    # Connecting the hp splitter to the connector instead
    # of the charge exchanger 
    m.fs.charge.esshp_to_connector = Arc(
        source=m.fs.charge.ess_hp_split.to_hxc,
        destination=m.fs.charge.connector.inlet
    )
    # Connecting the connector to the charge heat exchanger.
    # This would be commented and included in disjunction 1
    # m.fs.charge.connector_to_hxc = Arc(
    #     source=m.fs.charge.connector.outlet,
    #     destination=m.fs.charge.hxc.inlet_1
    # )
    # m.fs.charge.hxc_to_cooler = Arc(
    #     source=m.fs.charge.hxc.outlet_1,
    #     destination=m.fs.charge.cooler.inlet
    # )
    #--------
    m.fs.charge.cooler_to_hxpump = Arc(
        source=m.fs.charge.cooler.outlet,
        destination=m.fs.charge.hx_pump.inlet
    )
    m.fs.charge.hxpump_to_recyclemix = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer.from_hx_pump
    )

    # Connecting the bfp outlet to one inlet in the recycle mixer
    m.fs.charge.bfp_to_recyclemix = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.charge.recycle_mixer.from_bfw_out
    )
    m.fs.charge.recyclemix_to_fwh8 = Arc(
        source=m.fs.charge.recycle_mixer.outlet,
        destination=m.fs.fwh[8].inlet_2
    )

def set_model_input(m):

    # Model inputs / fixed variable or parameter values
    # assumed in this block, unless otherwise stated explicitly,
    # are either assumed or estimated for a total power out of 437 MW

    # These inputs will also fix all necessary inputs to the model
    # i.e. the degrees of freedom = 0

    ###########################################################################
    #  Charge Heat Exchanger section                                          #
    ###########################################################################
    # Heat Exchanger Area from supercritical plant model_input 
    # For conceptual design optimization, area is unfixed and optimized
    m.fs.charge.solar_salt_disjunct.hxc.area.fix(100)  # m2
    m.fs.charge.hitec_salt_disjunct.hxc.area.fix(100)  # m2

    # Salt conditions
    # Salt inlet flow is fixed during initialization but is unfixed and determined
    # during optimization
    m.fs.charge.solar_salt_disjunct.hxc.inlet_2.flow_mass.fix(100)   # kg/s
    m.fs.charge.solar_salt_disjunct.hxc.inlet_2.temperature.fix(513.15)  # K
    m.fs.charge.solar_salt_disjunct.hxc.inlet_2.pressure.fix(101325)  # Pa

    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.flow_mass.fix(100)   # kg/s
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.temperature.fix(513.15)  # K
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.pressure.fix(101325)  # Pa

    # Cooler outlet enthalpy is fixed during model build to ensure the inlet
    # to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler is
    # constrained in the model
    m.fs.charge.cooler.outlet.enth_mol[0].fix(10000)
    m.fs.charge.cooler.deltaP[0].fix(0)

    # Hx pump efficiecncy assumption
    m.fs.charge.hx_pump.efficiency_pump.fix(0.80)
    # m.fs.charge.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure * 1.1231)

    ###########################################################################
    #  ESS HP splitter                                                        #
    ###########################################################################
    # The model is built for a fixed flow of steam through the charger.
    # This flow of steam to the charger is unfixed and determine during
    # design optimization
    m.fs.charge.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.01)

    ###########################################################################
    #  Recycle mixer                                                          #
    ###########################################################################
    # m.fs.recycle_mix.ess.pressure[0].fix(24235081.4 * 1.15)
    # m.fs.charge.recycle_mixer.from_hx_pump.flow_mol.fix(0.01)
    # m.fs.charge.recycle_mixer.from_hx_pump.enth_mol.fix(103421.4)
    # m.fs.charge.recycle_mixer.from_hx_pump.pressure.fix(1131.69204)

    #-------- added by esrawli
    ###########################################################################
    #  Connector                                                         #
    ###########################################################################
    m.fs.charge.connector.heat_duty[0].fix(0) # Fix heat duty to zero for dummy conector
    #--------
    
def set_scaling_factors(m):
    """
    Scaling factors in the flowsheet
    """

    # Scaling factors for charge heat exchangers
    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc]:
        iscale.set_scaling_factor(salt_hxc.area, 1e-2)
        iscale.set_scaling_factor(salt_hxc.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(salt_hxc.shell.heat, 1e-6)
        iscale.set_scaling_factor(salt_hxc.tube.heat, 1e-6)

    # Scaling factors for storage section
    iscale.set_scaling_factor(m.fs.charge.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.cooler.control_volume.heat, 1e-6)
    # return m

def initialize(m, fileinput=None, outlvl=idaeslog.NOTSET,
               solver=None, optarg={}):

    iscale.calculate_scaling_factors(m)
    #  High pressure splitter initialization
    _set_port(m.fs.charge.ess_hp_split.inlet,
              m.fs.boiler.outlet)
    m.fs.charge.ess_hp_split.initialize(outlvl=outlvl,
                                        optarg=solver.options)

    #-------- added by esrawli
    #  Connector initialization
    _set_port(m.fs.charge.connector.inlet,
              m.fs.charge.ess_hp_split.to_hxc)
    # m.fs.charge.connector.inlet.flow_mol.fix()
    # m.fs.charge.connector.inlet.enth_mol.fix()
    # m.fs.charge.connector.inlet.pressure.fix()
    m.fs.charge.connector.initialize(outlvl=outlvl,
                                     optarg=solver.options)
    #--------
    
    #-------- modified by esrawli
    # Solar salt charge heat exchanger
    # We fix the charge steam inlet during initialization as note that
    # arcs were removed and replaced with disjuncts with equality constraints
    # Note that these should be unfixed during optimization
    _set_port(m.fs.charge.solar_salt_disjunct.hxc.inlet_1,
              m.fs.charge.connector.outlet)
    m.fs.charge.solar_salt_disjunct.hxc.inlet_1.flow_mol.fix()
    m.fs.charge.solar_salt_disjunct.hxc.inlet_1.enth_mol.fix()
    m.fs.charge.solar_salt_disjunct.hxc.inlet_1.pressure.fix()
    m.fs.charge.solar_salt_disjunct.hxc.initialize(outlvl=outlvl,
                                                   optarg=solver.options)

    #  Hitec salt charge heat exchanger initialization
    # We fix the charge steam inlet during initialization as note that
    # arcs were removed and replaced with disjuncts with equality constraints
    # Note that this should be unfixed during optimization
    _set_port(m.fs.charge.hitec_salt_disjunct.hxc.inlet_1,
              m.fs.charge.connector.outlet)
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_1.flow_mol.fix()
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_1.enth_mol.fix()
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_1.pressure.fix()
    m.fs.charge.hitec_salt_disjunct.hxc.initialize(
        outlvl=outlvl, optarg=solver.options)
    #--------
    
    #  Cooler
    _set_port(m.fs.charge.cooler.inlet,
              m.fs.charge.solar_salt_disjunct.hxc.outlet_1)
    m.fs.charge.cooler.initialize(outlvl=outlvl,
                                  optarg=solver.options)
    
    # HX pump
    _set_port(m.fs.charge.hx_pump.inlet,
              m.fs.charge.cooler.outlet)
    m.fs.charge.hx_pump.initialize(outlvl=outlvl,
                                   optarg=solver.options)

    #  Recycle mixer initialization
    _set_port(m.fs.charge.recycle_mixer.from_bfw_out,
              m.fs.bfp.outlet)
    _set_port(m.fs.charge.recycle_mixer.from_hx_pump,
              m.fs.charge.hx_pump.outlet)

    # fixing to a small value in mol/s
    # m.fs.charge.recycle_mixer.from_hx_pump.flow_mol.fix(0.01)
    # m.fs.charge.recycle_mixer.from_hx_pump.enth_mol.fix()
    # m.fs.charge.recycle_mixer.from_hx_pump.pressure.fix()
    m.fs.charge.recycle_mixer.initialize(outlvl=outlvl)#, optarg=solver.options)

    res = solver.solve(m)
    print("Charge Model Initialization = ",
          res.solver.termination_condition)
    print("***************   Charge Model Initialized   ********************")

def build_costing(m, solver=None, optarg={}):
    """
    This function adds the cost correlations for the storage design analysis
    This function does not compute the cost of power plant itself but has the 
    primary aim to estimate the capital and operatig cost of integrating an
    energy storage system.

    Specifically, this method contains cost correlations to estimate the 
    capital cost of charge heat exchanger, salt storage tank, molten salt 
    pump, and salt inventory. 

    All capital cost computed is annualized and operating cost is for 1 year
    In addition, operating savings in terms of annual coal cost are estimated
    based on the differential reduction of coal consumption as compared to
    ramped baseline power plant

    Unless other wise stated, the cost correlations used here (except IDAES
    costing method) are taken from 2nd Edition, Product & Process Design
    Principles, Seider et al.
    """

    
    # ###########################################################################
    # #  Solver options                                                         #
    # ###########################################################################
    # optarg = {
    #     "max_iter": 300,
    #     "halt_on_ampl_error": "yes",
    # }
    # solver = get_solver(solver, optarg)

    ###########################################################################
    #  Data                                                                   #
    ###########################################################################
    m.CE_index = 607.5  # Chemical engineering cost index for 2019

    # q_baseline_charge corresponds to heat duty of a plant with no storage
    # and producing 400 MW power
    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'q_baseline_charge': 838565942.4732262,
        'solar_salt_price': 0.49,
        'hitec_salt_price': 0.93,
        'storage_tank_material': 3.5,
        'storage_tank_insulation': 235,
        'storage_tank_foundation': 1210
    }

    m.data_salt_pump = {
        'FT':1.5,
        'FM':2.0,
        'head':3.281*5,
        'motor_FT':1,
        'nm':1
    }

    m.data_storage_tank = {
        'LbyD': 0.325,
        'tank_thickness':0.039,
        'material_density':7800
    }
    
    # Main flowsheet operation data
    m.fs.charge.coal_price = Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.charge.cooling_price = Param(
        initialize=m.data_cost['cooling_price'],
        doc='Cost of chilled water for cooler from Sieder et al. $/J')

    m.fs.q_baseline = Param(
        initialize=m.data_cost['q_baseline_charge'],
        doc='Boiler duty in Wth @ 699MW for baseline plant with no storage')

    m.fs.charge.solar_salt_price = Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')
    m.fs.charge.hitec_salt_price = Param(
        initialize=m.data_cost['hitec_salt_price'],
        doc='Hitec salt price in $/kg')


    ###########################################################################
    #  Operating hours                                                        #
    ###########################################################################
    m.number_hours_per_day = 6          
    m.number_of_years = 5

    m.fs.charge.hours_per_day = Var(
        initialize=m.number_hours_per_day,
        bounds=(0, 12),
        doc='Estimated number of hours of charging per day'
    )
    
    # Fixing number of hours of discharging to 6 in this problem
    m.fs.charge.hours_per_day.fix(m.number_hours_per_day)

    # Number of year over which the capital cost is annualized
    m.fs.charge.num_of_years = Param(
        initialize=m.number_of_years,
        doc='Number of years for capital cost annualization')


    ###########################################################################
    #  Capital cost                                                           #
    ###########################################################################

    #--------------------------------------------
    #  Solar salt inventory
    #--------------------------------------------
    # Total inventory of salt required
    m.fs.charge.solar_salt_disjunct.salt_amount = Expression(
        expr=(m.fs.charge.solar_salt_disjunct.hxc.inlet_2.flow_mass[0] * \
              m.fs.charge.hours_per_day * 3600),
        doc="Salt flow in gal per min"
    )

    # Purchase cost of salt
    m.fs.charge.solar_salt_disjunct.salt_purchase_cost = Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt purchase cost in $"
    )

    def solar_salt_purchase_cost_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.salt_purchase_cost * \
            m.fs.charge.num_of_years == (
                m.fs.charge.solar_salt_disjunct.salt_amount * \
                m.fs.charge.solar_salt_price
            )
        )
    m.fs.charge.solar_salt_disjunct.salt_purchase_cost_eq = Constraint(
        rule=solar_salt_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.salt_purchase_cost,
        m.fs.charge.solar_salt_disjunct.salt_purchase_cost_eq)

    #--------------------------------------------
    #  Hitec salt inventory
    #--------------------------------------------
    # Total inventory of salt required
    m.fs.charge.hitec_salt_disjunct.salt_amount = Expression(
        expr=(m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Salt flow in gal per min"
    )
    
    # Purchase cost of salt
    m.fs.charge.hitec_salt_disjunct.salt_purchase_cost = Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt purchase cost in $"
    )

    def hitec_salt_purchase_cost_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.salt_purchase_cost * \
            m.fs.charge.num_of_years
            == (m.fs.charge.hitec_salt_disjunct.salt_amount * \
                m.fs.charge.hitec_salt_price)
        )
    m.fs.charge.hitec_salt_disjunct.salt_purchase_cost_eq = \
        Constraint(rule=hitec_salt_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.salt_purchase_cost,
        m.fs.charge.hitec_salt_disjunct.salt_purchase_cost_eq)

    #--------------------------------------------
    #  Solar salt charge heat exchangers costing
    #--------------------------------------------
    # The charge heat exchanger cost is estimated using the IDAES costing
    # method with default options, i.e. a U-tube heat exchnager, stainless
    # steel material, and a tube length of 12ft. Refer to costing documentation
    # to change any of the default options
    # Purchase cost of heat exchanger has to be annualized when used
    m.fs.charge.solar_salt_disjunct.hxc.get_costing()
    m.fs.charge.solar_salt_disjunct.hxc.costing.CE_index = m.CE_index
    # Initializing the costing correlations
    icost.initialize(m.fs.charge.solar_salt_disjunct.hxc.costing)

    # Hitec salt heat exchanger
    m.fs.charge.hitec_salt_disjunct.hxc.get_costing()
    m.fs.charge.hitec_salt_disjunct.hxc.costing.CE_index = m.CE_index
    icost.initialize(m.fs.charge.hitec_salt_disjunct.hxc.costing)

    #--------------------------------------------
    #  Water pump
    #--------------------------------------------
    # The water pump is added as hx_pump before sending the feed water
    # to the discharge heat exchnager in order to increase its pressure.
    # The outlet pressure of hx_pump is set based on its return point
    # in the flowsheet
    # Purchase cost of hx_pump has to be annualized when used
    m.fs.charge.hx_pump.get_costing(
        Mat_factor="stain_steel",
        mover_type="compressor",
        compressor_type="centrifugal",
        driver_mover_type="electrical_motor",
        pump_type="centrifugal",
        pump_type_factor='1.4',
        pump_motor_type_factor='open'
        )
    m.fs.charge.hx_pump.costing.CE_index = m.CE_index
    # Initialize cost correlation
    icost.initialize(m.fs.charge.hx_pump.costing)

    #--------------------------------------------
    #  Salt-pump costing
    #--------------------------------------------
    # The salt pump is not explicitly modeled. Thus, the IDAES cost method
    # is not used for this equipment at this time.
    # The primary purpose of the salt pump is to move molten salt and not to
    # change the pressure. Thus the pressure head is computed assuming that
    # the salt is moved on an average of 5m linear distance.
    m.fs.charge.spump_FT = pyo.Param(
        initialize=m.data_salt_pump['FT'],
        doc='Pump Type Factor for vertical split case')
    m.fs.charge.spump_FM = pyo.Param(
        initialize=m.data_salt_pump['FM'],
        doc='Pump Material Factor Stainless Steel')
    m.fs.charge.spump_head = pyo.Param(
        initialize=m.data_salt_pump['head'],
        doc='Pump Head 5m in Ft.')
    m.fs.charge.spump_motorFT = pyo.Param(
        initialize=m.data_salt_pump['motor_FT'],
        doc='Motor Shaft Type Factor')
    m.fs.charge.spump_nm = pyo.Param(
        initialize=m.data_salt_pump['nm'],
        doc='Motor Shaft Type Factor')

    #--------------------------------------------
    #  Solar salt-pump costing
    # converting salt flow mas to flow vol in gal per min, Q_gpm
    m.fs.charge.solar_salt_disjunct.spump_Qgpm = pyo.Expression(
        expr=(m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].flow_mass * \
              264.17 * 60 / \
              (m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"])),
        doc="Salt flow in gal per min"
    )
    # Density in lb per ft3
    m.fs.charge.solar_salt_disjunct.dens_lbft3 = pyo.Expression(
        expr=m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].
        density["Liq"] * 0.062428,
        doc="pump size factor"
    )
    # Pump size factor
    m.fs.charge.solar_salt_disjunct.spump_sf = pyo.Expression(
        expr=m.fs.charge.solar_salt_disjunct.spump_Qgpm * (m.fs.charge.spump_head ** 0.5),
        doc="pump size factor"
    )
    # Pump purchase cost
    m.fs.charge.solar_salt_disjunct.pump_CP = pyo.Expression(
        expr=(m.fs.charge.spump_FT * m.fs.charge.spump_FM * \
              exp(9.2951
                  - 0.6019 * log(m.fs.charge.solar_salt_disjunct.spump_sf)
                  + 0.0519 * ((log(m.fs.charge.solar_salt_disjunct.spump_sf))**2)
              )
        ),
        doc="Salt Pump Base Cost in $"
    )
    
    # Costing motor
    m.fs.charge.solar_salt_disjunct.spump_np = pyo.Expression(
        expr=(-0.316
              + 0.24015 * log(m.fs.charge.solar_salt_disjunct.spump_Qgpm)
              - 0.01199 * ((log(m.fs.charge.solar_salt_disjunct.spump_Qgpm))**2)
        ),
        doc="fractional efficiency of the pump horse power"
    )
    # Motor power consumption
    m.fs.charge.solar_salt_disjunct.motor_pc = pyo.Expression(
        expr=((m.fs.charge.solar_salt_disjunct.spump_Qgpm * \
               m.fs.charge.spump_head * \
               m.fs.charge.solar_salt_disjunct.dens_lbft3) / \
              (33000 * m.fs.charge.solar_salt_disjunct.spump_np * \
               m.fs.charge.spump_nm)
        ),
        doc="Horsepower consumption"
    )
    # Motor purchase cost
    m.fs.charge.solar_salt_disjunct.motor_CP = pyo.Expression(
        expr=(m.fs.charge.spump_motorFT
              * exp(5.4866
                    + 0.13141 * log(m.fs.charge.solar_salt_disjunct.motor_pc)
                    + 0.053255 * ((log(m.fs.charge.solar_salt_disjunct.motor_pc))**2)
                    + 0.028628 * ((log(m.fs.charge.solar_salt_disjunct.motor_pc))**3)
                    - 0.0035549 * ((log(m.fs.charge.solar_salt_disjunct.motor_pc))**4))
              ),
        doc="Salt Pump's Motor Base Cost in $"
    )

    # Pump and motor purchase cost
    # pump toal cost constraint
    m.fs.charge.solar_salt_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt pump and motor purchase cost in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.spump_purchase_cost * \
            m.fs.charge.num_of_years
            == (m.fs.charge.solar_salt_disjunct.pump_CP
                + m.fs.charge.solar_salt_disjunct.motor_CP) * \
            (m.CE_index / 394)
        )
    m.fs.charge.solar_salt_disjunct.spump_purchase_cost_eq = pyo.Constraint(
        rule=solar_spump_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.spump_purchase_cost,
        m.fs.charge.solar_salt_disjunct.spump_purchase_cost_eq)

    #--------------------------------------------
    #  Hitec salt pump costing
    # The primary purpose of the salt pump is to move molten salt and not to
    # change the pressure. Thus the pressure head is computed assuming that
    # the salt is moved on an average of 5m linear distance.

    # converting salt flow mas to flow vol in gal per min, Q_gpm
    m.fs.charge.hitec_salt_disjunct.spump_Qgpm = Expression(
        expr=(m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].flow_mass * \
              264.17 * 60 / \
              (m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"])
        ),
        doc="Salt flow in gal per min"
    )
    # Density in lb per ft3
    m.fs.charge.hitec_salt_disjunct.dens_lbft3 = Expression(
        expr=m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].
        density["Liq"] * 0.062428,
        doc="pump size factor"
    )
    # Pump size factor
    m.fs.charge.hitec_salt_disjunct.spump_sf = Expression(
        expr=m.fs.charge.hitec_salt_disjunct.spump_Qgpm * \
        (m.fs.charge.spump_head ** 0.5),
        doc="pump size factor"
    )
    # Pump purchase cost
    m.fs.charge.hitec_salt_disjunct.pump_CP = Expression(
        expr=(m.fs.charge.spump_FT * m.fs.charge.spump_FM * exp(
            9.2951
            - 0.6019 * log(m.fs.charge.hitec_salt_disjunct.spump_sf)
            + 0.0519 * ((log(m.fs.charge.hitec_salt_disjunct.spump_sf))**2))
        ),
        doc="Salt Pump Base Cost in $"
    )
    # Costing motor
    m.fs.charge.hitec_salt_disjunct.spump_np = Expression(
        expr=(-0.316
              + 0.24015 * log(m.fs.charge.hitec_salt_disjunct.spump_Qgpm)
              - 0.01199 * ((log(m.fs.charge.hitec_salt_disjunct.spump_Qgpm))**2)
        ),
        doc="fractional efficiency of the pump horse power"
    )
    # Motor power consumption
    m.fs.charge.hitec_salt_disjunct.motor_pc = Expression(
        expr=(
            (m.fs.charge.hitec_salt_disjunct.spump_Qgpm * \
             m.fs.charge.spump_head * m.fs.charge.hitec_salt_disjunct.dens_lbft3)
            / (33000 * m.fs.charge.hitec_salt_disjunct.spump_np * m.fs.charge.spump_nm)
        ),
        doc="Horsepower consumption"
    )
    # Motor purchase cost
    m.fs.charge.hitec_salt_disjunct.motor_CP = Expression(
        expr=(m.fs.charge.spump_motorFT * exp(
            5.4866
            + 0.13141 * log(m.fs.charge.hitec_salt_disjunct.motor_pc)
            + 0.053255 * ((log(m.fs.charge.hitec_salt_disjunct.motor_pc))**2)
            + 0.028628 * ((log(m.fs.charge.hitec_salt_disjunct.motor_pc))**3)
            - 0.0035549 * ((log(m.fs.charge.hitec_salt_disjunct.motor_pc))**4))
        ),
        doc="Salt Pump's Motor Base Cost in $"
    )

    # Pump and motor purchase cost
    # pump toal cost constraint
    m.fs.charge.hitec_salt_disjunct.spump_purchase_cost = Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt pump and motor purchase cost in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.spump_purchase_cost * \
            m.fs.charge.num_of_years == (
                m.fs.charge.hitec_salt_disjunct.pump_CP 
                + m.fs.charge.hitec_salt_disjunct.motor_CP) * (m.CE_index / 394)
        )
    m.fs.charge.hitec_salt_disjunct.spump_purchase_cost_eq = Constraint(
        rule=solar_spump_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.spump_purchase_cost,
        m.fs.charge.hitec_salt_disjunct.spump_purchase_cost_eq)

    #--------------------------------------------
    #  Solar salt storage tank costing: vertical vessel
    #--------------------------------------------
    m.fs.charge.l_by_d = pyo.Param(
        initialize=m.data_storage_tank['LbyD'],
        doc='L by D assumption for computing storage tank dimensions')
    m.fs.charge.tank_thickness = pyo.Param(
        initialize=m.data_storage_tank['tank_thickness'],
        doc='Storage tank thickness assumed based on reference'
    )
    # Tank size and dimension computation
    m.fs.charge.solar_salt_disjunct.tank_volume = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank w/20% excess capacity")
    m.fs.charge.solar_salt_disjunct.tank_surf_area = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge.solar_salt_disjunct.tank_diameter = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge.solar_salt_disjunct.tank_height = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank [m]")
    m.fs.charge.solar_salt_disjunct.no_of_tanks = pyo.Var(
        initialize=1,
        bounds=(1, 3),
        doc='No of Tank units to use cost correlations')

    # Number of tanks change
    m.fs.charge.solar_salt_disjunct.no_of_tanks.fix()

    # Computing tank volume - jfr: editing to include 20% margin
    def solar_tank_volume_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.tank_volume * \
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"] == \
            m.fs.charge.solar_salt_disjunct.salt_amount * 1.10
        )
    m.fs.charge.solar_salt_disjunct.tank_volume_eq = pyo.Constraint(
        rule=solar_tank_volume_rule)

    # Compute surface area of tank: surf area of sides + top surf area
    # base area is accounted in foundation costs
    def solar_tank_surf_area_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.tank_surf_area == (
                pi * m.fs.charge.solar_salt_disjunct.tank_diameter * \
                m.fs.charge.solar_salt_disjunct.tank_height)
            + (pi * m.fs.charge.solar_salt_disjunct.tank_diameter**2) / 4
        )
    m.fs.charge.solar_salt_disjunct.tank_surf_area_eq = pyo.Constraint(
        rule=solar_tank_surf_area_rule)

    # Computing diameter for an assumed L by D
    def solar_tank_diameter_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.tank_diameter == (
                (4 * (m.fs.charge.solar_salt_disjunct.tank_volume / \
                      m.fs.charge.solar_salt_disjunct.no_of_tanks) / \
                 (m.fs.charge.l_by_d * pi))** (1 / 3))
        )
    m.fs.charge.solar_salt_disjunct.tank_diameter_eq = pyo.Constraint(
        rule=solar_tank_diameter_rule)

    # Computing height of tank
    def solar_tank_height_rule(b):
        return m.fs.charge.solar_salt_disjunct.tank_height == (
            m.fs.charge.l_by_d * m.fs.charge.solar_salt_disjunct.tank_diameter)
    m.fs.charge.solar_salt_disjunct.tank_height_eq = pyo.Constraint(
        rule=solar_tank_height_rule)

    # Initialize tanks design correlations
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.tank_volume,
        m.fs.charge.solar_salt_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.tank_diameter,
        m.fs.charge.solar_salt_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.tank_height,
        m.fs.charge.solar_salt_disjunct.tank_height_eq)

    # A dummy pyomo block for salt storage tank is declared for costing
    # The diameter and length for this tank is assumed
    # based on a number of tank (see the above for m.fs.no_of_tanks)
    # Costing for each vessel designed above
    # m.fs.charge.salt_tank = pyo.Block()
    m.fs.charge.solar_salt_disjunct.costing = pyo.Block()

    m.fs.charge.solar_salt_disjunct.costing.material_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_material'],
        doc='$/kg of SS316 material'
    )

    m.fs.charge.solar_salt_disjunct.costing.insulation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_insulation'],
        doc='$/m2'
    )

    m.fs.charge.solar_salt_disjunct.costing.foundation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_foundation'],
        doc='$/m2'
    )

    m.fs.charge.solar_salt_disjunct.costing.material_density = pyo.Param(
        initialize=m.data_storage_tank['material_density'],
        doc='Kg/m3'
    )

    m.fs.charge.solar_salt_disjunct.costing.tank_material_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    m.fs.charge.solar_salt_disjunct.costing.tank_insulation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    m.fs.charge.solar_salt_disjunct.costing.tank_foundation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def rule_tank_material_cost(b):
        return m.fs.charge.solar_salt_disjunct.costing.tank_material_cost == (
            m.fs.charge.solar_salt_disjunct.costing.material_cost * \
            m.fs.charge.solar_salt_disjunct.costing.material_density * \
            m.fs.charge.solar_salt_disjunct.tank_surf_area * \
            m.fs.charge.tank_thickness
        )
    m.fs.charge.solar_salt_disjunct.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_tank_material_cost)

    def rule_tank_insulation_cost(b):
        return m.fs.charge.solar_salt_disjunct.costing.tank_insulation_cost == (
            m.fs.charge.solar_salt_disjunct.costing.insulation_cost * \
            m.fs.charge.solar_salt_disjunct.tank_surf_area
        )
    m.fs.charge.solar_salt_disjunct.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_tank_insulation_cost)

    def rule_tank_foundation_cost(b):
        return m.fs.charge.solar_salt_disjunct.costing.tank_foundation_cost == (
            m.fs.charge.solar_salt_disjunct.costing.foundation_cost * \
            pi * m.fs.charge.solar_salt_disjunct.tank_diameter**2 / 4
        )
    m.fs.charge.solar_salt_disjunct.costing.eq_tank_foundation_cost = pyo.Constraint(
        rule=rule_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.charge.solar_salt_disjunct.costing.total_tank_cost = pyo.Expression(
        expr=m.fs.charge.solar_salt_disjunct.costing.tank_material_cost 
        + m.fs.charge.solar_salt_disjunct.costing.tank_foundation_cost 
        + m.fs.charge.solar_salt_disjunct.costing.tank_insulation_cost
    )

    #--------------------------------------------
    #  Hitec salt storage tank costing: vertical vessel
    #--------------------------------------------
    # Tank size and dimension computation
    m.fs.charge.hitec_salt_disjunct.tank_volume = Var(
        initialize=1000,
        bounds=(1, 10000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank w/20% excess capacity")
    m.fs.charge.hitec_salt_disjunct.tank_surf_area = Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge.hitec_salt_disjunct.tank_diameter = Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge.hitec_salt_disjunct.tank_height = Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank [m]")
    m.fs.charge.hitec_salt_disjunct.no_of_tanks = Var(
        initialize=1, bounds=(1, 4),
        doc='No of Tank units to use cost correlations')

    # Number of tanks change
    m.fs.charge.hitec_salt_disjunct.no_of_tanks.fix()

    # Computing tank volume with a 20% margin
    def hitec_tank_volume_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.tank_volume * \
            m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"] == \
            m.fs.charge.hitec_salt_disjunct.salt_amount * 1.10
        )
    m.fs.charge.hitec_salt_disjunct.tank_volume_eq = Constraint(
        rule=hitec_tank_volume_rule)

    # Compute surface area of tank: surf area of sides + top surf area
    # base area is accounted in foundation costs
    def hitec_tank_surf_area_rule(b):
        return m.fs.charge.hitec_salt_disjunct.tank_surf_area == (
            (pi * m.fs.charge.hitec_salt_disjunct.tank_diameter * \
             m.fs.charge.hitec_salt_disjunct.tank_height)
            + (pi*m.fs.charge.hitec_salt_disjunct.tank_diameter**2) / 4
        )
    m.fs.charge.hitec_salt_disjunct.tank_surf_area_eq = Constraint(
        rule=hitec_tank_surf_area_rule)

    # Computing diameter for an assumed L by D
    def hitec_tank_diameter_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.tank_diameter ==
            ((4 * (m.fs.charge.hitec_salt_disjunct.tank_volume /
                   m.fs.charge.hitec_salt_disjunct.no_of_tanks)
              / (m.fs.charge.l_by_d * pi)) ** (1 / 3))
        )
    m.fs.charge.hitec_salt_disjunct.tank_diameter_eq = \
        Constraint(rule=hitec_tank_diameter_rule)

    # Computing height of tank
    def hitec_tank_height_rule(b):
        return m.fs.charge.hitec_salt_disjunct.tank_height == \
            m.fs.charge.l_by_d * m.fs.charge.hitec_salt_disjunct.tank_diameter
    m.fs.charge.hitec_salt_disjunct.tank_height_eq = \
        Constraint(rule=hitec_tank_height_rule)

    # Initialize tanks design correlations
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.tank_volume,
        m.fs.charge.hitec_salt_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.tank_diameter,
        m.fs.charge.hitec_salt_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.tank_height,
        m.fs.charge.hitec_salt_disjunct.tank_height_eq)

    # A dummy pyomo block for salt storage tank is declared for costing
    # The diameter and length for this tank is assumed
    # based on a number of tank (see the above for m.fs.charge.no_of_tanks)
    # Costing for each vessel designed above
    m.fs.charge.hitec_salt_disjunct.costing = Block()

    m.fs.charge.hitec_salt_disjunct.costing.material_cost = Param(
        initialize=m.data_cost['storage_tank_material'],
        doc='$/kg of SS316 material'
    )

    m.fs.charge.hitec_salt_disjunct.costing.insulation_cost = Param(
        initialize=m.data_cost['storage_tank_insulation'],
        doc='$/m2'
    )

    m.fs.charge.hitec_salt_disjunct.costing.foundation_cost = Param(
        initialize=m.data_cost['storage_tank_foundation'],
        doc='$/m2'
    )

    m.fs.charge.hitec_salt_disjunct.costing.material_density = Param(
        initialize=m.data_storage_tank['material_density'],
        doc='Kg/m3'
    )

    m.fs.charge.hitec_salt_disjunct.costing.tank_material_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    m.fs.charge.hitec_salt_disjunct.costing.tank_insulation_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    m.fs.charge.hitec_salt_disjunct.costing.tank_foundation_cost = Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def rule_hitec_tank_material_cost(b):
        return m.fs.charge.hitec_salt_disjunct.costing.tank_material_cost == (
            m.fs.charge.hitec_salt_disjunct.costing.material_cost * \
            m.fs.charge.hitec_salt_disjunct.costing.material_density * \
            m.fs.charge.hitec_salt_disjunct.tank_surf_area * \
            m.fs.charge.tank_thickness
        )
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_material_cost = Constraint(
        rule=rule_hitec_tank_material_cost)

    def rule_hitec_tank_insulation_cost(b):
        return (m.fs.charge.hitec_salt_disjunct.costing.tank_insulation_cost ==
                m.fs.charge.hitec_salt_disjunct.costing.insulation_cost *
                m.fs.charge.hitec_salt_disjunct.tank_surf_area)
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_insulation_cost = Constraint(
        rule=rule_hitec_tank_insulation_cost)

    def rule_hitec_tank_foundation_cost(b):
        return (m.fs.charge.hitec_salt_disjunct.costing.tank_foundation_cost ==
                m.fs.charge.hitec_salt_disjunct.costing.foundation_cost *
                pi * m.fs.charge.hitec_salt_disjunct.tank_diameter**2 / 4)
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_foundation_cost = Constraint(
        rule=rule_hitec_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.charge.hitec_salt_disjunct.costing.total_tank_cost = Expression(
        expr=m.fs.charge.hitec_salt_disjunct.costing.tank_material_cost
        + m.fs.charge.hitec_salt_disjunct.costing.tank_foundation_cost 
        + m.fs.charge.hitec_salt_disjunct.costing.tank_insulation_cost
    )


    #--------------------------------------------
    # Total annualized capital cost for solar salt
    #--------------------------------------------
    # Capital cost var at flowsheet level to handle the salt capital cost
    # depending on the salt selected.
    m.fs.charge.capital_cost = pyo.Var(
        initialize=1000000,
        doc="Annualized capital cost")

    m.fs.charge.solar_salt_disjunct.capital_cost = Var(
        initialize=1000000,
        doc="Annualized capital cost for solar salt")


    # Annualized capital cost for the solar salt
    def solar_cap_cost_rule(b):
        return m.fs.charge.solar_salt_disjunct.capital_cost == (
            m.fs.charge.solar_salt_disjunct.salt_purchase_cost 
            + m.fs.charge.solar_salt_disjunct.spump_purchase_cost
            + (
                m.fs.charge.solar_salt_disjunct.hxc.costing.purchase_cost
                + m.fs.charge.hx_pump.costing.purchase_cost
                + m.fs.charge.solar_salt_disjunct.no_of_tanks
                * m.fs.charge.solar_salt_disjunct.costing.total_tank_cost
               )
            / m.fs.charge.num_of_years
        )
    m.fs.charge.solar_salt_disjunct.cap_cost_eq = pyo.Constraint(
        rule=solar_cap_cost_rule)
    
    # Adding constraint to link the fs capital cost var to
    # solar salt disjunct
    m.fs.charge.solar_salt_disjunct.fs_cap_cost_eq = Constraint(
        expr=m.fs.charge.capital_cost == m.fs.charge.solar_salt_disjunct.capital_cost)

    
    # Initialize
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.capital_cost,
        m.fs.charge.solar_salt_disjunct.cap_cost_eq)

    #--------------------------------------------
    # Total annualized capital cost for hitec salt
    #--------------------------------------------
    m.fs.charge.hitec_salt_disjunct.capital_cost = Var(
        initialize=1000000,
        doc="Annualized capital cost for solar salt")

    # Annualized capital cost for the solar salt
    def hitec_cap_cost_rule(b):
        return m.fs.charge.hitec_salt_disjunct.capital_cost == (
            m.fs.charge.hitec_salt_disjunct.salt_purchase_cost
            + m.fs.charge.hitec_salt_disjunct.spump_purchase_cost
            + (m.fs.charge.hitec_salt_disjunct.hxc.costing.purchase_cost
               + m.fs.charge.hx_pump.costing.purchase_cost
               + m.fs.charge.hitec_salt_disjunct.no_of_tanks * \
               m.fs.charge.hitec_salt_disjunct.costing.total_tank_cost)
            / m.fs.charge.num_of_years
        )
    m.fs.charge.hitec_salt_disjunct.cap_cost_eq = Constraint(
        rule=hitec_cap_cost_rule)

    # Adding constraint to link the fs capital cost var to
    # solar salt disjunct
    m.fs.charge.hitec_salt_disjunct.fs_cap_cost_eq = Constraint(
        expr=m.fs.charge.capital_cost == m.fs.charge.hitec_salt_disjunct.capital_cost)

    # Initialize
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.capital_cost,
        m.fs.charge.hitec_salt_disjunct.cap_cost_eq)


    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.charge.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.charge.hours_per_day,
        doc="Number of operating hours per year"
    )

    # Operating cost var
    m.fs.charge.operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(1e-5, None),
        doc="Annualized capital cost")

    def op_cost_rule(b):
        return m.fs.charge.operating_cost == (
            m.fs.charge.operating_hours * m.fs.charge.coal_price * \
            (m.fs.plant_heat_duty[0] * 1e6
                - m.fs.q_baseline)
            - (m.fs.charge.cooling_price * m.fs.charge.operating_hours * \
               m.fs.charge.cooler.heat_duty[0])
        )
    m.fs.charge.op_cost_eq = pyo.Constraint(rule=op_cost_rule)

    # Initialize
    calculate_variable_from_constraint(
        m.fs.charge.operating_cost,
        m.fs.charge.op_cost_eq)

    res = solver.solve(m,
                       tee=True,
                       symbolic_solver_labels=True)
    print("Cost Initialization = ",
          res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print('')
    print('')

    # return solver
#--------

def view_result(outfile, m):
    tags = {}

    usc.view_result(outfile, m)
    # Boiler
    tags['obj'] = ("%4.2f" % value(m.obj))

    # ESS HP Splitter
    tags['essvhp_Fout1'] = ("%4.3f" % (value(
        m.fs.charge.ess_hp_split.to_hp.flow_mol[0])*1e-3))
    tags['essvhp_Tout1'] = ("%4.2f" % (value(
        m.fs.charge.ess_hp_split.to_hp_state[0].temperature)))
    tags['essvhp_Pout1'] = ("%4.1f" % (value(
        m.fs.charge.ess_hp_split.to_hp.pressure[0])*1e-6))
    tags['essvhp_Hout1'] = ("%4.1f" % (value(
        m.fs.charge.ess_hp_split.to_hp.enth_mol[0])*1e-3))
    tags['essvhp_xout1'] = ("%4.4f" % (value(
        m.fs.charge.ess_hp_split.to_hp_state[0].vapor_frac)))
    tags['essvhp_Fout2'] = ("%4.3f" % (value(
        m.fs.charge.ess_hp_split.to_hxc.flow_mol[0])*1e-3))
    tags['essvhp_Tout2'] = ("%4.2f" % (value(
        m.fs.charge.ess_hp_split.to_hxc_state[0].temperature)))
    tags['essvhp_Pout2'] = ("%4.1f" % (value(
        m.fs.charge.ess_hp_split.to_hxc.pressure[0])*1e-6))
    tags['essvhp_Hout2'] = ("%4.1f" % (value(
        m.fs.charge.ess_hp_split.to_hxc.enth_mol[0])*1e-3))
    tags['essvhp_xout2'] = ("%4.4f" % (value(
        m.fs.charge.ess_hp_split.to_hxc_state[0].vapor_frac)))

    # Recycle mixer
    tags['rmix_Fout'] = ("%4.3f" % (value(
        m.fs.charge.recycle_mixer.outlet.flow_mol[0])*1e-3))
    tags['rmix_Tout'] = ("%4.2f" % (value(
        m.fs.charge.recycle_mixer.mixed_state[0].temperature)))
    tags['rmix_Pout'] = ("%4.1f" % (value(
        m.fs.charge.recycle_mixer.outlet.pressure[0])*1e-6))
    tags['rmix_Hout'] = ("%4.1f" % (value(
        m.fs.charge.recycle_mixer.outlet.enth_mol[0])*1e-3))
    tags['rmix_xout'] = ("%4.4f" % (value(
        m.fs.charge.recycle_mixer.mixed_state[0].vapor_frac)))

    # Charge heat exchanger
    # tags['hxsteam_Fout'] = ("%4.4f" % (value(
    #     m.fs.charge.hxc.outlet_1.flow_mol[0])*1e-3))
    # tags['hxsteam_Tout'] = ("%4.4f" % (value(
    #     m.fs.charge.hxc.side_1.properties_out[0].temperature)))
    # tags['hxsteam_Pout'] = ("%4.4f" % (
    #     value(m.fs.charge.hxc.outlet_1.pressure[0])*1e-6))
    # tags['hxsteam_Hout'] = ("%4.2f" % (
    #     value(m.fs.charge.hxc.outlet_1.enth_mol[0])))
    # tags['hxsteam_xout'] = ("%4.4f" % (
    #     value(m.fs.charge.hxc.side_1.properties_out[0].vapor_frac)))

    # (sub)Cooler
    tags['cooler_Fout'] = ("%4.4f" % (value(
        m.fs.charge.cooler.outlet.flow_mol[0])*1e-3))
    tags['cooler_Tout'] = ("%4.4f" % (
        value(m.fs.charge.cooler.control_volume.properties_out[0].temperature)))
    tags['cooler_Pout'] = ("%4.4f" % (
        value(m.fs.charge.cooler.outlet.pressure[0])*1e-6))
    tags['cooler_Hout'] = ("%4.2f" % (
        value(m.fs.charge.cooler.outlet.enth_mol[0])))
    tags['cooler_xout'] = ("%4.4f" % (
        value(m.fs.charge.cooler.control_volume.properties_out[0].vapor_frac)))


    # HX pump
    tags['hxpump_Fout'] = ("%4.4f" % (value(
        m.fs.charge.hx_pump.outlet.flow_mol[0])*1e-3))
    tags['hxpump_Tout'] = ("%4.4f" % (value(
        m.fs.charge.hx_pump.control_volume.properties_out[0].temperature)))
    tags['hxpump_Pout'] = ("%4.4f" % (value(
        m.fs.charge.hx_pump.outlet.pressure[0])*1e-6))
    tags['hxpump_Hout'] = ("%4.2f" % (value(
        m.fs.charge.hx_pump.outlet.enth_mol[0])))
    tags['hxpump_xout'] = ("%4.4f" % (value(
        m.fs.charge.hx_pump.control_volume.properties_out[0].vapor_frac)))


    original_svg_file = os.path.join(
        this_file_dir(), "pfd_ultra_supercritical_pc_gdp.svg")
    with open(original_svg_file, "r") as f:
        svg_tag(tags, f, outfile=outfile)

def add_bounds(m):
    """
    Adding bounds to all units in the charge heat exchanger section
    of the flowsheet

    The units for temperature, pressure, etc. are given below:
    Temperature in K
    Pressure in Pa
    Flow in mol/s
    Massic flow in kg/s
    Heat duty and heat in W
    """
    
    m.flow_max = m.main_flow * 1.2 # in mol/s
    m.salt_flow_max = 1000 # in kg/s

    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc]:
        salt_hxc.inlet_1.flow_mol.setlb(0)  
        salt_hxc.inlet_1.flow_mol.setub(0.2 * m.flow_max)  
        salt_hxc.inlet_2.flow_mass.setlb(0)  
        salt_hxc.inlet_2.flow_mass.setub(m.salt_flow_max)  
        #-------- added by esrawli
        salt_hxc.outlet_1.flow_mol.setlb(0)  
        salt_hxc.outlet_1.flow_mol.setub(0.2 * m.flow_max)  
        salt_hxc.outlet_2.flow_mass.setlb(0)  
        salt_hxc.outlet_2.flow_mass.setub(m.salt_flow_max)  
        salt_hxc.inlet_2.pressure.setlb(101320)  
        salt_hxc.inlet_2.pressure.setub(101330)  
        salt_hxc.outlet_2.pressure.setlb(101320) 
        salt_hxc.outlet_2.pressure.setub(101330) 
        salt_hxc.heat_duty.setlb(0)  
        salt_hxc.heat_duty.setub(200e6)  
        salt_hxc.shell.heat.setlb(-200e6)
        salt_hxc.shell.heat.setub(0)  
        salt_hxc.tube.heat.setlb(0)  
        salt_hxc.tube.heat.setub(200e6) 
        salt_hxc.tube.properties_in[0].enthalpy_mass.setlb(0)
        salt_hxc.tube.properties_in[0].\
            enthalpy_mass.setub(1.5e6)
        salt_hxc.tube.properties_out[0].enthalpy_mass.setlb(0)
        salt_hxc.tube.properties_out[0].\
            enthalpy_mass.setub(1.5e6)
        salt_hxc.overall_heat_transfer_coefficient.setlb(0)
        salt_hxc.overall_heat_transfer_coefficient.setub(10000)
        salt_hxc.area.setlb(0)
        salt_hxc.area.setub(5000)  # TODO: Check this value
        salt_hxc.costing.pressure_factor.setlb(0)  # no unit
        salt_hxc.costing.pressure_factor.setub(1e5)  # no unit
        salt_hxc.costing.purchase_cost.setlb(0)  # no unit
        salt_hxc.costing.purchase_cost.setub(1e7)  # no unit
        salt_hxc.costing.base_cost_per_unit.setlb(0)
        salt_hxc.costing.base_cost_per_unit.setub(1e6)
        salt_hxc.costing.material_factor.setlb(0)
        salt_hxc.costing.material_factor.setub(10)
        #--------
        salt_hxc.delta_temperature_out.setlb(10)  # K
        salt_hxc.delta_temperature_in.setlb(10)  # K
        salt_hxc.delta_temperature_out.setub(80)  
        salt_hxc.delta_temperature_in.setub(80) 

    #-------- added by esrawli
    # Adding missing bounds for the hx pump and cooler. For this flow,
    # 15% more than the main flow is used
    for unit_k in [m.fs.charge.connector, m.fs.charge.hx_pump,
                   m.fs.charge.cooler]:
        unit_k.inlet.flow_mol.setlb(0)
        unit_k.inlet.flow_mol.setub(0.2*m.main_flow*1.15)
        unit_k.outlet.flow_mol.setlb(0)
        unit_k.outlet.flow_mol.setub(0.2*m.main_flow*1.15)
    #---------
    m.fs.charge.cooler.heat_duty.setub(0)

    for split in [m.fs.charge.ess_hp_split]:
        split.to_hxc.flow_mol[:].setlb(0.1)
        split.to_hxc.flow_mol[:].setub(0.2 * m.flow_max)
        split.to_hp.flow_mol[:].setlb(0)
        split.to_hp.flow_mol[:].setub(m.flow_max)
        split.split_fraction[0.0, "to_hxc"].setlb(0)
        split.split_fraction[0.0, "to_hxc"].setub(1)
        split.split_fraction[0.0, "to_hp"].setlb(0)
        split.split_fraction[0.0, "to_hp"].setub(1)

    for mix in [m.fs.charge.recycle_mixer]:
        mix.from_bfw_out.flow_mol.setlb(0) 
        mix.from_bfw_out.flow_mol.setub(m.flow_max) 
        mix.from_hx_pump.flow_mol.setlb(0) 
        mix.from_hx_pump.flow_mol.setub(0.2* m.flow_max) 
        mix.outlet.flow_mol.setlb(0) 
        mix.outlet.flow_mol.setub(m.flow_max) 

    # Cost-related terms
    m.fs.charge.capital_cost.setlb(0)  # no units
    m.fs.charge.capital_cost.setub(1e7)
    m.fs.charge.hx_pump.costing.purchase_cost.setlb(0) 
    m.fs.charge.hx_pump.costing.purchase_cost.setub(1e7)

    for salt_cost in [m.fs.charge.solar_salt_disjunct,
                      m.fs.charge.hitec_salt_disjunct]:
        salt_cost.salt_purchase_cost.setlb(0)  
        salt_cost.salt_purchase_cost.setub(1e7)
        salt_cost.capital_cost.setlb(0)  
        salt_cost.capital_cost.setub(1e7)  
        salt_cost.spump_purchase_cost.setlb(0)  
        salt_cost.spump_purchase_cost.setub(1e7)

    return m


def main(m_usc):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_charge_model(m_usc)

    # Give all the required inputs to the model
    # Ensure that the degrees of freedom = 0 (model is complete)
    set_model_input(m)

    #-------- modified by esrawli
    # Assert that the model has no degree of freedom at this point
    # assert degrees_of_freedom(m) == 0
    print('DOF before init = ', degrees_of_freedom(m))
    #--------
    set_scaling_factors(m)
    # Initialize the model (sequencial initialization and custom routines)
    # Ensure after the model is initialized, the degrees of freedom = 0
    initialize(m, solver=solver)
    #-------- modified by esrawli
    # assert degrees_of_freedom(m) == 0
    print('DOF after init = ', degrees_of_freedom(m))
    #--------
    # m.fs.charge.hx_pump.report()
    # raise Exception("costing")
    build_costing(m, solver=solver)

    add_bounds(m)

    add_disjunction(m)

    return m, solver

#-------- added by esrawli
def run_nlps(m, solver=None,
             salt=None):
    """
    This function fixes the indicator variables for the two salts
    so to solve NLP problems 
    """

    # Disjunction 1: salt selection
    if salt == "solar":
        m.fs.charge.solar_salt_disjunct.indicator_var.fix(1)
        m.fs.charge.hitec_salt_disjunct.indicator_var.fix(0)
    else:
        m.fs.charge.solar_salt_disjunct.indicator_var.fix(0)
        m.fs.charge.hitec_salt_disjunct.indicator_var.fix(1)

    TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    print("The degrees of freedom after gdp transformation ",
          degrees_of_freedom(m))

    results = solver.solve(m,
                           tee=True,
                           symbolic_solver_labels=True,
                           options={
                               "max_iter": 300,
                               "halt_on_ampl_error": "yes"
                           }
    )

    return m, results


def run_gdp(m):
    """
    This function solves the GDP problem using the solver GDPopt
    """
    
    opt = SolverFactory('gdpopt')
    opt.CONFIG.strategy = 'LOA'  # RIC is an option
    opt.CONFIG.mip_solver = 'gurobi_direct'
    opt.CONFIG.nlp_solver = 'ipopt'
    opt.CONFIG.tee = True
    opt.CONFIG.init_strategy = "no_init"
    # opt.CONFIG.init_strategy = "set_covering"
    opt.CONFIG.time_limit = "2400"

    results = opt.solve(m,
                        tee=True,
                        nlp_solver_args=dict(tee=True,
                                             symbolic_solver_labels=True,
                                             options={
                                                 "max_iter": 300
                                             }
                        )
    )
    
    return m, results
#--------

def model_analysis(m, solver):

    ###########################################################################
    # Unfixing variables for analysis
    # (This section is deactived for the simulation of square model)
    ###########################################################################

    # Fixing the power plant produced power
    m.fs.plant_power_out.fix(400)

    # Unfixing the boiler flowrate
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s
    # m.fs.boiler.inlet.flow_mol.fix(m.main_flow)  # mol/s
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)

    # Unfix the hp steam split fraction to charge heat exchanger
    m.fs.charge.ess_hp_split.split_fraction[0, "to_hxc"].unfix()

    # Charge heat exchanger section
    # Unfix charge heat exchanger salt flow, steam flow, temperature,
    # and area
    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc]:
        salt_hxc.inlet_1.flow_mol.unfix()
        salt_hxc.inlet_1.enth_mol.unfix()
        salt_hxc.inlet_1.pressure.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()   # kg/s, 1 DOF
        # salt_hxc.inlet_2.temperature.unfix()  # K, 1 DOF
        salt_hxc.area.unfix() # 1 DOF
        # salt_hxc.heat_duty.fix(70*1e6)  # in W
        salt_hxc.heat_duty.fix(100*1e6)  # in W
    # m.fs.charge.solar_salt_disjunct.hxc.inlet_2.temperature.unfix()  # K
    m.fs.charge.solar_salt_disjunct.hxc.outlet_2.temperature.unfix()  # K
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.temperature.fix(435.15)  # K
    m.fs.charge.hitec_salt_disjunct.hxc.outlet_2.temperature.unfix()  # K

    m.fs.charge.cooler.outlet.enth_mol[0].unfix() # 1 DOF

    # Unfix the connector and cooler inlets
    # for unit in [m.fs.charge.connector]:
    #     unit.inlet.flow_mol.unfix()
    #     unit.inlet.enth_mol.unfix()
    #     unit.inlet.pressure.unfix()

    # adding a dummy objective for the simulation model
    m.obj = Objective(expr=m.fs.charge.capital_cost
                      + m.fs.charge.operating_cost)

    print('DOF before solution = ', degrees_of_freedom(m))

    #-------- modified by esrawli
    # Adding two functions to solve either an NLP problem
    # by fixing the disjuncts or the GDP problem
    # run_nlps(m,
    #          solver=solver,
    #          salt="solar")

    run_gdp(m)
    #--------

    print("***************** Printing Results ******************")
    print('')
    print("Obj =",
          value(m.obj))            
    print('')
    print("Objective (cap + op costs in $/y) =",
          pyo.value(m.obj))            
    print("Total capital cost ($/y) =",
          pyo.value(m.fs.charge.capital_cost))
    print("Operating costs ($/y) =",
          pyo.value(m.fs.charge.operating_cost))
    print('')
    print('Plant Power (MW) =',
          value(m.fs.plant_power_out[0]))
    print('')
    print("Boiler feed water flow (mol/s):",
          value(m.fs.boiler.inlet.flow_mol[0]))
    print("Boiler duty (MW_th):",
          value((m.fs.boiler.heat_duty[0]
                      + m.fs.reheater[1].heat_duty[0]
                      + m.fs.reheater[2].heat_duty[0])
                    * 1e-6))
    print('')
    print("HX pump cost ($/y) =",
          pyo.value(m.fs.charge.hx_pump.costing.purchase_cost / 5))
    print("Cooling duty (MW_th) =",
          pyo.value(m.fs.charge.cooler.heat_duty[0] * -1e-6))
    print('')
    print('====================================================================================')
    print("Disjunctions")
    for d in m.component_data_objects(ctype=Disjunct,
                                      active=True,
                                      sort=True, descend_into=True):
        if abs(d.indicator_var.value - 1) < 1e-6:
            print(d.name, ' should be selected!')
    print('')
    print('')
    print('====================================================================================')
    print(' ')
    if m.fs.charge.solar_salt_disjunct.indicator_var == 1:
        print("Salt: Solar salt is selected!")
        print("Heat exchanger area (m2):",
              value(m.fs.charge.solar_salt_disjunct.hxc.area))
        print("Heat exchanger cost ($/y):",
              value(m.fs.charge.solar_salt_disjunct.hxc.costing.purchase_cost / 15))
        print("Salt flow (kg/s):",
              value(m.fs.charge.solar_salt_disjunct.hxc.inlet_2.flow_mass[0]))
        print("Salt temperature in (K):",
              value(m.fs.charge.solar_salt_disjunct.hxc.inlet_2.temperature[0]))
        print("Salt temperature out (K):",
              value(m.fs.charge.solar_salt_disjunct.hxc.outlet_2.temperature[0]))
        print("Steam flow to storage (mol/s):",
              value(m.fs.charge.solar_salt_disjunct.hxc.inlet_1.flow_mol[0]))
        print("Water temperature in (K):",
              value(m.fs.charge.solar_salt_disjunct.
                    hxc.side_1.properties_in[0].temperature))
        print("Steam temperature out (K):",
              value(m.fs.charge.solar_salt_disjunct.
                    hxc.side_1.properties_out[0].temperature))
        print("delta temperature at inlet (K):",
              value(m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_in[0]))
        print("delta temperature at outlet (K):",
              value(m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_out[0]))
        print("Salt cost ($/y):",
              value(m.fs.charge.solar_salt_disjunct.salt_purchase_cost))
        print("Tank cost ($/y):",
              value(m.fs.charge.solar_salt_disjunct.costing.total_tank_cost / 15))
        print("Salt pump cost ($/y):",
              value(m.fs.charge.solar_salt_disjunct.spump_purchase_cost))
        print("")
        print("Salt storage tank volume in m3: ",
              value(m.fs.charge.solar_salt_disjunct.tank_volume))
        print("Salt density: ",
              value(m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"]))
        print("HXC heat duty: ",
              value(m.fs.charge.solar_salt_disjunct.hxc.heat_duty[0]) / 1e6)
    else:
        print("Salt: Hitec salt is selected")
        print("Heat exchanger area (m2):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.area))
        print("Heat exchanger cost ($/y):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.costing.purchase_cost / 15))
        print("Salt flow (kg/s):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.flow_mass[0]))
        print("Salt temperature in (K):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.temperature[0]))
        print("Salt temperature out (K):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.outlet_2.temperature[0]))
        print("Steam flow to storage (mol/s):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.inlet_1.flow_mol[0]))
        print("Water temperature in (K):",
              value(m.fs.charge.hitec_salt_disjunct.
                    hxc.side_1.properties_in[0].temperature))
        print("Steam temperature out (K):",
              value(m.fs.charge.hitec_salt_disjunct.
                    hxc.side_1.properties_out[0].temperature))
        print("delta temperature at inlet (K):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_in[0]))
        print("delta temperature at outlet (K):",
              value(m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_out[0]))
        print("Salt cost ($/y):",
              value(m.fs.charge.hitec_salt_disjunct.salt_purchase_cost))
        print("Tank cost ($/y):",
              value(m.fs.charge.hitec_salt_disjunct.costing.total_tank_cost / 15))
        print("Salt pump cost ($/y):",
              value(m.fs.charge.hitec_salt_disjunct.spump_purchase_cost))
        print("")
        print("Salt storage tank volume in m3: ",
              value(m.fs.charge.hitec_salt_disjunct.tank_volume))
        print("Salt density: ",
              value(m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"]))
        print("HXC heat duty: ",
              value(m.fs.charge.hitec_salt_disjunct.hxc.heat_duty[0]) / 1e6)
        print("")
        print("Solver details")
        print(results)
        print('')

    # for unit_k in [m.fs.boiler, m.fs.charge.ess_hp_split,
    #                m.fs.charge.solar_salt_disjunct.hxc,
    #                m.fs.charge.solar_salt_disjunct.hxc]:
    #     unit_k.report()
    # for k in pyo.RangeSet(m.number_turbines):
    #     m.fs.turbine[k].report()
    # for j in pyo.RangeSet(m.number_fwhs):
    #     m.fs.fwh[j].report()
    #     m.fs.condenser_mix.makeup.display()
    #     m.fs.condenser_mix.outlet.display()

    return m

if __name__ == "__main__":
    
    m_usc = usc.build_plant_model()
    solver = usc.initialize(m_usc)

    m_chg, solver = main(m_usc)

    m = model_analysis(m_chg, solver)
    # View results in a process flow diagram
    # view_result("pfd_usc_powerplant_gdp_results.svg", m_result)

    log_infeasible_constraints(m)
    log_close_to_bounds(m)
