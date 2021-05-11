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
This is an NLP model for the solution of an ultrasupercritical coal-fired
power plant based on a flowsheet presented in 1999 USDOE Report #DOE/FE-0400

This model uses some of the simple unit models from the power generation unit 
model library

Some of the parameters in the model, such as feed water heater areas,
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

__author__ = "Naresh Susarla, Soraya Rawlings"

import os

# Import Pyomo libraries
#-------- added by esrawli
from pyomo.environ import log, exp
#--------
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.util import get_solver, copy_port_values as _set_port
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (
    HeatExchanger,
    MomentumMixingType,
    Heater,
)
#--------added by esrawli
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
from pyomo.util.calc_var_value import calculate_variable_from_constraint
import idaes.core.util.unit_costing as icost
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
from idaes.generic_models.properties import iapws95

#-------- added by esrawli
from pyomo.util.infeasible import log_infeasible_constraints
import solarsalt_properties_new
#--------


def declare_unit_model():
    """Create flowsheet and add unit models.
    """
    ###########################################################################
    #  Flowsheet and Property Package                                         #
    ###########################################################################
    m = pyo.ConcreteModel(name="Steam Cycle Model")

    #-------- added by esrawli
    m.main_flow = 17854             # Main flow
    m.main_steam_pressure = 31125980
    m.number_turbines = 11          # number of turbines in the flowsheet
    m.number_turbine_splitters = 10 # number of turbine splitters in the flowsheet
    m.number_fwhs = 9               # Number of feedwater heaters in the flowsheet
    #--------

    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.prop_water = iapws95.Iapws95ParameterBlock()
    m.fs.salt_properties = solarsalt_properties_new.SolarsaltParameterBlock()

    ###########################################################################
    #   Turbine declarations                                   #
    ###########################################################################
    # A total of 11 single stage turbines are used to model
    # different multistage turbines
    m.fs.turbine = HelmTurbineStage(
        pyo.RangeSet(m.number_turbines),
        default={
            "property_package": m.fs.prop_water,
        }
    )

    #########################################################################
    #  Turbine outlet splitters                                  #
    #########################################################################
    # The default number of outlets for a splitter is 2. This can be changed
    # using the "num_outlets" argument.
    # In the USC flowsheet turbine_splitter[6] has 3 outlets. This is realized
    # by using the 'initialize' argument as shown below.
    m.fs.turbine_splitter = HelmSplitter(
        pyo.RangeSet(m.number_turbine_splitters),
        default = {
            "property_package": m.fs.prop_water
            },
        initialize={
            6:{
                "property_package": m.fs.prop_water,
                "num_outlets": 3
                },
        }
    )

    ###########################################################################
    #  Boiler section declarations:                                #
    ###########################################################################
    # Boiler section is set up using three heater blocks, as following:
    # 1) For the main steam the heater block is named 'boiler'
    # 2) For the 1st reheated steam the heater block is named 'reheater_1'
    # 3) For the 2nd reheated steam the heater block is named 'reheater_2'
    m.fs.boiler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )
    m.fs.reheater_1 = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )
    m.fs.reheater_2 = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    ###########################################################################
    #  Add Condenser Mixer, Condenser, and Condensate pump                    #
    ###########################################################################
    # condenser mix
    m.fs.condenser_mix = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["main", "bfpt", "drain", "makeup"],
            "property_package": m.fs.prop_water,
        }
    )

    # Condenser is set up as a heater block
    m.fs.condenser = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": False
        }
    )

    # condensate pump
    m.fs.cond_pump = HelmIsentropicCompressor(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    ###########################################################################
    #  Feedwater heater declaration                                     #
    ###########################################################################
    # Feed water heaters (FWHs) are declared as 0D heat exchangers
    # Shell side (side 1) is for steam condensing
    # Tube side (side 2) is for feed water heating 

    # Declaring indexed units for feed water heater mixers
    # The indices reflect the connection to corresponding feed water heaters
    # e. g. the outlet of fwh_mixer[1] is connected to fwh[1]
    # Note that there are no mixers before FWHs 5 and 9
    m.mixer_list = [1, 2, 3, 4, 6, 7, 8]
    m.fs.fwh_mixer = HelmMixer(
        m.mixer_list,
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )


    # Declaring indexed units for feed water heaters
    m.fs.fwh = HeatExchanger(
        pyo.RangeSet(m.number_fwhs),
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.prop_water,
                "material_balance_type": MaterialBalanceType.componentTotal,
                "has_pressure_change": True,
            },
            "tube": {
                "property_package": m.fs.prop_water,
                "material_balance_type": MaterialBalanceType.componentTotal,
                "has_pressure_change": True,
            },
        }
    )

    ###########################################################################
    #  Add deaerator                              #
    ###########################################################################
    m.fs.deaerator = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["steam", "drain", "feedwater"],
            "property_package": m.fs.prop_water,
        }
    )

    ###########################################################################
    #  Add auxiliary turbine, booster pump, and boiler feed pump (BFP)        #
    ###########################################################################
    m.fs.booster = HelmIsentropicCompressor(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    m.fs.bfp = HelmIsentropicCompressor(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    m.fs.bfpt = HelmTurbineStage(
        default={
            "property_package": m.fs.prop_water,
        }
    )

    #-------- added by esrawli
    ###########################################################################
    #  Add hp splitter                                                        #
    ###########################################################################
    # Defined to divert some steam from high pressure inlet to charge the
    # storage heat exchanger
    m.fs.ess_hp_split = Separator(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.total,
            "split_basis": SplittingType.totalFlow,
            "ideal_separation": False,
            "outlet_list": ["to_hxc", "to_hp"],
            "has_phase_equilibrium": False
        }
    )

    ###########################################################################
    #  Add charge heat exchanger for storage                                  #
    ###########################################################################
    m.fs.hxc = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.prop_water
            },
            "tube": {
                "property_package": m.fs.salt_properties
            }
        }
    )

    m.data_hxc = {
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
    m.fs.tube_thickness = pyo.Param(
        initialize=m.data_hxc['tube_thickness'],
        doc='Tube thickness [m]'
    )
    m.fs.hxc.tube_inner_dia = pyo.Param(
        initialize=m.data_hxc['tube_inner_dia'],
        doc='Tube inner diameter [m]'
    )
    m.fs.hxc.tube_outer_dia = pyo.Param(
        initialize=m.data_hxc['tube_outer_dia'],
        doc='Tube outer diameter [m]'
    )
    m.fs.hxc.k_steel = pyo.Param(
        initialize=m.data_hxc['k_steel'],
        doc='Thermal conductivity of steel [W/mK]'
    )
    m.fs.hxc.n_tubes = pyo.Param(
        initialize=m.data_hxc['number_tubes'],
        doc='Number of tubes'
    )
    m.fs.hxc.shell_inner_dia = pyo.Param(
        initialize=m.data_hxc['shell_inner_dia'],
        doc='Shell inner diameter [m]'
    )

    ###########################################################################
    #  Add cooler and hx pump                                                 #
    ###########################################################################
    # To ensure the outlet of charge heat exchanger is a subcooled liquid
    # before mixing it with the plant, a cooler is added after the heat
    # exchanger
    m.fs.cooler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    # A pump, if needed, is used to increase the pressure of the water to
    # allow mixing it at a desired location within the plant
    m.fs.hx_pump = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.pump,
        }
    )

    ###########################################################################
    #  Add recycle mixer                                                      #
    ###########################################################################
    m.fs.recycle_mixer = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["from_bfw_out", "from_hx_pump"],
            "property_package": m.fs.prop_water,
        }
    )

    #--------
    
    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m)
    _create_arcs(m)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)
    return m

def _make_constraints(m):
    """
    Define all model constraints except those included with the unit models
    """

    #########   Boiler section   #########

    # Following the reference DOE flowsheet, the outlet temperature for the
    # boiler unit, i. e. boiler, reheater_1, & reheater_2 is fixed to 866.15 K
    # Outlet temperature of boiler is set to 866.15 K
    @m.fs.boiler.Constraint(m.fs.time)
    def boiler_temperature_constraint(b, t):
        return (b.control_volume.properties_out[t].temperature == \
                866.15 * pyunits.K)

    # Outlet temperature of reheater_1 is set to 866.15 K
    @m.fs.reheater_1.Constraint(m.fs.time)
    def reheater1_temperature_constraint(b, t):
        return (b.control_volume.properties_out[t].temperature == \
                866.15 * pyunits.K)

    # Outlet temperature of reheater_2 is set to 866.15 K
    @m.fs.reheater_2.Constraint(m.fs.time)
    def reheater2_temperature_constraint(b, t):
        return (b.control_volume.properties_out[t].temperature == \
                866.15 * pyunits.K)

    #########   Condenser section   #########
    
    # The inlet 'main' refers to the main steam coming from the turbine train
    # Inlet 'bfpt' refers to the steam coming from the bolier feed pump turbine
    # Inlet 'drain' refers to the condensed steam from the feed water heater 1
    # Inlet 'makeup' refers to the make up water
    # The outlet pressure of condenser mixer is equal to the minimum pressure
    # Since the turbine (#11) outlet (or, mixer inlet 'main') pressure
    # has the minimum pressure, the following constraint sets the outlet
    # pressure of the condenser mixer to the pressure of the inlet 'main'
    @m.fs.condenser_mix.Constraint(m.fs.time)
    def mixer_pressure_constraint(b, t):
        return b.main_state[t].pressure == b.mixed_state[t].pressure

    # The outlet of condenser is assumed to be a saturated liquid
    # The following condensate enthalpy at the outlet of condeser equal to
    # that of a saturated liquid at that pressure
    @m.fs.condenser.Constraint(m.fs.time)
    def cond_vaporfrac_constraint(b, t):
        return (
            b.control_volume.properties_out[t].enth_mol == \
            b.control_volume.properties_out[t].enth_mol_sat_phase['Liq']
        )

    #########   Feed Water Heater section   #########

    # The condensing steam is assumed to leave the FWH as saturated liquid
    # Thus, each FWH is accompanied by 3 constraints, 2 for pressure drop
    # and 1 for the enthalpy.

    # The outlet pressure of fwh mixer is equal to the minimum pressure
    # Since the pressure of mixer inlet 'steam' has the minimum pressure,
    # the following constraint set the outlet pressure of fwh mixers
    # to be same as the pressure of the inlet 'steam'
    def fwhmixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure

    for p in m.mixer_list:
        b = m.fs.fwh_mixer[p]
        setattr(b,
                "mixer_press_constraint",
                pyo.Constraint(m.fs.config.time,
                               rule=fwhmixer_pressure_constraint))

    # Side 1 outlet of fwh is assumed to be a saturated liquid
    # The following constraint sets the side 1 outlet enthalpy to be 
    # same as that of saturated liquid
    def fwh_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )
    for i in pyo.RangeSet(m.number_fwhs):
        b = m.fs.fwh[i]
        setattr(b,
                "fwh_vfrac_constraint",
                pyo.Constraint(m.fs.config.time,
                               rule=fwh_vaporfrac_constraint))

    # Pressure drop on both sides are accounted for by setting the respective
    # outlet pressure based on the following assumptions:
    #     (1) Feed water side (side 2): A constant 4% pressure drop is assumed
    #           on the feedwater side for all FWHs. For this,
    #           the outlet pressure is set to 0.96 times the inlet pressure,
    #           on the feed water side for all FWHs
    #     (2) Steam condensing side (side 1): Going from high pressure to
    #           low pressure FWHs, the outlet pressure of
    #           the condensed steam in assumed to be 10% more than that
    #           of the pressure of steam extracted for the immediately
    #           next lower pressure feedwater heater.
    #           e.g. the outlet condensate pressure of FWH 'n'
    #           = 1.1 * pressure of steam extracted for FWH 'n-1'
    #           In case of FWH1 the FWH 'n-1' is used for Condenser,
    #           and in case of FWH6, FWH 'n-1' is for Deaerator. Here,
    #           the steam pressure for FWH 'n-1' is known because the
    #           pressure ratios for turbines are fixed.

    # Side 2 pressure drop constraint
    # Setting a 4% pressure drop on the feedwater side (P_out = 0.96 * P_in)
    def fwh_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )
    for i in pyo.RangeSet(m.number_fwhs):
        b = m.fs.fwh[i]
        setattr(b,
                "fwh_s2_delp_constraint",
                pyo.Constraint(m.fs.config.time,
                               rule=fwh_s2pdrop_constraint))

    # Side 1 pressure drop constraint
    # Setting the outlet pressure as described above. For this, the
    # relevant turbine stage pressure ratios are used (pressure_ratio_dict)
    # The pressure drop across the reheaters 1 and 2 are also accounted
    # in case of fwh[9] and fwh[7], respectively
    # For this, pressure_diffr_dict is defined

    # 0.204 is the pressure ratio for turbine #11 (see set_inputs)
    # 0.476 is the pressure ratio for turbine #10 (see set_inputs)
    # 0.572 is the pressure ratio for turbine #9 (see set_inputs)
    # 0.389 is the pressure ratio for turbine #8 (see set_inputs)
    # 0.514 is the pressure ratio for turbine #7 (see set_inputs)
    # 0.523 is the pressure ratio for turbine #5 (see set_inputs)
    # 0.609 is the pressure ratio for turbine #4 (see set_inputs)
    # 0.498 is the pressure ratio for turbine #3 (see set_inputs)
    # 0.774 is the pressure ratio for turbine #2 (see set_inputs)
    #-------- modified by esrawli
    # Renaming the dictionaries 
    m.data_pressure_ratio = {1: 0.204,
                             2: 0.476,
                             3: 0.572,
                             4: 0.389,
                             5: 0.514,
                             6: 0.523,
                             7: 0.609,
                             8: 0.498,
                             9: 0.774}

    # 742845 Pa is the pressure drop across reheater_1
    # 210952 Pa is the pressure drop across reheater_2
    m.data_pressure_diffr = {1: 0,
                             2: 0,
                             3: 0,
                             4: 0,
                             5: 0,
                             6: 210952,
                             7: 0,
                             8: 742845,
                             9: 0}
    #--------
    
    def fwh_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * b.turb_press_ratio *
            (b.side_1.properties_in[t].pressure - b.reheater_press_diff)
        )

    for i in pyo.RangeSet(m.number_fwhs):
        b = m.fs.fwh[i]
        b.turb_press_ratio = pyo.Param(initialize = m.data_pressure_ratio[i],
                                       units = pyunits.Pa/pyunits.Pa)
        b.reheater_press_diff = pyo.Param(initialize = m.data_pressure_diffr[i],
                                          units = pyunits.Pa)
        setattr(b, "s1_delp_constraint",
                pyo.Constraint(m.fs.config.time, rule=fwh_s1pdrop_constraint))

    # The outlet pressure of deaerator is equal to the minimum pressure
    # Since the pressure of deaerator inlet 'feedwater' has
    # the minimum pressure, the following constraint sets the outlet pressure
    # of deaerator to be same as the pressure of the inlet 'feedwater'
    @m.fs.deaerator.Constraint(m.fs.time)
    def damixer_pressure_constraint(b, t):
        return b.feedwater_state[t].pressure == b.mixed_state[t].pressure

    # The following constraint sets the outlet pressure of steam extracted
    # for boiler feed water turbine to be same as that of condenser
    @m.fs.Constraint(m.fs.time)
    def constraint_out_pressure(b, t):
        return (
            b.bfpt.control_volume.properties_out[t].pressure
            == b.condenser_mix.main_state[t].pressure
        )

    # The following constraint demands that the work done by the
    # boiler feed water pump is same as that of boiler feed water turbine
    # Essentially, this says that boiler feed water turbine produces just
    # enough power to meet the demand of boiler feed water pump
    @m.fs.Constraint(m.fs.time)
    def constraint_bfp_power(b, t):
        return (
            b.booster.control_volume.work[t]
            + b.bfp.control_volume.work[t]
            + b.bfpt.control_volume.work[t]
            == 0
        )

    #-------- added by esrawli

    #########   Charge heat exchanger section   #########
    
    # Salt side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.hxc.tube_cs_area = pyo.Expression(
        expr=(pi / 4) * (m.fs.hxc.tube_inner_dia ** 2),
        doc="Tube cross sectional area"
    )
    m.fs.hxc.tube_out_area = pyo.Expression(
        expr=(pi / 4) * (m.fs.hxc.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]"
    )
    m.fs.hxc.shell_eff_area = pyo.Expression(
        expr=(
            (pi / 4) * (m.fs.hxc.shell_inner_dia ** 2)
            - m.fs.hxc.n_tubes
            * m.fs.hxc.tube_out_area),
        doc="Effective shell cross sectional area [m2]"
    )

    # Salt (shell) side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.hxc.salt_reynolds_number = pyo.Expression(
        expr=(
            (m.fs.hxc.inlet_2.flow_mass[0] * m.fs.hxc.tube_outer_dia) / \
            (m.fs.hxc.shell_eff_area * \
             m.fs.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.hxc.salt_prandtl_number = pyo.Expression(
        expr=(
            m.fs.hxc.side_2.properties_in[0].cp_specific_heat["Liq"] * \
            m.fs.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] / \
            m.fs.hxc.side_2.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    m.fs.hxc.salt_prandtl_wall = pyo.Expression(
        expr=(
            (m.fs.hxc.side_2.properties_out[0].cp_specific_heat["Liq"] * \
             m.fs.hxc.side_2.properties_out[0].dynamic_viscosity["Liq"]) / \
            m.fs.hxc.side_2.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number at wall"
    )
    m.fs.hxc.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 * (m.fs.hxc.salt_reynolds_number**0.6) * \
            (m.fs.hxc.salt_prandtl_number**0.4) * \
            ((m.fs.hxc.salt_prandtl_number / \
              m.fs.hxc.salt_prandtl_wall)**0.25) * (2**0.2)
        ),
        doc="Salt Nusslet Number Sieder-Tate correlation"
    )

    # Steam side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.hxc.steam_reynolds_number = pyo.Expression(
        expr=(
           m.fs.hxc.inlet_1.flow_mol[0] * \
            m.fs.hxc.side_1.properties_in[0].mw * \
            m.fs.hxc.tube_inner_dia / \
            (m.fs.hxc.tube_cs_area * m.fs.hxc.n_tubes * \
             m.fs.hxc.side_1.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (m.fs.hxc.side_1.properties_in[0].cp_mol / \
             m.fs.hxc.side_1.properties_in[0].mw) * \
            m.fs.hxc.side_1.properties_in[0].visc_d_phase["Vap"] / \
            m.fs.hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.hxc.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 * (m.fs.hxc.steam_reynolds_number**0.8) * \
            (m.fs.hxc.steam_prandtl_number**(0.33)) * \
            ((m.fs.hxc.side_1.properties_in[0].visc_d_phase["Vap"] / \
              m.fs.hxc.side_1.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number Sieder-Tate correlation"
    )

    # Salt and steam side heat transfer coefficients
    m.fs.hxc.h_salt = pyo.Expression(
        expr=(
            m.fs.hxc.side_2.properties_in[0].thermal_conductivity["Liq"] * \
            m.fs.hxc.salt_nusselt_number / m.fs.hxc.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    m.fs.hxc.h_steam = pyo.Expression(
        expr=(
            m.fs.hxc.side_1.properties_in[0].therm_cond_phase["Vap"] * \
            m.fs.hxc.steam_nusselt_number / m.fs.hxc.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    # Computing overall heat transfer coefficient
    # OHTC constraint is rewritten to avoid having a denominator in the equation
    # OHTC = _________________________________1___________________________________
    #        __1__  + _Tout_dia_*_log(Tout_dia/Tin_dia)_  + __(Tout_dia/Tin_dia)__
    #        Hsalt              2 k_steel                           Hsteam
    #
    m.fs.hxc.tube_dia_relation = (
        m.fs.hxc.tube_outer_dia / m.fs.hxc.tube_inner_dia
    )
    m.fs.hxc.log_tube_dia_relation = log(m.fs.hxc.tube_dia_relation)    
    @m.fs.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        # return (
        #     m.fs.hxc.overall_heat_transfer_coefficient[t]
        #     == 1 / ((1 / m.fs.hxc.h_salt)
        #             + ((m.fs.hxc.tube_outer_dia * \
        #                 m.fs.hxc.log_tube_dia_relation) / \
        #                (2 * m.fs.hxc.k_steel))
        #             + (m.fs.hxc.tube_dia_relation / m.fs.hxc.h_steam))
        # )
        #-------- modified by esrawli: equation rewritten to avoid having denominators
        return (
            m.fs.hxc.overall_heat_transfer_coefficient[t] * \
            (2 * m.fs.hxc.k_steel * m.fs.hxc.h_steam
              + m.fs.hxc.tube_outer_dia * m.fs.hxc.log_tube_dia_relation *\
              m.fs.hxc.h_salt * m.fs.hxc.h_steam
              + m.fs.hxc.tube_dia_relation * m.fs.hxc.h_salt * \
              2 * m.fs.hxc.k_steel)        
        ) == 2 * m.fs.hxc.k_steel * m.fs.hxc.h_salt * m.fs.hxc.h_steam

    # Cooler
    # The temperature at the outlet of the cooler is required to be subcooled
    # by at least 5 degrees
    @m.fs.cooler.Constraint(m.fs.time)
    def constraint_cooler_enth2(b, t):
        return (
            m.fs.cooler.control_volume.properties_out[t].temperature <=
            (m.fs.cooler.control_volume.properties_out[t].temperature_sat - 5)
        )

    # hx pump
    # The outlet pressure of hx_pump is then fixed to be the same as
    # boiler feed pump's outlet pressure
    # @m.fs.Constraint(m.fs.time)
    # def constraint_hxpump_presout(b, t):
    #     return m.fs.hx_pump.outlet.pressure[t] >= \
    #         (m.main_steam_pressure * 1.1231)

    # Recycle mixer
    # The outlet pressure of the recycle mixer is same as
    # the outlet pressure of the boiler feed water, i.e. inlet 'bfw_out'
    @m.fs.recycle_mixer.Constraint(m.fs.time)
    def recyclemixer_pressure_constraint(b, t):
        return b.from_bfw_out_state[t].pressure == b.mixed_state[t].pressure
    
    #--------


def _create_arcs(m):

    #-------- modified by esrawli
    # boiler to turb
    # m.fs.boiler_to_turb1 = Arc(
    #     source=m.fs.boiler.outlet, destination=m.fs.turbine[1].inlet
    # )
    # Connecting the boiler to the ess hp splitter instead of turbine 1
    m.fs.boiler_to_turb1 = Arc(
        source=m.fs.boiler.outlet,
        destination=m.fs.ess_hp_split.inlet
    )
    m.fs.esshp_to_turb1 = Arc(
        source=m.fs.ess_hp_split.to_hp,
        destination=m.fs.turbine[1].inlet
    )

    # Connections to integrate storage
    m.fs.esshp_to_hxc = Arc(
        source=m.fs.ess_hp_split.to_hxc,
        destination=m.fs.hxc.inlet_1
    )
    m.fs.hxc_to_cooler = Arc(
        source=m.fs.hxc.outlet_1,
        destination=m.fs.cooler.inlet
    )
    m.fs.cooler_to_hxpump = Arc(
        source=m.fs.cooler.outlet,
        destination=m.fs.hx_pump.inlet
    )
    m.fs.hxpump_to_recyclemix = Arc(
        source=m.fs.hx_pump.outlet,
        destination=m.fs.recycle_mixer.from_hx_pump
    )
    #--------

    # turbine1 splitter
    m.fs.turb1_to_t1split = Arc(
        source=m.fs.turbine[1].outlet,
        destination=m.fs.turbine_splitter[1].inlet
    )
    m.fs.t1split_to_turb2 = Arc(
        source=m.fs.turbine_splitter[1].outlet_1,
        destination=m.fs.turbine[2].inlet
    )
    m.fs.t1split_to_fwh10 = Arc(
        source=m.fs.turbine_splitter[1].outlet_2,
        destination=m.fs.fwh[9].inlet_1
    )

    # turbine2 splitter
    m.fs.turb2_to_t2split = Arc(
        source=m.fs.turbine[2].outlet,
        destination=m.fs.turbine_splitter[2].inlet
    )
    m.fs.t2split_to_rh1 = Arc(
        source=m.fs.turbine_splitter[2].outlet_1,
        destination=m.fs.reheater_1.inlet
    )
    m.fs.t2split_to_fwh9mix = Arc(
        source=m.fs.turbine_splitter[2].outlet_2,
        destination=m.fs.fwh_mixer[8].steam
    )

    # reheater_1 to turbine_3
    m.fs.rh1_to_turb3 = Arc(
        source=m.fs.reheater_1.outlet,
        destination=m.fs.turbine[3].inlet
    )

    # turbine3 splitter
    m.fs.turb3_to_t3split = Arc(
        source=m.fs.turbine[3].outlet,
        destination=m.fs.turbine_splitter[3].inlet
    )
    m.fs.t3split_to_turb4 = Arc(
        source=m.fs.turbine_splitter[3].outlet_1,
        destination=m.fs.turbine[4].inlet
    )
    m.fs.t3split_to_fwh8mix = Arc(
        source=m.fs.turbine_splitter[3].outlet_2,
        destination=m.fs.fwh_mixer[7].steam
    )

    # turbine4 splitter
    m.fs.turb4_to_t4split = Arc(
        source=m.fs.turbine[4].outlet,
        destination=m.fs.turbine_splitter[4].inlet
    )
    m.fs.t4split_to_rh2 = Arc(
        source=m.fs.turbine_splitter[4].outlet_1,
        destination=m.fs.reheater_2.inlet
    )
    m.fs.t4split_to_fwh7mix = Arc(
        source=m.fs.turbine_splitter[4].outlet_2,
        destination=m.fs.fwh_mixer[6].steam
    )

    # reheater_2 to turbine_5
    m.fs.rh2_to_turb5 = Arc(
        source=m.fs.reheater_2.outlet,
        destination=m.fs.turbine[5].inlet
    )

    # turbine5 splitter
    m.fs.turb5_to_t5split = Arc(
        source=m.fs.turbine[5].outlet,
        destination=m.fs.turbine_splitter[5].inlet
    )
    m.fs.t5split_to_turb6 = Arc(
        source=m.fs.turbine_splitter[5].outlet_1,
        destination=m.fs.turbine[6].inlet
    )
    m.fs.t5split_to_fwh6da = Arc(
        source=m.fs.turbine_splitter[5].outlet_2,
        destination=m.fs.deaerator.steam
    )

    # turbine6 splitter
    m.fs.turb6_to_t6split = Arc(
        source=m.fs.turbine[6].outlet,
        destination=m.fs.turbine_splitter[6].inlet
    )
    m.fs.t6split_to_turb7 = Arc(
        source=m.fs.turbine_splitter[6].outlet_1,
        destination=m.fs.turbine[7].inlet
    )
    m.fs.t6split_to_fwh5 = Arc(
        source=m.fs.turbine_splitter[6].outlet_2,
        destination=m.fs.fwh[5].inlet_1
    )
    m.fs.t6split_to_bfpt = Arc(
        source=m.fs.turbine_splitter[6].outlet_3,
        destination=m.fs.bfpt.inlet
    )

    # turbine7 splitter
    m.fs.turb7_to_t7split = Arc(
        source=m.fs.turbine[7].outlet,
        destination=m.fs.turbine_splitter[7].inlet
    )
    m.fs.t7split_to_turb8 = Arc(
        source=m.fs.turbine_splitter[7].outlet_1,
        destination=m.fs.turbine[8].inlet
    )
    m.fs.t7split_to_fwh4mix = Arc(
        source=m.fs.turbine_splitter[7].outlet_2,
        destination=m.fs.fwh_mixer[4].steam
    )

    # turbine8 splitter
    m.fs.turb8_to_t8split = Arc(
        source=m.fs.turbine[8].outlet,
        destination=m.fs.turbine_splitter[8].inlet
    )
    m.fs.t8split_to_turb9 = Arc(
        source=m.fs.turbine_splitter[8].outlet_1,
        destination=m.fs.turbine[9].inlet
    )
    m.fs.t8split_to_fwh3mix = Arc(
        source=m.fs.turbine_splitter[8].outlet_2,
        destination=m.fs.fwh_mixer[3].steam
    )

    # turbine9 splitter
    m.fs.turb9_to_t9split = Arc(
        source=m.fs.turbine[9].outlet,
        destination=m.fs.turbine_splitter[9].inlet
    )
    m.fs.t9split_to_turb10 = Arc(
        source=m.fs.turbine_splitter[9].outlet_1,
        destination=m.fs.turbine[10].inlet
    )
    m.fs.t9split_to_fwh2mix = Arc(
        source=m.fs.turbine_splitter[9].outlet_2,
        destination=m.fs.fwh_mixer[2].steam
    )

    # turbine10 splitter
    m.fs.turb10_to_t10split = Arc(
        source=m.fs.turbine[10].outlet,
        destination=m.fs.turbine_splitter[10].inlet
    )
    m.fs.t10split_to_turb11 = Arc(
        source=m.fs.turbine_splitter[10].outlet_1,
        destination=m.fs.turbine[11].inlet
    )
    m.fs.t10split_to_fwh1mix = Arc(
        source=m.fs.turbine_splitter[10].outlet_2,
        destination=m.fs.fwh_mixer[1].steam
    )

    # condenser mixer to condensate pump
    m.fs.turb11_to_cmix = Arc(
        source=m.fs.turbine[11].outlet,
        destination=m.fs.condenser_mix.main
    )
    m.fs.drain_to_cmix = Arc(
        source=m.fs.fwh[1].outlet_1,
        destination=m.fs.condenser_mix.drain
    )
    m.fs.bfpt_to_cmix = Arc(
        source=m.fs.bfpt.outlet,
        destination=m.fs.condenser_mix.bfpt
    )
    m.fs.cmix_to_cond = Arc(
        source=m.fs.condenser_mix.outlet,
        destination=m.fs.condenser.inlet
    )
    m.fs.cond_to_cpump = Arc(
        source=m.fs.condenser.outlet,
        destination=m.fs.cond_pump.inlet
    )

    # fwh1
    m.fs.pump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2
    )
    m.fs.fwh2_to_fwh1mix = Arc(
        source=m.fs.fwh[2].outlet_1,
        destination=m.fs.fwh_mixer[1].drain
    )
    m.fs.mix1_to_fwh1 = Arc(
        source=m.fs.fwh_mixer[1].outlet,
        destination=m.fs.fwh[1].inlet_1
    )

    # fwh2
    m.fs.fwh3_to_fwh2mix = Arc(
        source=m.fs.fwh[3].outlet_1,
        destination=m.fs.fwh_mixer[2].drain
    )
    m.fs.mix2_to_fwh2 = Arc(
        source=m.fs.fwh_mixer[2].outlet,
        destination=m.fs.fwh[2].inlet_1
    )
    m.fs.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2,
        destination=m.fs.fwh[2].inlet_2
    )

    # fwh3
    m.fs.fwh4_to_fwh3mix = Arc(
        source=m.fs.fwh[4].outlet_1,
        destination=m.fs.fwh_mixer[3].drain
    )
    m.fs.mix3_to_fwh3 = Arc(
        source=m.fs.fwh_mixer[3].outlet,
        destination=m.fs.fwh[3].inlet_1
    )
    m.fs.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2,
        destination=m.fs.fwh[3].inlet_2
    )

    # fwh4
    m.fs.fwh5_to_fwh4mix = Arc(
        source=m.fs.fwh[5].outlet_1,
        destination=m.fs.fwh_mixer[4].drain
    )
    m.fs.mix4_to_fwh4 = Arc(
        source=m.fs.fwh_mixer[4].outlet,
        destination=m.fs.fwh[4].inlet_1
    )
    m.fs.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2,
        destination=m.fs.fwh[4].inlet_2
    )

    # fwh5
    m.fs.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2
    )

    # Deaerator
    m.fs.fwh5_to_fwh6da = Arc(
        source=m.fs.fwh[5].outlet_2,
        destination=m.fs.deaerator.feedwater
    )
    m.fs.fwh7_to_fwh6da = Arc(
        source=m.fs.fwh[6].outlet_1,
        destination=m.fs.deaerator.drain
    )

    # Booster Pump
    m.fs.fwh6da_to_booster = Arc(
        source=m.fs.deaerator.outlet,
        destination=m.fs.booster.inlet
    )

    # fwh7
    m.fs.fwh7_to_fwh6mix = Arc(
        source=m.fs.fwh[7].outlet_1,
        destination=m.fs.fwh_mixer[6].drain
    )
    m.fs.mix6_to_fwh6 = Arc(
        source=m.fs.fwh_mixer[6].outlet,
        destination=m.fs.fwh[6].inlet_1
    )
    m.fs.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2
    )

    # fwh8
    m.fs.fwh8_to_fwh7mix = Arc(
        source=m.fs.fwh[8].outlet_1,
        destination=m.fs.fwh_mixer[7].drain
    )
    m.fs.mix7_to_fwh7 = Arc(
        source=m.fs.fwh_mixer[7].outlet,
        destination=m.fs.fwh[7].inlet_1
    )
    m.fs.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].outlet_2,
        destination=m.fs.fwh[7].inlet_2
    )

    # BFW Pump
    m.fs.fwh7_to_bfp = Arc(
        source=m.fs.fwh[7].outlet_2,
        destination=m.fs.bfp.inlet
    )
    
    # fwh9
    m.fs.fwh9_to_fwh8mix = Arc(
        source=m.fs.fwh[9].outlet_1,
        destination=m.fs.fwh_mixer[8].drain
    )
    m.fs.mix8_to_fwh8 = Arc(
        source=m.fs.fwh_mixer[8].outlet,
        destination=m.fs.fwh[8].inlet_1
    )
    #-------- modified by esrawli
    # m.fs.bfp_to_fwh8 = Arc(
    #     source=m.fs.bfp.outlet,
    #     destination=m.fs.fwh[8].inlet_2
    # )
    # Connecting the bfp outlet to one inlet in the recycle mixer
    m.fs.bfp_to_recyclemix = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.recycle_mixer.from_bfw_out
    )
    m.fs.recyclemix_to_fwh8 = Arc(
        source=m.fs.recycle_mixer.outlet,
        destination=m.fs.fwh[8].inlet_2
    )
    #--------

    # fwh10
    m.fs.fwh9_to_fwh10 = Arc(
        source=m.fs.fwh[8].outlet_2,
        destination=m.fs.fwh[9].inlet_2
    )

    # boiler
    m.fs.fwh10_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def set_model_input(m):
    """
    Model inputs and fixed variables or parameter values
    assumed in this block, unless otherwise stated explicitly,
    are either assumed or estimated for a total power out of 437 MW

    These inputs will also fix all necessary inputs to the model
    i.e. the degrees of freedom = 0
    """

    ###########################################################################
    #  Turbine input                                                          #
    ###########################################################################
    #  Turbine inlet conditions
    # main_steam_pressure = 31125980  # Pa
    m.fs.boiler.inlet.flow_mol.fix(m.main_flow)  # mol/s
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)

    # Reheater section pressure drop estimated
    # for a total power out of 437 MW
    m.fs.reheater_1.deltaP.fix(-742845)  # Pa
    m.fs.reheater_2.deltaP.fix(-210952)  # Pa

    #-------- modified by esrawli
    # Adding a dictionary with the turbine ratio pressure and efficiency data
    # The efficiency and pressure ratios of all turbines were estimated
    # for a total power out of 437 MW
    m.data_turbine_ratioP = {
        1: 0.388,
        2: 0.774,
        3: 0.498,
        4: 0.609,
        5: 0.523,
        6: 0.495,
        7: 0.514,
        8: 0.389,
        9: 0.572,
        10: 0.476,
        11: 0.204
    }

    m.data_turbine_eff = {
        1: 0.94,
        2: 0.94,
        3: 0.94,
        4: 0.94,
        5: 0.88,
        6: 0.88,
        7: 0.78,
        8: 0.78,
        9: 0.78,
        10: 0.78,
        11: 0.78
    }

    for k in pyo.RangeSet(m.number_turbines):
        m.fs.turbine[k].ratioP.fix(m.data_turbine_ratioP[k])
        m.fs.turbine[k].efficiency_isentropic.fix(m.data_turbine_eff[k])
    #--------

    ###########################################################################
    #  Condenser section                                                      #
    ###########################################################################
    m.fs.cond_pump.efficiency_isentropic.fix(0.80)
    m.fs.cond_pump.deltaP.fix(2313881)

    # Make up stream to condenser
    m.fs.condenser_mix.makeup.flow_mol.value = 9.0E-12  # mol/s
    m.fs.condenser_mix.makeup.pressure.fix(103421.4)  # Pa
    m.fs.condenser_mix.makeup.enth_mol.fix(1131.69204)  # J/mol

    ###########################################################################
    #  FWHs section inputs                                                    #
    ###########################################################################
    #-------- modified by esrawli
    # Adding a dictionary with the FWHs area and ohtc data. Now, all the FWHs
    # are included here and are not separated between "low" and "high" pressure
    m.data_fwh_area = {
        1: 250,
        2: 195,
        3: 164,
        4: 208,
        5: 152,
        6: 207,
        7: 202,
        8: 715,
        9: 175
        
    }
    m.data_fwh_ohtc = {}
    for k in pyo.RangeSet(m.number_fwhs):
        m.data_fwh_ohtc[k] = 3000

    for k in pyo.RangeSet(m.number_fwhs):
        m.fs.fwh[k].area.fix(m.data_fwh_area[k])
        m.fs.fwh[k].overall_heat_transfer_coefficient.fix(m.data_fwh_ohtc[k])
    #--------
    
    #########################################################################
    #  Deaerator and boiler feed pump (BFP) Input                           #
    #########################################################################
    # Unlike the feedwater heaters the steam extraction flow to the deaerator
    # is not constrained by the saturated liquid constraint. Thus, the flow
    # to the deaerator is assumed to be fixed in this model.
    m.fs.turbine_splitter[5].split_fraction[:, "outlet_2"].fix(0.017885)

    m.fs.booster.efficiency_isentropic.fix(0.80)
    m.fs.booster.deltaP.fix(5715067)
    # BFW Pump pressure is assumed based on referece report
    m.fs.bfp.outlet.pressure[:].fix(m.main_steam_pressure * 1.1231)  # Pa
    m.fs.bfp.efficiency_isentropic.fix(0.80)

    m.fs.bfpt.efficiency_isentropic.fix(0.80)

    #-------- added by esrawli
    ###########################################################################
    #  Charge Heat Exchanger section                                          #
    ###########################################################################
    # Heat Exchanger Area from supercritical plant model_input 
    # For conceptual design optimization, area is unfixed and optimized
    m.fs.hxc.area.fix(100)  # RR10

    # Salt conditions
    # Salt inlet flow is fixed during initialization but is unfixed and determined
    # during optimization
    m.fs.hxc.inlet_2.flow_mass.fix(100)   # kg/s
    m.fs.hxc.inlet_2.temperature.fix(513.15)  # K
    m.fs.hxc.inlet_2.pressure.fix(101325)  # Pa

    # Cooler outlet enthalpy is fixed during model build to ensure the inlet
    # to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler is
    # constrained in the model
    m.fs.cooler.outlet.enth_mol[0].fix(10000)
    m.fs.cooler.deltaP[0].fix(0)

    # Hx pump efficiecncy assumption
    m.fs.hx_pump.efficiency_pump.fix(0.80)
    m.fs.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure * 1.1231)

    ###########################################################################
    #  ESS HP splitter                                                        #
    ###########################################################################
    # The model is built for a fixed flow of steam through the charger.
    # This flow of steam to the charger is unfixed and determine during
    # design optimization
    m.fs.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.05)

    ###########################################################################
    #  Recycle mixer                                                          #
    ###########################################################################
    # m.fs.recycle_mix.ess.pressure[0].fix(24235081.4 * 1.15)
    # m.fs.recycle_mixer.from_hx_pump.flow_mol.fix(0.01) # fixing to a small value in mol/s
    # m.fs.recycle_mixer.from_hx_pump.enth_mol.fix(103421.4)
    # m.fs.recycle_mixer.from_hx_pump.pressure.fix(1131.69204)
    #--------



def set_scaling_factors(m):

    # FWHs
    for i in pyo.RangeSet(m.number_fwhs):
        b = m.fs.fwh[i]
        iscale.set_scaling_factor(b.area, 1e-2)
        iscale.set_scaling_factor(b.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(b.shell.heat, 1e-6)
        iscale.set_scaling_factor(b.tube.heat, 1e-6)

    # Turbines
    for j in pyo.RangeSet(m.number_turbines):
        b = m.fs.turbine[j]
        iscale.set_scaling_factor(b.control_volume.work, 1e-6)
    
    iscale.set_scaling_factor(m.fs.boiler.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reheater_1.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reheater_2.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.cond_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.booster.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.bfp.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.bfpt.control_volume.work, 1e-6)

    # Charge heat exchanger section
    iscale.set_scaling_factor(m.fs.hxc.area, 1e-2)
    iscale.set_scaling_factor(
        m.fs.hxc.overall_heat_transfer_coefficient, 1e-3)
    iscale.set_scaling_factor(m.fs.hxc.shell.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.hxc.tube.heat, 1e-6)
    
    iscale.set_scaling_factor(m.fs.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.cooler.control_volume.heat, 1e-6)

    # return m

def initialize(m, fileinput=None, outlvl=idaeslog.NOTSET,
               solver=None, optarg={}):

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    iscale.calculate_scaling_factors(m)
    
    # Initializing the boiler
    m.fs.boiler.inlet.pressure.fix(32216913)
    m.fs.boiler.inlet.enth_mol.fix(23737)
    m.fs.boiler.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.boiler.inlet.pressure.unfix()
    m.fs.boiler.inlet.enth_mol.unfix()

    #-------- added by esrawli
    # High pressure splitter initialization
    _set_port(m.fs.ess_hp_split.inlet,  m.fs.boiler.outlet)
    m.fs.ess_hp_split.initialize(outlvl=outlvl, optarg=solver.options)

    # Charge heat exchanger
    _set_port(m.fs.hxc.inlet_1, m.fs.ess_hp_split.to_hxc)
    m.fs.hxc.initialize(outlvl=outlvl, optarg=solver.options)

    #  Cooler
    _set_port(m.fs.cooler.inlet,  m.fs.hxc.outlet_1)
    m.fs.cooler.initialize(outlvl=outlvl, optarg=solver.options)
    
    # HX pump
    _set_port(m.fs.hx_pump.inlet,  m.fs.cooler.outlet)
    m.fs.hx_pump.initialize(outlvl=outlvl, optarg=solver.options)
    #--------

    # Initialization routine for the turbine train

    # Deactivating constraints that fix enthalpy at FWH outlet
    # This lets us initialize the model using the fixed split_fractions
    # for steam extractions for all the feed water heaters except deaerator
    # These split fractions will be unfixed later and the constraints will
    # be activated
    #-------- modified by esrawli
    # Adding a dictionary with the turbine splitter data. 
    m.data_turbine_split = {
        1: 0.073444,
        2: 0.140752,
        3: 0.032816,
        4: 0.012425,
        7: 0.036058,
        8: 0.026517,
        9: 0.029888,
        10: 0.003007
    }
    m.turbine_split_list = [1, 2, 3, 4, 7, 8, 9, 10]
    for k in m.turbine_split_list:
        m.fs.turbine_splitter[k].split_fraction[:, "outlet_2"].fix(
            m.data_turbine_split[k])
    m.fs.turbine_splitter[6].split_fraction[:, "outlet_2"].fix(0.081155)
    m.fs.turbine_splitter[6].split_fraction[:, "outlet_3"].fix(0.091274)
    #--------
    
    m.fs.constraint_bfp_power.deactivate()
    m.fs.constraint_out_pressure.deactivate()
    for i in pyo.RangeSet(m.number_fwhs):
        m.fs.fwh[i].fwh_vfrac_constraint.deactivate()


    # solving the turbine, splitter, and reheaters
    #-------- modified by esrawli
    # _set_port(m.fs.turbine[1].inlet,  m.fs.boiler.outlet)
    # Connecting the turbine 1 inlet with the ess hp splitter
    _set_port(m.fs.turbine[1].inlet, m.fs.ess_hp_split.to_hp)
    m.fs.turbine[1].initialize(outlvl=outlvl, optarg=solver.options)
    #--------
    
    _set_port(m.fs.turbine_splitter[1].inlet, m.fs.turbine[1].outlet)
    m.fs.turbine_splitter[1].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[2].inlet, m.fs.turbine_splitter[1].outlet_1)
    m.fs.turbine[2].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[2].inlet, m.fs.turbine[2].outlet)
    m.fs.turbine_splitter[2].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.reheater_1.inlet, m.fs.turbine_splitter[2].outlet_1)
    m.fs.reheater_1.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[3].inlet, m.fs.reheater_1.outlet)
    m.fs.turbine[3].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[3].inlet, m.fs.turbine[3].outlet)
    m.fs.turbine_splitter[3].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[4].inlet, m.fs.turbine_splitter[3].outlet_1)
    m.fs.turbine[4].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[4].inlet, m.fs.turbine[4].outlet)
    m.fs.turbine_splitter[4].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.reheater_2.inlet, m.fs.turbine_splitter[4].outlet_1)
    m.fs.reheater_2.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[5].inlet, m.fs.reheater_2.outlet)
    m.fs.turbine[5].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[5].inlet, m.fs.turbine[5].outlet)
    m.fs.turbine_splitter[5].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[6].inlet, m.fs.turbine_splitter[5].outlet_1)
    m.fs.turbine[6].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[6].inlet, m.fs.turbine[6].outlet)
    m.fs.turbine_splitter[6].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[7].inlet, m.fs.turbine_splitter[6].outlet_1)
    m.fs.turbine[7].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[7].inlet, m.fs.turbine[7].outlet)
    m.fs.turbine_splitter[7].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[8].inlet, m.fs.turbine_splitter[7].outlet_1)
    m.fs.turbine[8].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[8].inlet, m.fs.turbine[8].outlet)
    m.fs.turbine_splitter[8].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[9].inlet, m.fs.turbine_splitter[8].outlet_1)
    m.fs.turbine[9].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[9].inlet, m.fs.turbine[9].outlet)
    m.fs.turbine_splitter[9].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[10].inlet, m.fs.turbine_splitter[9].outlet_1)
    m.fs.turbine[10].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[10].inlet, m.fs.turbine[10].outlet)
    m.fs.turbine_splitter[10].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[11].inlet, m.fs.turbine_splitter[10].outlet_1)
    m.fs.turbine[11].initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the boiler feed pump turbine.
    _set_port(m.fs.bfpt.inlet, m.fs.turbine_splitter[6].outlet_3)
    m.fs.bfpt.outlet.pressure.fix(6896)
    m.fs.bfpt.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.bfpt.outlet.pressure.unfix()

    ###########################################################################
    #  Condenser                                                              #
    ###########################################################################
    _set_port(m.fs.condenser_mix.bfpt, m.fs.bfpt.outlet)
    _set_port(m.fs.condenser_mix.main, m.fs.turbine[11].outlet)
    m.fs.condenser_mix.drain.flow_mol.fix(2102)
    m.fs.condenser_mix.drain.pressure.fix(7586)
    m.fs.condenser_mix.drain.enth_mol.fix(3056)
    m.fs.condenser_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.condenser_mix.drain.unfix()

    _set_port(m.fs.condenser.inlet, m.fs.condenser_mix.outlet)
    m.fs.condenser.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.cond_pump.inlet, m.fs.condenser.outlet)
    m.fs.cond_pump.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  Low pressure FWH section                                               #
    ###########################################################################

    # fwh1
    _set_port(m.fs.fwh_mixer[1].steam,  m.fs.turbine_splitter[10].outlet_2)
    m.fs.fwh_mixer[1].drain.flow_mol.fix(2072)
    m.fs.fwh_mixer[1].drain.pressure.fix(37187)
    m.fs.fwh_mixer[1].drain.enth_mol.fix(5590)
    m.fs.fwh_mixer[1].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[1].drain.unfix()

    _set_port(m.fs.fwh[1].inlet_1, m.fs.fwh_mixer[1].outlet)
    _set_port(m.fs.fwh[1].inlet_2, m.fs.cond_pump.outlet)
    m.fs.fwh[1].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh2
    _set_port(m.fs.fwh_mixer[2].steam,  m.fs.turbine_splitter[9].outlet_2)
    m.fs.fwh_mixer[2].drain.flow_mol.fix(1762)
    m.fs.fwh_mixer[2].drain.pressure.fix(78124)
    m.fs.fwh_mixer[2].drain.enth_mol.fix(7009)
    m.fs.fwh_mixer[2].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[2].drain.unfix()

    _set_port(m.fs.fwh[2].inlet_1, m.fs.fwh_mixer[2].outlet)
    _set_port(m.fs.fwh[2].inlet_2, m.fs.fwh[1].outlet_2)
    m.fs.fwh[2].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh3
    _set_port(m.fs.fwh_mixer[3].steam,  m.fs.turbine_splitter[8].outlet_2)
    m.fs.fwh_mixer[3].drain.flow_mol.fix(1480)
    m.fs.fwh_mixer[3].drain.pressure.fix(136580)
    m.fs.fwh_mixer[3].drain.enth_mol.fix(8203)
    m.fs.fwh_mixer[3].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[3].drain.unfix()

    _set_port(m.fs.fwh[3].inlet_1, m.fs.fwh_mixer[3].outlet)
    _set_port(m.fs.fwh[3].inlet_2, m.fs.fwh[2].outlet_2)
    m.fs.fwh[3].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh4
    _set_port(m.fs.fwh_mixer[4].steam,  m.fs.turbine_splitter[7].outlet_2)
    m.fs.fwh_mixer[4].drain.flow_mol.fix(1082)
    m.fs.fwh_mixer[4].drain.pressure.fix(351104)
    m.fs.fwh_mixer[4].drain.enth_mol.fix(10534)
    m.fs.fwh_mixer[4].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[4].drain.unfix()

    _set_port(m.fs.fwh[4].inlet_1, m.fs.fwh_mixer[4].outlet)
    _set_port(m.fs.fwh[4].inlet_2, m.fs.fwh[3].outlet_2)
    m.fs.fwh[4].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh5
    _set_port(m.fs.fwh[5].inlet_2, m.fs.fwh[4].outlet_2)
    _set_port(m.fs.fwh[5].inlet_1, m.fs.turbine_splitter[6].outlet_2)
    m.fs.fwh[5].initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  boiler feed pump and deaerator                                         #
    ###########################################################################
    # Deaerator
    _set_port(m.fs.deaerator.feedwater, m.fs.fwh[5].outlet_2)
    _set_port(m.fs.deaerator.steam, m.fs.turbine_splitter[5].outlet_2)
    m.fs.deaerator.drain.flow_mol[:].fix(4277)
    m.fs.deaerator.drain.pressure[:].fix(1379964)
    m.fs.deaerator.drain.enth_mol[:].fix(14898)
    m.fs.deaerator.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.deaerator.drain.unfix()

    # Booster pump
    _set_port(m.fs.booster.inlet, m.fs.deaerator.outlet)
    m.fs.booster.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  High-pressure feedwater heaters                                        #
    ###########################################################################
    # fwh6
    _set_port(m.fs.fwh_mixer[6].steam, m.fs.turbine_splitter[4].outlet_2)
    m.fs.fwh_mixer[6].drain.flow_mol.fix(4106)
    m.fs.fwh_mixer[6].drain.pressure.fix(2870602)
    m.fs.fwh_mixer[6].drain.enth_mol.fix(17959)
    m.fs.fwh_mixer[6].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[6].drain.unfix()

    _set_port(m.fs.fwh[6].inlet_1, m.fs.fwh_mixer[6].outlet)
    _set_port(m.fs.fwh[6].inlet_2, m.fs.booster.outlet)
    m.fs.fwh[6].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh7
    _set_port(m.fs.fwh_mixer[7].steam, m.fs.turbine_splitter[3].outlet_2)
    m.fs.fwh_mixer[7].drain.flow_mol.fix(3640)
    m.fs.fwh_mixer[7].drain.pressure.fix(4713633)
    m.fs.fwh_mixer[7].drain.enth_mol.fix(20472)
    m.fs.fwh_mixer[7].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[7].drain.unfix()

    _set_port(m.fs.fwh[7].inlet_1, m.fs.fwh_mixer[7].outlet)
    _set_port(m.fs.fwh[7].inlet_2, m.fs.fwh[6].outlet_2)
    m.fs.fwh[7].initialize(outlvl=outlvl, optarg=solver.options)

    # Boiler feed pump
    _set_port(m.fs.bfp.inlet, m.fs.fwh[7].outlet_2)
    m.fs.bfp.initialize(outlvl=outlvl, optarg=solver.options)

    #-------- added by esrawli
    #  Recycle mixer initialization
    _set_port(m.fs.recycle_mixer.from_bfw_out, m.fs.bfp.outlet)
    _set_port(m.fs.recycle_mixer.from_hx_pump, m.fs.hx_pump.outlet)
    # fixing to a small value in mol/s
    # m.fs.recycle_mixer.from_hx_pump.flow_mol.fix(0.01)
    # m.fs.recycle_mixer.from_hx_pump.enth_mol.fix()
    # m.fs.recycle_mixer.from_hx_pump.pressure.fix()
    m.fs.recycle_mixer.initialize(outlvl=outlvl, optarg=solver.options)
    #--------

    # fwh8
    _set_port(m.fs.fwh_mixer[8].steam, m.fs.turbine_splitter[2].outlet_2)
    m.fs.fwh_mixer[8].drain.flow_mol.fix(1311)
    m.fs.fwh_mixer[8].drain.pressure.fix(10282256)
    m.fs.fwh_mixer[8].drain.enth_mol.fix(25585)
    m.fs.fwh_mixer[8].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[8].drain.unfix()

    _set_port(m.fs.fwh[8].inlet_1, m.fs.fwh_mixer[8].outlet)
    #-------- modified by esrawli
    # fwh8 inlet 2 should be connected to bfp outlet
    # _set_port(m.fs.fwh[8].inlet_2, m.fs.fwh[7].outlet_2) # in original model
    # _set_port(m.fs.fwh[8].inlet_2, m.fs.bfp.outlet) # corrected
    _set_port(m.fs.fwh[8].inlet_2, m.fs.recycle_mixer.outlet) 
    #----------
    m.fs.fwh[8].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh9
    _set_port(m.fs.fwh[9].inlet_2, m.fs.fwh[8].outlet_2)
    _set_port(m.fs.fwh[9].inlet_1, m.fs.turbine_splitter[1].outlet_2)
    m.fs.fwh[9].initialize(outlvl=outlvl, optarg=solver.options)

    #########################################################################
    #  Model Initialization with Square Problem Solve                       #
    #########################################################################
    #  Unfix split fractions and activate vapor fraction constraints
    #  Vaporfrac constraints set condensed steam enthalpy at the condensing
    #  side outlet to be that of a saturated liquid
    # Then solve the square problem again for an initilized model
    for i in pyo.RangeSet(m.number_turbine_splitters):
        m.fs.turbine_splitter[i].split_fraction[:, "outlet_2"].unfix()

    # keeping the extraction to deareator to be fixed
    # unfixing the extraction to bfpt
    m.fs.turbine_splitter[5].split_fraction[:, "outlet_2"].fix()
    m.fs.turbine_splitter[6].split_fraction[:, "outlet_3"].unfix()

    m.fs.constraint_bfp_power.activate()
    m.fs.constraint_out_pressure.activate()
    for j in pyo.RangeSet(m.number_fwhs):
        m.fs.fwh[j].fwh_vfrac_constraint.activate()

    res = solver.solve(m)
    print("Model Initialization = ",
          res.solver.termination_condition)
    print("*********************Model Initialized**************************")
    print('')
    print('')

    return solver

#-------- added by esrawli
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

    
    ###########################################################################
    #  Solver options                                                         #
    ###########################################################################
    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    ###########################################################################
    #  Data                                                              #
    ###########################################################################
    m.number_hours_per_day = 6          
    m.number_of_years = 5

    #--------------------------------------------
    #  General costing data
    #--------------------------------------------
    m.CE_index = 607.5  # Chemical engineering cost index for 2019

    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'q_baseline_charge': 917930546.8932868,
        'solar_salt_price': 0.49,
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
    m.fs.coal_price = pyo.Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.cooling_price = pyo.Param(
        initialize=m.data_cost['cooling_price'],
        doc='Cost of chilled water for cooler from Sieder et al. $/J')
    m.fs.q_baseline = pyo.Param(
        initialize=m.data_cost['q_baseline_charge'],
        doc='Boiler duty in Wth @ 699MW for baseline plant with no storage')

    # Data for solar molten salt 
    m.fs.solar_salt_price = pyo.Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')

    #--------------------------------------------
    #  Operating hours
    #--------------------------------------------
    m.fs.hours_per_day = pyo.Var(
        initialize=m.number_hours_per_day,
        bounds=(0, 12),
        doc='Estimated number of hours of charging per day'
    )
    
    # Fixing number of hours of discharging to 6 in this problem
    m.fs.hours_per_day.fix(m.number_hours_per_day)

    # Number of year over which the capital cost is annualized
    m.fs.num_of_years = pyo.Param(
        initialize=m.number_of_years,
        doc='Number of years for capital cost annualization')

    ###########################################################################
    #  Capital cost                                                           #
    ###########################################################################

    #--------------------------------------------
    #  Solar salt inventory
    #--------------------------------------------
    # Total inventory of salt required
    m.fs.salt_amount = pyo.Expression(
        expr=(m.fs.hxc.inlet_2.flow_mass[0] * \
              m.fs.hours_per_day * 3600),
        doc="Salt flow in gal per min"
    )

    # Purchase cost of salt
    m.fs.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt purchase cost in $"
    )

    def solar_salt_purchase_cost_rule(b):
        return (
            m.fs.salt_purchase_cost * m.fs.num_of_years == (
                m.fs.salt_amount * m.fs.solar_salt_price
            )
        )
    m.fs.salt_purchase_cost_eq = pyo.Constraint(
        rule=solar_salt_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.salt_purchase_cost,
        m.fs.salt_purchase_cost_eq)

    #--------------------------------------------
    #  Solar salt charge heat exchangers costing
    #--------------------------------------------
    # The charge heat exchanger cost is estimated using the IDAES costing
    # method with default options, i.e. a U-tube heat exchnager, stainless
    # steel material, and a tube length of 12ft. Refer to costing documentation
    # to change any of the default options
    # Purchase cost of heat exchanger has to be annualized when used
    m.fs.hxc.get_costing()
    m.fs.hxc.costing.CE_index = m.CE_index
    # Initializing the costing correlations
    icost.initialize(m.fs.hxc.costing)

    #--------------------------------------------
    #  Water pump
    #--------------------------------------------
    # The water pump is added as hx_pump before sending the feed water
    # to the discharge heat exchnager in order to increase its pressure.
    # The outlet pressure of hx_pump is set based on its return point
    # in the flowsheet
    # Purchase cost of hx_pump has to be annualized when used
    m.fs.hx_pump.get_costing(
        Mat_factor="stain_steel",
        mover_type="compressor",
        compressor_type="centrifugal",
        driver_mover_type="electrical_motor",
        pump_type="centrifugal",
        pump_type_factor='1.4',
        pump_motor_type_factor='open'
        )
    m.fs.hx_pump.costing.CE_index = m.CE_index
    # Initialize cost correlation
    icost.initialize(m.fs.hx_pump.costing)

    #--------------------------------------------
    #  Solar salt pump costing
    #--------------------------------------------
    # The salt pump is not explicitly modeled. Thus, the IDAES cost method
    # is not used for this equipment at this time.
    # The primary purpose of the salt pump is to move molten salt and not to
    # change the pressure. Thus the pressure head is computed assuming that
    # the salt is moved on an average of 5m linear distance.
    m.fs.spump_FT = pyo.Param(
        initialize=m.data_salt_pump['FT'],
        doc='Pump Type Factor for vertical split case')
    m.fs.spump_FM = pyo.Param(
        initialize=m.data_salt_pump['FM'],
        doc='Pump Material Factor Stainless Steel')
    m.fs.spump_head = pyo.Param(
        initialize=m.data_salt_pump['head'],
        doc='Pump Head 5m in Ft.')
    m.fs.spump_motorFT = pyo.Param(
        initialize=m.data_salt_pump['motor_FT'],
        doc='Motor Shaft Type Factor')
    m.fs.spump_nm = pyo.Param(
        initialize=m.data_salt_pump['nm'],
        doc='Motor Shaft Type Factor')

    # converting salt flow mas to flow vol in gal per min, Q_gpm
    m.fs.spump_Qgpm = pyo.Expression(
        expr=(m.fs.hxc.side_2.properties_in[0].flow_mass * 264.17 * 60
              / (m.fs.hxc.side_2.properties_in[0].density["Liq"])),
        doc="Salt flow in gal per min"
    )
    # Density in lb per ft3
    m.fs.dens_lbft3 = pyo.Expression(
        expr=m.fs.hxc.side_2.properties_in[0].density["Liq"]*0.062428,
        doc="pump size factor"
    )
    # Pump size factor
    m.fs.spump_sf = pyo.Expression(
        expr=m.fs.spump_Qgpm * (m.fs.spump_head ** 0.5),
        doc="pump size factor"
    )
    # Pump purchase cost
    m.fs.pump_CP = pyo.Expression(
        expr=(m.fs.spump_FT * m.fs.spump_FM * \
              exp(9.2951
                  - 0.6019 * log(m.fs.spump_sf)
                  + 0.0519 * ((log(m.fs.spump_sf))**2)
              )
        ),
        doc="Salt Pump Base Cost in $"
    )
    
    # Costing motor
    m.fs.spump_np = pyo.Expression(
        expr=(-0.316
              + 0.24015 * log(m.fs.spump_Qgpm)
              - 0.01199 * ((log(m.fs.spump_Qgpm))**2)
        ),
        doc="fractional efficiency of the pump horse power"
    )
    # Motor power consumption
    m.fs.motor_pc = pyo.Expression(
        expr=((m.fs.spump_Qgpm * m.fs.spump_head * \
               m.fs.dens_lbft3) / \
              (33000 * m.fs.spump_np * m.fs.spump_nm)),
        doc="Horsepower consumption"
    )
    # Motor purchase cost
    m.fs.motor_CP = pyo.Expression(
        expr=(m.fs.spump_motorFT
              * exp(5.4866 + 0.13141 * log(m.fs.motor_pc)
                    + 0.053255 * ((log(m.fs.motor_pc))**2)
                    + 0.028628 * ((log(m.fs.motor_pc))**3)
                    - 0.0035549 * ((log(m.fs.motor_pc))**4))
              ),
        doc="Salt Pump's Motor Base Cost in $"
    )

    # Pump and motor purchase cost
    # pump toal cost constraint
    m.fs.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt pump and motor purchase cost in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            m.fs.spump_purchase_cost * m.fs.num_of_years
            == (m.fs.pump_CP + m.fs.motor_CP) * \
            (m.CE_index / 394)
        )
    m.fs.spump_purchase_cost_eq = pyo.Constraint(
        rule=solar_spump_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.spump_purchase_cost,
        m.fs.spump_purchase_cost_eq)

    #--------------------------------------------
    #  Solar salt storage tank costing: vertical vessel
    #--------------------------------------------
    m.fs.l_by_d = pyo.Param(
        initialize=m.data_storage_tank['LbyD'],
        doc='L by D assumption for computing storage tank dimensions')
    m.fs.tank_thickness = pyo.Param(
        initialize=m.data_storage_tank['tank_thickness'],
        doc='Storage tank thickness assumed based on reference'
    )
    # Tank size and dimension computation
    m.fs.tank_volume = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank w/20% excess capacity")
    m.fs.tank_surf_area = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.tank_diameter = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.tank_height = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank [m]")
    m.fs.no_of_tanks = pyo.Var(
        initialize=1,
        bounds=(1, 3),
        doc='No of Tank units to use cost correlations')

    # Number of tanks change
    m.fs.no_of_tanks.fix()

    # Computing tank volume - jfr: editing to include 20% margin
    def solar_tank_volume_rule(b):
        return (
            m.fs.tank_volume * \
            m.fs.hxc.side_2.properties_in[0].density["Liq"] == \
            m.fs.salt_amount * 1.10
        )
    m.fs.tank_volume_eq = pyo.Constraint(
        rule=solar_tank_volume_rule)

    # Compute surface area of tank: surf area of sides + top surf area
    # base area is accounted in foundation costs
    def solar_tank_surf_area_rule(b):
        return (
            m.fs.tank_surf_area == (
                pi * m.fs.tank_diameter * m.fs.tank_height)
            + (pi*m.fs.tank_diameter**2)/4
        )
    m.fs.tank_surf_area_eq = pyo.Constraint(
        rule=solar_tank_surf_area_rule)

    # Computing diameter for an assumed L by D
    def solar_tank_diameter_rule(b):
        return (
            m.fs.tank_diameter == (
                (4 * (m.fs.tank_volume / m.fs.no_of_tanks) / \
                 (m.fs.l_by_d * pi))** (1 / 3))
        )
    m.fs.tank_diameter_eq = pyo.Constraint(
        rule=solar_tank_diameter_rule)

    # Computing height of tank
    def solar_tank_height_rule(b):
        return m.fs.tank_height == (
            m.fs.l_by_d * m.fs.tank_diameter)
    m.fs.tank_height_eq = pyo.Constraint(
        rule=solar_tank_height_rule)

    # Initialize tanks design correlations
    calculate_variable_from_constraint(
        m.fs.tank_volume,
        m.fs.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.tank_diameter,
        m.fs.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.tank_height,
        m.fs.tank_height_eq)

    # A dummy pyomo block for salt storage tank is declared for costing
    # The diameter and length for this tank is assumed
    # based on a number of tank (see the above for m.fs.no_of_tanks)
    # Costing for each vessel designed above
    m.fs.salt_tank = pyo.Block()
    m.fs.salt_tank.costing = pyo.Block()

    m.fs.salt_tank.costing.material_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_material'],
        doc='$/kg of SS316 material'
    )

    m.fs.salt_tank.costing.insulation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_insulation'],
        doc='$/m2'
    )

    m.fs.salt_tank.costing.foundation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_foundation'],
        doc='$/m2'
    )

    m.fs.salt_tank.costing.material_density = pyo.Param(
        initialize=m.data_storage_tank['material_density'],
        doc='Kg/m3'
    )

    m.fs.salt_tank.costing.tank_material_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    m.fs.salt_tank.costing.tank_insulation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    m.fs.salt_tank.costing.tank_foundation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def rule_tank_material_cost(b):
        return m.fs.salt_tank.costing.tank_material_cost == (
            m.fs.salt_tank.costing.material_cost * \
            m.fs.salt_tank.costing.material_density * \
            m.fs.tank_surf_area * \
            m.fs.tank_thickness
        )
    m.fs.salt_tank.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_tank_material_cost)

    def rule_tank_insulation_cost(b):
        return m.fs.salt_tank.costing.tank_insulation_cost == (
            m.fs.salt_tank.costing.insulation_cost * \
            m.fs.tank_surf_area
        )
    m.fs.salt_tank.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_tank_insulation_cost)

    def rule_tank_foundation_cost(b):
        return m.fs.salt_tank.costing.tank_foundation_cost == (
            m.fs.salt_tank.costing.foundation_cost * \
            pi * m.fs.tank_diameter**2 / 4
        )
    m.fs.salt_tank.costing.eq_tank_foundation_cost = pyo.Constraint(
        rule=rule_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.salt_tank.costing.total_tank_cost = pyo.Expression(
        expr=m.fs.salt_tank.costing.tank_material_cost 
        + m.fs.salt_tank.costing.tank_foundation_cost 
        + m.fs.salt_tank.costing.tank_insulation_cost
    )

    #--------------------------------------------
    # Total annualized capital cost for solar salt
    #--------------------------------------------
    # Capital cost var at flowsheet level to handle the salt capital cost
    # depending on the salt selected.
    m.fs.capital_cost = pyo.Var(
        initialize=1000000,
        doc="Annualized capital cost")

    # Annualized capital cost for the solar salt
    def solar_cap_cost_rule(b):
        return m.fs.capital_cost == (
            m.fs.salt_purchase_cost 
            + m.fs.spump_purchase_cost
            + (m.fs.hxc.costing.purchase_cost
               + m.fs.hx_pump.costing.purchase_cost
               + m.fs.no_of_tanks * \
               m.fs.salt_tank.costing.total_tank_cost)
            / m.fs.num_of_years
        )
    m.fs.cap_cost_eq = pyo.Constraint(
        rule=solar_cap_cost_rule)
    
    # Initialize
    calculate_variable_from_constraint(
        m.fs.capital_cost,
        m.fs.cap_cost_eq)

    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.hours_per_day,
        doc="Number of operating hours per year"
    )

    # Operating cost var
    m.fs.operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, None),
        doc="Annualized capital cost")

    #-------- added by esrawli
    m.fs.plant_power_out = pyo.Var(
        m.fs.time,
        domain=pyo.Reals,
        bounds=(300, 600), # values recommended by Naresh
        initialize=400,
        doc="Net Power MWe out from the power plant",
        units=pyunits.MW
    )

    #   Constraint on Plant Power Output
    #   Plant Power Out = Total Turbine Power
    @m.fs.Constraint(m.fs.time)
    def production_cons(b, t):
        return (
            (-1*sum(m.fs.turbine[p].work_mechanical[t]
                 for p in pyo.RangeSet(m.number_turbines))
             )
            == m.fs.plant_power_out[t]*1e6*(pyunits.W/pyunits.MW)
        )
    #--------

    def op_cost_rule(b):
        return m.fs.operating_cost == (
            m.fs.operating_hours * m.fs.coal_price * \
            (m.fs.boiler.heat_duty[0]
             + m.fs.reheater_1.heat_duty[0]
             + m.fs.reheater_2.heat_duty[0]
             - m.fs.q_baseline)
            - (m.fs.cooling_price * m.fs.operating_hours * \
               m.fs.cooler.heat_duty[0])
        )
    m.fs.op_cost_eq = pyo.Constraint(rule=op_cost_rule)

    # Initialize
    calculate_variable_from_constraint(
        m.fs.operating_cost,
        m.fs.op_cost_eq)

    res = solver.solve(m, tee=True)
    print("Cost Initialization = ",
          res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print('')
    print('')

    # return solver
#--------

def view_result(outfile, m):
    """
    Print the results in a process flow diagram previously given in a .svg file
    """
    
    tags = {}

    # Boiler
    tags['obj'] = ("%4.2f" % pyo.value(m.obj))
    tags['power_out'] = ("%4.2f" % pyo.value(m.fs.plant_power_out[0]))
    
    tags['boiler_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.boiler.inlet.flow_mol[0])*1e-3))
    tags['boiler_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.boiler.control_volume.properties_in[0].temperature)))
    tags['boiler_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.inlet.pressure[0])*1e-6))
    tags['boiler_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.inlet.enth_mol[0])*1e-3))
    tags['boiler_xin'] = ("%4.4f" % (pyo.value(
        m.fs.boiler.control_volume.properties_in[0].vapor_frac)))
    tags['boiler_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.boiler.outlet.flow_mol[0])*1e-3))
    tags['boiler_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.boiler.control_volume.properties_out[0].temperature)))
    tags['boiler_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.outlet.pressure[0])*1e-6))
    tags['boiler_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.outlet.enth_mol[0])*1e-3))
    tags['boiler_xout'] = ("%4.4f" % (pyo.value(
        m.fs.boiler.control_volume.properties_out[0].vapor_frac)))

    # Reheater 1 & 2
    tags['turb3_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.turbine[3].inlet.flow_mol[0])*1e-3))
    tags['turb3_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.turbine[3].control_volume.properties_in[0].temperature)))
    tags['turb3_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[3].inlet.pressure[0])*1e-6))
    tags['turb3_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[3].inlet.enth_mol[0])*1e-3))
    tags['turb3_xin'] = ("%4.4f" % (pyo.value(
        m.fs.turbine[3].control_volume.properties_in[0].vapor_frac)))

    tags['turb5_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.turbine[5].inlet.flow_mol[0])*1e-3))
    tags['turb5_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.turbine[5].control_volume.properties_in[0].temperature)))
    tags['turb5_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[5].inlet.pressure[0])*1e-6))
    tags['turb5_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[5].inlet.enth_mol[0])*1e-3))
    tags['turb5_xin'] = ("%4.4f" % (pyo.value(
        m.fs.turbine[5].control_volume.properties_in[0].vapor_frac)))

    # Turbine out
    tags['turb11_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.turbine[11].outlet.flow_mol[0])*1e-3))
    tags['turb11_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.turbine[11].control_volume.properties_out[0].temperature)))
    tags['turb11_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[11].outlet.pressure[0])*1e-6))
    tags['turb11_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[11].outlet.enth_mol[0])*1e-3))
    tags['turb11_xout'] = ("%4.4f" % (pyo.value(
        m.fs.turbine[11].control_volume.properties_out[0].vapor_frac)))

    # Condenser
    tags['cond_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.condenser.outlet.flow_mol[0])*1e-3))
    tags['cond_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.condenser.control_volume.properties_out[0].temperature)))
    tags['cond_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.condenser.outlet.pressure[0])*1e-6))
    tags['cond_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.condenser.outlet.enth_mol[0])*1e-3))
    tags['cond_xout'] = ("%4.4f" % (pyo.value(
        m.fs.condenser.control_volume.properties_out[0].vapor_frac)))

    # Feed water heaters
    tags['fwh9shell_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[9].shell_inlet.flow_mol[0])*1e-3))
    tags['fwh9shell_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[9].shell.properties_in[0].temperature)))
    tags['fwh9shell_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[9].shell_inlet.pressure[0])*1e-6))
    tags['fwh9shell_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[9].shell_inlet.enth_mol[0])*1e-3))
    tags['fwh9shell_xin'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[9].shell.properties_in[0].vapor_frac)))

    tags['fwh7tube_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[7].tube_outlet.flow_mol[0])*1e-3))
    tags['fwh7tube_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[7].tube.properties_out[0].temperature)))
    tags['fwh7tube_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[7].tube_outlet.pressure[0])*1e-6))
    tags['fwh7tube_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[7].tube_outlet.enth_mol[0])*1e-3))
    tags['fwh7tube_xout'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[7].tube.properties_out[0].vapor_frac)))

    tags['fwh6shell_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[6].shell_outlet.flow_mol[0])*1e-3))
    tags['fwh6shell_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[6].shell.properties_out[0].temperature)))
    tags['fwh6shell_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[6].shell_outlet.pressure[0])*1e-6))
    tags['fwh6shell_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[6].shell_outlet.enth_mol[0])*1e-3))
    tags['fwh6shell_xout'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[6].shell.properties_out[0].vapor_frac)))

    tags['fwh5tube_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[5].tube_outlet.flow_mol[0])*1e-3))
    tags['fwh5tube_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[5].tube.properties_out[0].temperature)))
    tags['fwh5tube_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].tube_outlet.pressure[0])*1e-6))
    tags['fwh5tube_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].tube_outlet.enth_mol[0])*1e-3))
    tags['fwh5tube_xout'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[5].tube.properties_out[0].vapor_frac)))

    tags['fwh5shell_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[5].shell_inlet.flow_mol[0])*1e-3))
    tags['fwh5shell_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[5].shell.properties_in[0].temperature)))
    tags['fwh5shell_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].shell_inlet.pressure[0])*1e-6))
    tags['fwh5shell_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].shell_inlet.enth_mol[0])*1e-3))
    tags['fwh5shell_xin'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[5].shell.properties_in[0].vapor_frac)))

    # Deareator
    tags['da_steam_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.deaerator.steam.flow_mol[0])*1e-3))
    tags['da_steam_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.deaerator.steam_state[0].temperature)))
    tags['da_steam_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.steam.pressure[0])*1e-6))
    tags['da_steam_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.steam.enth_mol[0])*1e-3))
    tags['da_steam_xin'] = ("%4.4f" % (pyo.value(
        m.fs.deaerator.steam_state[0].vapor_frac)))
    tags['da_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.deaerator.outlet.flow_mol[0])*1e-3))
    tags['da_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.deaerator.mixed_state[0].temperature)))
    tags['da_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.outlet.pressure[0])*1e-6))
    tags['da_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.outlet.enth_mol[0])*1e-3))
    tags['da_xout'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.mixed_state[0].vapor_frac)))

    # Feed water heaters mixers
    for i in m.mixer_list:
        tags['fwh'+str(i)+'mix_steam_Fin'] = ("%4.3f" % (pyo.value(
            m.fs.fwh_mixer[i].steam.flow_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_steam_Tin'] = ("%4.2f" % (pyo.value(
            m.fs.fwh_mixer[i].steam_state[0].temperature)))
        tags['fwh'+str(i)+'mix_steam_Pin'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].steam.pressure[0])*1e-6))
        tags['fwh'+str(i)+'mix_steam_Hin'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].steam.enth_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_steam_xin'] = ("%4.4f" % (pyo.value(
            m.fs.fwh_mixer[i].steam_state[0].vapor_frac)))
        tags['fwh'+str(i)+'mix_Fout'] = ("%4.3f" % (pyo.value(
            m.fs.fwh_mixer[i].outlet.flow_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_Tout'] = ("%4.2f" % (pyo.value(
            m.fs.fwh_mixer[i].mixed_state[0].temperature)))
        tags['fwh'+str(i)+'mix_Pout'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].outlet.pressure[0])*1e-6))
        tags['fwh'+str(i)+'mix_Hout'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].outlet.enth_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_xout'] = ("%4.4f" % (pyo.value(
            m.fs.fwh_mixer[i].mixed_state[0].vapor_frac)))

    # BFP
    tags['bfp_power'] = ("%4.2f" % (pyo.value(
        m.fs.bfp.control_volume.work[0])*1e-6))
    tags['booster_power'] = ("%4.2f" % (pyo.value(
        m.fs.booster.control_volume.work[0])*1e-6))
    tags['bfpt_power'] = ("%4.2f" % (pyo.value(
        m.fs.bfpt.control_volume.work[0])*-1e-6))
    tags['cond_power'] = ("%4.2f" % (pyo.value(
        m.fs.cond_pump.control_volume.work[0])*1e-6))

    # ESS HP Splitter
    tags['essvhp_Fout1'] = ("%4.3f" % (pyo.value(
        m.fs.ess_hp_split.to_hp.flow_mol[0])*1e-3))
    tags['essvhp_Tout1'] = ("%4.2f" % (pyo.value(
        m.fs.ess_hp_split.to_hp_state[0].temperature)))
    tags['essvhp_Pout1'] = ("%4.1f" % (pyo.value(
        m.fs.ess_hp_split.to_hp.pressure[0])*1e-6))
    tags['essvhp_Hout1'] = ("%4.1f" % (pyo.value(
        m.fs.ess_hp_split.to_hp.enth_mol[0])*1e-3))
    tags['essvhp_xout1'] = ("%4.4f" % (pyo.value(
        m.fs.ess_hp_split.to_hp_state[0].vapor_frac)))
    tags['essvhp_Fout2'] = ("%4.3f" % (pyo.value(
        m.fs.ess_hp_split.to_hxc.flow_mol[0])*1e-3))
    tags['essvhp_Tout2'] = ("%4.2f" % (pyo.value(
        m.fs.ess_hp_split.to_hxc_state[0].temperature)))
    tags['essvhp_Pout2'] = ("%4.1f" % (pyo.value(
        m.fs.ess_hp_split.to_hxc.pressure[0])*1e-6))
    tags['essvhp_Hout2'] = ("%4.1f" % (pyo.value(
        m.fs.ess_hp_split.to_hxc.enth_mol[0])*1e-3))
    tags['essvhp_xout2'] = ("%4.4f" % (pyo.value(
        m.fs.ess_hp_split.to_hxc_state[0].vapor_frac)))

    # Recycle mixer
    tags['rmix_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.recycle_mixer.outlet.flow_mol[0])*1e-3))
    tags['rmix_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.recycle_mixer.mixed_state[0].temperature)))
    tags['rmix_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.recycle_mixer.outlet.pressure[0])*1e-6))
    tags['rmix_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.recycle_mixer.outlet.enth_mol[0])*1e-3))
    tags['rmix_xout'] = ("%4.4f" % (pyo.value(
        m.fs.recycle_mixer.mixed_state[0].vapor_frac)))

    # Charge heat exchanger
    tags['hxsteam_Fout'] = ("%4.4f" % (pyo.value(
        m.fs.hxc.outlet_1.flow_mol[0])*1e-3))
    tags['hxsteam_Tout'] = ("%4.4f" % (pyo.value(
        m.fs.hxc.side_1.properties_out[0].temperature)))
    tags['hxsteam_Pout'] = ("%4.4f" % (
        pyo.value(m.fs.hxc.outlet_1.pressure[0])*1e-6))
    tags['hxsteam_Hout'] = ("%4.2f" % (
        pyo.value(m.fs.hxc.outlet_1.enth_mol[0])))
    tags['hxsteam_xout'] = ("%4.4f" % (
        pyo.value(m.fs.hxc.side_1.properties_out[0].vapor_frac)))

    ## (sub)Cooler
    tags['cooler_Fout'] = ("%4.4f" % (pyo.value(
        m.fs.cooler.outlet.flow_mol[0])*1e-3))
    tags['cooler_Tout'] = ("%4.4f" % (
        pyo.value(m.fs.cooler.control_volume.properties_out[0].temperature)))
    tags['cooler_Pout'] = ("%4.4f" % (
        pyo.value(m.fs.cooler.outlet.pressure[0])*1e-6))
    tags['cooler_Hout'] = ("%4.2f" % (
        pyo.value(m.fs.cooler.outlet.enth_mol[0])))
    tags['cooler_xout'] = ("%4.4f" % (
        pyo.value(m.fs.cooler.control_volume.properties_out[0].vapor_frac)))


    # HX pump
    tags['hxpump_Fout'] = ("%4.4f" % (pyo.value(
        m.fs.hx_pump.outlet.flow_mol[0])*1e-3))
    tags['hxpump_Tout'] = ("%4.4f" % (pyo.value(
        m.fs.hx_pump.control_volume.properties_out[0].temperature)))
    tags['hxpump_Pout'] = ("%4.4f" % (pyo.value(
        m.fs.hx_pump.outlet.pressure[0])*1e-6))
    tags['hxpump_Hout'] = ("%4.2f" % (pyo.value(
        m.fs.hx_pump.outlet.enth_mol[0])))
    tags['hxpump_xout'] = ("%4.4f" % (pyo.value(
        m.fs.hx_pump.control_volume.properties_out[0].vapor_frac)))


    original_svg_file = os.path.join(
        this_file_dir(), "pfd_ultra_supercritical_pc_nlp.svg")
    with open(original_svg_file, "r") as f:
        svg_tag(tags, f, outfile=outfile)

#-------- added by esrawli
def add_bounds(m):
    """
    Add bounds to the units in the flowsheet
    """
    
    m.flow_max = m.main_flow * 1.2 # number from Naresh
    m.salt_flow_max = 1000 # in kg/s

    for unit_k in [m.fs.boiler, m.fs.reheater_1,
                   m.fs.reheater_2, m.fs.cond_pump,
                   m.fs.bfp, m.fs.bfpt]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s

    # Turbines
    for k in pyo.RangeSet(m.number_turbines):
        m.fs.turbine[k].inlet.flow_mol[:].setlb(0)
        m.fs.turbine[k].inlet.flow_mol[:].setub(m.flow_max)
        m.fs.turbine[k].outlet.flow_mol[:].setlb(0)
        m.fs.turbine[k].outlet.flow_mol[:].setub(m.flow_max)

    # FWHs
    for k in pyo.RangeSet(m.number_fwhs):
        m.fs.fwh[k].inlet_1.flow_mol[:].setlb(0)
        m.fs.fwh[k].inlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.fwh[k].inlet_2.flow_mol[:].setlb(0)
        m.fs.fwh[k].inlet_2.flow_mol[:].setub(m.flow_max)
        m.fs.fwh[k].outlet_1.flow_mol[:].setlb(0)
        m.fs.fwh[k].outlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.fwh[k].outlet_2.flow_mol[:].setlb(0)
        m.fs.fwh[k].outlet_2.flow_mol[:].setub(m.flow_max)

    # Mixers
    for k in m.mixer_list:
        m.fs.fwh_mixer[k].steam.flow_mol[:].setlb(0)
        m.fs.fwh_mixer[k].steam.flow_mol[:].setub(m.flow_max)
        m.fs.fwh_mixer[k].drain.flow_mol[:].setlb(0)
        m.fs.fwh_mixer[k].drain.flow_mol[:].setub(m.flow_max)
    m.fs.recycle_mixer.from_bfw_out.flow_mol.setlb(0) # mol/s
    m.fs.recycle_mixer.from_bfw_out.flow_mol.setub(m.flow_max) # mol/s
    m.fs.recycle_mixer.from_hx_pump.flow_mol.setlb(0) # mol/s
    m.fs.recycle_mixer.from_hx_pump.flow_mol.setub(0.2* m.flow_max) # mol/s
    m.fs.recycle_mixer.outlet.flow_mol.setlb(0) # mol/s
    m.fs.recycle_mixer.outlet.flow_mol.setub(m.flow_max) # mol/s        
    m.fs.condenser_mix.main.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.main.flow_mol[:].setub(m.flow_max) # esrawli - it causes "Restoration Failed!" error when uncommented
    m.fs.condenser_mix.bfpt.flow_mol[:].setlb(0)
    m.fs.condenser_mix.bfpt.flow_mol[:].setub(m.flow_max)
    m.fs.condenser_mix.drain.flow_mol[:].setlb(0)
    m.fs.condenser_mix.drain.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser_mix.makeup.flow_mol[:].setlb(0)
    m.fs.condenser_mix.makeup.flow_mol[:].setub(m.flow_max)
    m.fs.deaerator.steam.flow_mol[:].setlb(0)
    m.fs.deaerator.steam.flow_mol[:].setub(m.flow_max)
    m.fs.deaerator.drain.flow_mol[:].setlb(0)
    m.fs.deaerator.drain.flow_mol[:].setub(m.flow_max)
    m.fs.deaerator.feedwater.flow_mol[:].setlb(0)
    m.fs.deaerator.feedwater.flow_mol[:].setub(m.flow_max)

    # Turbine splitters
    for k in pyo.RangeSet(m.number_turbine_splitters):
        m.fs.turbine_splitter[k].split_fraction[0.0, "outlet_1"].setlb(0)
        m.fs.turbine_splitter[k].split_fraction[0.0, "outlet_1"].setub(1)
        m.fs.turbine_splitter[k].split_fraction[0.0, "outlet_2"].setlb(0)
        m.fs.turbine_splitter[k].split_fraction[0.0, "outlet_2"].setub(1)

    # Splitter
    for split in [m.fs.ess_hp_split]:
        split.to_hxc.flow_mol[:].setlb(0)
        split.to_hxc.flow_mol[:].setub(0.2 * m.flow_max)
        split.to_hp.flow_mol[:].setlb(0)
        split.to_hp.flow_mol[:].setub(m.flow_max)
        split.split_fraction[0.0, "to_hxc"].setlb(0)
        split.split_fraction[0.0, "to_hxc"].setub(1)
        split.split_fraction[0.0, "to_hp"].setlb(0)
        split.split_fraction[0.0, "to_hp"].setub(1)

    # Heat exchanger
    m.fs.hxc.inlet_1.flow_mol.setlb(0)
    m.fs.hxc.inlet_1.flow_mol.setub(0.2 * m.flow_max)
    m.fs.hxc.outlet_1.flow_mol.setlb(0)
    m.fs.hxc.outlet_1.flow_mol.setub(0.2 * m.flow_max)
    m.fs.hxc.inlet_2.flow_mass.setlb(0)  # kg/s
    m.fs.hxc.inlet_2.flow_mass.setub(m.salt_flow_max)  # kg/s
    m.fs.hxc.outlet_2.flow_mass.setlb(0)  # kg/s
    m.fs.hxc.outlet_2.flow_mass.setub(m.salt_flow_max)  # kg/s
    m.fs.hxc.inlet_2.pressure.setlb(101320)  # Pa
    m.fs.hxc.inlet_2.pressure.setub(101330)  # Pa
    m.fs.hxc.outlet_2.pressure.setlb(101320)  # Pa
    m.fs.hxc.outlet_2.pressure.setub(101330)  # Pa
    m.fs.hxc.heat_duty.setlb(0)  # W
    m.fs.hxc.heat_duty.setub(200e6)  # W
    m.fs.hxc.shell.heat.setlb(-200e6)  # W
    m.fs.hxc.shell.heat.setub(0)  # W
    m.fs.hxc.tube.heat.setlb(0)  # W
    m.fs.hxc.tube.heat.setub(200e6)  # W
    m.fs.hxc.tube.properties_in[0].enthalpy_mass.setlb(0)
    m.fs.hxc.tube.properties_in[0].\
        enthalpy_mass.setub(1.5e6)
    m.fs.hxc.tube.properties_out[0].enthalpy_mass.setlb(0)
    m.fs.hxc.tube.properties_out[0].\
        enthalpy_mass.setub(1.5e6)
    m.fs.hxc.overall_heat_transfer_coefficient.setlb(0)
    m.fs.hxc.overall_heat_transfer_coefficient.setub(10000)
    m.fs.hxc.area.setlb(0)
    m.fs.hxc.area.setub(5000)  # TODO: Check this value
    m.fs.hxc.delta_temperature_in.setlb(10)  # K
    # m.fs.hxc.delta_temperature_in.setub(80)  # K
    m.fs.hxc.delta_temperature_out.setlb(10)  # K
    # m.fs.hxc.delta_temperature_out.setub(80)  # K

    # Cooler and hx pump 
    for k in [m.fs.cooler, m.fs.hx_pump]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(0.2 * m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        # unit_k.outlet.flow_mol[:].setub(0.2 * m.flow_max)  # esrawli - "Restoration Failed!" error when uncommented
    m.fs.cooler.heat_duty.setub(0)

    # Cost-related terms
    m.fs.salt_purchase_cost.setlb(0)  # no units
    m.fs.salt_purchase_cost.setub(1e7)  # no units
    m.fs.capital_cost.setlb(0)  # no units
    # m.fs.capital_cost.setub(1e7)  # esrawli - "Restoration Failed!" error when uncommented
    m.fs.spump_purchase_cost.setlb(0)  # no units
    m.fs.spump_purchase_cost.setub(1e7)  # no units

    m.fs.hx_pump.costing.purchase_cost.setlb(0)  # no units
    m.fs.hx_pump.costing.purchase_cost.setub(1e7)  # no units

    return m
#--------


def build_plant_model(initialize_from_file=None, store_initialization=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = declare_unit_model()

    # Give all the required inputs to the model
    # Ensure that the degrees of freedom = 0 (model is complete)
    set_model_input(m)
    # Assert that the model has no degree of freedom at this point
    assert degrees_of_freedom(m) == 0

    set_scaling_factors(m)
    # Initialize the model (sequencial initialization and custom routines)
    # Ensure after the model is initialized, the degrees of freedom = 0
    solver = initialize(m)
    assert degrees_of_freedom(m) == 0

    #-------- added by esrawli
    build_costing(m)

    # Unfix the hp steam split fraction to charge heat exchanger
    # m.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()

    # Unfix salt flow to charge heat exchanger, temperature, and area
    # of charge heat exchanger
    m.fs.hxc.inlet_2.flow_mass.unfix()   # kg/s
    m.fs.hxc.inlet_2.temperature.unfix()  # K, 1 DOF
    m.fs.hxc.outlet_2.temperature.unfix()  # K
    m.fs.hxc.area.unfix() # 1 DOF
    # m.fs.hxc.heat_duty.fix(150*1e6)  # in W (obtained from supercritical plant)

    # The cooler outlet enthaply was fixed during model build to ensure liquid
    # at the inlet of hx_pump and keep the model square. This is now unfixed
    # and temperature of the outlet is constrained through
    #  'm.fs.cooler.constraint_cooler_enth2' in the flowsheet
    m.fs.cooler.outlet.enth_mol[0].unfix() # 1 DOF

    add_bounds(m)
    #--------

    # The power plant with storage for a charge scenario is now ready

    #-------- modified by esrawli
    # I relocated this constraint to the function build_costing
    #  Declaring a plant power out variable for easy analysis of various
    #  design and operating scenarios
    # m.fs.plant_power_out = pyo.Var(
    #     m.fs.time,
    #     domain=pyo.Reals,
    #     #-------- added by esrawli
    #     bounds=(300, 600), # values recommended by Naresh
    #     #-------- 
    #     initialize=400,
    #     doc="Net Power MWe out from the power plant",
    #     units=pyunits.MW
    # )

    # #   Constraint on Plant Power Output
    # #   Plant Power Out = Total Turbine Power
    # @m.fs.Constraint(m.fs.time)
    # def production_cons(b, t):
    #     return (
    #         (-1*sum(m.fs.turbine[p].work_mechanical[t]
    #              for p in pyo.RangeSet(m.number_turbines))
    #          )
    #         == m.fs.plant_power_out[t]*1e6*(pyunits.W/pyunits.MW)
    #     )

    # added by esrawli
    # Adding an objective function
    # m.obj = pyo.Objective(expr=1)
    m.obj = pyo.Objective(expr=m.fs.capital_cost + m.fs.operating_cost)
    #--------

    return m, solver


def model_analysis(m, solver):

#   Solving the flowsheet and check result
#   At this time one can make chnages to the model for further analysis
    flow_frac_list = [1.0]
    pres_frac_list = [1.0]
    for i in flow_frac_list:
        for j in pres_frac_list:
            #-------- modified by esrawli
            # m.fs.plant_power_out[0].fix(400)  # MW
            m.fs.boiler.inlet.flow_mol.fix(i*m.main_flow)  # mol/s
            #--------
            m.fs.boiler.outlet.pressure.fix(j*m.main_steam_pressure)
            #-------- added by esrawli
            dof = degrees_of_freedom(m)
            print('DOF before solution = ', dof)
            #--------
            solver.solve(m,
                         tee=True,
                         symbolic_solver_labels=True,
                         options={
                             "max_iter": 300,
                             "halt_on_ampl_error": "yes"}
            )
            print("***************** Printing Results ******************")
            print('')
            print('Plant Power (MW) =',
                  pyo.value(m.fs.plant_power_out[0]))
            print('')
            print("Objective (cap + op costs in $/y) =",
                  pyo.value(m.obj))            
            print("Total capital cost ($/y) =",
                  pyo.value(m.fs.capital_cost))
            print("Operating costs ($/y) =",
                  pyo.value(m.fs.operating_cost))
            print('')
            print("Heat exchanger area (m2) =",
                  pyo.value(m.fs.hxc.area))
            print("Salt flow (kg/s) =",
                  pyo.value(m.fs.hxc.inlet_2.flow_mass[0]))
            print("Salt temperature in (K) =",
                  pyo.value(m.fs.hxc.inlet_2.temperature[0]))
            print("Salt temperature out (K) =",
                  pyo.value(m.fs.hxc.outlet_2.temperature[0]))
            print('')
            print("Steam flow to storage (mol/s) =",
                  pyo.value(m.fs.hxc.inlet_1.flow_mol[0]))
            print("Water temperature in (K) =",
                  pyo.value(m.fs.hxc.side_1.properties_in[0].temperature))
            print("Steam temperature out (K) =",
                  pyo.value(m.fs.hxc.side_1.properties_out[0].temperature))
            print('')
            print("Boiler feed water flow (mol/s):",
                  pyo.value(m.fs.boiler.inlet.flow_mol[0]))
            print("Boiler duty (MW_th):",
                  pyo.value((m.fs.boiler.heat_duty[0]
                             + m.fs.reheater_1.heat_duty[0]
                             + m.fs.reheater_2.heat_duty[0])
                            * 1e-6))
            print('')
            print("Salt cost ($/y) =",
                  pyo.value(m.fs.salt_purchase_cost))
            print("Tank cost ($/y) =",
                  pyo.value(m.fs.salt_tank.costing.total_tank_cost / 15))
            print("Salt pump cost ($/y) =",
                  pyo.value(m.fs.spump_purchase_cost))
            print("HX pump cost ($/y) =",
                  pyo.value(m.fs.hx_pump.costing.purchase_cost / 15))
            print("Heat exchanger cost ($/y) =",
                  pyo.value(m.fs.hxc.costing.purchase_cost / 15))
            print("Cooling duty (MW_th) =",
                  pyo.value(m.fs.cooler.heat_duty[0] * -1e-6))

            # for unit_k in [m.fs.boiler, m.fs.ess_hp_split,
            #                m.fs.reheater_1, m.fs.reheater_2,
            #                m.fs.hxc, m.fs.cooler, m.fs.hx_pump,
            #                m.fs.recycle_mixer]:
            #     unit_k.report()
            # for k in pyo.RangeSet(m.number_turbines):
            #     m.fs.turbine[k].report()
            # for j in pyo.RangeSet(m.number_fwhs):
            #     m.fs.fwh[j].report()
            #     m.fs.condenser_mix.makeup.display()
            #     m.fs.condenser_mix.outlet.display()

    return m

if __name__ == "__main__":
    m, solver = build_plant_model(initialize_from_file=None,
                                  store_initialization=None)
    # User can import the model from build_plant_model for analysis
    # A sample analysis function is called below
    m_result = model_analysis(m, solver)
    # View results in a process flow diagram
    view_result("pfd_usc_powerplant_nlp_results.svg", m_result)
    log_infeasible_constraints(m)
