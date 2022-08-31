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
                           TransformationFactory, Expression, value)
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
#--------


def create_charge_model(m):
    """Create flowsheet and add unit models.
    """

    # Create a block to add charge storage model
    m.fs.charge = Block()
    # Add molten salt properties (Solar salt)
    m.fs.salt_properties = solarsalt_properties_new.SolarsaltParameterBlock()

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

    ###########################################################################
    #  Add charge heat exchanger for storage                                  #
    ###########################################################################
    m.fs.charge.hxc = HeatExchanger(
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

    m.fs.charge.hxc_data = {
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
    m.fs.charge.tube_thickness = Param(
        initialize=m.fs.charge.hxc_data['tube_thickness'],
        doc='Tube thickness [m]'
    )
    m.fs.charge.hxc.tube_inner_dia = Param(
        initialize=m.fs.charge.hxc_data['tube_inner_dia'],
        doc='Tube inner diameter [m]'
    )
    m.fs.charge.hxc.tube_outer_dia = Param(
        initialize=m.fs.charge.hxc_data['tube_outer_dia'],
        doc='Tube outer diameter [m]'
    )
    m.fs.charge.hxc.k_steel = Param(
        initialize=m.fs.charge.hxc_data['k_steel'],
        doc='Thermal conductivity of steel [W/mK]'
    )
    m.fs.charge.hxc.n_tubes = Param(
        initialize=m.fs.charge.hxc_data['number_tubes'],
        doc='Number of tubes'
    )
    m.fs.charge.hxc.shell_inner_dia = Param(
        initialize=m.fs.charge.hxc_data['shell_inner_dia'],
        doc='Shell inner diameter [m]'
    )

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

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m)
    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)
    return m

def _make_constraints(m):
    # Define all constraints for the charge model

    #########   Charge heat exchanger section   #########
    # Salt side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.charge.hxc.tube_cs_area = Expression(
        expr=(pi / 4) * (m.fs.charge.hxc.tube_inner_dia ** 2),
        doc="Tube cross sectional area"
    )
    m.fs.charge.hxc.tube_out_area = Expression(
        expr=(pi / 4) * (m.fs.charge.hxc.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]"
    )
    m.fs.charge.hxc.shell_eff_area = Expression(
        expr=(
            (pi / 4) * (m.fs.charge.hxc.shell_inner_dia ** 2)
            - m.fs.charge.hxc.n_tubes
            * m.fs.charge.hxc.tube_out_area),
        doc="Effective shell cross sectional area [m2]"
    )

    # Salt (shell) side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.charge.hxc.salt_reynolds_number = Expression(
        expr=(
            (m.fs.charge.hxc.inlet_2.flow_mass[0] * m.fs.charge.hxc.tube_outer_dia) / \
            (m.fs.charge.hxc.shell_eff_area * \
             m.fs.charge.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.charge.hxc.salt_prandtl_number = Expression(
        expr=(
            m.fs.charge.hxc.side_2.properties_in[0].cp_specific_heat["Liq"] * \
            m.fs.charge.hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] / \
            m.fs.charge.hxc.side_2.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    m.fs.charge.hxc.salt_prandtl_wall = Expression(
        expr=(
            (m.fs.charge.hxc.side_2.properties_out[0].cp_specific_heat["Liq"] * \
             m.fs.charge.hxc.side_2.properties_out[0].dynamic_viscosity["Liq"]) / \
            m.fs.charge.hxc.side_2.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number at wall"
    )
    m.fs.charge.hxc.salt_nusselt_number = Expression(
        expr=(
            0.35 * (m.fs.charge.hxc.salt_reynolds_number**0.6) * \
            (m.fs.charge.hxc.salt_prandtl_number**0.4) * \
            ((m.fs.charge.hxc.salt_prandtl_number / \
              m.fs.charge.hxc.salt_prandtl_wall)**0.25) * (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126"
    )

    # Steam side: Reynolds number, Prandtl number, and Nusselt number
    m.fs.charge.hxc.steam_reynolds_number = Expression(
        expr=(
           m.fs.charge.hxc.inlet_1.flow_mol[0] * \
            m.fs.charge.hxc.side_1.properties_in[0].mw * \
            m.fs.charge.hxc.tube_inner_dia / \
            (m.fs.charge.hxc.tube_cs_area * m.fs.charge.hxc.n_tubes * \
             m.fs.charge.hxc.side_1.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.charge.hxc.steam_prandtl_number = Expression(
        expr=(
            (m.fs.charge.hxc.side_1.properties_in[0].cp_mol / \
             m.fs.charge.hxc.side_1.properties_in[0].mw) * \
            m.fs.charge.hxc.side_1.properties_in[0].visc_d_phase["Vap"] / \
            m.fs.charge.hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.charge.hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 * (m.fs.charge.hxc.steam_reynolds_number**0.8) * \
            (m.fs.charge.hxc.steam_prandtl_number**(0.33)) * \
            ((m.fs.charge.hxc.side_1.properties_in[0].visc_d_phase["Vap"] / \
              m.fs.charge.hxc.side_1.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Salt and steam side heat transfer coefficients
    m.fs.charge.hxc.h_salt = Expression(
        expr=(
            m.fs.charge.hxc.side_2.properties_in[0].thermal_conductivity["Liq"] * \
            m.fs.charge.hxc.salt_nusselt_number / m.fs.charge.hxc.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    m.fs.charge.hxc.h_steam = Expression(
        expr=(
            m.fs.charge.hxc.side_1.properties_in[0].therm_cond_phase["Vap"] * \
            m.fs.charge.hxc.steam_nusselt_number / m.fs.charge.hxc.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    # Computing overall heat transfer coefficient
    # OHTC constraint is rewritten to avoid having a denominator in the equation
    # OHTC = _________________________________1___________________________________
    #        __1__  + _Tout_dia_*_log(Tout_dia/Tin_dia)_  + __(Tout_dia/Tin_dia)__
    #        Hsalt              2 k_steel                           Hsteam
    #
    m.fs.charge.hxc.tube_dia_relation = (
        m.fs.charge.hxc.tube_outer_dia / m.fs.charge.hxc.tube_inner_dia
    )
    m.fs.charge.hxc.log_tube_dia_relation = log(m.fs.charge.hxc.tube_dia_relation)    
    @m.fs.charge.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        return (
            m.fs.charge.hxc.overall_heat_transfer_coefficient[t]
            == 1 / ((1 / m.fs.charge.hxc.h_salt)
                    + ((m.fs.charge.hxc.tube_outer_dia * \
                        m.fs.charge.hxc.log_tube_dia_relation) / \
                        (2 * m.fs.charge.hxc.k_steel))
                    + (m.fs.charge.hxc.tube_dia_relation / m.fs.charge.hxc.h_steam))
        )
        #-------- modified by esrawli: equation rewritten to avoid having denominators
        # return (
        #     m.fs.charge.hxc.overall_heat_transfer_coefficient[t] * \
        #     (2 * m.fs.charge.hxc.k_steel * m.fs.charge.hxc.h_steam
        #       + m.fs.charge.hxc.tube_outer_dia * m.fs.charge.hxc.log_tube_dia_relation *\
        #       m.fs.charge.hxc.h_salt * m.fs.charge.hxc.h_steam
        #       + m.fs.charge.hxc.tube_dia_relation * m.fs.charge.hxc.h_salt * \
        #       2 * m.fs.charge.hxc.k_steel)        
        # ) == 2 * m.fs.charge.hxc.k_steel * m.fs.charge.hxc.h_salt * m.fs.charge.hxc.h_steam

    # Cooler
    # The temperature at the outlet of the cooler is required to be subcooled
    # by at least 5 degrees
    @m.fs.charge.cooler.Constraint(m.fs.time)
    def constraint_cooler_enth2(b, t):
        return (
            m.fs.charge.cooler.control_volume.properties_out[t].temperature ==
            (m.fs.charge.cooler.control_volume.properties_out[t].temperature_sat - 5)
        )

    # hx pump
    # The outlet pressure of hx_pump is then fixed to be the same as
    # boiler feed pump's outlet pressure
    # @m.fs.Constraint(m.fs.time)
    # def constraint_hxpump_presout(b, t):
    #     return m.fs.charge.hx_pump.outlet.pressure[t] >= \
    #         (m.main_steam_pressure * 1.1231)

    # Recycle mixer
    # The outlet pressure of the recycle mixer is same as
    # the outlet pressure of the boiler feed water, i.e. inlet 'bfw_out'
    @m.fs.charge.recycle_mixer.Constraint(m.fs.time)
    def recyclemixer_pressure_constraint(b, t):
        return b.from_bfw_out_state[t].pressure == b.mixed_state[t].pressure


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
    m.fs.charge.esshp_to_hxc = Arc(
        source=m.fs.charge.ess_hp_split.to_hxc,
        destination=m.fs.charge.hxc.inlet_1
    )
    m.fs.charge.hxc_to_cooler = Arc(
        source=m.fs.charge.hxc.outlet_1,
        destination=m.fs.charge.cooler.inlet
    )
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
    m.fs.charge.hxc.area.fix(100)  # m2

    # Salt conditions
    # Salt inlet flow is fixed during initialization but is unfixed and determined
    # during optimization
    m.fs.charge.hxc.inlet_2.flow_mass.fix(100)   # kg/s
    m.fs.charge.hxc.inlet_2.temperature.fix(513.15)  # K
    m.fs.charge.hxc.inlet_2.pressure.fix(101325)  # Pa

    # Cooler outlet enthalpy is fixed during model build to ensure the inlet
    # to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler is
    # constrained in the model
    # m.fs.charge.cooler.outlet.enth_mol[0].fix(10000)
    m.fs.charge.cooler.deltaP[0].fix(0)

    # Hx pump efficiecncy assumption
    m.fs.charge.hx_pump.efficiency_pump.fix(0.80)
    m.fs.charge.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure * 1.1231)

    ###########################################################################
    #  ESS HP splitter                                                        #
    ###########################################################################
    # The model is built for a fixed flow of steam through the charger.
    # This flow of steam to the charger is unfixed and determine during
    # design optimization
    m.fs.charge.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.05)

    ###########################################################################
    #  Recycle mixer                                                          #
    ###########################################################################
    # m.fs.recycle_mix.ess.pressure[0].fix(24235081.4 * 1.15)
    # m.fs.charge.recycle_mixer.from_hx_pump.flow_mol.fix(0.01)
    # m.fs.charge.recycle_mixer.from_hx_pump.enth_mol.fix(103421.4)
    # m.fs.charge.recycle_mixer.from_hx_pump.pressure.fix(1131.69204)


def set_scaling_factors(m):
    # scaling factors in the flowsheet

    iscale.set_scaling_factor(m.fs.charge.hxc.area, 1e-2)
    iscale.set_scaling_factor(
        m.fs.charge.hxc.overall_heat_transfer_coefficient, 1e-3)

    # scaling factors for storage
    iscale.set_scaling_factor(m.fs.charge.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.hxc.shell.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.hxc.tube.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.cooler.control_volume.heat, 1e-6)
    # return m

def initialize(m, fileinput=None, outlvl=idaeslog.NOTSET,
               solver=None, optarg={}):

    iscale.calculate_scaling_factors(m)

    #  High pressure splitter initialization
    _set_port(m.fs.charge.ess_hp_split.inlet,  m.fs.boiler.outlet)
    m.fs.charge.ess_hp_split.initialize(outlvl=outlvl, optarg=solver.options)

    # Charge heat exchanger
    _set_port(m.fs.charge.hxc.inlet_1, m.fs.charge.ess_hp_split.to_hxc)
    m.fs.charge.hxc.initialize(outlvl=outlvl, optarg=solver.options)

    #  Cooler
    _set_port(m.fs.charge.cooler.inlet,  m.fs.charge.hxc.outlet_1)
    m.fs.charge.cooler.initialize(outlvl=outlvl, optarg=solver.options)
    
    # HX pump
    _set_port(m.fs.charge.hx_pump.inlet,  m.fs.charge.cooler.outlet)
    m.fs.charge.hx_pump.initialize(outlvl=outlvl, optarg=solver.options)

    #  Recycle mixer initialization
    _set_port(m.fs.charge.recycle_mixer.from_bfw_out, m.fs.bfp.outlet)
    _set_port(m.fs.charge.recycle_mixer.from_hx_pump, m.fs.charge.hx_pump.outlet)

    # fixing to a small value in mol/s
    # m.fs.charge.recycle_mixer.from_hx_pump.flow_mol.fix(0.01)
    # m.fs.charge.recycle_mixer.from_hx_pump.enth_mol.fix()
    # m.fs.charge.recycle_mixer.from_hx_pump.pressure.fix()
    m.fs.charge.recycle_mixer.initialize(outlvl=outlvl)#, optarg=solver.options)

    res = solver.solve(m)
    print("Charge Model Initialization = ",
          res.solver.termination_condition)
    print("*********************Model Initialized**************************")


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
    tags['hxsteam_Fout'] = ("%4.4f" % (value(
        m.fs.charge.hxc.outlet_1.flow_mol[0])*1e-3))
    tags['hxsteam_Tout'] = ("%4.4f" % (value(
        m.fs.charge.hxc.side_1.properties_out[0].temperature)))
    tags['hxsteam_Pout'] = ("%4.4f" % (
        value(m.fs.charge.hxc.outlet_1.pressure[0])*1e-6))
    tags['hxsteam_Hout'] = ("%4.2f" % (
        value(m.fs.charge.hxc.outlet_1.enth_mol[0])))
    tags['hxsteam_xout'] = ("%4.4f" % (
        value(m.fs.charge.hxc.side_1.properties_out[0].vapor_frac)))

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
        this_file_dir(), "pfd_ultra_supercritical_pc_nlp.svg")
    with open(original_svg_file, "r") as f:
        svg_tag(tags, f, outfile=outfile)

def add_bounds(m):
    
    m.flow_max = m.main_flow * 1.2 # number from Naresh
    m.salt_flow_max = 1000 # in kg/s

    # Charge heat exchanger section
    m.fs.charge.hxc.inlet_1.flow_mol.setlb(0)  # mol/s
    m.fs.charge.hxc.inlet_1.flow_mol.setub(0.2 * m.flow_max)  # mol/s
    m.fs.charge.hxc.inlet_2.flow_mass.setlb(0)  # kg/s
    m.fs.charge.hxc.inlet_2.flow_mass.setub(m.salt_flow_max)  # kg/s
    m.fs.charge.hxc.delta_temperature_out.setlb(10)  # K
    m.fs.charge.hxc.delta_temperature_in.setlb(10)  # K
    # m.fs.charge.hxc.delta_temperature_out.setub(80)  # K
    # m.fs.charge.hxc.delta_temperature_in.setub(80)  # K

    m.fs.charge.cooler.heat_duty.setub(0)

    for split in [m.fs.charge.ess_hp_split]:
        split.to_hxc.flow_mol[:].setlb(0)
        split.to_hxc.flow_mol[:].setub(0.2 * m.flow_max)
        split.to_hp.flow_mol[:].setlb(0)
        split.to_hp.flow_mol[:].setub(m.flow_max)
        split.split_fraction[0.0, "to_hxc"].setlb(0)
        split.split_fraction[0.0, "to_hxc"].setub(1)
        split.split_fraction[0.0, "to_hp"].setlb(0)
        split.split_fraction[0.0, "to_hp"].setub(1)

    for mix in [m.fs.charge.recycle_mixer]:
        mix.from_bfw_out.flow_mol.setlb(0) # mol/s
        mix.from_bfw_out.flow_mol.setub(m.flow_max) # mol/s
        mix.from_hx_pump.flow_mol.setlb(0) # mol/s
        mix.from_hx_pump.flow_mol.setub(0.2* m.flow_max) # mol/s
        mix.outlet.flow_mol.setlb(0) # mol/s
        mix.outlet.flow_mol.setub(m.flow_max) # mol/s        

    return m


def main(m_usc):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_charge_model(m_usc)

    # Give all the required inputs to the model
    # Ensure that the degrees of freedom = 0 (model is complete)
    set_model_input(m)
    # Assert that the model has no degree of freedom at this point
    assert degrees_of_freedom(m) == 0

    set_scaling_factors(m)
    # Initialize the model (sequencial initialization and custom routines)
    # Ensure after the model is initialized, the degrees of freedom = 0
    initialize(m, solver=solver)
    assert degrees_of_freedom(m) == 0

    add_bounds(m)

    return m, solver


def model_analysis(m, solver):

    ###########################################################################
    # Unfixing variables for analysis
    # (This section is deactived for the simulation of square model)
    ###########################################################################
    # Unfix the hp steam split fraction to charge heat exchanger
    # m.fs.charge.ess_hp_split.split_fraction[0, "to_hxc"].unfix()

    # Unfix salt flow to charge heat exchanger, temperature, and area
    # of charge heat exchanger
    # m.fs.charge.hxc.inlet_2.flow_mass.unfix()   # kg/s
    # m.fs.charge.hxc.inlet_2.temperature.unfix()  # K, 1 DOF
    # m.fs.charge.hxc.outlet_2.temperature.unfix()  # K
    # m.fs.charge.hxc.area.unfix() # 1 DOF
    # m.fs.charge.hxc.heat_duty.fix(150*1e6)  # in W

    # adding a dummy objective for the simulation model
    m.obj = Objective(expr=1)

    m.fs.boiler.inlet.flow_mol.fix(m.main_flow)  # mol/s
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)

    print('DOF before solution = ', degrees_of_freedom(m))
    solver.solve(m,
                 tee=True,
                 symbolic_solver_labels=True,
                 options={
                     "max_iter": 300,
                     "halt_on_ampl_error": "yes"}
    )
    print("***************** Printing Results ******************")
    print('')
    print("Obj =",
          value(m.obj))            
    print("Heat exchanger area (m2) =",
          value(m.fs.charge.hxc.area))
    print('')
    print("Salt flow (kg/s) =",
          value(m.fs.charge.hxc.inlet_2.flow_mass[0]))
    print("Salt temperature in (K) =",
          value(m.fs.charge.hxc.inlet_2.temperature[0]))
    print("Salt temperature out (K) =",
          value(m.fs.charge.hxc.outlet_2.temperature[0]))
    print('')
    print("Steam flow to storage (mol/s) =",
          value(m.fs.charge.hxc.inlet_1.flow_mol[0]))
    print("Water temperature in (K) =",
          value(m.fs.charge.hxc.side_1.properties_in[0].temperature))
    print("Steam temperature out (K) =",
          value(m.fs.charge.hxc.side_1.properties_out[0].temperature))
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
    # for unit_k in [m.fs.boiler, m.fs.ess_hp_split]:
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

    m, solver = main(m_usc)

    # User can import the model from build_plant_model for analysis
    # A sample analysis function is called below
    m_result = model_analysis(m, solver)
    # View results in a process flow diagram
    # view_result("pfd_usc_powerplant_nlp_results.svg", m_result)
    log_infeasible_constraints(m)
