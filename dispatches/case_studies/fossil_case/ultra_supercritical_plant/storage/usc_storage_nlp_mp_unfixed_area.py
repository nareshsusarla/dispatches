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

"""

__author__ = "Naresh Susarla and Soraya Rawlings"

# Import Python libraries
from math import pi
import logging
import json

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var)
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES libraries
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import (HeatExchanger,
                                      MomentumMixingType,
                                      Heater)
# import idaes.core.util.unit_costing as icost
from idaes.core import UnitModelCostingBlock
from idaes.models.costing.SSLW import SSLWCosting, SSLWCostingData

from idaes.core.util import model_serializer as ms

# Import IDAES Libraries
from idaes.models.unit_models import PressureChanger
from idaes.models_extra.power_generation.unit_models.helm import (HelmMixer,
                                                                  HelmTurbineStage,
                                                                  HelmSplitter)
from idaes.models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import ultra supercritical power plant model
# from dispatches.models.fossil_case.ultra_supercritical_plant import (
#     ultra_supercritical_powerplant_mixcon as usc)
# from dispatches.models.fossil_case.ultra_supercritical_plant import (
#     ultra_supercritical_powerplant as usc)
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
# from dispatches.models.fossil_case.ultra_supercritical_plant.storage import (
#     solarsalt_properties)
# import solarsalt_properties
# from dispatches.models.fossil_case.properties import solarsalt_properties
from dispatches.properties import solarsalt_properties

from pyomo.network.plugins import expand_arcs

from IPython import embed
logging.basicConfig(level=logging.INFO)


scaling_obj = 1e-3

def create_charge_model(m,
                        method=None,
                        pmax=None,
                        max_salt_amount=None):
    """Create flowsheet and add unit models.
    """

    # Create a block to add charge storage model
    # m.fs.charge = Block()
    # m.fs.discharge = Block()

    # Add data
    add_data(m, max_salt_amount=max_salt_amount)

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()

    ###########################################################################
    #  Add hp and ip splitters                                                #
    ###########################################################################
    # Declared to divert some steam from high pressure inlet and
    # intermediate pressure inlet to charge the storage heat exchanger
    m.fs.ess_hp_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_hxc", "to_turbine"],
        }
    )
    m.fs.ess_bfp_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_hxd", "to_recyclemix"],
        }
    )

    ###########################################################################
    #  Add cooler and hx pump                                                 #
    ###########################################################################
    # To ensure the outlet of charge heat exchanger is a subcooled
    # liquid before mixing it with the plant, a cooler is added after
    # the heat exchanger
    m.fs.cooler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    # A pump, if needed, is used to increase the pressure of the water
    # to allow mixing it at a desired location within the plant
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
    m.fs.recycle_mixer = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["from_bfw_out", "from_hx_pump"],
            "property_package": m.fs.prop_water,
        }
    )

    # Add charge heat exchanger
    m.fs.hxc = HeatExchanger(
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

    # Add discharge heat exchanger
    m.fs.hxd = HeatExchanger(
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

    # Data to compute overall heat transfer coefficient for the charge
    # heat exchanger using the Sieder-Tate Correlation. Parameters for
    # tube diameter and thickness assumed from the data in (2017) He
    # et al., Energy Procedia 105, 980-985

    # -------- Charge Heat Exchanger Heat Transfer Coefficient --------
    m.fs.hxc.salt_reynolds_number = pyo.Expression(
        expr=(
            (m.fs.hxc.inlet_2.flow_mass[0] *
             m.fs.tube_outer_dia) /
            (m.fs.shell_eff_area *
             m.fs.hxc.side_2.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number")
    m.fs.hxc.salt_prandtl_number = pyo.Expression(
        expr=(
            m.fs.hxc.side_2.properties_in[0].cp_mass["Liq"] *
            m.fs.hxc.side_2.properties_in[0].visc_d_phase["Liq"] /
            m.fs.hxc.side_2.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number")
    m.fs.hxc.salt_prandtl_wall = pyo.Expression(
        expr=(
            m.fs.hxc.side_2.properties_out[0].cp_mass["Liq"] *
            m.fs.hxc.side_2.properties_out[0].visc_d_phase["Liq"] /
            m.fs.hxc.side_2.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number at wall")
    m.fs.hxc.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 *
            (m.fs.hxc.salt_reynolds_number**0.6) *
            (m.fs.hxc.salt_prandtl_number**0.4) *
            ((m.fs.hxc.salt_prandtl_number /
              m.fs.hxc.salt_prandtl_wall) ** 0.25) *
            (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")
    m.fs.hxc.steam_reynolds_number = pyo.Expression(
        expr=(
            m.fs.hxc.inlet_1.flow_mol[0] *
            m.fs.hxc.side_1.properties_in[0].mw *
            m.fs.tube_inner_dia /
            (m.fs.tube_cs_area *
             m.fs.n_tubes *
             m.fs.hxc.side_1.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")
    m.fs.hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (m.fs.hxc.side_1.properties_in[0].cp_mol /
             m.fs.hxc.side_1.properties_in[0].mw) *
            m.fs.hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
            m.fs.hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")
    m.fs.hxc.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 *
            (m.fs.hxc.steam_reynolds_number**0.8) *
            (m.fs.hxc.steam_prandtl_number**(0.33)) *
            ((m.fs.hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
              m.fs.hxc.side_1.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of charge heat exchanger
    m.fs.hxc.h_salt = pyo.Expression(
        expr=(
            m.fs.hxc.side_2.properties_in[0].therm_cond_phase["Liq"] *
            m.fs.hxc.salt_nusselt_number /
            m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    m.fs.hxc.h_steam = pyo.Expression(
        expr=(
            m.fs.hxc.side_1.properties_in[0].therm_cond_phase["Vap"] *
            m.fs.hxc.steam_nusselt_number /
            m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")

    # Rewrite overall heat transfer coefficient constraint to avoid
    # denominators

    @m.fs.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        # return (
        #     m.fs.charge.hxc.overall_heat_transfer_coefficient[t]
        #     == 1 / ((1 / m.fs.charge.hxc.h_salt)
        #             + ((m.fs.charge.hxc.tube_outer_dia *
        #                 m.fs.charge.hxc.log_tube_dia_ratio) /
        #                 (2 * m.fs.charge.hxc.k_steel))
        #             + (m.fs.charge.hxc.tube_dia_ratio /
        #                m.fs.charge.hxc.h_steam))
        # )
        return (
            m.fs.hxc.overall_heat_transfer_coefficient[t] *
            (2 * m.fs.k_steel *
             m.fs.hxc.h_steam +
             m.fs.tube_outer_dia *
             m.fs.log_tube_dia_ratio *
             m.fs.hxc.h_salt *
             m.fs.hxc.h_steam +
             m.fs.tube_dia_ratio *
             m.fs.hxc.h_salt *
             2 * m.fs.k_steel)
        ) == (2 * m.fs.k_steel *
              m.fs.hxc.h_salt *
              m.fs.hxc.h_steam)

    m.fs.hxc_to_cooler = Arc(
        source=m.fs.hxc.outlet_1,
        destination=m.fs.cooler.inlet,
        doc="Connection from cooler to solar charge heat exchanger"
    )

    # ------- Discharge Heat Exchanger Heat Transfer Coefficient -------
    # Discharge heat exchanger salt and steam side constraints to
    # calculate Reynolds number, Prandtl number, and Nusselt number
    m.fs.hxd.salt_reynolds_number = pyo.Expression(
        expr=(
            m.fs.hxd.inlet_1.flow_mass[0]
            * m.fs.tube_outer_dia
            / (m.fs.shell_eff_area
               * m.fs.hxd.side_1.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.hxd.salt_prandtl_number = pyo.Expression(
        expr=(
            m.fs.hxd.side_1.properties_in[0].cp_mass["Liq"]
            * m.fs.hxd.side_1.properties_in[0].visc_d_phase["Liq"]
            / m.fs.hxd.side_1.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    # Assuming that the wall conditions are same as those at the outlet
    m.fs.hxd.salt_prandtl_wall = pyo.Expression(
        expr=(
            m.fs.hxd.side_1.properties_out[0].cp_mass["Liq"]
            * m.fs.hxd.side_1.properties_out[0].visc_d_phase["Liq"]
            / m.fs.hxd.side_1.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Wall Salt Prandtl Number"
    )
    m.fs.hxd.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 * (m.fs.hxd.salt_reynolds_number**0.6)
            * (m.fs.hxd.salt_prandtl_number**0.4)
            * ((m.fs.hxd.salt_prandtl_number
                / m.fs.hxd.salt_prandtl_wall)**0.25)
            * (2**0.2)
        ),
        doc="Solar Salt Nusslet Number from 2019, App Ener (233-234), 126"
    )
    m.fs.hxd.steam_reynolds_number = pyo.Expression(
        expr=(
            m.fs.hxd.inlet_2.flow_mol[0]
            * m.fs.hxd.side_2.properties_in[0].mw
            * m.fs.tube_inner_dia
            / (m.fs.tube_cs_area
               * m.fs.n_tubes
               * m.fs.hxd.side_2.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.hxd.steam_prandtl_number = pyo.Expression(
        expr=(
            (m.fs.hxd.side_2.properties_in[0].cp_mol
             / m.fs.hxd.side_2.properties_in[0].mw)
            * m.fs.hxd.side_2.properties_in[0].visc_d_phase["Liq"]
            / m.fs.hxd.side_2.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.hxd.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 * (m.fs.hxd.steam_reynolds_number ** 0.8)
            * (m.fs.hxd.steam_prandtl_number ** (0.33))
            * ((m.fs.hxd.side_2.properties_in[0].visc_d_phase["Liq"]
                / m.fs.hxd.side_2.properties_out[0].visc_d_phase["Vap"]
                ) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Discharge heat exchanger salt and steam side heat transfer
    # coefficients
    m.fs.hxd.h_salt = pyo.Expression(
        expr=(
            m.fs.hxd.side_1.properties_in[0].therm_cond_phase["Liq"]
            * m.fs.hxd.salt_nusselt_number / m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    m.fs.hxd.h_steam = pyo.Expression(
        expr=(
            m.fs.hxd.side_2.properties_in[0].therm_cond_phase["Liq"]
            * m.fs.hxd.steam_nusselt_number / m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    @m.fs.hxd.Constraint(m.fs.time,
                         doc="Overall heat transfer coefficient for hxd")
    def constraint_hxd_ohtc(b, t):
        return (
            m.fs.hxd.overall_heat_transfer_coefficient[t] *
            (2 * m.fs.k_steel *
             m.fs.hxd.h_steam +
             m.fs.tube_outer_dia *
             m.fs.log_tube_dia_ratio *
             m.fs.hxd.h_salt *
             m.fs.hxd.h_steam +
             m.fs.tube_dia_ratio *
             m.fs.hxd.h_salt *
             2 * m.fs.k_steel)
        ) == (2 * m.fs.k_steel *
              m.fs.hxd.h_salt *
              m.fs.hxd.h_steam)

    m.fs.es_turbine = HelmTurbineStage(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m, method=method, pmax=pmax)
    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def add_data(m, max_salt_amount=None):

    # Chemical engineering cost index for 2019
    m.CE_index = 607.5

    #  Define the data for the design of the storage heat
    # exchangers. The design is: Shell-n-tube counter-flow heat
    # exchanger design parameters. Data to compute overall heat
    # transfer coefficient for the charge heat exchanger using the
    # Sieder-Tate Correlation. Parameters for tube diameter and
    # thickness assumed from the data in (2017) He et al., Energy
    # Procedia 105, 980-985.
    m.fs.data_storage_hx = {
        'tube_thickness': 0.004,
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    m.fs.tube_thickness = pyo.Param(
        initialize=m.fs.data_storage_hx['tube_thickness'],
        doc='Tube thickness [m]')
    m.fs.tube_inner_dia = pyo.Param(
        initialize=m.fs.data_storage_hx['tube_inner_dia'],
        doc='Tube inner diameter [m]')
    m.fs.tube_outer_dia = pyo.Param(
        initialize=m.fs.data_storage_hx['tube_outer_dia'],
        doc='Tube outer diameter [m]')
    m.fs.k_steel = pyo.Param(
        initialize=m.fs.data_storage_hx['k_steel'],
        doc='Thermal conductivity of steel [W/mK]')
    m.fs.n_tubes = pyo.Param(
        initialize=m.fs.data_storage_hx['number_tubes'],
        doc='Number of tubes')
    m.fs.shell_inner_dia = pyo.Param(
        initialize=m.fs.data_storage_hx['shell_inner_dia'],
        doc='Shell inner diameter [m]')
    m.fs.tube_cs_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.tube_inner_dia ** 2),
        doc="Tube cross sectional area")
    m.fs.tube_out_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]")
    m.fs.shell_eff_area = pyo.Expression(
        expr=(
            (pi / 4) *
            (m.fs.shell_inner_dia ** 2) -
            m.fs.n_tubes *
            m.fs.tube_out_area),
        doc="Effective shell cross sectional area [m2]")

    m.fs.tube_dia_ratio = (m.fs.tube_outer_dia / m.fs.tube_inner_dia)
    m.fs.log_tube_dia_ratio = log(m.fs.tube_dia_ratio)

    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'solar_salt_price': 0.49,
    }

    # Main flowsheet operation data
    m.fs.coal_price = pyo.Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.cooling_price = pyo.Param(
        initialize=m.data_cost['cooling_price'],
        doc='Cost of chilled water for cooler from Sieder et al. $/J')
    m.fs.solar_salt_price = pyo.Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')

    # Define the number of hours per day to operate the storage system
    # and the number of years over which the capital costs are
    # annualized
    m.fs.hours_per_day = pyo.Param(
        initialize=24,
        doc='Estimated number of hours of charging per day'
    )
    m.fs.num_of_years = pyo.Param(
        initialize=30,
        doc='Number of years for capital cost annualization')

    # Define the amount of storage material
    m.fs.salt_amount = pyo.Param(
        initialize=max_salt_amount,
        doc="Solar salt amount in mton"
    )

    # m.fs.hxc_salt_design_flow = pyo.Param(
    #     initialize=312,
    #     doc='Design flow of salt through hxc')
    # m.fs.hxc_salt_design_density = pyo.Param(
    #     initialize=1937.36,
    #     doc='Design density of salt through hxc')

    # m.fs.hxd_salt_design_flow = pyo.Param(
    #     initialize=362.2,
    #     doc='Design flow of salt through hxd')
    # m.fs.hxd_salt_design_density = pyo.Param(
    #     initialize=1721.12,
    #     doc='Design density of salt through hxd')


def _make_constraints(m, method=None, pmax=None):
    """Declare the constraints for the charge model
    """

    # Cooler
    @m.fs.cooler.Constraint(m.fs.time,
                            doc="Cooler outlet temperature to be subcooled")
    def constraint_cooler_enth2(b, t):
        return (
            b.control_volume.properties_out[t].temperature <=
            (b.control_volume.properties_out[t].temperature_sat - 5)
        )

    # HX pump
    @m.fs.Constraint(m.fs.time,
                     doc="HX pump out pressure equal to BFP out pressure")
    def constraint_hxpump_presout(b, t):
        return (m.fs.hx_pump.outlet.pressure[t] ==
                m.main_steam_pressure * 1.1231)

    # Recycle mixer
    @m.fs.recycle_mixer.Constraint(
        m.fs.time,
        doc="Recycle mixer outlet pressure equal to min inlet pressure")
    def recyclemixer_pressure_constraint(b, t):
        return b.from_bfw_out_state[t].pressure == b.mixed_state[t].pressure

    m.fs.production_cons.deactivate()
    @m.fs.Constraint(m.fs.time)
    def production_cons_with_storage(b, t):
        return (
            (-1 * sum(m.fs.turbine[p].work_mechanical[t]
                      for p in m.set_turbine)
             - m.fs.hx_pump.control_volume.work[0]
             # + ((-1) * b.es_turbine.work_mechanical[0]) # add es turbine work
            ) ==
            m.fs.plant_power_out[t] * 1e6 * (pyunits.W/pyunits.MW)
        )
    m.fs.net_power = pyo.Expression(
        expr=(m.fs.plant_power_out[0]
              + (-1e-6) * m.fs.es_turbine.work_mechanical[0])
    )

    m.fs.boiler_efficiency = pyo.Expression(
        expr=0.2143 * (m.fs.net_power / pmax)
        + 0.7357,
        doc="Boiler efficiency in fraction"
    )
    # m.fs.boiler_efficiency = pyo.Var(
    #     initialize=0.9,
    #     bounds=(0, 1)
    # )
    # m.fs.boiler_efficiency_eq = pyo.Constraint(
    #     expr=(0.2143 * (m.fs.net_power / pmax)
    #           + 0.7357)
    # )
    m.fs.coal_heat_duty = pyo.Var(
        initialize=1000,
        bounds=(0, 1e5),
        doc="Coal heat duty supplied to boiler (MW)")

    if method == "with_efficiency":
        def coal_heat_duty_rule(b):
            return m.fs.coal_heat_duty * m.fs.boiler_efficiency == (
                m.fs.plant_heat_duty[0])
        m.fs.coal_heat_duty_eq = Constraint(rule=coal_heat_duty_rule)
        # @m.fs.Expression()
        # def coal_heat_duty(b):
        #     return (m.fs.plant_heat_duty[0] / m.fs.boiler_efficiency)

    else:
        def coal_heat_duty_rule(b):
            return m.fs.coal_heat_duty == (
                m.fs.plant_heat_duty[0])
        m.fs.coal_heat_duty_eq = Constraint(rule=coal_heat_duty_rule)
        # @m.fs.Expression()
        # def coal_heat_duty(b):
        #     return m.fs.plant_heat_duty[0]

    m.fs.cycle_efficiency = pyo.Expression(
        expr=m.fs.net_power / m.fs.coal_heat_duty * 100,
        doc="Cycle efficiency in %"
    )
    # m.fs.cycle_efficiency = pyo.Var(
    #     initialize=0.4,
    #     bounds=(0, 1)
    # )
    # m.fs.cycle_efficiency_eq = pyo.Constraint(
    #     expr=m.fs.net_power / m.fs.coal_heat_duty,
    #     doc="Cycle efficiency in fraction"
    # )


def _create_arcs(m):
    """Create arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the charge heat exchanger
    for arc_s in [m.fs.bfp_to_fwh8,
                  m.fs.rh1_to_turb3]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()

    # m.fs.charge.t4split_to_rh2_recon = Arc(
    #     source=m.fs.turbine_splitter[4].outlet_1,
    #     destination=m.fs.reheater[2].inlet
    # )
    # m.fs.charge.condpump_to_fwh1_recon = Arc(
    #     source=m.fs.cond_pump.outlet,
    #     destination=m.fs.fwh[1].inlet_2
    # )

    m.fs.rh1_to_esshp = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.ess_hp_split.inlet,
        doc="Connection from reheater to ip splitter"
    )
    m.fs.esshp_to_turb3 = Arc(
        source=m.fs.ess_hp_split.to_turbine,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from HP splitter to turbine 3"
    )
    m.fs.esshp_to_hxc = Arc(
        source=m.fs.ess_hp_split.to_hxc,
        destination=m.fs.hxc.inlet_1,
        doc="Connection from HP splitter to HXC inlet 1"
    )
    m.fs.cooler_to_hxpump = Arc(
        source=m.fs.cooler.outlet,
        destination=m.fs.hx_pump.inlet,
        doc="Connection from cooler to HX pump"
    )
    m.fs.hxpump_to_recyclemix = Arc(
        source=m.fs.hx_pump.outlet,
        destination=m.fs.recycle_mixer.from_hx_pump,
        doc="Connection from HX pump to recycle mixer"
    )
    m.fs.bfp_to_essbfp = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.ess_bfp_split.inlet,
        doc="Connection from BFP outlet to discharge split"
    )
    m.fs.essbfp_to_hxd = Arc(
        source=m.fs.ess_bfp_split.to_hxd,
        destination=m.fs.hxd.inlet_2,
        doc="Connection from BFP outlet to discharge split"
    )
    m.fs.essbfp_to_recyclemix = Arc(
        source=m.fs.ess_bfp_split.to_recyclemix,
        destination=m.fs.recycle_mixer.from_bfw_out,
        doc="Connection from BFP outlet to recycle mixer"
    )
    m.fs.hxd_to_esturbine = Arc(
        source=m.fs.hxd.outlet_2,
        destination=m.fs.es_turbine.inlet,
        doc="Connection from BFP outlet to discharge split"
    )
    m.fs.recyclemix_to_fwh8 = Arc(
        source=m.fs.recycle_mixer.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from Recycle Mixer to FWH8 tube side"
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
    m.fs.hxc.area.fix(2500)  # m2
    m.fs.hxd.area.fix(2000)  # m2

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.hxc.inlet_2.flow_mass.fix(140)   # kg/s
    m.fs.hxc.inlet_2.temperature.fix(513.15)  # K
    m.fs.hxc.inlet_2.pressure.fix(101325)  # Pa

    m.fs.hxd.inlet_1.flow_mass.fix(250)  # 250
    m.fs.hxd.inlet_1.temperature.fix(853.15)
    m.fs.hxd.inlet_1.pressure.fix(101325)

    # Cooler outlet enthalpy is fixed during model build to ensure the
    # inlet to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler
    # is constrained in the model
    m.fs.cooler.outlet.enth_mol[0].fix(10000)
    m.fs.cooler.deltaP[0].fix(0)

    # HX pump efficiecncy assumption
    m.fs.hx_pump.efficiency_pump.fix(0.80)
    # m.fs.charge.hx_pump.outlet.pressure[0].fix(
    # m.main_steam_pressure * 1.1231)

    m.fs.es_turbine.ratioP.fix(0.0286)
    m.fs.es_turbine.efficiency_isentropic.fix(0.5)
    ###########################################################################
    #  ESS VHP and HP splitters                                               #
    ###########################################################################
    # The model is built for a fixed flow of steam through the
    # charger.  This flow of steam to the charger is unfixed and
    # determine during design optimization
    m.fs.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.1)
    m.fs.ess_bfp_split.split_fraction[0, "to_hxd"].fix(0.1)  # 0.1


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

    """

    # Include scaling factors for solar, hitec, and thermal oil charge
    # heat exchangers
    for fluid in [m.fs.hxc, m.fs.hxd]:
        iscale.set_scaling_factor(fluid.area, 1e-2)
        iscale.set_scaling_factor(
            fluid.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(fluid.shell.heat, 1e-6)
        iscale.set_scaling_factor(fluid.tube.heat, 1e-6)

    iscale.set_scaling_factor(m.fs.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.es_turbine.control_volume.work, 1e-6)

    for k in [m.fs.cooler]:
        iscale.set_scaling_factor(k.control_volume.heat, 1e-6)


def initialize(m,
               solver=None,
               outlvl=idaeslog.WARNING,
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

    # Initialize splitters
    propagate_state(m.fs.rh1_to_esshp)
    m.fs.ess_hp_split.initialize(outlvl=outlvl,
                                 optarg=solver.options)

    # Re-initialize turbines connected to splitters since the flow is
    # not the same as before
    propagate_state(m.fs.esshp_to_turb3)
    m.fs.turbine[3].initialize(outlvl=outlvl,
                               optarg=solver.options)

    # Initialize connector
    # Fixing the charge steam inlet during initialization as note that
    # arcs were removed and replaced with disjuncts with equality constraints
    # Note that these should be unfixed during optimization
    propagate_state(m.fs.esshp_to_hxc)
    m.fs.hxc.initialize(outlvl=outlvl,
                        optarg=solver.options)

    # Initialize cooler
    propagate_state(m.fs.hxc_to_cooler)
    m.fs.cooler.initialize(outlvl=outlvl,
                           optarg=solver.options)

    # Initialize HX pump
    propagate_state(m.fs.cooler_to_hxpump)
    m.fs.hx_pump.initialize(outlvl=outlvl,
                            optarg=solver.options)

    # Initialize BFP split
    propagate_state(m.fs.bfp_to_essbfp)
    m.fs.ess_bfp_split.initialize(outlvl=outlvl,
                                  optarg=solver.options)

    #  Recycle mixer initialization
    propagate_state(m.fs.essbfp_to_recyclemix)
    propagate_state(m.fs.hxpump_to_recyclemix)
    m.fs.recycle_mixer.initialize(outlvl=outlvl)

    propagate_state(m.fs.essbfp_to_hxd)
    # m.fs.hxd.report()
    m.fs.hxd.initialize(outlvl=outlvl,
                        optarg=solver.options)
    # m.fs.hxd.report()
    # raise Exception()
    propagate_state(m.fs.hxd_to_esturbine)
    m.fs.es_turbine.initialize(outlvl=outlvl,
                               optarg=solver.options)
    print('DOFs before init solution =', degrees_of_freedom(m))
    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)

    print("Charge Model Initialization = ",
          res.solver.termination_condition)
    print("***************   Charge Model Initialized   ********************")


def build_costing(m,
                  solver=None,
                  optarg={"tol": 1e-8, "max_iter": 300}):
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
    solver = get_solver('ipopt', optarg)

    ##############################################################
    # Add capital cost
    # 1. Calculate change and discharge heat exchangers costs
    # 2. Calculate total capital cost for charge and discharge heat
    # exchangers
    ##############################################################

    # Add IDAES costing method
    m.fs.costing = SSLWCosting()

    ###### 1. Calculate charge and discharge heat exchangers costs
    # Calculate and initialize the Solar salt charge and discharge
    # heat exchangers coss, which are estimated using the IDAES
    # costing method with default options, i.e., a U-tube heat
    # exchanger, stainless steel material, and a tube length of
    # 12ft. Refer to costing documentation to change any of the
    # default options. The purchase cost of heat exchanger has to be
    # annualized when used
    for storage_hx in [m.fs.hxc,
                       m.fs.hxd]:
        storage_hx.costing = UnitModelCostingBlock(
            default={"flowsheet_costing_block": m.fs.costing,
                     "costing_method": SSLWCostingData.cost_heat_exchanger}
        )

    ###### 2. Calculate total capital cost for charge and discharge
    ###### heat exchangers
    m.fs.storage_capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e9),
        doc="Capital cost for solar salt heat exchangers in $/hour")

    def solar_cap_cost_rule(b):
        return m.fs.storage_capital_cost == (
            (m.fs.hxc.costing.capital_cost
             + m.fs.hxd.costing.capital_cost)
            / (m.fs.num_of_years * 365 * 24)
        )
    m.fs.cap_cost_eq = Constraint(
        rule=solar_cap_cost_rule)

    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.fuel_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Fuel (coal) cost in $/hour")  # add units

    # Modified to remove q_baseline, this now is the fuel cost (if no cooler)
    def fuel_cost_rule(b):
        return m.fs.fuel_cost == (
            m.fs.operating_hours *
            m.fs.coal_price *
            (m.fs.coal_heat_duty * 1e6) -
            (m.fs.cooling_price *
             m.fs.operating_hours *
             m.fs.cooler.heat_duty[0])
        ) / (365 * 24)
    m.fs.fuel_cost_eq = Constraint(rule=fuel_cost_rule)


    ###########################################################################
    #  Annual capital and operating cost for full plant
    ###########################################################################
    # Capital cost for power plant
    # m.fs.plant_capital_cost = pyo.Var(
    #     initialize=1000000,
    #     bounds=(0, 1e12),
    #     doc="Annualized capital cost for the plant in $")
    m.fs.plant_fixed_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $/hour")
    m.fs.plant_variable_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $/hour")

    # def plant_cap_cost_rule(b):
    #     return m.fs.plant_capital_cost == (
    #         (2688973 * m.fs.plant_power_out[0]  # in MW
    #          + 618968072) /
    #         m.fs.num_of_years
    #     ) * (m.CE_index / 575.4)
    # m.fs.plant_cap_cost_eq = Constraint(rule=plant_cap_cost_rule)

    def op_fixed_plant_cost_rule(b):
        return m.fs.plant_fixed_operating_cost == (
            (16657.5 * m.fs.plant_power_out[0]  # in MW
             + 6109833.3) /
            (m.fs.num_of_years * 365 * 24)
        ) * (m.CE_index / 575.4)
    m.fs.op_fixed_plant_cost_eq = Constraint(
        rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return m.fs.plant_variable_operating_cost == (
            (31754.7 * m.fs.plant_power_out[0]) *
            (m.CE_index / 575.4)
        ) / (365 * 24)
    m.fs.op_variable_plant_cost_eq = Constraint(
        rule=op_variable_plant_cost_rule)

    return m


def initialize_with_costing(m, solver=None):
    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.fuel_cost,
        m.fs.fuel_cost_eq)

    # # Initialize capital cost of power plant
    # calculate_variable_from_constraint(
    #     m.fs.plant_capital_cost,
    #     m.fs.plant_cap_cost_eq)

    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.plant_fixed_operating_cost,
        m.fs.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.plant_variable_operating_cost,
        m.fs.op_variable_plant_cost_eq)

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True)
    print("Cost Initialization = ",
          res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print('')
    print('')


def add_bounds(m):
    """Add bounds to units in charge model

    """

    # calculate_bounds(m)

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W

    # Maximum flow/heat is calculated solving no storage case with power
    # max = 436. Minimum flow/heat is calculated solving no storage mode
    # with min power = 283 (0.65*p_max)
    m.flow_max = m.main_flow * 1.5  # in mol/s
    m.flow_min = 11804  # in mol/s
    m.boiler_heat_max = 918e6  # in W
    m.boiler_heat_min = 586e6  # in W
    m.salt_flow_max = 500  # in kg/s
    m.fs.heat_duty_max = 200e6  # in MW
    m.factor = 2

    # Charge heat exchanger
    m.fs.hxc.inlet_1.flow_mol.setlb(0)
    m.fs.hxc.inlet_1.flow_mol.setub(0.2 * m.flow_max)
    m.fs.hxc.inlet_2.flow_mass.setlb(0)
    m.fs.hxc.inlet_2.flow_mass.setub(m.salt_flow_max)
    m.fs.hxc.outlet_1.flow_mol.setlb(0)
    m.fs.hxc.outlet_1.flow_mol.setub(0.2 * m.flow_max)
    m.fs.hxc.outlet_2.flow_mass.setlb(0)
    m.fs.hxc.outlet_2.flow_mass.setub(m.salt_flow_max)
    m.fs.hxc.inlet_2.pressure.setlb(101320)
    m.fs.hxc.inlet_2.pressure.setub(101330)
    m.fs.hxc.outlet_2.pressure.setlb(101320)
    m.fs.hxc.outlet_2.pressure.setub(101330)
    m.fs.hxc.heat_duty.setlb(0)
    m.fs.hxc.heat_duty.setub(m.fs.heat_duty_max)
    m.fs.hxc.shell.heat.setlb(-m.fs.heat_duty_max)
    m.fs.hxc.shell.heat.setub(0)
    m.fs.hxc.tube.heat.setlb(0)
    m.fs.hxc.tube.heat.setub(m.fs.heat_duty_max)
    m.fs.hxc.tube.properties_in[0].enth_mass.setlb(0)
    m.fs.hxc.tube.properties_in[0].enth_mass.setub(1.5e6)
    m.fs.hxc.tube.properties_out[0].enth_mass.setlb(0)
    m.fs.hxc.tube.properties_out[0].enth_mass.setub(1.5e6)

    m.fs.hxc.overall_heat_transfer_coefficient.setlb(0)
    m.fs.hxc.overall_heat_transfer_coefficient.setub(10000)
    m.fs.hxc.area.setlb(100)
    m.fs.hxc.area.setub(6000)  # 5000
    # Testing for NS
    m.fs.hxc.delta_temperature_in.setlb(9)
    # m.fs.hxc.delta_temperature_out.setlb(10)
    m.fs.hxc.delta_temperature_out.setlb(5)
    m.fs.hxc.delta_temperature_in.setub(80.5)
    m.fs.hxc.delta_temperature_out.setub(81)

    # Discharge heat exchanger
    m.fs.hxd.inlet_2.flow_mol.setlb(0)
    m.fs.hxd.inlet_2.flow_mol.setub(0.2 * m.flow_max)
    m.fs.hxd.inlet_1.flow_mass.setlb(0)
    m.fs.hxd.inlet_1.flow_mass.setub(m.salt_flow_max)
    m.fs.hxd.outlet_2.flow_mol.setlb(0)
    m.fs.hxd.outlet_2.flow_mol.setub(0.2 * m.flow_max)
    m.fs.hxd.outlet_1.flow_mass.setlb(0)
    m.fs.hxd.outlet_1.flow_mass.setub(m.salt_flow_max)
    m.fs.hxd.inlet_1.pressure.setlb(101320)
    m.fs.hxd.inlet_1.pressure.setub(101330)
    m.fs.hxd.outlet_1.pressure.setlb(101320)
    m.fs.hxd.outlet_1.pressure.setub(101330)
    m.fs.hxd.heat_duty.setlb(0)
    m.fs.hxd.heat_duty.setub(m.fs.heat_duty_max)

    m.fs.hxd.tube.heat.setub(m.fs.heat_duty_max)
    m.fs.hxd.tube.heat.setlb(0)
    m.fs.hxd.shell.heat.setub(0)
    m.fs.hxd.shell.heat.setlb(-m.fs.heat_duty_max)

    m.fs.hxd.shell.properties_in[0].enth_mass.setlb(0)
    m.fs.hxd.shell.properties_in[0].enth_mass.setub(1.5e6)
    m.fs.hxd.shell.properties_out[0].enth_mass.setlb(0)
    m.fs.hxd.shell.properties_out[0].enth_mass.setub(1.5e6)
    m.fs.hxd.overall_heat_transfer_coefficient.setlb(0)
    m.fs.hxd.overall_heat_transfer_coefficient.setub(10000)
    m.fs.hxd.area.setlb(100)
    m.fs.hxd.area.setub(6000)  # 5000
    m.fs.hxd.delta_temperature_in.setlb(4.9)
    m.fs.hxd.delta_temperature_out.setlb(10)
    m.fs.hxd.delta_temperature_in.setub(300)
    m.fs.hxd.delta_temperature_out.setub(300)

    # Add bounds for the HX pump and Cooler
    for unit_k in [m.fs.hx_pump,
                   m.fs.cooler]:
        unit_k.inlet.flow_mol.setlb(0)
        unit_k.inlet.flow_mol.setub(0.2*m.flow_max)
        unit_k.outlet.flow_mol.setlb(0)
        unit_k.outlet.flow_mol.setub(0.2*m.flow_max)
    m.fs.cooler.heat_duty.setub(0)

    # Add bounds needed HP splitter
    for split in [m.fs.ess_hp_split]:
        split.to_hxc.flow_mol[:].setlb(0)
        split.to_hxc.flow_mol[:].setub(0.2 * m.flow_max)
        split.split_fraction[0.0, "to_hxc"].setlb(0)
        split.split_fraction[0.0, "to_hxc"].setub(1)
        split.split_fraction[0.0, "to_turbine"].setlb(0)
        split.split_fraction[0.0, "to_turbine"].setub(1)
        split.inlet.flow_mol[:].setlb(0)
        split.inlet.flow_mol[:].setub(m.flow_max)

    for split in [m.fs.ess_bfp_split]:
        split.to_hxd.flow_mol[:].setlb(0)
        split.to_hxd.flow_mol[:].setub(0.2 * m.flow_max)
        split.split_fraction[0.0, "to_hxd"].setlb(0)
        split.split_fraction[0.0, "to_hxd"].setub(1)
        split.split_fraction[0.0, "to_recyclemix"].setlb(0)
        split.split_fraction[0.0, "to_recyclemix"].setub(1)
        split.inlet.flow_mol[:].setlb(0)
        split.inlet.flow_mol[:].setub(m.flow_max)

    for mix in [m.fs.recycle_mixer]:
        mix.from_bfw_out.flow_mol.setlb(0)
        mix.from_bfw_out.flow_mol.setub(m.flow_max)
        mix.from_hx_pump.flow_mol.setlb(0)
        mix.from_hx_pump.flow_mol.setub(0.2 * m.flow_max)
        mix.outlet.flow_mol.setlb(0)
        mix.outlet.flow_mol.setub(m.flow_max)

    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e10)
        m.fs.turbine[k].work.setub(0)
    m.fs.hx_pump.control_volume.work[0].setlb(0)
    m.fs.hx_pump.control_volume.work[0].setub(1e10)

    # m.fs.plant_power_out[0].setlb(100)
    # m.fs.plant_power_out[0].setub(450)

    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s

    # Adding bounds on turbine splitters flow
    for k in m.set_turbine_splitter:
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setub(m.flow_max)


def main(method=None,
         pmax=None,
         load_from_file=None,
         solver=None,
         max_salt_amount=None):

    if load_from_file is not None:

        # build plant model
        m = usc.build_plant_model()

        # Create a flowsheet, add properties, unit models, and arcs
        m = create_charge_model(m,
                                method=method,
                                pmax=pmax,
                                max_salt_amount=max_salt_amount)

        # Give all the required inputs to the model
        set_model_input(m)
        # print('DOF after build: ', degrees_of_freedom(m))
        # Add scaling factor
        set_scaling_factors(m)

        # Add cost correlations
        m = build_costing(m)
        # print('DOF after costing: ', degrees_of_freedom(m))
        # raise Exception()

        # Initialize with bounds
        ms.from_json(m, fname=load_from_file)
    else:

        m = usc.build_plant_model()
        usc.initialize(m)

        # Create a flowsheet, add properties, unit models, and arcs
        m = create_charge_model(m,
                                method=method,
                                pmax=pmax,
                                max_salt_amount=max_salt_amount)

        # Give all the required inputs to the model
        set_model_input(m)
        print('DOF after build: ', degrees_of_freedom(m))
        # Add scaling factor
        set_scaling_factors(m)

        # Initialize the model with a sequential initialization and custom
        # routines
        print('DOF before initialization: ', degrees_of_freedom(m))
        initialize(m)
        print('DOF after initialization: ', degrees_of_freedom(m))

        # Add cost correlations
        m = build_costing(m)
        print('DOF after costing: ', degrees_of_freedom(m))
        # raise Exception()

        # Initialize with bounds
        initialize_with_costing(m, solver=solver)

    # Add bounds
    add_bounds(m)
    # print('DOF after bounds: ', degrees_of_freedom(m))

    # # storing the initialization file
    # ms.to_json(m, fname='initialized_usc_storage_nlp_mp.json')
    return m


def print_results(m, results):

    print('================================')
    print('')
    print('')
    print("***************** Optimization Results ******************")
    print('Obj ($/h): {:.6f}'.format(
        value(m.obj) / scaling_obj))
    print('Revenue ($/h): {:.6f}'.format(
        value(m.fs.revenue)))
    print('Storage Capital Cost ($/h): {:.6f}'.format(
        value(m.fs.storage_capital_cost)))
    print('Fuel Cost ($/h): {:.6f}'.format(
        value(m.fs.fuel_cost)/(365*24)))
    # print('Plant capital cost ($/h): {:.6f}'.format(
    #     value(m.fs.plant_capital_cost)))
    print('Plant fixed operating costs ($/h): {:.6f}'.format(
        value(m.fs.plant_fixed_operating_cost)))
    print('Plant variable operating costs ($/h): {:.6f}'.format(
        value(m.fs.plant_variable_operating_cost)))
    print('')
    print('')
    print("***************** Costing Results ******************")
    print('Hot Previous Salt Inventory (mton): {:.6f}'.format(
        value(m.fs.previous_salt_inventory_hot[0])))
    print('Hot Salt Inventory (mton): {:.6f}'.format(
        value(m.fs.salt_inventory_hot[0])))
    print('Cold Previous Salt Inventory (mton): {:.6f}'.format(
        value(m.fs.previous_salt_inventory_cold[0])))
    print('Cold Salt Inventory (mton): {:.6f}'.format(
        value(m.fs.salt_inventory_cold[0])))
    print('Salt Amount (mton): {:.6f}'.format(
        value(m.fs.salt_amount)))
    print('')
    print('')
    print("***************** Power Plant Operation ******************")
    print('')
    print('Net Power (MW): {:.6f}'.format(
        value(m.fs.net_power)))
    print('Plant Power (MW): {:.6f}'.format(
        value(m.fs.plant_power_out[0])))
    print('ES turbine Power (MW): {:.6f}'.format(
        value(m.fs.es_turbine.work[0]) * (-1e-6)))
    print('Boiler feed water flow (mol/s): {:.6f}'.format(
        value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler duty (MW_th): {:.6f}'.format(
        value(m.fs.plant_heat_duty[0])))
    print('Cooling duty (MW_th): {:.6f}'.format(
        value(m.fs.cooler.heat_duty[0]) * -1e-6))
    print('HXC heat duty (MW): {:.6f}'.format(
        value(m.fs.hxc.heat_duty[0]) * 1e-6))
    print('HXD heat duty (MW): {:.6f}'.format(
        value(m.fs.hxd.heat_duty[0]) * 1e-6))
    print('Makeup water flow: {:.6f}'.format(
        value(m.fs.condenser_mix.makeup.flow_mol[0])))
    print('')
    print('')
    print("***************** Charge Heat Exchanger (HXC) ******************")
    print('')
    print('HXC area (m2): {:.6f}'.format(
        value(m.fs.hxc.area)))
    print('HXC cost ($/h): {:.6f}'.format(
        value(m.fs.hxc.costing.capital_cost) / (m.fs.num_of_years * 365 * 24)))
    print('HXC Salt flow (kg/s): {:.6f}'.format(
        value(m.fs.hxc.inlet_2.flow_mass[0])))
    print('HXC Salt temperature in (K): {:.6f}'.format(
        value(m.fs.hxc.inlet_2.temperature[0])))
    print('HXC Salt temperature out (K): {:.6f}'.format(
        value(m.fs.hxc.outlet_2.temperature[0])))
    print('HXC Steam flow to storage (mol/s): {:.6f}'.format(
        value(m.fs.hxc.inlet_1.flow_mol[0])))
    print('HXC Water temperature in (K): {:.6f}'.format(
        value(m.fs.hxc.side_1.properties_in[0].temperature)))
    print('HXC Steam temperature out (K): {:.6f}'.format(
        value(m.fs.hxc.side_1.properties_out[0].temperature)))
    print('HXC Delta temperature at inlet (K): {:.6f}'.format(
        value(m.fs.hxc.delta_temperature_in[0])))
    print('HXC Delta temperature at outlet (K): {:.6f}'.format(
        value(m.fs.hxc.delta_temperature_out[0])))
    print('')
    print('')
    print("*************** Discharge Heat Exchanger (HXD) ****************")
    print('')
    print('HXD area (m2): {:.6f}'.format(
        value(m.fs.hxd.area)))
    print('HXD cost ($/h): {:.6f}'.format(
        value(m.fs.hxd.costing.capital_cost) / (m.fs.num_of_years * 365 * 24)))
    print('HXD Salt flow (kg/s): {:.6f}'.format(
        value(m.fs.hxd.inlet_1.flow_mass[0])))
    print('HXD Salt temperature in (K): {:.6f}'.format(
        value(m.fs.hxd.inlet_1.temperature[0])))
    print('HXD Salt temperature out (K): {:.6f}'.format(
        value(m.fs.hxd.outlet_1.temperature[0])))
    print('HXD Steam flow to storage (mol/s): {:.6f}'.format(
        value(m.fs.hxd.inlet_2.flow_mol[0])))
    print('HXD Water temperature in (K): {:.6f}'.format(
        value(m.fs.hxd.side_2.properties_in[0].temperature)))
    print('HXD Steam temperature out (K): {:.6f}'.format(
        value(m.fs.hxd.side_2.properties_out[0].temperature)))
    print('HXD Delta temperature at inlet (K): {:.6f}'.format(
        value(m.fs.hxd.delta_temperature_in[0])))
    print('HXD Delta temperature at outlet (K): {:.6f}'.format(
        value(m.fs.hxd.delta_temperature_out[0])))
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


def model_analysis(m,
                   solver=None,
                   power=None,
                   pmax=None,
                   pmin=None,
                   pmax_storage=None,
                   pmin_storage=None,
                   tank_status=None,
                   fix_power=None,
                   max_salt_amount=None,
                   factor_mton=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    if fix_power:
        m.fs.power_demand_eq = Constraint(
            expr=m.fs.net_power == power
        )
    else:
        m.fs.plant_power_min = Constraint(
            expr=m.fs.plant_power_out[0] >= pmin
        )
        m.fs.plant_power_max = Constraint(
            expr=m.fs.plant_power_out[0] <= pmax
        )
        m.fs.storage_power_min = Constraint(
            expr=m.fs.es_turbine.work[0] * (-1e-6) >= pmin_storage
        )
        m.fs.storage_power_max = Constraint(
            expr=m.fs.es_turbine.work[0] * (-1e-6) <= pmax_storage
        )

    # Fix variables in the flowsheet
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s

    # Unfix all data
    m.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.fs.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()
        salt_hxc.area.unfix()

    for salt_hxd in [m.fs.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()
        salt_hxd.area.unfix()

    for unit in [m.fs.cooler]:
        unit.inlet.unfix()
    m.fs.cooler.outlet.enth_mol[0].unfix()

    # Fix storage heat exchangers area and salt temperatures
    hot_salt_temperature = design_data_dict["hot_salt_temperature"]
    cold_salt_temperature = design_data_dict["cold_salt_temperature"]
    hxc_area = design_data_dict["hxc_area"]
    hxd_area = design_data_dict["hxd_area"]
    m.fs.hxc.area.fix(hxc_area)
    m.fs.hxd.area.fix(hxd_area)
    m.fs.hxc.outlet_2.temperature.fix(hot_salt_temperature)
    m.fs.hxd.inlet_1.temperature.fix(hot_salt_temperature)
    m.fs.hxd.outlet_1.temperature.fix(cold_salt_temperature)

    # Add salt inventory mass balances
    max_inventory = 1e7 * factor_mton
    m.fs.previous_salt_inventory_hot = pyo.Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, max_inventory),
        doc="Hot salt at the beginning of the time period in mton"
        )
    m.fs.salt_inventory_hot = pyo.Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the end of the time period in mton"
        )
    m.fs.previous_salt_inventory_cold = pyo.Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, max_inventory),
        doc="Cold salt at the beginning of the time period in mton"
        )
    m.fs.salt_inventory_cold = pyo.Var(
        m.fs.time,
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the end of the time period in mton"
        )

    @m.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            b.salt_inventory_hot[0] == (
                b.previous_salt_inventory_hot[0]
                + (3600 * b.hxc.inlet_2.flow_mass[0]
                   - 3600 * b.hxd.inlet_1.flow_mass[0]) * factor_mton # in mton
            )
        )

    @m.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            b.salt_inventory_hot[0] +
            b.salt_inventory_cold[0] == b.salt_amount)

    # Fix the previous salt inventory based on the tank scenario
    tank_min = 1 * factor_mton # in mton
    tank_max = max_salt_amount  # in mton
    if tank_status == "hot_empty":
        m.fs.previous_salt_inventory_hot[0].fix(tank_min)
        m.fs.previous_salt_inventory_cold[0].fix(tank_max - tank_min)
    elif tank_status == "hot_half_full":
        m.fs.previous_salt_inventory_hot[0].fix(tank_max / 2)
        m.fs.previous_salt_inventory_cold[0].fix(tank_max / 2)
    elif tank_status == "hot_full":
        m.fs.previous_salt_inventory_hot[0].fix(tank_max - tank_min)
        m.fs.previous_salt_inventory_cold[0].fix(tank_min)
    else:
        print('Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full')


    # Calculate revenue
    m.fs.revenue = pyo.Expression(
        expr=(m.fs.lmp * m.fs.net_power),
        doc="Revenue function in $/h assuming 1 hr operation"
    )

    # Objective function: total costs
    m.obj = Objective(
        expr=(
            m.fs.revenue
            - ((m.fs.fuel_cost
                + m.fs.plant_fixed_operating_cost
                + m.fs.plant_variable_operating_cost))
            - (
                m.fs.storage_capital_cost
                # + m.fs.plant_capital_cost
            )
        ) * scaling_obj,
        sense=maximize
    )

    print('DOF before solve = ', degrees_of_freedom(m))

    # Solve the design optimization model
    results = solver.solve(
        m,
        tee=True,
        symbolic_solver_labels=True,
        options={
            "linear_solver": "ma27",
            "max_iter": 150
        }
    )
    m.fs.condenser_mix.makeup.display()
    print_results(m, results)
    # print_reports(m)

    log_close_to_bounds(m)
    log_infeasible_constraints(m)


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Add design data from .json file
    # data_path = 'uscp_nlp_design_data.json'
    data_path = 'uscp_design_data.json'
    with open(data_path) as design_data:
        design_data_dict = json.load(design_data)

    pmax = design_data_dict["plant_max_power"]
    pmin = design_data_dict["plant_min_power"]
    pmin_storage = design_data_dict["min_discharge_turbine_power"]
    pmax_storage = design_data_dict["max_discharge_turbine_power"]
    factor_mton = design_data_dict["factor_mton"]
    max_salt_amount = design_data_dict["max_salt_amount"] * factor_mton

    # Tank scenarios: "hot_empty", "hot_full", "hot_half_full"
    power_demand = 436 # in MW
    method = "with_efficiency"
    tank_status = "hot_empty"
    fix_power = False
    load_from_file = None

    m_nlp = main(method=method,
                 pmax=pmax,
                 load_from_file=load_from_file,
                 solver=solver,
                 max_salt_amount=max_salt_amount)

    m_nlp.fs.lmp = pyo.Param(
        initialize=22,
        doc="Hourly LMP in $/MWh"
        )

    m = model_analysis(m_nlp,
                       solver,
                       power=power_demand,
                       pmax=pmax,
                       pmin=pmin,
                       pmax_storage=pmax_storage,
                       pmin_storage=pmin_storage,
                       tank_status=tank_status,
                       fix_power=fix_power,
                       max_salt_amount=max_salt_amount,
                       factor_mton=factor_mton)
