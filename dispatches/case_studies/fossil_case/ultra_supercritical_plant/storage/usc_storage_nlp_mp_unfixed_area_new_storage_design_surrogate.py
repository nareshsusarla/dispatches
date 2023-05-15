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

"""This is an NLP model for the conceptual design of an ultra
supercritical coal-fired power plant integrated with an energy storage
system

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

__author__ = "Soraya Rawlings and Naresh Susarla"

# Import Python libraries
from math import pi
import logging
import json

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (ConcreteModel, Block, Param, Constraint, Objective, Reals,
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
from idaes.models.costing.SSLW import (SSLWCosting, SSLWCostingData,
                                       PumpType, PumpMaterial, PumpMotorType)

from idaes.core.util import model_serializer as ms

# Import IDAES Libraries
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import PressureChanger
from idaes.models_extra.power_generation.unit_models.helm import (HelmMixer,
                                                                  HelmTurbineStage,
                                                                  HelmSplitter)
from idaes.models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.exceptions import ConfigurationError

from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
from dispatches.properties import solarsalt_properties

from pyomo.network.plugins import expand_arcs

from IPython import embed
logging.basicConfig(level=logging.INFO)
logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)

# Import Property Packages (IAPWS95 for Water/Steam)
from idaes.models.properties import iapws95

scaling_obj = 1

# Add design data from .json file
with open('uscp_design_data_new_storage_design.json') as design_data:
    design_data_dict = json.load(design_data)


def create_integrated_model(m=None, method=None):
    """Create flowsheet and add unit models.
    """

    if m is None:
        m = ConcreteModel(name="Ultra Supercritical Power Plant Model")

    m.fs = FlowsheetBlock(dynamic=False)
    # Add water and molten salt properties (Solar salt)
    m.fs.prop_water = iapws95.Iapws95ParameterBlock()
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()

    # Add data
    add_data(m)


    # Add units for charge storage system: (a) Add charge heat
    # exchanger and (b) add pump used to increase the pressure of the
    # water to allow mixing it at a desired location within the plant
    m.fs.hxc = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.prop_water},
        tube={"property_package": m.fs.solar_salt_properties}
    )
    m.fs.hx_pump = PressureChanger(
        property_package=m.fs.prop_water,
        material_balance_type=MaterialBalanceType.componentTotal,
        thermodynamic_assumption=ThermodynamicAssumption.pump,
    )

    # Add untis for discharge storage: (a) Add discharge heat
    # exchanger and (b) add turbine for storage discharge
    m.fs.hxd = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.solar_salt_properties},
        tube={"property_package": m.fs.prop_water}
    )
    m.fs.es_turbine = HelmTurbineStage(property_package=m.fs.prop_water)

    # Compute overall heat transfer coefficient for the charge and
    # discharge heat exchanger using the Sieder-Tate Correlation. The
    # parameters for tube diameter and thickness assumed from the data
    # in (2017) He et al., Energy Procedia 105, 980-985

    # -------- Charge Heat Exchanger Heat Transfer Coefficient --------
    m.fs.hxc.salt_reynolds_number = pyo.Expression(
        expr=(
            (m.fs.hxc.tube_inlet.flow_mass[0]*m.fs.tube_outer_dia)/
            (m.fs.shell_eff_area*m.fs.hxc.cold_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number")
    m.fs.hxc.salt_prandtl_number = pyo.Expression(
        expr=(
            m.fs.hxc.cold_side.properties_in[0].cp_mass["Liq"]*
            m.fs.hxc.cold_side.properties_in[0].visc_d_phase["Liq"]/
            m.fs.hxc.cold_side.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number")
    m.fs.hxc.salt_prandtl_wall = pyo.Expression(
        expr=(
            m.fs.hxc.cold_side.properties_out[0].cp_mass["Liq"]*
            m.fs.hxc.cold_side.properties_out[0].visc_d_phase["Liq"]/
            m.fs.hxc.cold_side.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number at wall")
    m.fs.hxc.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 *
            (m.fs.hxc.salt_reynolds_number**0.6)*(m.fs.hxc.salt_prandtl_number**0.4)*
            ((m.fs.hxc.salt_prandtl_number/m.fs.hxc.salt_prandtl_wall)**0.25)*(2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")
    m.fs.hxc.steam_reynolds_number = pyo.Expression(
        expr=(
            m.fs.hxc.shell_inlet.flow_mol[0]*m.fs.hxc.hot_side.properties_in[0].mw*
            m.fs.tube_inner_dia/
            (m.fs.tube_cs_area*m.fs.n_tubes*m.fs.hxc.hot_side.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")
    m.fs.hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (m.fs.hxc.hot_side.properties_in[0].cp_mol/m.fs.hxc.hot_side.properties_in[0].mw)*
            m.fs.hxc.hot_side.properties_in[0].visc_d_phase["Vap"]/
            m.fs.hxc.hot_side.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")
    m.fs.hxc.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 *
            (m.fs.hxc.steam_reynolds_number**0.8)*(m.fs.hxc.steam_prandtl_number**(0.33))*
            ((m.fs.hxc.hot_side.properties_in[0].visc_d_phase["Vap"]/
              m.fs.hxc.hot_side.properties_out[0].visc_d_phase["Liq"])**0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of charge heat exchanger
    m.fs.hxc.h_salt = pyo.Expression(
        expr=(
            m.fs.hxc.cold_side.properties_in[0].therm_cond_phase["Liq"]*
            m.fs.hxc.salt_nusselt_number/m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    m.fs.hxc.h_steam = pyo.Expression(
        expr=(
            m.fs.hxc.hot_side.properties_in[0].therm_cond_phase["Vap"]*
            m.fs.hxc.steam_nusselt_number/m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")

    # Calculate overall heat transfer coefficient
    @m.fs.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        return (
            b.overall_heat_transfer_coefficient[t]*
            (2*m.fs.k_steel*b.h_steam +
             m.fs.tube_outer_dia*m.fs.log_tube_dia_ratio*b.h_salt*b.h_steam +
             m.fs.tube_dia_ratio*b.h_salt*2*m.fs.k_steel)
        ) == 2*m.fs.k_steel*b.h_salt*b.h_steam

    # ------- Discharge Heat Exchanger Heat Transfer Coefficient -------
    # Discharge heat exchanger salt and steam side constraints to
    # calculate Reynolds number, Prandtl number, and Nusselt number
    m.fs.hxd.salt_reynolds_number = pyo.Expression(
        expr=(m.fs.hxd.shell_inlet.flow_mass[0]*m.fs.tube_outer_dia/
              (m.fs.shell_eff_area*m.fs.hxd.hot_side.properties_in[0].visc_d_phase["Liq"])),
        doc="Salt Reynolds Number"
    )
    m.fs.hxd.salt_prandtl_number = pyo.Expression(
        expr=(m.fs.hxd.hot_side.properties_in[0].cp_mass["Liq"]*
              m.fs.hxd.hot_side.properties_in[0].visc_d_phase["Liq"]/
              m.fs.hxd.hot_side.properties_in[0].therm_cond_phase["Liq"]),
        doc="Salt Prandtl Number"
    )
    # Assuming that the wall conditions are same as those at the outlet
    m.fs.hxd.salt_prandtl_wall = pyo.Expression(
        expr=(m.fs.hxd.hot_side.properties_out[0].cp_mass["Liq"]*
              m.fs.hxd.hot_side.properties_out[0].visc_d_phase["Liq"]/
              m.fs.hxd.hot_side.properties_out[0].therm_cond_phase["Liq"]),
        doc="Wall Salt Prandtl Number"
    )
    m.fs.hxd.salt_nusselt_number = pyo.Expression(
        expr=(0.35*(m.fs.hxd.salt_reynolds_number**0.6)*(m.fs.hxd.salt_prandtl_number**0.4)*
              ((m.fs.hxd.salt_prandtl_number/m.fs.hxd.salt_prandtl_wall)**0.25)*(2**0.2)),
        doc="Solar Salt Nusslet Number from 2019, App Ener (233-234), 126"
    )
    m.fs.hxd.steam_reynolds_number = pyo.Expression(
        expr=(m.fs.hxd.tube_inlet.flow_mol[0]*m.fs.hxd.cold_side.properties_in[0].mw*
              m.fs.tube_inner_dia/
              (m.fs.tube_cs_area*m.fs.n_tubes*m.fs.hxd.cold_side.properties_in[0].visc_d_phase["Vap"])),
        doc="Steam Reynolds Number"
    )
    m.fs.hxd.steam_prandtl_number = pyo.Expression(
        expr=((m.fs.hxd.cold_side.properties_in[0].cp_mol/m.fs.hxd.cold_side.properties_in[0].mw)*
              m.fs.hxd.cold_side.properties_in[0].visc_d_phase["Vap"]/
              m.fs.hxd.cold_side.properties_in[0].therm_cond_phase["Vap"]),
        doc="Steam Prandtl Number"
    )
    m.fs.hxd.steam_nusselt_number = pyo.Expression(
        expr=(0.023*(m.fs.hxd.steam_reynolds_number**0.8)*(m.fs.hxd.steam_prandtl_number**(0.33))*
              ((m.fs.hxd.cold_side.properties_in[0].visc_d_phase["Vap"]/
                m.fs.hxd.cold_side.properties_out[0].visc_d_phase["Liq"])**0.14)),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Discharge heat exchanger salt and steam side heat transfer
    # coefficients
    m.fs.hxd.h_salt = pyo.Expression(
        expr=(m.fs.hxd.hot_side.properties_in[0].therm_cond_phase["Liq"]*
              m.fs.hxd.salt_nusselt_number/m.fs.tube_outer_dia),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    m.fs.hxd.h_steam = pyo.Expression(
        expr=(m.fs.hxd.cold_side.properties_in[0].therm_cond_phase["Vap"]*
              m.fs.hxd.steam_nusselt_number/m.fs.tube_inner_dia),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    @m.fs.hxd.Constraint(m.fs.time,
                         doc="Overall heat transfer coefficient for hxd")
    def constraint_hxd_ohtc(b, t):
        return (
            b.overall_heat_transfer_coefficient[t]*
            (2*m.fs.k_steel*b.h_steam +
             m.fs.tube_outer_dia*m.fs.log_tube_dia_ratio*b.h_salt*b.h_steam +
             m.fs.tube_dia_ratio*b.h_salt*2*m.fs.k_steel)
        ) == 2*m.fs.k_steel*b.h_salt*b.h_steam

    ###########################################################################
    #  Add model variables for salt inventory and model surrogates            #
    ###########################################################################
    m.flow_max_storage = 7141.6
    m.flow_max = 35708
    m.flow_min = 1e-3
    m.flow_min_boiler = 13390.5

    # Define the amount of storage material
    m.fs.salt_amount = pyo.Var(initialize=m.max_salt_amount,
                               bounds=(0, m.max_inventory),
                               units=pyunits.metric_ton,
                               doc="Solar salt amount")
    m.fs.salt_amount.fix(m.max_salt_amount)

    m.fs.boiler_flow = pyo.Var(m.fs.time,
                               domain=Reals,
                               bounds=(m.flow_min_boiler, m.flow_max),
                               initialize=m.main_flow,
                               doc="Boiler flow in mol/s",
                               units=pyunits.mol/pyunits.second)
    m.fs.plant_power_out = pyo.Var(m.fs.time,
                                   domain=NonNegativeReals,
                                   initialize=436,
                                   doc="Net Power MWe out from the power plant",
                                   units=pyunits.MW)
    m.fs.plant_heat_duty = pyo.Var(m.fs.time,
                                   domain=Reals,
                                   initialize=500,
                                   doc="Plant heat duty in MWh from the power plant",
                                   units=pyunits.MW)
    m.fs.steam_to_storage = pyo.Var(m.fs.time,
                                    domain=Reals,
                                    bounds=(m.flow_min, m.flow_max_storage),
                                    initialize=4000,
                                    doc="Steam flow to storage in mol/s",
                                    units=pyunits.mol/pyunits.second)
    m.fs.steam_to_discharge = pyo.Var(m.fs.time,
                                      domain=Reals,
                                      bounds=(m.flow_min, m.flow_max_storage),
                                      initialize=2000,
                                      doc="Steam flow to discharge in mol/s",
                                      units=pyunits.mol/pyunits.second)

    m.fs.hx_pump_work_MW = pyo.units.convert(m.fs.hx_pump.control_volume.work[0],
                                             to_units=pyunits.MW)
    @m.fs.Constraint(m.fs.time)
    def constraint_plant_power_out(b, t):
        return b.plant_power_out[t] == (
            (0.23559794386721039094468e-1*m.fs.boiler_flow[0] -
             0.27437447836144223528576e-1*m.fs.steam_to_storage[0] +
             0.60710771863505590655069e-7*m.fs.boiler_flow[0]**2 -
             3.5422970651897642824224) -
            m.fs.hx_pump_work_MW
        )

    @m.fs.Constraint(m.fs.time)
    def constraint_plant_heat_duty(b, t):
        return b.plant_heat_duty[t] == (
            0.49537255180412791133460e-1*m.fs.boiler_flow[0] -
            0.78959880803533399190597e-2*m.fs.steam_to_storage[0] +
            0.18537169317994028979120e-6*m.fs.boiler_flow[0]**2 -
            25.517607633969475955382
        )

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m, method=method)
    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def add_data(m):

    # Data from ultra-supercritical power plant model
    m.main_flow = 17854
    m.main_steam_pressure = 31125980

    # From models used to generate data for surrogate. For reference,
    # the inlet temperature for the steam during charge is 866.15K and
    # during discharge is 484K
    m.charge_inlet_pressure = 8604660.305760002*pyunits.Pa
    m.charge_inlet_enth_mol = 65223.26974470633*pyunits.J/pyunits.mol
    m.discharge_inlet_pressure = 2320703*pyunits.Pa
    m.discharge_inlet_enth_mol = 2974.4090577310344*pyunits.J/pyunits.mol

    m.pmin_storage = design_data_dict["min_discharge_turbine_power"]*pyunits.MW
    m.pmax_storage = design_data_dict["max_discharge_turbine_power"]*pyunits.MW
    m.max_salt_amount = pyo.units.convert(design_data_dict["max_salt_amount"]*pyunits.kg,
                                          to_units=pyunits.metric_ton)
    m.min_storage_duty = design_data_dict["min_storage_duty"]*pyunits.MW
    m.max_storage_duty = design_data_dict["max_storage_duty"]*pyunits.MW
    m.min_area = design_data_dict["min_storage_area_design"]*pyunits.m**2
    m.max_area = design_data_dict["max_storage_area_design"]*pyunits.m**2
    m.hxc_area_init = design_data_dict["hxc_area"]*pyunits.m**2
    m.hxd_area_init = design_data_dict["hxd_area"]*pyunits.m**2
    m.max_salt_flow = design_data_dict["max_salt_flow"]*(pyunits.kg/pyunits.seconds)
    m.hot_salt_temp_init = design_data_dict["hot_salt_temperature"]*pyunits.K
    m.min_salt_temp = design_data_dict["min_solar_salt_temperature"]*pyunits.K
    m.max_salt_temp = design_data_dict["max_solar_salt_temperature"]*pyunits.K
    m.max_inventory = pyo.units.convert(1e7*pyunits.kg,
                                        to_units=pyunits.metric_ton)
    m.max_discharge_power = design_data_dict["max_discharge_turbine_power"]*pyunits.MW

    # Chemical engineering cost index for 2019
    m.CE_index = 607.5

    #  Define the data for the design of the storage heat
    # exchangers. The design is: Shell-n-tube counter-flow heat
    # exchanger design parameters. Data to compute overall heat
    # transfer coefficient for the charge heat exchanger using the
    # Sieder-Tate Correlation. Parameters for tube diameter and
    # thickness assumed from the data in (2017) He et al., Energy
    # Procedia 105, 980-985.
    m.fs.data_storage_hx = {'tube_thickness': 0.004,
                            'tube_inner_dia': 0.032,
                            'tube_outer_dia': 0.036,
                            'k_steel': 21.5,
                            'number_tubes': 20,
                            'shell_inner_dia': 1}

    m.fs.tube_thickness = pyo.Param(initialize=m.fs.data_storage_hx['tube_thickness'],
                                    doc='Tube thickness [m]')
    m.fs.tube_inner_dia = pyo.Param(initialize=m.fs.data_storage_hx['tube_inner_dia'],
                                    doc='Tube inner diameter [m]')
    m.fs.tube_outer_dia = pyo.Param(initialize=m.fs.data_storage_hx['tube_outer_dia'],
                                    doc='Tube outer diameter [m]')
    m.fs.k_steel = pyo.Param(initialize=m.fs.data_storage_hx['k_steel'],
                             doc='Thermal conductivity of steel [W/mK]')
    m.fs.n_tubes = pyo.Param(initialize=m.fs.data_storage_hx['number_tubes'],
                             doc='Number of tubes')
    m.fs.shell_inner_dia = pyo.Param(initialize=m.fs.data_storage_hx['shell_inner_dia'],
                                     doc='Shell inner diameter [m]')
    m.fs.tube_cs_area = pyo.Expression(expr=(pi/4)*(m.fs.tube_inner_dia**2),
                                       doc="Tube cross sectional area")
    m.fs.tube_out_area = pyo.Expression(expr=(pi/4)*(m.fs.tube_outer_dia**2),
                                        doc="Tube cross sectional area including thickness [m2]")
    m.fs.shell_eff_area = pyo.Expression(expr=((pi/4)*(m.fs.shell_inner_dia**2) -
                                               m.fs.n_tubes *m.fs.tube_out_area),
                                         doc="Effective shell cross sectional area [m2]")

    m.fs.tube_dia_ratio = (m.fs.tube_outer_dia/m.fs.tube_inner_dia)
    m.fs.log_tube_dia_ratio = log(m.fs.tube_dia_ratio)

    m.data_cost = {'coal_price': 2.11e-9,
                   'cooling_price': 3.3e-9}

    # Main flowsheet operation data
    m.fs.coal_price = pyo.Param(initialize=m.data_cost['coal_price'],
                                doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.hours_per_day = pyo.Param(initialize=24,
                                   doc='Estimated number of hours of charging per day')
    m.fs.num_of_years = pyo.Param(initialize=30,
                                  doc='Number of years for capital cost annualization')

    # Add data as Param to calculate salt pump costing. Since the pump
    # units are not explicitly modeled, the IDAES cost method is not
    # used for this equipment.  The primary purpose of the salt pump
    # is to move the storage material without changing the pressure.
    # Thus, the pressure head is computed assuming that the salt is
    # moved on an average of 5 m (i.e., 16.41 ft) linear distance.
    m.data_salt_pump = {'FT': 1.5,
                        'FM': 2.0,
                        'head': 16.41,  # in ft, eqvt to 5 m
                        'motor_FT': 1,
                        'nm': 1}
    m.fs.spump_FT = pyo.Param(initialize=m.data_salt_pump['FT'],
                              doc='Pump Type Factor for vertical split case')
    m.fs.spump_FM = pyo.Param(initialize=m.data_salt_pump['FM'],
                              doc='Pump Material Factor Stainless Steel')
    m.fs.spump_head = pyo.Param(initialize=m.data_salt_pump['head'],
                                doc='Pump Head 5m in ft.')
    m.fs.spump_motorFT = pyo.Param(initialize=m.data_salt_pump['motor_FT'],
                                   doc='Motor Shaft Type Factor')
    m.fs.spump_nm = pyo.Param(initialize=m.data_salt_pump['nm'],
                              doc='Motor Shaft Type Factor')



def _make_constraints(m, method=None):
    """Declare the constraints for the charge model
    """

    # ES turbine temperature constraint
    m.fs.eq_turbine_temperature_out = pyo.Constraint(
        expr=(
            m.fs.es_turbine.control_volume.properties_out[0].temperature ==
            m.fs.es_turbine.control_volume.properties_out[0].temperature_sat + 1*pyunits.K
        )
    )

    # HX pump
    @m.fs.Constraint(m.fs.time,
                     doc="HX pump out pressure equal to BFP out pressure")
    def constraint_hxpump_presout(b, t):
        return b.hx_pump.outlet.pressure[t] >= m.main_steam_pressure*1.1231

    m.fs.es_turbine_MW = pyo.units.convert((-1)*m.fs.es_turbine.work_mechanical[0],
                                           to_units=pyunits.MW)
    @m.fs.Expression(m.fs.time)
    def net_power(b, t):
        return b.plant_power_out[t] + b.es_turbine_MW

    m.fs.max_boiler_duty = pyo.Param(initialize=940,
                                     mutable=False,
                                     units=pyunits.MW,
                                     doc='Maximum boiler thermal power at maximum plant power')
    m.fs.boiler_efficiency = pyo.Var(initialize=0.95,
                                     bounds=(0, 1),
                                     doc="Boiler efficiency")
    def rule_boiler_efficiency(b):
        return m.fs.boiler_efficiency == (
            0.2143*(b.plant_heat_duty[0]/m.fs.max_boiler_duty)+
            0.7357)
    m.fs.eq_boiler_efficiency = pyo.Constraint(rule=rule_boiler_efficiency,
                                               doc="Boiler efficiency in fraction")

    m.fs.coal_heat_duty = pyo.Var(initialize=1e3,
                                  bounds=(0, 1e5),
                                  units=pyunits.MW,
                                  doc="Coal heat duty supplied to Boiler")

    if method == "with_efficiency":
        def rule_coal_heat_duty(b):
            return b.coal_heat_duty*b.boiler_efficiency == b.plant_heat_duty[0]
        m.fs.coal_heat_duty_eq = pyo.Constraint(rule=rule_coal_heat_duty)
    else:
        def coal_heat_duty_rule(b):
            return b.coal_heat_duty == b.plant_heat_duty[0]
        m.fs.coal_heat_duty_eq = pyo.Constraint(rule=coal_heat_duty_rule)

    m.fs.cycle_efficiency = pyo.Var(initialize=0.4,
                                    bounds=(0, 1),
                                    doc="Cycle efficiency")
    def rule_cycle_efficiency(b):
        return m.fs.cycle_efficiency == (b.net_power[0]/b.coal_heat_duty)
    m.fs.eq_cycle_efficiency = pyo.Constraint(rule=rule_cycle_efficiency,
                                              doc="Cycle efficiency")


def _create_arcs(m):
    """Create arcs"""

    @m.fs.Constraint()
    def constraint_flow_mol_to_hxc(b):
        return b.hxc.shell_inlet.flow_mol[0] == b.steam_to_storage[0]
    @m.fs.Constraint()
    def constraint_pressure_to_hxc(b):
        return b.hxc.shell_inlet.pressure[0] == m.charge_inlet_pressure
    @m.fs.Constraint()
    def constraint_enth_mol_to_hxc(b):
        return b.hxc.shell_inlet.enth_mol[0] == m.charge_inlet_enth_mol

    m.fs.hxc_to_hxpump = Arc(source=m.fs.hxc.shell_outlet,
                             destination=m.fs.hx_pump.inlet,
                             doc="Connection from solar charge heat exchanger to HX pump")

    @m.fs.Constraint()
    def constraint_flow_mol_to_hxd(b):
        return b.hxd.tube_inlet.flow_mol[0] == b.steam_to_discharge[0]
    @m.fs.Constraint()
    def constraint_pressure_to_hxd(b):
        return b.hxd.tube_inlet.pressure[0] == m.discharge_inlet_pressure
    @m.fs.Constraint()
    def constraint_enth_mol_to_hxd(b):
        return b.hxd.tube_inlet.enth_mol[0] == m.discharge_inlet_enth_mol

    m.fs.hxd_to_esturbine = Arc(source=m.fs.hxd.tube_outlet,
                                destination=m.fs.es_turbine.inlet,
                                doc="Connection from discharge heat exchanger to storage turbine")


def set_model_input(m):
    """Define model inputs and fixed variables or parameter values
    """

    # All the parameter values in this block, unless otherwise stated
    # explicitly, are either assumed or estimated for a total power
    # out of 437 MW

    # These inputs will also fix all necessary inputs to the model
    # i.e. the degrees of freedom = 0

    # Data from usc model
    m.fs.boiler_flow.fix(m.main_flow)  # mol/s
    m.fs.steam_to_storage.fix(m.main_flow*0.14) # mol/s
    m.fs.steam_to_discharge.fix(m.main_flow*0.1) # mol/s

    ###########################################################################
    #  Charge Heat Exchanger section                                          #
    ###########################################################################
    # Add heat exchanger area from supercritical plant model_input. For
    # conceptual design optimization, area is unfixed and optimized
    m.fs.hxc.area.fix(2000)  # m2
    m.fs.hxd.area.fix(1200)  # m2

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.hxc.tube_inlet.flow_mass.fix(200)   # kg/s
    m.fs.hxc.tube_inlet.temperature.fix(513.15)  # K
    m.fs.hxc.tube_inlet.pressure.fix(101325)  # Pa

    m.fs.hxd.shell_inlet.flow_mass[0].fix(240)  # 250
    m.fs.hxd.shell_inlet.temperature.fix(853.15)
    m.fs.hxd.shell_inlet.pressure.fix(101325)

    # HX pump efficiency assumption
    m.fs.hx_pump.efficiency_pump.fix(0.8)
    m.fs.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure*1.1231)

    # m.fs.es_turbine.ratioP.fix(0.0286)
    m.fs.es_turbine.efficiency_isentropic.fix(0.8)


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

    """

    # Include scaling factors for solar, hitec, and thermal oil charge
    # heat exchangers
    for fluid in [m.fs.hxc,
                  m.fs.hxd]:
        iscale.set_scaling_factor(fluid.area, 1e-2)
        iscale.set_scaling_factor(
            fluid.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(fluid.shell.heat, 1e-6)
        iscale.set_scaling_factor(fluid.tube.heat, 1e-6)

    iscale.set_scaling_factor(m.fs.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.es_turbine.control_volume.work, 1e-6)


def initialize(m,
               solver=None,
               outlvl=idaeslog.NOTSET,
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

    # Initialize feed flow in HXC and HXD using the constraints in
    # create_arcs
    calculate_variable_from_constraint(m.fs.hxc.shell_inlet.flow_mol[0],
                                       m.fs.constraint_flow_mol_to_hxc)
    calculate_variable_from_constraint(m.fs.hxc.shell_inlet.pressure[0],
                                       m.fs.constraint_pressure_to_hxc)
    calculate_variable_from_constraint(m.fs.hxc.shell_inlet.enth_mol[0],
                                       m.fs.constraint_enth_mol_to_hxc)

    calculate_variable_from_constraint(m.fs.hxd.tube_inlet.flow_mol[0],
                                       m.fs.constraint_flow_mol_to_hxd)
    calculate_variable_from_constraint(m.fs.hxd.tube_inlet.pressure[0],
                                       m.fs.constraint_pressure_to_hxd)
    calculate_variable_from_constraint(m.fs.hxd.tube_inlet.enth_mol[0],
                                       m.fs.constraint_enth_mol_to_hxd)

    m.fs.hxc.initialize(outlvl=outlvl,
                        optarg=solver.options)

    propagate_state(m.fs.hxc_to_hxpump)
    m.fs.hx_pump.initialize(outlvl=outlvl,
                            optarg=solver.options)

    m.fs.hxd.initialize(outlvl=outlvl,
                        optarg=solver.options)

    propagate_state(m.fs.hxd_to_esturbine)
    m.fs.es_turbine.initialize(outlvl=outlvl,
                               optarg=solver.options)

    # Check and raise an error if the degrees of freedom are not 0
    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building the model are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m))
            )

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)

    print("Initialization of Integrated Model = ",
          res.solver.termination_condition)
    print("***************   Initialized   ********************")


def build_costing(m,
                  solver=None,
                  optarg={"max_iter": 300}):
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

    # Calculate plant operating costs
    m.fs.operating_hours = pyo.Param(initialize=(365*3600*m.fs.hours_per_day),
                                     doc="Number of operating hours per year")
    m.fs.fuel_cost = pyo.Var(initialize=1e4,
                             bounds=(0, 1e6),
                             doc="Fuel (coal) cost in $/hour")
    m.fs.coal_heat_duty_W = pyo.units.convert(m.fs.coal_heat_duty,
                                              to_units=pyunits.W)
    def rule_fuel_cost(b):
        return b.fuel_cost == (
            b.operating_hours*b.coal_price*b.coal_heat_duty_W
        )/(365*24)
    m.fs.eq_fuel_cost = pyo.Constraint(rule=rule_fuel_cost)

    # Calculate annualized capital and operating costs for full plant
    # m.fs.plant_capital_cost = pyo.Var(initialize=1e4,
    #                                   bounds=(0, 1e6),
    #                                   doc="Plant capital cost in $/h")
    # def rule_plant_cap_cost(b):
    #     return b.plant_capital_cost*(365*24) == (
    #         (2688973*m.fs.plant_power_out[0]  # in MW
    #          + 618968072)/b.num_of_years
    #     )*(m.CE_index/575.4)
    # m.fs.eq_plant_cap_cost = pyo.Constraint(rule=rule_plant_cap_cost)

    m.fs.plant_operating_cost = pyo.Var(initialize=1e4,
                                        bounds=(0, 1e6),
                                        doc="Plant variable and fixed operating cost in $/hour")
    def rule_plant_op_cost(b):
        return b.plant_operating_cost == (
            (
                (16657.5*b.plant_power_out[0] + 6109833.3)/b.num_of_years + # fixed cost
                (31754.7*b.plant_power_out[0]) # variable cost
            )*(m.CE_index/575.4)
        )/(365*24)
    m.fs.eq_plant_op_cost = pyo.Constraint(rule=rule_plant_op_cost)

    return m


def initialize_with_costing(m, solver=None):

    # Initialize operating and capital costs
    calculate_variable_from_constraint(m.fs.fuel_cost,
                                       m.fs.eq_fuel_cost)

    # calculate_variable_from_constraint(m.fs.plant_capital_cost,
    #                                    m.fs.eq_plant_cap_cost)
    calculate_variable_from_constraint(m.fs.plant_operating_cost,
                                       m.fs.eq_plant_op_cost)

    res = solver.solve(m, tee=False, symbolic_solver_labels=True)
    print("Cost Initialization = ", res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print()


def calculate_bounds(m):
    m.fs.temperature_degrees = 5

    # Calculate bounds for solar salt from properties expressions
    m.fs.solar_salt_temperature_max = 853.15 + m.fs.temperature_degrees # in K
    m.fs.solar_salt_temperature_min = 513.15 - m.fs.temperature_degrees # in K
    # Note: min/max interchanged because at max temperature we obtain the min value
    m.fs.solar_salt_enth_mass_max = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.solar_salt_temperature_max - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value*0.5*\
           (m.fs.solar_salt_temperature_max - 273.15)**2)
    )
    m.fs.solar_salt_enth_mass_min = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.solar_salt_temperature_min - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value*0.5*\
           (m.fs.solar_salt_temperature_min - 273.15)**2)
    )

    m.fs.salt_enth_mass_max = m.fs.solar_salt_enth_mass_max
    m.fs.salt_enth_mass_min = m.fs.solar_salt_enth_mass_min


def add_bounds(m):
    """Add bounds to units in charge model

    """

    calculate_bounds(m)

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W
    m.heat_duty_max = (pyo.units.convert(m.max_storage_duty, to_units=pyunits.W))
    m.power_max_storage = (pyo.units.convert(-m.max_discharge_power, to_units=pyunits.W))
    m.factor = 2

    # Charge heat exchanger
    m.fs.hxc.shell_inlet.flow_mol.setlb(m.flow_min)
    m.fs.hxc.shell_inlet.flow_mol.setub(m.flow_max_storage)
    m.fs.hxc.tube_inlet.flow_mass.setlb(m.flow_min)
    m.fs.hxc.tube_inlet.flow_mass.setub(m.max_salt_flow)
    m.fs.hxc.shell_outlet.flow_mol.setlb(m.flow_min)
    m.fs.hxc.shell_outlet.flow_mol.setub(m.flow_max_storage)
    m.fs.hxc.tube_outlet.flow_mass.setlb(m.flow_min)
    m.fs.hxc.tube_outlet.flow_mass.setub(m.max_salt_flow)
    m.fs.hxc.tube_inlet.pressure.setlb(101320)
    m.fs.hxc.tube_inlet.pressure.setub(101330)
    m.fs.hxc.tube_outlet.pressure.setlb(101320)
    m.fs.hxc.tube_outlet.pressure.setub(101330)
    m.fs.hxc.heat_duty.setlb(0)
    m.fs.hxc.heat_duty.setub(m.heat_duty_max)
    m.fs.hxc.shell.heat.setlb(-m.heat_duty_max)
    m.fs.hxc.shell.heat.setub(0)
    m.fs.hxc.tube.heat.setlb(0)
    m.fs.hxc.tube.heat.setub(m.heat_duty_max)
    m.fs.hxc.tube.properties_in[:].enth_mass.setlb(m.fs.salt_enth_mass_min/m.factor)
    m.fs.hxc.tube.properties_in[:].enth_mass.setub(m.fs.salt_enth_mass_max*m.factor)
    m.fs.hxc.tube.properties_out[:].enth_mass.setlb(m.fs.salt_enth_mass_min/m.factor)
    m.fs.hxc.tube.properties_out[:].enth_mass.setub(m.fs.salt_enth_mass_max*m.factor)
    m.fs.hxc.overall_heat_transfer_coefficient.setlb(1)
    m.fs.hxc.overall_heat_transfer_coefficient.setub(10000)
    m.fs.hxc.area.setlb(m.min_area)
    m.fs.hxc.area.setub(m.max_area)
    m.fs.hxc.delta_temperature_in.setlb(9)
    m.fs.hxc.delta_temperature_out.setlb(5)
    m.fs.hxc.delta_temperature_in.setub(80.5)
    m.fs.hxc.delta_temperature_out.setub(81)

    # Discharge heat exchanger
    m.fs.hxd.tube_inlet.flow_mol.setlb(m.flow_min)
    m.fs.hxd.tube_inlet.flow_mol.setub(m.flow_max_storage)
    m.fs.hxd.shell_inlet.flow_mass.setlb(m.flow_min)
    m.fs.hxd.shell_inlet.flow_mass.setub(m.max_salt_flow)
    m.fs.hxd.tube_outlet.flow_mol.setlb(m.flow_min)
    m.fs.hxd.tube_outlet.flow_mol.setub(m.flow_max_storage)
    m.fs.hxd.shell_outlet.flow_mass.setlb(m.flow_min)
    m.fs.hxd.shell_outlet.flow_mass.setub(m.max_salt_flow)
    m.fs.hxd.shell_inlet.pressure.setlb(101320)
    m.fs.hxd.shell_inlet.pressure.setub(101330)
    m.fs.hxd.shell_outlet.pressure.setlb(101320)
    m.fs.hxd.shell_outlet.pressure.setub(101330)
    m.fs.hxd.heat_duty.setlb(0)
    m.fs.hxd.heat_duty.setub(m.heat_duty_max)
    m.fs.hxd.tube.heat.setub(m.heat_duty_max)
    m.fs.hxd.tube.heat.setlb(0)
    m.fs.hxd.shell.heat.setub(0)
    m.fs.hxd.shell.heat.setlb(-m.heat_duty_max)
    m.fs.hxd.shell.properties_in[:].enth_mass.setlb(m.fs.salt_enth_mass_min/m.factor)
    m.fs.hxd.shell.properties_in[:].enth_mass.setub(m.fs.salt_enth_mass_max*m.factor)
    m.fs.hxd.shell.properties_out[:].enth_mass.setlb(m.fs.salt_enth_mass_min/m.factor)
    m.fs.hxd.shell.properties_out[:].enth_mass.setub(m.fs.salt_enth_mass_max*m.factor)
    m.fs.hxd.overall_heat_transfer_coefficient.setlb(1)
    m.fs.hxd.overall_heat_transfer_coefficient.setub(10000)
    m.fs.hxd.area.setlb(m.min_area)
    m.fs.hxd.area.setub(m.max_area)
    m.fs.hxd.delta_temperature_in.setlb(4.9)
    m.fs.hxd.delta_temperature_out.setlb(10)
    m.fs.hxd.delta_temperature_in.setub(300)
    m.fs.hxd.delta_temperature_out.setub(300)

    # Add bounds for the HX pump
    for unit_k in [m.fs.hx_pump]:
        unit_k.inlet.flow_mol.setlb(m.flow_min)
        unit_k.inlet.flow_mol.setub(m.flow_max_storage)
        unit_k.outlet.flow_mol.setlb(m.flow_min)
        unit_k.outlet.flow_mol.setub(m.flow_max_storage)
        unit_k.control_volume.work[0].setlb(0)
        unit_k.control_volume.work[0].setub(1e12)

    for es_turb in [m.fs.es_turbine]:
        es_turb.inlet.flow_mol.setlb(m.flow_min)
        es_turb.inlet.flow_mol.setub(m.flow_max_storage)
        es_turb.outlet.flow_mol.setlb(m.flow_min)
        es_turb.outlet.flow_mol.setub(m.flow_max_storage)
        es_turb.control_volume.work[0].setlb(m.power_max_storage)
        es_turb.control_volume.work[0].setub(0)


def main(method=None,
         pmin=None,
         pmax=None,
         solver=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_integrated_model(method=method)

    # Give all the required inputs to the model
    set_model_input(m)

    # Add scaling factor
    set_scaling_factors(m)

    # Initialize the model with a sequential initialization and custom
    # routines
    initialize(m)

    # Add cost correlations
    m = build_costing(m)

    # Initialize with bounds
    initialize_with_costing(m, solver=solver)

    # Add bounds
    add_bounds(m)

    # Store initialization file
    ms.to_json(m, fname='initialized_usc_storage_nlp_mp_unfixed_area.json')
    print()
    print('>>>>> Saving initialization .json file')

    return m


def print_results(m, results):

    print('================= Optimization Results ===============')
    print('Variable                       Value')
    print(' Obj ($/h) {:>30.4f}'.format(value(m.obj)/scaling_obj))
    print(' Net Power (MW) {:>25.4f}'.format(value(m.fs.net_power[0])))
    print(' Plant Power (MW) {:>23.4f}'.format(value(m.fs.plant_power_out[0])))
    print(' ES turbine Power (MW) {:>18.4f}'.format(value(m.fs.es_turbine_MW)))
    print(' HX pump power (MW) {:>20.4f}'.format(value(m.fs.hx_pump_work_MW)))
    print(' Boiler water flow (mol/s) {:>14.4f}'.format(value(m.fs.boiler_flow[0])))
    print(' Plant heat duty (MWth) {:>17.4f}'.format(value(m.fs.plant_heat_duty[0])))
    print(' Fuel Cost ($/h) {:>26.6f}'.format(value(m.fs.fuel_cost)/(365*24)))
    print(' Revenue ($/h) {:>28.6f}'.format(value(m.fs.revenue)))
    print(' Steam(v) to charge (mol/s) {:>13.4f}'.format(value(m.fs.steam_to_storage[0])))
    print(' Steam(l) to discharge (mol/s) {:>10.4f}'.format(value(m.fs.steam_to_discharge[0])))
    print(' Salt amount (mton) {:>21.4f}'.format(value(m.fs.salt_amount)))
    print(' Salt Inventory')
    print('  (now) Hot [Cold] (mton) {:>15.4f} [{:>2.4f}]'.format(
        value(m.fs.salt_inventory_hot[0]),
        value(m.fs.salt_inventory_cold[0])))
    print('  (previous) Hot [Cold] (mton) {:>6.4f} [{:>2.4f}]'.format(
        value(m.fs.previous_salt_inventory_hot[0]),
        value(m.fs.previous_salt_inventory_cold[0])))
    print()
    print(' -------------------------------------------------------------')
    print('                                HXC             HXD')
    print('                                -----------------------------')
    print(' Area (m2) {:>30.4f} {:>15.4f}'.format(
        value(m.fs.hxc.area),
        value(m.fs.hxd.area)))
    print(' U {:>38.4f} {:>15.4f}'.format(
        value(m.fs.hxc.overall_heat_transfer_coefficient[0]),
        value(m.fs.hxd.overall_heat_transfer_coefficient[0])))
    print(' Heat duty (MW) {:>25.4f} {:>15.4f}'.format(
        value(m.fs.hxc.heat_duty[0])*1e-6,
        value(m.fs.hxd.heat_duty[0])*1e-6))
    print(' Salt flow in (kg/s) {:>20.4f} {:>15.4f}'.format(
        value(m.fs.hxc.tube_inlet.flow_mass[0]),
        value(m.fs.hxd.shell_inlet.flow_mass[0])))
    print(' Water flow to storage (mol/s) {:>10.4f} {:>15.4f}'.format(
        value(m.fs.hxc.shell_inlet.flow_mol[0]),
        value(m.fs.hxd.tube_inlet.flow_mol[0])))
    print(' Salt temperature in/out (K) {:>10.2f}/{:.2f} {:>8.2f}/{:.2f}'.format(
        value(m.fs.hxc.tube_inlet.temperature[0]),
        value(m.fs.hxc.tube_outlet.temperature[0]),
        value(m.fs.hxd.shell_inlet.temperature[0]),
        value(m.fs.hxd.shell_outlet.temperature[0])))
    print(' Water temperature in/out (K) {:>9.2f}/{:.2f} {:>8.2f}/{:.2f}'.format(
        value(m.fs.hxc.hot_side.properties_in[0].temperature),
        value(m.fs.hxc.hot_side.properties_out[0].temperature),
        value(m.fs.hxd.cold_side.properties_in[0].temperature),
        value(m.fs.hxd.cold_side.properties_out[0].temperature)))
    print(' Delta temperature in/out (K) {:>9.2f}/{:.2f} {:>9.2f}/{:.2f}'.format(
        value(m.fs.hxc.delta_temperature_in[0]),
        value(m.fs.hxc.delta_temperature_out[0]),
        value(m.fs.hxd.delta_temperature_in[0]),
        value(m.fs.hxd.delta_temperature_out[0])))
    print('==============================================================')
    print('')
    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')


def model_analysis(m,
                   solver=None,
                   power=None,
                   pmax=None,
                   pmin=None,
                   tank_status=None,
                   constant_salt=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    m.ramp_rate = design_data_dict["ramp_rate"]*pyunits.MW
    m.pmax_total = pmax + m.pmax_storage
    m.min_temp = design_data_dict["min_solar_salt_temperature"]*pyunits.K
    m.max_temp = design_data_dict["max_solar_salt_temperature"]*pyunits.K
    m.min_inventory = pyo.units.convert(75000*pyunits.kg,
                                        to_units=pyunits.metric_ton)
    m.tank_max = m.max_salt_amount # in mton
    m.tank_min = 1e-3*pyunits.metric_ton

    m.fs.plant_power_min = pyo.Constraint(expr=m.fs.plant_power_out[0] >= pmin)
    m.fs.plant_power_max = pyo.Constraint(expr=m.fs.plant_power_out[0] <= pmax)

    m.fs.hxc_heat_duty_MW = pyo.units.convert(m.fs.hxc.heat_duty[0],
                                              to_units=pyunits.MW)
    m.fs.hxd_heat_duty_MW = pyo.units.convert(m.fs.hxd.heat_duty[0],
                                              to_units=pyunits.MW)
    m.fs.charge_storage_lb_eq = pyo.Constraint(
        expr=m.fs.hxc_heat_duty_MW >= m.min_storage_duty)
    m.fs.charge_storage_ub_eq = pyo.Constraint(
        expr=m.fs.hxc_heat_duty_MW <= m.max_storage_duty)
    m.fs.discharge_storage_lb_eq = pyo.Constraint(
        expr=m.fs.hxd_heat_duty_MW >= m.min_storage_duty)
    m.fs.discharge_storage_ub_eq = pyo.Constraint(
        expr=m.fs.hxd_heat_duty_MW <= m.max_storage_duty*(1 - 0.01))

    # Fix variables in the flowsheet
    m.fs.boiler_flow.unfix()
    m.fs.steam_to_storage.unfix()
    m.fs.steam_to_discharge.unfix()

    m.fs.hx_pump.outlet.pressure[0].unfix()
    
    if not constant_salt:
        m.fs.salt_amount.unfix()

    # Unfix all data
    for salt_hxc in [m.fs.hxc]:
        salt_hxc.shell_inlet.unfix()
        salt_hxc.tube_inlet.flow_mass.unfix()
        salt_hxc.area.unfix()
        salt_hxc.tube_outlet.temperature[0].unfix()
        salt_hxc.tube_outlet.temperature.fix(853.15)

    for salt_hxd in [m.fs.hxd]:
        salt_hxd.tube_inlet.unfix()
        salt_hxd.shell_inlet.flow_mass.unfix()
        salt_hxd.area.unfix()
        salt_hxd.shell_inlet.temperature[0].unfix()

    # # Values to test model using the results from integrated model
    # # with rigorous power plant model
    # m.fs.hxc.area.fix(1877)
    # m.fs.hxd.area.fix(900)
    # m.fs.steam_to_storage.fix(3232)
    # m.fs.steam_to_discharge.fix(2005)

    # Fix storage heat exchangers area and salt temperatures
    m.cold_salt_temperature = design_data_dict["cold_salt_temperature"]*pyunits.K
    m.fs.hxd.shell_outlet.temperature.fix(m.cold_salt_temperature)

    m.fs.previous_power = pyo.Var(domain=NonNegativeReals,
                                  initialize=436,
                                  bounds=(pmin, m.pmax_total),
                                  units=pyunits.MW,
                                  doc="Previous period power")
    m.fs.previous_power.fix(447.66*pyunits.MW)

    @m.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (b.previous_power - m.ramp_rate) <= b.plant_power_out[0]

    @m.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (b.previous_power + m.ramp_rate) >= b.plant_power_out[0]

    # Add salt inventory mass balances
    m.fs.previous_salt_inventory_hot = pyo.Var(m.fs.time,
                                               domain=NonNegativeReals,
                                               initialize=m.min_inventory,
                                               bounds=(0, m.max_inventory),
                                               units=pyunits.metric_ton,
                                               doc="Hot salt at the beginning of the time period")
    m.fs.salt_inventory_hot = pyo.Var(m.fs.time,
                                      domain=NonNegativeReals,
                                      initialize=m.min_inventory,
                                      bounds=(0, m.max_inventory),
                                      units=pyunits.metric_ton,
                                      doc="Hot salt inventory at the end of the time period")
    m.fs.previous_salt_inventory_cold = pyo.Var(m.fs.time,
                                                domain=NonNegativeReals,
                                                initialize=m.tank_max - m.min_inventory,
                                                bounds=(0, m.max_inventory),
                                                units=pyunits.metric_ton,
                                                doc="Cold salt at the beginning of the time period")
    m.fs.salt_inventory_cold = pyo.Var(m.fs.time,
                                       domain=NonNegativeReals,
                                       initialize=m.tank_max - m.min_inventory,
                                       bounds=(0, m.max_inventory),
                                       units=pyunits.metric_ton,
                                       doc="Cold salt inventory at the end of the time period")
    # Fix the previous salt inventory based on the tank scenario
    if tank_status == "hot_empty":
        m.fs.tank_init = pyo.units.convert(1103053.48*pyunits.kg,
                                           to_units=pyunits.metric_ton)
        m.fs.previous_salt_inventory_hot.fix(m.fs.tank_init)
        m.fs.previous_salt_inventory_cold.fix(m.tank_max - m.fs.tank_init)
    elif tank_status == "hot_half_full":
        m.fs.previous_salt_inventory_hot[0].fix(m.tank_max/2)
        m.fs.previous_salt_inventory_cold[0].fix(m.tank_max/2)
    elif tank_status == "hot_full":
        m.fs.previous_salt_inventory_hot[0].fix(m.tank_max - m.tank_min)
        m.fs.previous_salt_inventory_cold[0].fix(m.tank_min)
    else:
        print('Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full')

    m.fs.hxc_flow_mass = pyo.units.convert(m.fs.hxc.tube_inlet.flow_mass[0],
                                           to_units=pyunits.metric_ton/pyunits.hour)
    m.fs.hxd_flow_mass = pyo.units.convert(m.fs.hxd.shell_inlet.flow_mass[0],
                                           to_units=pyunits.metric_ton/pyunits.hour)
    scaling_const = 1e-3
    @m.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            scaling_const*b.salt_inventory_hot[0] == (
                b.previous_salt_inventory_hot[0]
                + (b.hxc_flow_mass - b.hxd_flow_mass) # in mton/h
            )*scaling_const
        )

    @m.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return scaling_const*b.salt_amount == (b.salt_inventory_hot[0] +
                                               b.salt_inventory_cold[0])*scaling_const

    @m.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return b.hxd_flow_mass <= b.previous_salt_inventory_hot[0]

    @m.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return b.hxc_flow_mass <= b.previous_salt_inventory_cold[0]


    @m.fs.Constraint(doc="Salt temperature in discharge heat exchanger")
    def constraint_discharge_temperature(b):
        return b.hxd.shell_inlet.temperature[0] == b.hxc.tube_outlet.temperature[0]

    # Calculate revenue
    m.fs.revenue = pyo.Expression(
        expr=(m.fs.lmp*m.fs.net_power[0]),
        doc="Revenue function in $/h assuming 1 hr operation"
    )

    # Objective function: total costs
    m.obj = pyo.Objective(
        expr=-(
            m.fs.revenue -
            (m.fs.fuel_cost +
             m.fs.plant_operating_cost)
        )*scaling_obj
    )

    print('DOF before solve = ', degrees_of_freedom(m))

    # Solve the design optimization model
    results = solver.solve(m,
                           tee=True,
                           symbolic_solver_labels=True,
                           options={"linear_solver": "ma27",
                                    "max_iter": 300})

    print_results(m, results)

    log_close_to_bounds(m)
    log_infeasible_constraints(m)


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    pmin = design_data_dict["plant_min_power"]*pyunits.MW
    pmax_plant = design_data_dict["plant_max_power"]*pyunits.MW

    # Tank scenarios: "hot_empty", "hot_full", "hot_half_full"
    power_demand = 436*pyunits.MW
    method = "with_efficiency"
    tank_status = "hot_empty"
    constant_salt = True

    m_nlp = main(method=method,
                 pmin=pmin,
                 pmax=pmax_plant,
                 solver=solver)

    m_nlp.fs.lmp = pyo.Param(
        initialize=22,
        doc="Hourly LMP in $/MWh"
        )

    m = model_analysis(m_nlp,
                       solver,
                       power=power_demand,
                       pmax=pmax_plant,
                       pmin=pmin,
                       tank_status=tank_status,
                       constant_salt=constant_salt)
