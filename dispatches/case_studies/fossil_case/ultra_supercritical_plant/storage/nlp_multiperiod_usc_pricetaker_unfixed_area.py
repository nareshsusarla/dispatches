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

"""This script uses the IDAES multiperiod class to create a steady
state NLP multiperiod model for an integrated ultra-supercritical
power plant model. The purpose of this script is to create an NLP
multiperiod model that can be use for market analysis using a
pricetaker assumption. The integrated storage with ultra-supercritical
power plant model is used a steady state model for creating the
multiperiod model.

"""

# Import Python libraries
import json

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var)
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)

# Import IDAES libraries
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers.get_solver import get_solver


# Import integrated ultrasupercritical power plant model depending on
# the storage design. Note that when this should match the call
new_design = True
if new_design:
    print('>>>>> Solving for new storage design')
    import usc_storage_nlp_mp_unfixed_area_new_storage_design as usc
    # Add design data from .json file
    data_path = 'uscp_design_data_new_storage_design.json'
else:
    print('>>>>> Solving for original storage design')
    import usc_storage_nlp_mp_unfixed_area as usc
    # Add design data from .json file
    data_path = 'uscp_design_data.json'

with open(data_path) as design_data:
    design_data_dict = json.load(design_data)

pmax = design_data_dict["plant_max_power"]
pmin = design_data_dict["plant_min_power"]
pmax_storage = design_data_dict["max_discharge_turbine_power"]
min_area = design_data_dict["min_storage_area_design"]
max_area = design_data_dict["max_storage_area_design"]

def create_ss_model():

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Add data from .json file
    # min_storage_heat_duty = design_data_dict["min_storage_heat_duty"]
    min_storage_heat_duty = design_data_dict["min_storage_heat_duty"]
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"]
    factor_mton = design_data_dict["factor_mton"]
    max_salt_amount = design_data_dict["max_salt_amount"] * factor_mton
    max_salt_flow = design_data_dict["max_salt_flow"] # in kg/s

    # Add options needed in the integrated model
    method = "with_efficiency" # adds boiler and cycle efficiencies
    # load_from_file = 'initialized_usc_storage_nlp_mp.json'
    load_from_file = None

    m = pyo.ConcreteModel()
    m.usc = usc.main(method=method,
                     pmax=pmax,
                     load_from_file=load_from_file,
                     solver=solver,
                     max_salt_amount=max_salt_amount,
                     max_storage_heat_duty=max_storage_heat_duty,
                     min_area=min_area,
                     max_area=max_area,
                     max_salt_flow=max_salt_flow)

    # Set bounds for power produced by the plant without storage
    m.usc.fs.plant_min_power_eq = pyo.Constraint(
        expr=m.usc.fs.plant_power_out[0] >= pmin
    )
    m.usc.fs.plant_max_power_eq = pyo.Constraint(
        expr=m.usc.fs.plant_power_out[0] <= pmax
    )

    # Set lower and upper bounds to charge and discharge heat
    # exchangers
    hxc_heat_duty = (1e-6) * (pyunits.MW / pyunits.W) * m.usc.fs.hxc.heat_duty[0]
    hxd_heat_duty = (1e-6) * (pyunits.MW / pyunits.W) * m.usc.fs.hxd.heat_duty[0]
    m.usc.fs.charge_storage_lb_eq = pyo.Constraint(
        expr=hxc_heat_duty >= min_storage_heat_duty
    )
    m.usc.fs.discharge_storage_lb_eq = pyo.Constraint(
        expr=hxd_heat_duty >= min_storage_heat_duty
    )
    m.usc.fs.charge_storage_ub_eq = pyo.Constraint(
        expr=hxc_heat_duty <= max_storage_heat_duty
    )
    m.usc.fs.discharge_storage_ub_eq = pyo.Constraint(
        expr=hxd_heat_duty <= max_storage_heat_duty * (1 - 0.01)
    )

    # Unfix boiler flow fixed during during initialization
    m.usc.fs.boiler.inlet.flow_mol[0].unfix()

    # Unfix storage system data
    m.usc.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.usc.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for charge_hxc in [m.usc.fs.hxc]:
        charge_hxc.inlet_1.unfix()
        charge_hxc.inlet_2.flow_mass.unfix()
        charge_hxc.area.unfix()
        charge_hxc.outlet_2.temperature[0].unfix()

    for discharge_hxd in [m.usc.fs.hxd]:
        discharge_hxd.inlet_2.unfix()
        discharge_hxd.inlet_1.flow_mass.unfix()
        discharge_hxd.area.unfix()
        discharge_hxd.inlet_1.temperature[0].unfix()

    if not new_design:
        for unit in [m.usc.fs.cooler]:
            unit.inlet.unfix()
        m.usc.fs.cooler.outlet.enth_mol[0].unfix()

    # Fix storage heat exchangers area and salt temperatures
    cold_salt_temperature = design_data_dict["cold_salt_temperature"]
    m.usc.fs.hxd.outlet_1.temperature[0].fix(cold_salt_temperature)

    return m


def create_mp_block():

    print('>>> Creating USC model and initialization for each time period')

    m = create_ss_model()
    b1 = m.usc

    # print('DOFs within mp create 1 =', degrees_of_freedom(m))

    # Add data from .json file
    ramp_rate = design_data_dict["ramp_rate"]
    # min_area = design_data_dict["min_storage_area"]
    # min_area = design_data_dict["min_storage_area_design"]
    # min_area = 100
    # max_area = design_data_dict["max_storage_area"]
    # max_area = design_data_dict["max_storage_area_design"]
    pmax_total = pmax + pmax_storage
    factor_mton = design_data_dict["factor_mton"]

    # Add coupling variables
    b1.previous_power = pyo.Var(
        domain=NonNegativeReals,
        initialize=400,
        bounds=(pmin, pmax_total),
        doc="Previous period power in MW"
        )

    inventory_max = 1e7 * factor_mton # in mton
    b1.previous_salt_inventory_hot = pyo.Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, inventory_max),
        doc="Hot salt at the beginning of the time period in mton"
        )
    b1.salt_inventory_hot = pyo.Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, inventory_max),
        doc="Hot salt inventory at the end of the time period in mton"
        )
    b1.previous_salt_inventory_cold = pyo.Var(
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, inventory_max),
        doc="Cold salt at the beginning of the time period in mton"
        )
    b1.salt_inventory_cold = pyo.Var(
        domain=NonNegativeReals,
        initialize=80,
        bounds=(0, inventory_max),
        doc="Cold salt inventory at the end of the time period in mton"
        )

    @b1.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            b1.previous_power - ramp_rate <=
            b.plant_power_out[0])

    @b1.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            b1.previous_power + ramp_rate >=
            b.plant_power_out[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            1e-3 * b1.salt_inventory_hot == (
                b1.previous_salt_inventory_hot
                + (3600 * b.hxc.inlet_2.flow_mass[0]
                   - 3600 * b.hxd.inlet_1.flow_mass[0]) * factor_mton # in mton
            ) * 1e-3
        )

    @b1.fs.Constraint(doc="Maximum salt inventory at any time in mton")
    def constraint_salt_inventory(b):
        return (
            1e-3 * b.salt_amount == (
                b1.salt_inventory_hot
                + b1.salt_inventory_cold
            ) * 1e-3
        )

    # Add area coupling variables
    b1.previous_charge_area = pyo.Var(
        domain=NonNegativeReals,
        initialize=1900,
        bounds=(min_area, max_area),
        doc="Previous area of charge heat exchanger in m2"
        )
    b1.previous_discharge_area = pyo.Var(
        domain=NonNegativeReals,
        initialize=1000,
        bounds=(min_area, max_area),
        doc="Previous area  of discharge heat exchanger in m2"
        )

    @b1.fs.Constraint(doc="Charge heat exchanger area constraint")
    def constraint_charge_area(b):
        return b1.previous_charge_area == b.hxc.area

    @b1.fs.Constraint(doc="Discharge heat exchanger area constraint")
    def constraint_discharge_area(b):
        return b1.previous_discharge_area == b.hxd.area

    # Add charge and discharge salt temperature
    min_temp = design_data_dict["min_solar_salt_temperature"]
    max_temp = design_data_dict["max_solar_salt_temperature"]
    b1.previous_charge_temperature = pyo.Var(
        domain=NonNegativeReals,
        initialize=design_data_dict["hot_salt_temperature"],
        bounds=(min_temp, max_temp),
        doc="Previous salt temperature in K"
        )

    @b1.fs.Constraint(doc="Salt temperature in charge heat exchanger")
    def constraint_charge_temperature(b):
        return b1.previous_charge_temperature == b.hxc.outlet_2.temperature[0]

    @b1.fs.Constraint(doc="Salt temperature in discharge heat exchanger")
    def constraint_discharge_temperature(b):
        return b.hxd.inlet_1.temperature[0] == b1.previous_charge_temperature


    return m


# The tank level and power output are linked between contiguous time
# periods
def get_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [
        (b1.usc.salt_inventory_hot, b2.usc.previous_salt_inventory_hot),
        (b1.usc.fs.plant_power_out[0], b2.usc.previous_power),
        (b1.usc.fs.hxc.area, b2.usc.previous_charge_area),
        (b1.usc.fs.hxd.area, b2.usc.previous_discharge_area),
        (b1.usc.fs.hxc.outlet_2.temperature[0], b2.usc.previous_charge_temperature)
    ]

# The tank level at the end of the last period must be the same as the
# level at the beginning of the first period and power output must be
# the same as the initial tank level.
def get_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [
        (b1.usc.salt_inventory_hot, b2.usc.previous_salt_inventory_hot),
    ]


# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.
def create_nlp_multiperiod_usc_model(n_time_points=None, pmin=None, pmax=None):
    """Create a multiperiod usc_mp cycle object. This object contains a
    Pyomo model with a block for each time instance

    n_time_points: Number of time blocks to create

    """

    multiperiod_usc = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=create_mp_block,
        linking_variable_func=get_link_variable_pairs,
        periodic_variable_func=get_periodic_variable_pairs
    )

    # If you have no arguments, you don't actually need to pass in
    # anything. NOTE: building the model will initialize each time block
    multiperiod_usc.build_multi_period_model()

    return multiperiod_usc
