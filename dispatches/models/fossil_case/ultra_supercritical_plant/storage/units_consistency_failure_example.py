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

"""This is a an example to demonstrate the issue regarding units consistency.

"""
# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_equivalent
from idaes.core.solvers.get_solver import get_solver

m = pyo.ConcreteModel(name="Test Model")

m.var_1 = pyo.Var(
    initialize=400,
    units=((pyunits.J**0.4) *
           (pyunits.kg**0.2) *
           (pyunits.W**0.6) /
           pyunits.K /
           (pyunits.m**2.2) /
           (pyunits.Pa**0.2) /
           (pyunits.s**0.8))
)
m.var_1.fix()
m.var_2 = pyo.Var(
    initialize=400,
    units=pyunits.kg/pyunits.s**3/pyunits.K
)
m.equality_constraint = pyo.Constraint(
    expr=m.var_1 == m.var_2)
m.obj = pyo.Objective(expr=m.var_2)

try:
    solver = pyo.SolverFactory("ipopt")
except:
    solver = get_solver('ipopt')
solver.solve(m)
assert_units_equivalent(m.var_1, m.var_2)
