import pandas as pd
import numpy as np
import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.homotopy as homotopy
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
import os


class flowsheet_simulation_object():

    def __init__(self):
        pass

    @staticmethod
    def fix_dof(m, flowrate=17384, pressure=34e6, at_current_value=False):
        '''
        Fix all the degrees of freedom at given values.

        Arguments:
            m: the power plant model.
            flowrate: fix the inlet float rate [float]
            pressure: fix the inlet pressure at this value [float]
        '''
        # TODO: user input
        if at_current_value:
            m.fs.boiler.inlet.flow_mol.fix()
            m.fs.boiler.outlet.pressure.fix()
        else:
            m.fs.boiler.inlet.flow_mol.fix(flowrate)
            m.fs.boiler.outlet.pressure.fix(pressure)

        return

    def record_results_for_all_horizon(self,\
                                       m,\
                                       load_profile,
                                       input_dict = None,
                                       ipopt_exit_condition = None,\
                                       homotopy_exit_condition = None):

        '''
        This function records the results for one time step in NMPC whose horizon is
        the length of 'load_profile'.

        Arguments:

            m: the big pyomo model for NMPC load tracking
            load_profile: the loads we try to track [list]
            input_dict: input data we want to record in addition to the information
                        in the models, e.g. load profiles. [dict]
            ipopt_exit_condition: IPOPT exit condition
            homotopy_exit_condition: IDAES homotopy meta-solver exit condition if
                                     available.

        Returns:
            a pandas DataFrame that contains the results from each instances in the
            planning horizon. Each instance is a row in the DataFrame.

        '''
        df_list = []
        #for idx, sub_m in enumerate(m.sub_model_list):
        for idx, load in enumerate(load_profile):

            t = m.load_time_dict[load]

            df = self.record_results(m.ssmodel[t],\
                                input_dict = input_dict,\
                                ipopt_exit_condition = ipopt_exit_condition,\
                                homotopy_exit_condition = homotopy_exit_condition)

            df['Horizon'] = idx + 1

            df_list.append(df)

        return pd.concat(df_list)

    # function to record results
    def record_results(self,\
                       m,\
                       input_dict = None,
                       ipopt_exit_condition = None,\
                       ipopt_solver_message = None, \
                       ipopt_iteration_count = None,
                       ipopt_solution_time = None,
                       ipopt_number_variables = None, \
                       ipopt_number_constraints = None, \
                       **kwargs):

        '''
        This function records the following information into a pandas DataFrame from
        one instance of the power plant model:
            1. Flow rate
            2. Inlet Pressure
            3. Power output
            4. Steam Cycle Efficiency
            5. Boiler Heat
            7. IPOPT Exit Condition
            8. Homotopy Solver Exit Condition
        And additional information in the 'input_dict' and **kwargs.

        Arguments:
            result_df: a pandas DataFrame that has the above rows

        Returns:
            a 1-row pandas DataFrame that contains the results from one model instance.
        '''

        result_dict = {}

        for input_name in input_dict:
            result_dict[input_name] = input_dict[input_name]

        # record results
        # TODO: user input
        result_dict['Boiler Rate [mol/s]'] = pyo.value(m.fs.boiler.inlet.flow_mol[0])
        result_dict['Boiler Pressure [Pa]'] = pyo.value(m.fs.boiler.outlet.pressure[0])
        result_dict['Net Power [MW]'] = pyo.value(m.fs.plant_power_out[0])
        result_dict['IPOPT Exit Condition'] = ipopt_exit_condition
        result_dict['IPOPT Solver Message'] = ipopt_solver_message
        result_dict['IPOPT Iteration Count'] = ipopt_iteration_count
        result_dict['IPOPT Solution Time'] = ipopt_solution_time
        result_dict['Number of Variables'] = ipopt_number_variables
        result_dict['Number of Constraints'] = ipopt_number_constraints

        for k,v in kwargs.items():
            result_dict[k] = v

        result_df = pd.DataFrame.from_dict(result_dict,orient = 'index')

        return result_df.T


class robustness_simulation(flowsheet_simulation_object):

    '''
    This class contains the necessary methods for minimizing-cost simulation.
    '''

    def __init__(self):
        # inherit from network data class
        super().__init__()

        self.square_problem = True

    @staticmethod
    def build_model():
        '''
        This method builds a IDAES model that solves square problems. First it
        initialize the supercritical power plant.

        Arguments:
            None
        Returns:
            m: the built IDAES power plant model
            solver: an IPOPT solver object
        '''

        # if direct solve failed, use homotopy
        # TODO: user input
        m = usc.build_plant_model()
        usc.initialize(m)

        # specifiy solver options
        solver = pyo.SolverFactory("ipopt")
        solver.options = {
            "max_iter": 300,
            "halt_on_ampl_error": "yes",
            'bound_push': 1e-23,
            "mu_init": 1e-6,
            }

        return m, solver

    def set_input(self,model,input_dict):
        '''
        This function sets the input to the model.

        Arguments:
            model: the IDAES power plant model
            input_dict: a dictionary that stores the inputs to the model.
        Returns
            None
        '''

        # TODO: user input
        self.fix_dof(m=model,
                     flowrate=input_dict['Boiler Flowrate Set Pt [mol per s]'],
                     pressure=input_dict['Boiler Pressure Set Pt [Pa]'])

        return

    @staticmethod
    def assemble_json_file_name(json_file_dir,\
                                input_dict, \
                                optimal = True):
        '''
        This function assemble the json file name using the input we had for the
        model.

        Arguments:
            json_file_dir: a folder directory to store the json files
            input_dict: the input dict we used
            optimal: True if an optimal solution was reached
        Returns:
            the resulted json file name
        '''

        json_file_name = ''
        for input_name in input_dict:
            json_file_name += input_name.replace(" ","")
            json_file_name += '_' + str(round(input_dict[input_name],2)) + '_'

        if optimal:
            json_file_name += 'optimal'
        else:
            json_file_name += 'non-optimal'

        json_file_name += '.json.gz'

        return os.path.join(json_file_dir,json_file_name)

    def try_homotopy(self,m,input_dict,solver):

        '''
        If the direct solve did not reaches an optimal solution, this method will
        be called to use the homotopy meta-solver.

        Arguments:
            model: the IDAES power plant model
            input_dict: a dictionary that stores the inputs to the model.
        Returns:
            None
        '''

        # if direct solve failed, use homotopy
        # TODO: user input
        self.fix_dof(m ,  #= model
                    flowrate = 17384,\
                    pressure = 34e6)
        solver.solve(m,tee = True)

        # use homotopy solver
        print("")
        print("################################")
        print("Using homotopy to initialize!!!!")
        print("################################")
        print("")

        # list of variables
        # TODO: user inputs
        var_list = [m.fs.boiler.inlet.flow_mol[0],m.fs.boiler.outlet.pressure[0]]

        # list of var targets
        # TODO: user inputs
        flowrate = input_dict['Boiler Flowrate Set Pt [mol per s]'],\
        pressure = input_dict['Boiler Pressure Set Pt [Pa]']
        target_list = [flowrate, pressure]

        try:
            termination, \
            solver_prog, \
            num_iter = homotopy(model = m, variables = var_list, targets = target_list, \
                                  max_solver_iterations=100, max_solver_time=100, step_init=0.1,\
                                  step_cut=0.5, iter_target=2, step_accel=0.5, max_step=1, \
                                  min_step=0.05, max_eval=200)
        except:
            homotopy_exit_condition = 'Failed'
        else:
            homotopy_exit_condition = str(termination)

        return homotopy_exit_condition

