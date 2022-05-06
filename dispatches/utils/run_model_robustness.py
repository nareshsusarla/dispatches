import numpy as np
import pandas as pd
import os
from idaes.core.util import model_serializer as ms
import datetime
import idaes.core.util.convergence.convergence_base as cb
import matplotlib.pyplot as plt
from data_analysis_utils import *
from simulation_object import *

current_date = str(datetime.datetime.now().date())

def solve_and_record(m,\
                     solver,\
                     input_dict,\
                     simulation_object,\
                     previous_sln_condition = 'optimal', \
                     previous_optimal_json_file = None,\
                     json_file_dir = None):

    '''
    This function takes an IDAES flowsheet model and tries to solve it. First,
    the function loads the input/set point into the model and tries to solve it
    directly and/or use homotopy meta-solver. If not optimal, the function
    rebuilds the model and solve. After solving the model, the function saves the
    result into pandas DataFrame and return.

    Arguments:
        m: IDAES power plant model
        solver: Pyomo solver object, usually IPOPT
        input_dict: a dictionary contains the inputs to the model, e.g. load of
        plant
        simulation_object: an object that has necessary methods for the simulation
        previous_sln_condition: the solve exit condition from last solve
        previous_optimal_json_file: the directory to the previous optimal model
        json file
        json_file_dir: which folder to save the json files

    Returns:
        result_df: pandas DataFrame that saves the results
        ipopt_exit_condition: ipopt exit condition for this solve
        previous_optimal_json_file: previous optimal json file
    '''

    simulation_object.set_input(m,input_dict)

    try:
        r1, r2, r3, r4 = cb._run_ipopt_with_stats(m, solver)
        homotopy_exit_condition = None
        ipopt_exit_condition = str(r1.solver.termination_condition)
        ipopt_solver_message = str(r1.solver.message)
        ipopt_iteration_count = r3
        ipopt_solution_time = r4
        ipopt_number_variables = r1.problem.number_of_variables
        ipopt_number_constraints = r1.problem.number_of_constraints
        
    except:
        ipopt_exit_condition = 'Failed'
        ipopt_solver_message = 'Failed'
        r3 = 0
        r4 = 0
        ipopt_number_variables = 0
        ipopt_number_constraints = 0
    # save the model condition to json
    if json_file_dir is None:
        json_file_dir = 'model_json_{}/'.format(current_date)
        if not os.path.isdir(json_file_dir):
            os.mkdir(json_file_dir)

    file_name = simulation_object.assemble_json_file_name(json_file_dir,\
                                                          input_dict, \
                                                          optimal = False)
    ms.to_json(m,fname = file_name,gz = True, human_read = True)

    # record the results in a pandas dataframe
    result_df = simulation_object.record_results(m,\
                                     input_dict = input_dict,\
                                     ipopt_exit_condition = ipopt_exit_condition,\
                                     ipopt_solver_message = ipopt_solver_message, \
                                     ipopt_iteration_count = r3,
                                     ipopt_solution_time = r4,
                                     ipopt_number_variables = ipopt_number_variables, \
                                     ipopt_number_constraints = ipopt_number_constraints, \
                                     json_file_dir = file_name)

    return result_df,ipopt_exit_condition,previous_optimal_json_file


def plot_results():

    # TODO: user input
    da = FlowsheetData(
        data_dir='robustness_data/robustness_2var_{}.csv'.format(current_date))

    fig, ax = plt.subplots(figsize = (12,10))
    cbar = da.plot_efficiency_contour(fig = fig, ax = ax)
    ax.set_xlabel('Flowrate [mol/s]',fontsize = 20)
    ax.set_ylabel('Pressure [Pa]',fontsize = 20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    cbar.ax.tick_params(labelsize = 20)
    cbar.set_label('Net Power [MW]',size = 20)
    plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1), fontsize = 20, ncol = 2)
    plt.savefig('robustness_contour_plot')


def convergence_tester():
    # TODO: user input
    flowrate_range = np.flip(np.around(np.linspace(15000, 20000, 2), 2))
    pressure_range = np.flip(np.array([32., 34.5]))*1e6
    
    # initialize simulation object
    robustness_sim_obj = robustness_simulation()
    
    # set up directories to store results
    json_file_dir = os.path.join('robustness_2var_json_{}'.format(current_date))
    if not os.path.isdir(json_file_dir):
        os.mkdir(json_file_dir)
    
    result_dir = os.path.join('robustness_data')
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    result_file_name = 'robustness_2var_{}.csv'.format(current_date)
    
    # declare a list to store the result df
    df_list = []
    
    for pres_idx in range(len(pressure_range)):
    
        for flow_idx in range(len(flowrate_range)):
    
            # build the model
            # TODO: user input
            m,solver = robustness_sim_obj.build_model()
            input_dict = {'Boiler Flowrate Set Pt [mol per s]': flowrate_range[flow_idx],\
                          'Boiler Pressure Set Pt [Pa]':pressure_range[pres_idx]}
    
            print("")
            print("The flowrate is set to be {} mol/s".format(flowrate_range[flow_idx]))
            print("The pressure is set to be {0:.1f} Pa".format(pressure_range[pres_idx]))
            print("")
    
            result_df,\
            ipopt_exit_condition,\
            previous_optimal_json_file = solve_and_record(m,\
                                                          solver,\
                                                          # jcn,\
                                                          input_dict = input_dict,\
                                                          simulation_object = robustness_sim_obj,\
                                                          previous_sln_condition = 'optimal', \
                                                          previous_optimal_json_file = None,\
                                                          json_file_dir = json_file_dir)
            df_list.append(result_df)
    
    pd.concat(df_list).to_csv(os.path.join(result_dir,result_file_name),index = False)

if __name__ == "__main__":

    convergence_tester()
    plot_results()
