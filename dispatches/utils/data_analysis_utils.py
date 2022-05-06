import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from os.path import isfile, join

class data_analysis:
    '''
    This class reads result data from the flowsheet model and manipulate the
    data to plot.
    '''

    def __init__(self,data_dir):
        '''
        This init function reads all the data from a study and store the data as
        class properties.

        Arguments:
            data_dir: the directory to data file [str]
        '''

        self.df = pd.read_csv(data_dir)

    def read_data_to_dict(self, dof_name_list):
        '''
        This function reads all the result files and store output vars
        into dictionaries.

        Arguments:
            dof_name_list: a list of DOF names (str) in the corresponding model
        Returns:
            var_dict: key: (dof1, dof2 ....)
                        value: power output in [MW]
            solver_condition_dict: key: (dof1, dof2 ....)
                             value: solver exit conditon [str]
        '''

        # TODO: user input
        power_dict = {}
        solver_condition_dict = {}

        # loop thru the rows in the data
        for index, row in self.df.iterrows():

            # using dof to assemble dict keys
            key = tuple((round(float(row[dof_name]),2) for dof_name in dof_name_list))

            power_dict[key] = round(float(row['Net Power [MW]']),2)
            solver_condition_dict[key] = str(row['IPOPT Exit Condition'])
        return power_dict, solver_condition_dict

    @staticmethod
    def plot_contour(fig, ax, dof1_tuple, dof2_tuple, result_tuple):
        '''
        This function makes contour plots.

        Arguments:
            fig: a matplotlib figure object
            ax: a matplotlib axes object
            dof1_tuple: (dof1 name [str],dof1 range [list/np arr])
            dof2_tuple: (dof2 name [str],dof2 range [list/np arr])
            result_tuple: (result name [str],result dict [dict])

            result dict: keys: (dof1_value,dof2_value) values: corresponding output

        Returns:
            None
        '''

        dof1_name = dof1_tuple[0]
        dof2_name = dof2_tuple[0]
        result_name = result_tuple[0]

        dof1_list = dof1_tuple[1]
        dof2_list = dof2_tuple[1]
        result = result_tuple[1]

        # create mesh grid
        [X,Y] = np.meshgrid(dof1_list,dof2_list)

        # assemble z arr
        Z = np.zeros((len(dof2_list),len(dof1_list)))

        for i in range(len(dof2_list)):
            for j in range(len(dof1_list)):
                try:
                    Z[i, j] = result[(dof1_list[j], dof2_list[i])]
                except KeyError:
                    Z[i,j] = np.nan

        # plot the contour
        cset = ax.contourf(X, Y, Z, alpha = 0.9, cmap = cm.get_cmap('Blues'))
        contour = ax.contour(X, Y, Z, colors = 'black')

        ax.clabel(contour,fontsize=15,colors='k',fmt='%.1f')
        cbar = fig.colorbar(cset)
        cbar.ax.set_ylabel(result_name)
        ax.set_xlabel(dof1_name)
        ax.set_ylabel(dof2_name)

        return cbar

    @staticmethod
    def scatter_non_optimal_points(ax,solver_condition_dict):

        '''
        This function marks the non-optimal points on the contours.

        Arguemnts:
            ax: a matplotlib axes object
            solver_condition_dict: a dict that stores the solver exit condtions
            as its values and input to the simulation as the key.

        Returns:
            None
        '''

        max_iter_x = []
        max_iter_y = []

        failed_x = []
        failed_y = []

        # record the coordinates of the points
        for key in solver_condition_dict:

            if solver_condition_dict[key] == 'maxIterations':
                max_iter_x.append(key[0])
                max_iter_y.append(key[1])

            elif solver_condition_dict[key] == 'Failed':
                failed_x.append(key[0])
                failed_y.append(key[1])

        # scatter the points
        ax.scatter(max_iter_x,max_iter_y, s = 300, marker = 'o',label = 'MaxIter',color = 'black')
        ax.scatter(failed_x,failed_y, s = 300, marker = '^', label = 'Failed',color = 'red')

        return

    @staticmethod
    def plot_time_series(ax,data,dof_name,marker = 'o',time = None):

        '''
        This function plots given degrees of freedom as time series.

        Arguments:
            ax: the axes for the plot
            data: the time series being plotted
            dof_name: the name of degree of freedom. This should match the name
                      recorded in the results [str]
            marker: the marker of choice [str]

        Returns:
            ax
        '''

        if time is None:
            time = range(len(data))
        dof = data

        ax.plot(time, dof,linestyle = '-', marker = marker,label = dof_name)

        # set the locator
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))

        # set labels
        ax.set_xlabel('Time [Hr]',fontsize = 20)
        ax.set_ylabel(dof_name,fontsize = 20)

        ax.tick_params(labelsize= 20)
        ax.legend(fontsize = 20)
        ax.grid(True)

        return ax


class FlowsheetData(data_analysis):

    def __init__(self, data_dir, flow_rate_range=None, pressure_range=None):
        '''
        This init function reads all the data from a study and store the data
        as class properties.

        Arguments:
            data_dir: the directory to the folder that stores the data [str]
            flow_rate_range: the boiler inlet flow rate (mol/s) range used in
            the study. [list]
            pressure_range: the boiler outlet pressure range in MPa used in the
            study. [list]
        '''

        # inherit from basic data class
        super().__init__(data_dir)

        # TODO: user input
        self.power_dict, self.solver_condition_dict = (
            self.read_data_to_dict(
            dof_name_list=['Boiler Flowrate Set Pt [mol per s]',
                           'Boiler Pressure Set Pt [Pa]']))

        self.get_input_range()

    def get_input_range(self, input_dict=None):
        '''
        This function gets the range of each degrees of freedom.

        Arguemnts:
            input_dict: a dictionary whose keys are the 'range name' and values
            are the corresponding names in the DataFrames.
        Returns:
            None
        '''

        # TODO: user input
        if input_dict is None:
            input_dict = {
                'flowrate_range': ['Boiler Flowrate Set Pt [mol per s]'],
                'pressure_range': ['Boiler Pressure Set Pt [Pa]'],
                'net_power_range': ['Net Power [MW]']}

        for var_name in input_dict:

            try:
                value = np.around(
                    np.unique(
                        self.df[input_dict[var_name]].mean(axis=1).values), 2)
            except:
                print('Missing data for ' + var_name)
            else:
                setattr(self,var_name,value)

        return

    def plot_efficiency_contour(self, fig, ax):
        '''
        This function plots the contour of efficiency.

        Arguments:
            fig: a matplotlib figure object
            ax: a matplotlib axes object

        Returns:
            None
        '''

        # TODO: user input
        cbar = self.plot_contour(
            fig,
            ax,
            dof1_tuple=('Boiler Flowrate Set Pt [mol per s]', self.flowrate_range),
            dof2_tuple=('Boiler Pressure Set Pt [Pa]', self.pressure_range),
            result_tuple=('Net Power [MW]', self.power_dict))

        self.scatter_non_optimal_points(ax, self.solver_condition_dict)

        return cbar
