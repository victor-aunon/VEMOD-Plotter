# -*- coding: utf-8 -*-
"""
This file is part of VEMOD plotter
"""

# Imports
from collections import OrderedDict


class Variable:
    """A class that defines an engine variable

    Attributes
    ----------
    name : str
        The name of the variable
    model_col : str
        The column name to look for in the model results
    exp_col : str or None
        The column name to look for in the experimental data
    conv_factor : float
        The conversion factor to pass from experimental units to model ones
    units: str
        The physical units of the variable
    limits: tuple or None
        The graphic limits of the variable
    values: dict
        The experimental and/or model values of the variable. The case names
        are the keys of this dictionary
    plot_trends: bool
        Plot or not trend figures (model vs experiment)
    time_evolution: bool
        Plot or not time evolution figures. Only for steady-state
    is_pollutant: bool
        True if the variable represents a pollutant emission
    extra: dict or None
        A dictionary with specific attributes of the class, e.g., tau,
        transient cycle divisions

    Methods
    -------
    set_exp_column(col)
        Returns None if col is '-'

    set_values(values_dict)
        Stores the variable's numeric values

    get_values()
        Returns the variable's numeric values in a formatted dictionary style
    """

    def __init__(self, variable_dict):
        """
        Parameters
        ----------
        variable_dict: dict, required
            A dictionary with the initialization attributes of the variable
        """

        self.name = variable_dict['name']
        self.model_col = variable_dict['model_col']

        self.plot_trends = variable_dict['plot_trends']
        self.time_evolution = variable_dict['plot_time_evolution']
        self.is_pollutant = variable_dict['is_pollutant']

        self.exp_col = self.set_exp_column(variable_dict['exp_col']) if\
            'exp_col' in variable_dict.keys() else None
        self.conv_factor = variable_dict['conv_factor'] if 'conv_factor' in\
            variable_dict.keys() else 1
        self.units = variable_dict['units'] if 'units' in\
            variable_dict.keys() else '-'
        self.limits = variable_dict['limits'] if 'limits' in\
            variable_dict.keys() else None
        self.extra = variable_dict['extra'] if 'extra' in\
            variable_dict.keys() else None
        self.values = OrderedDict()

    def set_exp_column(self, col):
        """Return None if col is '-'"""

        return col if col != '-' else None

    def set_values(self, values_dict):
        """Set variable values

        Parameters
        ----------
        values_dict: dict, required
            A dictionary with experimental and/or model values
            - keys: case names
            - values: dictionaries with model and experimental values
        """

        for key, vals in values_dict.items():
            self.values[key] = vals

    def get_values(self) -> str:
        """Get variable formatted values

        Returns
        -------
        str: the variable name and its values for each case
        """

        res = "{} [{}]\n".format(self.name, self.units)
        for k, v in self.values.items():
            res = res + "{}: {}\n".format(k, v).replace('{', '').replace(
                '}', '').replace("'", '')
        return res
