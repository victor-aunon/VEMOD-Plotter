# -*- coding: utf-8 -*-
"""
This file is part of VEMOD plotter
"""

# Imports
import math
import logging as log
from collections import OrderedDict

# Libs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage.filters import gaussian_filter

# Own modules
from .stats import resize

# Globals
mpl.use('Agg')
golden_ratio = (1 + np.sqrt(5)) / 2.
markers = ('v', 'd', 's', 'o', '^', 'p', 'X', '*', 'P', 'H', 'D', '<', '>',
           'h', '8', '1', '2', '3', '4')
colors = ('#3742fa', '#2ed573', '#ff4757', '#ffa502', '#70A1FF', '#747d8c',
          '#5352ed', '#7bed9f', '#ff6b81', '#a4b0be', '#ff7f50', '#2f3542',
          '#1e90ff', '#ECCC68', '#6D214F', '#55E6C1', '#B33771', '#82589F',
          '#D6A2E8')
subplots_dict = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (3, 2),
                 6: (3, 2), 7: (4, 2), 8: (4, 2), 9: (5, 2), 10: (5, 2),
                 11: (6, 2), 12: (6, 2), 13: (7, 2), 14: (7, 2)}
"""A dictionary that defines the subplots grid according to the number of
set of axes, e,g, 5 set of axes -> 320 (3 rows, 2 columns)"""


def cm_to_inches(x) -> float:
    """Convert centimetres to inches"""

    return x / 2.54


def process_limits(limits) -> tuple:
    """Return None if limits are NA, else return the limits values"""

    return (None if math.isnan(limits[0]) else limits[0],
            None if math.isnan(limits[1]) else limits[1])


def get_case_labels(dict) -> list:
    """Format the case names to be printed on graphs

    Parameters
    ----------
    dict: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...

    Returns
    -------
    list: a formatted list of the case names
    """

    case_list = []
    for case, props in dict.items():
        name = ''
        if 'speed' in props.keys():
            name += ' ' + props['speed'].lstrip('0') + 'rpm'
        if 'load' in props.keys():
            load = props['load'].lstrip('0')
            load = '0' if load == '' else load
            name += ' ' + load + '%'
        if 'BMEP' in props.keys():
            name += ' ' + props['BMEP'].lstrip('0') +\
                ('bar' if 'bar' not in props['BMEP'].lstrip('0') else '')
        if 'torque' in props.keys():
            name += ' ' + props['torque'].lstrip('0') + 'Nm'
        if 'EGR' in props.keys():
            name += ' egr=' + props['EGR'].lstrip('0')
        if 'undefined' in props.keys():
            name += ' ' + props['undefined']
        case_list.append(name)

    return case_list


def new_ax(axis_width=2.67, x_0=0.5, y_0=0.5, extra_y=0,
           ratio=golden_ratio, extra_x=0.) -> (plt.figure, plt.axes):
    """Create a figure with one set axes

    Parameters
    ----------
    axis_width: float, optional, default: 2.67
        The width of the set axes inside the figure
    x_0: float, optional, default: 0.5
        The horizontal margin between the set axes and the figure limits
    y_0: float, optional, default: 0.5
        The vertical margin between the set axes and the figure limits
    extra_y: float, optional, default: 0
        The vertical extra margin between the set axes and the figure limits
    ratio: float, optional, default: golden ratio
        The ratio between the set axes width and height
    extra_x: float, optional, default: 0
        The horizontal extra margin between the set axes and the figure limits

    Returns
    -------
    A matplotlib.pyplot.figure object and a matplotlib.pyplot.axes object
    """

    axis_height = axis_width / ratio
    fig_width = axis_width + x_0 + extra_x
    dx = axis_width / fig_width
    fig_height = y_0 + axis_height + extra_y
    fig = plt.figure(figsize=(fig_width, fig_height))
    dy = axis_height / fig_height
    ax = plt.axes([x_0 / fig_width, y_0 / fig_height, dx, dy])
    fig.patch.set_alpha(0.)
    return fig, ax


def plot_combustion(data, cases, styles, path, units='mg', limits=None, r=1):
    """Plot a bar graph with the injected, burned, premix burned and
    premix unburned fuel for each steady-state case

    Parameters
    ----------
    data: dict, required
        The dictionary with fuel data and case names
    cases: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    units: str, optional, default: 'mg'
        The physical units of the variable, default mg
    limits: tuple or None, default: None
        The Y axis limits of the variable
    r: int, optional, default: 1
        The recursive mode: 1-> standard figure, 2-> figure without x labels,
        3-> figure without legend
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]

    fig, ax = new_ax(cm_to_inches(20), cm_to_inches(2.6), cm_to_inches(2.5),
                     cm_to_inches(1.5))
    ind = np.arange(len(data.keys()))
    width = 0.25  # bar width
    rects0 = ax.bar(ind - 2*width, [v['model']['injected'] for (k, v) in
                    data.items()], width, color='limegreen')
    rects1 = ax.bar(ind - 1*width,  [v['model']['burned'] for (k, v) in
                    data.items()], width, color='k')
    rects2 = ax.bar(ind + 0*width, [v['model']['premix_burned'] for (k, v) in
                    data.items()], width, color='salmon')
    rects3 = ax.bar(ind + 1*width, [v['model']['premix_unburned'] for (k, v) in
                    data.items()], width, color='maroon')
    # This set of bars represent the gap between series
    ax.bar(ind + 2*width, np.zeros(len(ind)), width)

    ax.set_ylabel('{} [{}]'.format('Fuel', units))
    if limits is not None:
        ax.set_ylim(process_limits(limits))

    # Set legend
    if r != 3:
        ax.legend((rects0[0], rects1[0], rects2[0], rects3[0]),
                  ('Injected fuel', 'Burned fuel', 'Premix burned fuel',
                   'Premix unburned fuel'), ncol=2, bbox_to_anchor=(0.5, 0.95),
                  loc='lower center')

    # Process case names
    test_names = get_case_labels(cases)
    x_lab_size = 7 if len(test_names[0]) > 13 else 10

    # Set x labels
    ax.xaxis.set_ticks(ind)
    ax.set_xlim(-1, len(ind))
    if r == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(test_names, size=x_lab_size, ha='right')
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    # Save figures
    subfolder = ''
    if r == 2:
        subfolder = 'no_xticklabels/'
    elif r == 3:
        subfolder = 'no_legend/'
    fig.savefig('{}/img/{}{}.pdf'.format(path, subfolder, 'Combustion'))
    fig.savefig('{}/img/{}{}.png'.format(path, subfolder, 'Combustion'),
                dpi=300)
    plt.close(fig)

    # Recursively enter into the function
    if r < 3:
        plot_combustion(data, cases, styles, path, limits, r=r + 1)
    else:
        return


def plot_stacked_bars(data, group1, group2, cases, styles, path, name,
                      units='-', limits=None, r=1):
    """Plot a stacked bars graph with different grouped variables for
    each steady-state case

    Parameters
    ----------
    data: dict, required
        The dictionary with fuel data and case names
    group1: list, required
        The list of keys in data corresponding to the first stack
    group2: list, required
        The list of keys in data corresponding to the second stack
    cases: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable group
    units: str, optional, default: '-'
        The physical units of the variable group, default -
    limits: tuple or None, default: None
        The Y axis limits of the variable
    r: int, optional, default: 1
        The recursive mode: 1-> standard figure, 2-> figure without x labels,
        3-> figure without legend
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]

    fig, ax = new_ax(cm_to_inches(20), cm_to_inches(2.6), cm_to_inches(2.5),
                     cm_to_inches(1.5))
    ind = np.arange(len(cases.keys()))
    width = 0.35  # bar width

    vars_map = {'ambient': {'color': 'mediumseagreen', 'legend': 'To ambient'},
                'oil': {'color': 'orange', 'legend': 'To oil'},
                'turbine': {'color': 'firebrick', 'legend': 'Turbine'},
                'compressor': {'color': 'blue', 'legend': 'Compressor'},
                'compressor+': {'color': 'blue', 'legend': 'Compressor'},
                'compressor-': {'color': 'blue', 'legend': 'Compressor'},
                'mech_losses': {'color': 'darkgrey', 'legend': 'Mech. losses'}}

    # stack1 and stack2 are the stacked lists of values for each case
    stack1, stack2 = [], []
    stack1.append(np.zeros(len(ind)))
    stack2.append(np.zeros(len(ind)))
    # rects1 and rects2 are the stacked list of bars for each case
    # rects is a list merging both sets of bars
    # legends is a list of legend labels
    rects1, rects2, rects, legends = [], [], [], []

    for i, var in enumerate(group1):
        values = [v['model'][var] for (k, v) in data.items()]
        stack1.append(stack1[-1] + values)
        rects1.append(ax.bar(ind - width, values, width,
                      color=vars_map[var]['color'], bottom=stack1[i]))
        rects.append(rects1[i])
        leg = vars_map[var]['legend']
        if leg not in legends:
            legends.append(leg)

    for i, var in enumerate(group2):
        values = [v['model'][var] for (k, v) in data.items()]
        stack2.append(stack2[-1] + values)
        rects2.append(ax.bar(ind, values, width,
                      color=vars_map[var]['color'], bottom=stack2[i]))
        rects.append(rects2[i])
        leg = vars_map[var]['legend']
        if leg not in legends:
            legends.append(leg)

    # Set legend
    if r != 3:
        ax.legend(rects, legends, ncol=len(group1) + len(group2),
                  bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=12)

    ax.set_ylabel('{} [{}]'.format(name, units))
    if limits is not None:
        ax.set_ylim(process_limits(limits))

    # Process case names
    test_names = get_case_labels(cases)
    x_lab_size = 7 if len(test_names[0]) > 13 else 10

    # Set x labels
    ax.xaxis.set_ticks(ind)
    ax.set_xlim(-1, len(ind))
    if r == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(test_names, size=x_lab_size, ha='right')
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    # Save figures
    subfolder = ''
    if r == 2:
        subfolder = 'no_xticklabels/'
    elif r == 3:
        subfolder = 'no_legend/'
    fig.savefig('{}/img/{}{}.pdf'.format(path, subfolder, name))
    fig.savefig('{}/img/{}{}.png'.format(path, subfolder, name), dpi=300)
    plt.close(fig)

    # Recursively enter into the function
    if r < 3:
        plot_stacked_bars(data, group1, group2, cases, styles, path, name,
                          units, limits, r=r + 1)
    else:
        return


def plot_bars(data, cases, styles, path, name, units, limits=None, r=1):
    """Plot a bar graph with model and reference values for each
    steady-state case

    Parameters
    ----------
    data: dict, required
        The dictionary with fuel data and case names
    cases: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable
    units: str, required
        The physical units of the variable
    limits: tuple or None, default: None
        The Y axis limits of the variable
    r: int, optional, default: 1
        The recursive mode: 1-> standard figure, 2-> figure without x labels,
        3-> figure without legend
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]
    legend = [styles['@labelExp'] if '@labelExp' in styles.keys()
              else 'Reference', styles['@labelModel'] if '@labelModel' in
              styles.keys() else 'VEMOD']
    colors = [styles['@colorExp'] if '@colorExp' in styles.keys()
              else '#E24A33', styles['@colorModel'] if '@colorModel' in
              styles.keys() else '#348ABD']

    fig, ax = new_ax(cm_to_inches(20), cm_to_inches(2.6), cm_to_inches(2.5),
                     cm_to_inches(1.5))
    ind = np.arange(len(data.keys()))
    width = 0.35  # bar width
    rects1 = ax.bar(ind - width, [v['exp'] for (k, v) in data.items()],
                    width, color=colors[0])
    rects2 = ax.bar(ind, [v['model'] for (k, v) in data.items()],
                    width, color=colors[1])

    ax.set_ylabel('{} [{}]'.format(name, units))
    # If the variable is mass fraction or turbocharger speed
    # (big and small numbers), change Y axis to scientific notation
    if '_y' in name.lower():
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    if limits is not None:
        ax.set_ylim(process_limits(limits))

    # Set legend
    if r != 3:
        ax.legend((rects1[0], rects2[0]), legend, ncol=2,
                  bbox_to_anchor=(0.5, 1), loc='lower center')

    # Process case names
    test_names = get_case_labels(cases)
    x_lab_size = 7 if len(test_names[0]) > 13 else 10

    # Set x labels
    ax.xaxis.set_ticks(ind)
    ax.set_xlim(-1, len(ind))
    if r == 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(test_names, size=x_lab_size, ha='right')
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    # Save figures
    subfolder = ''
    if r == 2:
        subfolder = 'no_xticklabels/'
    elif r == 3:
        subfolder = 'no_legend/'
    fig.savefig('{}/img/{}{}.pdf'.format(path, subfolder, name))
    fig.savefig('{}/img/{}{}.png'.format(path, subfolder, name),
                dpi=300)
    plt.close(fig)

    # Recursively enter into the function
    if r < 3:
        plot_bars(data, cases, styles, path, name, units, limits, r=r + 1)
    else:
        return


def plot_trends(data, cases, styles, path, name, units, r=1):
    """Plot a scatter graph model vs experimental for each steady-state case

    Parameters
    ----------
    data: dict, required
        The dictionary with fuel data and case names
    cases: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable
    units: str, required
        The physical units of the variable
    r: int, optional, default: 1
        The recursive mode: 1-> standard figure, 2-> figure without legend
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]
    x_label = styles['@labelExp'] if '@labelExp' in\
        styles.keys() else 'Reference'
    y_label = styles['@labelModel'] if '@labelModel' in\
        styles.keys() else 'VEMOD'

    fig, ax = new_ax(cm_to_inches(20), cm_to_inches(4.5), cm_to_inches(2.5),
                     cm_to_inches(1.5))
    model = [v['model'] for (k, v) in data.items()]
    exp = [v['exp'] for (k, v) in data.items()]

    # Calculate the separation between data and axes
    if np.nanmin(model) < np.nanmin(exp):
        margin_l = np.nanmin(model)
    else:
        margin_l = np.nanmin(exp)
    if np.nanmax(model) > np.nanmax(exp):
        margin_r = np.nanmax(model)
    else:
        margin_r = np.nanmax(exp)

    # Group the cases by their engine speed
    legend = []
    for case, values in cases.items():
        if values['speed'] != '':
            if values['speed'] not in legend:
                legend.append(values['speed'])
        else:
            legend.append(values['undefined'])

    # Plot points
    rects = []
    for i, group in enumerate(legend):
        # This for loop plots each point in the group increasing the marker
        # size for each point in the group
        for j, (e, m) in enumerate(zip(
                [e for e, case in zip(exp, cases.keys()) if group in case],
                [m for m, case in zip(model, cases.keys()) if group in case])):
            if j == 0:
                rects.append(ax.plot(e, m, color=colors[i], marker=markers[i],
                                     ls='', ms=10))
            else:
                ax.plot(e, m, color=colors[i], marker=markers[i],
                        ls='', ms=10 + 2*j)

    # Add a line whose slope is 1
    rects.append(ax.plot(np.arange(-1000, 1000000, 1000),
                         np.arange(-1000, 1000000, 1000), color='r', zorder=0))

    ax.set_xlabel(x_label)
    ax.set_axisbelow(True)
    ax.set_ylabel('{} [{}]\n{}'.format(name, units, y_label))
    # If the variable is mass fraction or turbocharger speed
    # (big and small numbers), change Y axis to scientific notation
    if ('_y' in name.lower()) or ('turbocharger speed' in name.lower()):
        plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    # Set legend
    if r != 3:
        ax.legend([r[0] for r in rects[:-1]], legend, ncol=10,
                  bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=11)

    # Set limits
    ax.set_xlim(margin_l - (margin_r - margin_l) / 12,
                margin_r + (margin_r - margin_l) / 12)
    ax.set_ylim(margin_l - (margin_r - margin_l) / 12,
                margin_r + (margin_r - margin_l) / 12)

    # Save figures
    subfolder = ''
    if r == 2:
        subfolder = 'no_legend/'
    fig.savefig('{}/trends/{}{}_Trend.pdf'.format(path, subfolder, name))
    fig.savefig('{}/trends/{}{}_Trend.png'.format(path, subfolder, name),
                dpi=300)
    plt.close(fig)

    # Recursively enter into the function
    if r < 2:
        plot_trends(data, cases, styles, path, name, units, r=r + 1)
    else:
        return


def plot_evolution(data, cases, styles, path, name, units):
    """Plot a model variable versus the simulation time for each steady-state
    case. The cases are grouped by their engine speed in different subplots

    Parameters
    ----------
    data: dict, required
        The dictionary with the variable data and case names
    cases: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable
    units: str, required
        The physical units of the variable
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] - 2 if '@fontSize' in\
        styles.keys() else 12
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]

    # Get variable label for legends
    labels = data.extra['legend']

    # Group the cases by their engine speed
    speed_dict = {}
    for case, values in cases.items():
        if values['speed'] != '':
            if values['speed'] not in speed_dict.keys():
                speed_dict[values['speed']] = [case]
            else:
                speed_dict[values['speed']].append(case)
        else:
            speed_dict[values['undefined']] = case

    for speed, names in speed_dict.items():
        # Create a figure for each speed and variable
        fig = plt.figure(figsize=(8.8967, 6.44))
        fig.patch.set_alpha(0.)
        # List which contains all the subplots
        axes = []
        try:
            titles = get_case_labels(OrderedDict((k, v) for (k, v)
                                     in cases.items() if v['speed'] == speed))
        except Exception:
            titles = get_case_labels(OrderedDict((k, v) for (k, v)
                                     in cases.items()
                                     if v['undefined'] == speed))
        for i, (case, title) in enumerate(zip(names, titles)):
            # Create as many subplots as values has each speed in speed_dict
            axes.append(fig.add_subplot(subplots_dict[len(names)][0],
                                        subplots_dict[len(names)][1], i + 1))
            axes[-1].set_axisbelow(True)
            axes[-1].set_title(title, fontsize=12, weight="bold")
            for subvar, color, label in zip(data.values[case]['model'],
                                            colors, labels):
                # Create one trace for each subvar
                axes[-1].plot(data.values[case]['time'], subvar, color=color,
                              linewidth=2, label=label)
            axes[-1].xaxis.set_ticklabels([])

        # Write Y labels
        # Just one common Y label
        fig.text(0.02, 0.5, "{} [{}]".format(name, units), va='center',
                 ha='center', rotation='vertical', fontsize=14)

        # Write X labels
        # Just one common X label
        fig.text(0.5, 0.04, "Time [s]", va='center', ha='center', fontsize=14)
        axes[-1].xaxis.set_ticklabels([int(x) for x in plt.gca().get_xticks()])
        try:
            axes[-2].xaxis.set_ticklabels([int(x) for x
                                           in plt.gca().get_xticks()])
        except Exception:
            pass
        # Set Legend
        if pd.isna(labels[0]) is False:
            axes[0].legend(fontsize=9, loc='upper left')

        # Save figures
        fig.savefig('{}/time_evolution/{}_{}.pdf'.format(path, name, speed))
        fig.savefig('{}/time_evolution/{}_{}.png'.format(path, name, speed),
                    dpi=300)
        plt.close(fig)


def plot_cylinder_var(data, cases, styles, path, name, units):
    """Plot model and reference values of an instantaneous cylinder variable
    for each steady-state case. The cases are grouped by their engine speed
    in different subplots

    Parameters
    ----------
    data: dict, required
        The dictionary with the variable data and case names
    cases: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable
    units: str, required
        The physical units of the variable
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]
    legend = [styles['@labelExp'] if '@labelExp' in styles.keys()
              else 'Reference', styles['@labelModel'] if '@labelModel' in
              styles.keys() else 'VEMOD']
    linestyles = [styles['@lineStyleExp'] if '@lineStyleExp' in styles.keys()
                  else '-', styles['@lineStyleModel'] if '@lineStyleModel' in
                  styles.keys() else '-']
    colors = [styles['@colorExp'] if '@colorExp' in styles.keys()
              else '#E24A33', styles['@colorModel'] if '@colorModel' in
              styles.keys() else '#348ABD']

    # Group the cases by their engine speed
    speed_dict = {}
    for case, values in cases.items():
        if values['speed'] != '':
            if values['speed'] not in speed_dict.keys():
                speed_dict[values['speed']] = [case]
            else:
                speed_dict[values['speed']].append(case)
        else:
            speed_dict[values['undefined']] = case

    for speed, names in speed_dict.items():
        # Create a figure for each speed and variable
        fig = plt.figure(figsize=(8.8967, 6.44))
        fig.patch.set_alpha(0.)
        # List which contains all the subplots
        axes = []
        try:
            titles = get_case_labels(OrderedDict((k, v) for (k, v)
                                     in cases.items() if v['speed'] == speed))
        except Exception:
            titles = get_case_labels(OrderedDict((k, v) for (k, v)
                                     in cases.items()
                                     if v['undefined'] == speed))
        for i, (case, title) in enumerate(zip(names, titles)):
            # Create as many subplots as values has each speed in speed_dict
            axes.append(fig.add_subplot(subplots_dict[len(names)][0],
                                        subplots_dict[len(names)][1], i + 1))
            axes[-1].set_axisbelow(True)
            axes[-1].set_title(title, fontsize=12, weight="bold")
            # MODEL
            if name == 'Fuel':
                axes[-1].plot(data.values[case]['model']['x'],
                              data.values[case]['model']['y']['injected'],
                              c='limegreen', zorder=2, label='Injected')
                axes[-1].plot(data.values[case]['model']['x'],
                              data.values[case]['model']['y']['burned'],
                              c='k', zorder=2, label='Burned')
                axes[-1].plot(data.values[case]['model']['x'],
                              data.values[case]['model']['y']['premix_burned'],
                              c='salmon', zorder=2, label='Premix burned')
                axes[-1].plot(data.values[case]['model']['x'],
                              data.values[case]['model']['y'][
                                  'premix_unburned'],
                              c='maroon', zorder=2, label='Premix unburned')
                y_label = "{} [{}]".format(name, units)
                x_label = 'Angle [cad]'
            elif name == 'Fuel injected - burned norm. vs O2':
                index0 = data.values[case]['model']['y'][
                    'injected'].idxmin() + 1
                index1 = data.values[case]['model']['y'][
                    'O2'][index0:].idxmin()
                indexes = np.arange(index0, index1, 1)
                X = data.values[case]['model']['y']['O2']
                Y = (data.values[case]['model']['y']['injected'] -
                     data.values[case]['model']['y']['burned']) /\
                    data.values[case]['model']['y']['injected']
                axes[-1].scatter(np.take(X.values, indexes), np.take(Y.values,
                                 indexes), marker='o', s=3, color='b')
                axes[-1].set_ylim([0, 0.5])
                y_label = 'Fuel injected - burned, relative to injected [0-1]'
                x_label = r'$Y O_2$'
            else:
                axes[-1].plot(data.values[case]['model']['x'][1:-2],
                              data.values[case]['model']['y'][1:-2],
                              c=colors[1], ls=linestyles[1], zorder=2,
                              label=legend[1])
                y_label = "{} [{}]".format(name, units)
                x_label = 'Angle [cad]'
            # REFERENCE
            if data.exp_col is not None:
                for i, (angle, cyl_data) in enumerate(zip(data.values[case][
                        'exp']['x'], data.values[case]['exp']['y'])):
                    axes[-1].plot(angle, cyl_data, c=colors[0],
                                  ls=linestyles[0], zorder=1,
                                  alpha=1-i*0.15, label=legend[0] if i == 0
                                  else '_nolegend_')
            axes[-1].xaxis.set_ticklabels([])
            if data.limits is not None:
                axes[-1].set_xlim(process_limits(data.limits))

        # Write Y labels
        # Just one common Y label
        fig.text(0.04, 0.5, y_label, va='center',
                 ha='center', rotation='vertical', fontsize=14)

        # Write X labels
        # Just one common X label
        fig.text(0.5, 0.04, x_label, va='center',
                 ha='center', fontsize=14)
        if name == 'Fuel injected - burned norm. vs O2':
            axes[-1].xaxis.set_ticklabels(['{:.2f}'.format(x) for x in
                                           plt.gca().get_xticks()])
        else:
            axes[-1].xaxis.set_ticklabels([int(x) for x
                                           in plt.gca().get_xticks()])
        try:
            if name == 'Fuel injected - burned norm. vs O2':
                axes[-2].xaxis.set_ticklabels(['{:.2f}'.format(x) for x
                                               in plt.gca().get_xticks()])
            else:
                axes[-2].xaxis.set_ticklabels([int(x) for x
                                               in plt.gca().get_xticks()])
        except Exception:
            pass
        # Set Legend
        if name != 'Fuel injected - burned norm. vs O2':
            axes[0].legend(fontsize=9, loc='upper left')

        # Save figures
        fig.savefig('{}/img_ins/{}_{}.pdf'.format(path, name, speed))
        fig.savefig('{}/img_ins/{}_{}.png'.format(path, name, speed),
                    dpi=300)
        plt.close(fig)


def plot_transient_var(data, styles, path, name, units, profile, r=1):
    """Plot model and reference values of a variable along the transient
    test. A second subplot is drawn showing the absolute error for
    cumulative values. It also plots only model variables

    Parameters
    ----------
    data: dict, required
        The dictionary with the variable data
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable
    units: str, required
        The physical units of the variable
    profile: dict, required
        The dictionary with the transient vehicle speed, the transient,
        duration, the model signal delay and the transient divisions
    r: int, optional, default: 1
        The recursive mode: 1-> standard figure, 2-> figure without x labels,
        3-> figure without legend
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]
    legend = [styles['@labelExp'] if '@labelExp' in styles.keys()
              else 'Reference', styles['@labelModel'] if '@labelModel' in
              styles.keys() else 'VEMOD']
    linestyles = [styles['@lineStyleExp'] if '@lineStyleExp' in styles.keys()
                  else '-', styles['@lineStyleModel'] if '@lineStyleModel' in
                  styles.keys() else '-']
    linewidths = [styles['@lineWidthExp'] if '@lineWidthExp' in styles.keys()
                  else '1.5', styles['@lineWidthModel'] if '@lineWidthModel' in
                  styles.keys() else '1.5']
    _colors = [styles['@colorExp'] if '@colorExp' in styles.keys()
               else '#E24A33', styles['@colorModel'] if '@colorModel' in
               styles.keys() else '#348ABD']

    if (('acc' in name.lower()) or (units == 'g')) and\
            data.time_evolution is False:
        # Create a figure which shows also the absolute error
        # Create a specific figure with 2 subplots
        axis_height = cm_to_inches(30) / 3
        fig_width = cm_to_inches(30) + cm_to_inches(3.5) + 0.4
        fig_height = cm_to_inches(2.5) + axis_height + cm_to_inches(1.5)
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(top=0.94, bottom=0.10, left=0.10, right=0.965)
        fig.patch.set_alpha(0.)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax3 = ax1.twinx()  # Vehicle speed will be plotted in this axis
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        # Plot the vehicle speed in the secondary axis
        ax3.fill_between(data.values['exp']['x'], 0, profile['veh_speed'],
                         facecolor='gray', alpha=0.2, zorder=-1)
        ax3.set_axisbelow(True)

        # Plot the accumulate value in the first subplot
        ax1.plot(data.values['exp']['x'], data.values['exp']['y'],
                 c=_colors[0], ls=linestyles[0], lw=linewidths[0], zorder=2)
        ax1.plot(data.values['model']['x'] + profile['delay'],
                 data.values['model']['y'],
                 c=_colors[1], ls=linestyles[1], lw=linewidths[1], zorder=2)
        ax1.yaxis.set_label_coords(-0.08, 0.5)

        # Resize the model data in order to compare when the model is
        # over the reference, or viceversa
        (xdm, ydm), (xde, yde) = resize(data.values['model']['x'] +
                                        profile['delay'],
                                        data.values['model']['y'],
                                        data.values['exp']['x'],
                                        data.values['exp']['y'])
        # Remove nans
        mask = ~np.isnan(ydm) & ~np.isnan(yde)
        ydm, yde, xde = ydm[mask], yde[mask], xde[mask]

        # Plot the absolute error in the second subplot
        ax2.fill_between(xde, 0, ydm-yde, where=ydm-yde >= 0,
                         facecolor=_colors[1], interpolate=True)
        ax2.fill_between(xde, 0, ydm-yde, where=ydm-yde <= 0,
                         facecolor=_colors[0], interpolate=True)
        ax2.yaxis.set_label_coords(-0.08, 0.5)

        # Plot divisions
        for div in profile['divisions']:
            ax1.axvline(div['@start'], c='g', alpha=0.5, lw=1.5, zorder=1)
            ax1.axvline(div['@end'], c='g', alpha=0.5, lw=1.5, zorder=1)
            ax3.text(np.mean([div['@start'], div['@end']]), -4, div['@name'],
                     {'color': 'g', 'fontsize': 11, 'ha': 'center',
                      'va': 'center'})

        y_label = name.replace('_g_s', '').replace('_g', '')
        ax1.set_ylabel('{} [{}]'.format(y_label, units))
        if data.limits is not None:
            ax1.set_ylim(process_limits(data.limits))
        # Just plot xtick labels on the second subplot
        ax1.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
        ax3.tick_params(length=0)  # Do not show ticks on the secondary axis
        textXpos = profile['max_time'] * 11 / 12
        ax3.text(textXpos, 2, "Vehicle speed",
                 {'color': 'gray', 'fontsize': 10, 'ha': 'center',
                  'va': 'center'})
        ax3.grid(b=False)  # Remove the secondary axis grid
        ax2.set_ylabel('Mod - Ref [{}]'.format(units))

        # Set x label and ticks
        if r == 2:
            ax2.set_xticklabels([])
        else:
            ax2.set_xlabel('Time [s]')
        ax3.spines['top'].set_visible(False)

        # Set legend
        if r != 3:
            ax1.legend(legend, bbox_to_anchor=(0.5, 1.1), loc='upper center',
                       ncol=len(legend), fancybox=True, shadow=True,
                       fontsize=14).set_zorder(100)
        ax1.set_xlim([0, profile['max_time']])
        ax2.set_xlim([0, profile['max_time']])

    else:
        # Create a simple figure
        fig, ax = new_ax(cm_to_inches(30), cm_to_inches(3.5),
                         cm_to_inches(2.5), cm_to_inches(1.5), ratio=3,
                         extra_x=0.4)

        ax.set_axisbelow(True)

        # Smooth signals
        try:
            if data.extra['tau'] != 0:
                y_mod = np.convolve(data.values['model']['y'], np.ones((
                    data.extra['tau'] * 20,)) / data.extra['tau'] / 20,
                    mode='same')
                y_mod = np.concatenate((np.ones(data.extra['tau'] * 10) *
                                       data.values['model']['y'][0],
                                        y_mod[:(-data.extra['tau'] * 10)]))
            else:
                y_mod = gaussian_filter(data.values['model']['y'], sigma=2)
        except Exception:
            # When data.extra is None
            y_mod = gaussian_filter(data.values['model']['y'], sigma=2)

        if data.time_evolution is False:
            y_exp = gaussian_filter(data.values['exp']['y'], sigma=2)
            ax.plot(data.values['exp']['x'], y_exp,
                    c=_colors[0], ls=linestyles[0], lw=linewidths[0], zorder=2)
            ax.plot(data.values['model']['x'] + profile['delay'], y_mod,
                    c=_colors[1], ls=linestyles[1], lw=linewidths[1], zorder=2)
        else:
            legend = data.extra['legend']
            # Plot model variables
            for i, y_mod in enumerate(data.values['model']['y']):
                ax.plot(data.values['model']['x'] + profile['delay'], y_mod,
                        c=colors[i], ls='-', lw=1.5, zorder=2)

        ax.yaxis.set_label_coords(-0.08, 0.5)
        y_label = name.replace('_g_s', '').replace('_g', '')
        ax.set_ylabel('{} [{}]'.format(y_label, units))
        if data.limits is not None:
            ax.set_ylim(process_limits(data.limits))

        # Plot divisions
        for div in profile['divisions']:
            ax.axvline(div['@start'], c='g', alpha=0.5, lw=1.5, zorder=1)
            ax.axvline(div['@end'], c='g', alpha=0.5, lw=1.5, zorder=1)
            x_pos = np.mean([div['@start'], div['@end']]) / profile['max_time']
            ax.text(x_pos, -0.11, div['@name'],
                    {'color': 'g', 'fontsize': 11, 'ha': 'center',
                     'va': 'center', 'transform': ax.transAxes})

        # Set x label and ticks
        if r == 2:
            ax.set_xticklabels([])
        else:
            if len(profile['divisions']) > 0:
                ax.xaxis.set_label_coords(0.5, -0.17)
                ax.set_xlabel('Time [s]')
            else:
                ax.set_xlabel('Time [s]')

        # Set legend
        if r != 3:
            if pd.isna(legend[0]) is False:
                ax.legend(legend, bbox_to_anchor=(0.5, 1.1),
                          loc='upper center', ncol=len(legend), fancybox=True,
                          shadow=True, fontsize=14).set_zorder(100)
        ax.set_xlim([0, profile['max_time']])

    # Save figures
    subfolder = ''
    if r == 2:
        subfolder = 'no_xticklabels/'
    elif r == 3:
        subfolder = 'no_legend/'
    fig.savefig('{}/img/{}{}.pdf'.format(path, subfolder, name))
    fig.savefig('{}/img/{}{}.png'.format(path, subfolder, name), dpi=300)
    plt.close(fig)

    # Recursively enter into the function
    if r < 3:
        plot_transient_var(data, styles, path, name, units, profile, r=r + 1)
    else:
        return


def plot_transient_trends(data, styles, path, name, units, profile):
    """Plot a scatter graph model vs experimental data of a variable along
    the transient test

    Parameters
    ----------
    data: dict, required
        The dictionary with the variable data
    styles: dict, required
        The graphic styles: font size, font family, matplotlib style...
    path: str, required
        The path where the figures have to be stored
    name: str, required
        The name of the variable
    units: str, required
        The physical units of the variable
    profile: dict, required
        The dictionary with the transient vehicle speed, the transient,
        duration, the model signal delay and the transient divisions
    """

    plt.style.use(styles['@style'] if '@style' in styles.keys() else 'ggplot')
    mpl.rcParams['font.size'] = styles['@fontSize'] if '@fontSize' in\
        styles.keys() else 14
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.family'] = styles['@fontFamily'] if '@fontFamily' in\
        styles.keys() else "sans-serif"
    mpl.rcParams['font.{}'.format(mpl.rcParams['font.family'][0])] = [
        styles['@font'] if '@font' in styles.keys() else "DejaVu Sans"]
    x_label = styles['@labelExp'] if '@labelExp' in\
        styles.keys() else 'Reference'
    y_label = styles['@labelModel'] if '@labelModel' in\
        styles.keys() else 'VEMOD'

    fig, ax = new_ax(cm_to_inches(20), cm_to_inches(4.5), cm_to_inches(2.5),
                     cm_to_inches(1.5))

    # Ignore the first 2% of the points. For some variables it is useful
    # because measured values start at a very low value
    start = int(len(data.values['exp']['y']) / 50)

    # Resize the model data in order to compare when the model is
    # over the reference
    (xdm, ydm), (xde, yde) = resize(data.values['model']['x'] +
                                    profile['delay'],
                                    data.values['model']['y'],
                                    data.values['exp']['x'],
                                    data.values['exp']['y'],
                                    limits=(0, profile['max_time']))
    xdm, xde = xdm[start:], xde[start:]
    ydm, yde = ydm[start:], yde[start:]

    # Smooth signals
    try:
        if data.extra['tau'] != 0:
            ydm = np.convolve(ydm, np.ones((data.extra['tau'] * 20,)) /
                              data.extra['tau'] / 20, mode='same')
            ydm = np.concatenate((np.ones(data.extra['tau'] * 10) *
                                  data.values['model']['y'][0],
                                  ydm[:(-data.extra['tau'] * 10)]))
            # Trim the first values
            ydm = ydm[(data.extra['tau'] * 10):]
            yde = yde[(data.extra['tau'] * 10):]
            xde = xde[(data.extra['tau'] * 10):]
        else:
            ydm = gaussian_filter(ydm, sigma=2)
    except Exception:
        # When data.extra is None
        ydm = gaussian_filter(ydm, sigma=2)
    yde = gaussian_filter(yde, sigma=2)

    if len(profile['divisions']) > 0:
        legend = []
        rects = []
        for i, div in enumerate(profile['divisions']):
            y_exp = np.take(yde, np.where((xde >= div['@start']) &
                                          (xde <= div['@end']))[0])
            y_mod = np.take(ydm, np.where((xde >= div['@start']) &
                                          (xde <= div['@end']))[0])
            rects.append(ax.scatter(y_exp, y_mod, color=colors[i], s=15))
            legend.append(div['@name'])
    else:
        ax.scatter(yde, ydm, color='b', s=15)
        legend = None

    # Add a line whose slope is 1
    ax.plot(np.arange(-1000, 1000000, 1000),
            np.arange(-1000, 1000000, 1000), color='r', zorder=0)

    # Mask to remove nans in order to calculate the coefficient of
    # determination
    mask = ~np.isnan(ydm) & ~np.isnan(yde)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(ydm[mask],
                                                                 yde[mask])
    except Exception:
        log.error("{} trend could not be plotted. Either model or experiment "
                  "are NaN".format(name))
        plt.close(fig)
        return

    ax.annotate(r'$R^2 = $ ' + str(round(np.power(r_value, 2), 3)), xy=(1, 0),
                xytext=(-15, +15), fontsize=16, xycoords='axes fraction',
                textcoords='offset points', bbox=dict(facecolor='white',
                alpha=0.8), horizontalalignment='right',
                verticalalignment='bottom')

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_axisbelow(True)
    ax.set_ylabel('{} [{}]\n{}'.format(name, units, y_label))

    # Set legend
    if legend is not None:
        ax.legend(rects, legend, ncol=10,
                  bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=11,
                  markerscale=3)

    # Calculate the separation between data and axes
    if np.nanmin(ydm) < np.nanmin(yde):
        margin_l = np.nanmin(ydm)
    else:
        margin_l = np.nanmin(yde)
    if np.nanmax(ydm) > np.nanmax(yde):
        margin_r = np.nanmax(ydm)
    else:
        margin_r = np.nanmax(yde)

    # Set limits
    ax.set_xlim(margin_l - (margin_r - margin_l) / 12,
                margin_r + (margin_r - margin_l) / 12)
    ax.set_ylim(margin_l - (margin_r - margin_l) / 12,
                margin_r + (margin_r - margin_l) / 12)

    # Save figures
    fig.savefig('{}/trends/{}_Trend.pdf'.format(path, name))
    fig.savefig('{}/trends/{}_Trend.png'.format(path, name), dpi=300)
    plt.close(fig)
