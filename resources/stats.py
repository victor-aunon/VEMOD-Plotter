# -*- coding: utf-8 -*-
"""
This file is part of VEMOD plotter
"""

# Imports
import os
import logging as log
from collections import OrderedDict

# Libs
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.ndimage.filters import gaussian_filter
from joblib import Parallel, delayed


def resize(x_mod, y_mod, x_exp, y_exp, limits=None) -> (tuple, tuple):
    """Resize an array to the shortest size of two arrays. Remove non
    coincident values from the two arrays

    Parameters
    ----------
    x_mod: list or numpy.array, required
        The X values of the model variable
    y_mod: list or numpy.array, required
        The Y values of the model variable
    x_exp: list or numpy.array, required
        The X values of the experimental variable
    y_exp: list or numpy.array, required
        The Y values of the experimental variable
    limits: tuple or None, optional, default: None
        The X axis limits of the resized array

    Returns
    -------
    tuple, tuple: (resized experimental X, interpolated Y model values),
                  (resized experimental X, resized experimental Y)
    """

    if limits is None:
        # Get the X values where Model and Experiment intersect
        x_common = sorted(list(set(np.round(x_mod, 2)) &
                               set(np.round(x_exp, 2))))
        limits = (0.0, x_common[-1])
    xde = np.take(x_exp, np.where((x_exp >= limits[0]) &
                                  (x_exp <= limits[1]))[0])
    yde = np.take(y_exp, np.where((x_exp >= limits[0]) &
                                  (x_exp <= limits[1]))[0])
    xdm = np.take(x_mod, np.where((x_mod >= limits[0]) &
                                  (x_mod <= limits[1]))[0])
    ydm = np.take(y_mod, np.where((x_mod >= limits[0]) &
                                  (x_mod <= limits[1]))[0])
    ypm = np.interp(xde, xdm, ydm)
    return (xde, ypm), (xde, yde)


def get_delay(x_model, y_model, x_exp, y_exp, limits=None) -> float:
    """Get the delay between two signals representing the same variable.
    Requires the resize function

    Parameters
    ----------
    x_model: numpy.array, required
        The model X values
    y_model: numpy.array, required
        The model Y values
    x_exp: numpy.array, required
        The experimental X values
    y_exp: numpy.array, required
        The experimental X values
    limits: tuple, optional, default: None
        The stretch along the X axis where the delay must be calculated

    Returns
    -------
    float: the delay between model and experimental signals in seconds
    """

    mod, exp = resize(x_model, y_model, x_exp, y_exp, limits)
    ymf = np.fft.fft(mod[1])
    yef = np.fft.fft(exp[1])
    c = np.fft.ifft(ymf * np.conj(yef))
    time_shift = np.argmax(abs(c))
    if (len(exp[0]) - time_shift) > time_shift:
        delay = 0 - exp[0][time_shift]
    else:
        delay = exp[0][len(exp[0]) - time_shift]
    return delay


def get_cp(T) -> np.array:
    """Return the air isobar specific heat capacity in kJ/Kg.k
    at a given temperature temperature (T) in Kelvin"""

    # Specific Heat Capacities of Air
    cp = [1.003, 1.005, 1.008, 1.013, 1.020, 1.029, 1.040, 1.051, 1.063,
          1.075, 1.087, 1.099]  # kJ/kg.K
    temp = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]  # K
    return np.interp(T, temp, cp)


def smape(model, exp) -> np.array:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE). It has been
    as an indicator of the relative error, since the common relative error
    definition gives unrealistic values when measured values are close to zero.
    This SMAPE may have a negative sign when model is higher than experiment

    Parameters
    ----------
    model: list or numpy.array, required
        The model or forecast values
    exp: list or numpy.array, required
        The experimental or real values

    Returns
    -------
    numpy.array: the SMAPE between forecast and real values
    """

    numerator = [abs(x - y) for x, y in zip(model, exp)]
    denominator = [x + y for x, y in zip(model, exp)]
    return np.divide(np.sum(numerator), np.sum(denominator)) * 100


def create_summary(args, var_list, path, mod_label=None, exp_label=None,
                   profile=None, num_cores=1):
    """Create a summary Excel file with relevant data for each variable

    Parameters
    ----------
    args: argparse.ArgumentParser.parse_args, required
        The arguments passed to the main function of the program
    var_list: list, required
        The list with all the engine variables
    path: str, required
        The path where the summary has to be stored
    mod_label: str, optional, default: None
        The model label
    exp_label: str, optional, default: None
        The reference label
    profile: dict, optional, default: None
        The dictionary with the transient vehicle speed, the transient,
        duration, the model signal delay and the transient divisions
    num_cores: int, optional, default: 1
        The number of logical CPU cores of the machine.
        Used for multiprocessing
    """

    if args.mode == 'steady-avg':
        summary = pd.DataFrame(index=var_list[0].values.keys(),
                               columns=pd.MultiIndex.from_product(
                                [[v.name + ' [{}]'.format(v.units) for v in
                                  var_list],
                                 [mod_label, exp_label, 'abs error',
                                  'SMAPE %']]))
        for key in var_list[0].values.keys():
            for var in var_list:
                sum_col = var.name + ' [{}]'.format(var.units)
                summary.loc[key, (sum_col, mod_label)] = var.values[
                    key]['model']
                summary.loc[key, (sum_col, exp_label)] = var.values[key]['exp']
                summary.loc[key, (sum_col, 'abs error')] = var.values[
                    key]['model'] - var.values[key]['exp']
                summary.loc[key, (sum_col, 'SMAPE %')] = smape(
                    [var.values[key]['model']], [var.values[key]['exp']])
        average = {}
        for var in var_list:
            sum_col = var.name + ' [{}]'.format(var.units)
            average[(sum_col, mod_label)] = ''
            average[(sum_col, exp_label)] = ''
            average[(sum_col, 'abs error')] = summary[
                (sum_col, 'abs error')].mean()
            average[(sum_col, 'SMAPE %')] = summary[
                (sum_col, 'SMAPE %')].mean()
        summary.loc['AVERAGE', :] = average
        writer = pd.ExcelWriter(os.path.join(path, 'summary_averages.xlsx'))
        summary.to_excel(writer, 'Sheet1')
        writer.save()

    elif args.mode == 'steady-ins':
        summary = pd.DataFrame(index=var_list[0].values.keys(),
                               columns=pd.MultiIndex.from_product(
                                [[v.name + ' [{}]'.format(v.units) for v in
                                  var_list],
                                 ['Max model', 'Max reference', 'Min model',
                                  'Min reference', 'abs error', 'SMAPE %',
                                  'std', 'std rel %', 'R squared']]))
        for key in var_list[0].values.keys():
            for var in var_list:
                sum_col = var.name + ' [{}]'.format(var.units)
                model, experiment = resize(
                    var.values[key]['model']['x'],
                    var.values[key]['model']['y'],
                    np.mean(var.values[key]['exp']['x'], axis=0),
                    np.mean(var.values[key]['exp']['y'], axis=0),
                    limits=(-100, 100))
                summary.loc[key, (sum_col, 'Max model')] = np.nanmax(
                    var.values[key]['model']['y'])
                summary.loc[key, (sum_col, 'Min model')] = np.nanmin(
                    var.values[key]['model']['y'])
                summary.loc[key, (sum_col, 'Max reference')] = np.nanmax(
                        np.mean(var.values[key]['exp']['y'], axis=0))
                summary.loc[key, (sum_col, 'Min reference')] = np.nanmin(
                        np.mean(var.values[key]['exp']['y'], axis=0))
                summary.loc[key, (sum_col, 'abs error')] = np.mean(
                    model[1] - experiment[1])
                summary.loc[key, (sum_col, 'SMAPE %')] = smape(
                    model[1], experiment[1])
                summary.loc[key, (sum_col, 'std')] = np.std(
                    model[1] - experiment[1])
                summary.loc[key, (sum_col, 'std rel %')] = summary.at[
                    key, (sum_col, 'std')] / abs(summary.at[
                        key, (sum_col, 'abs error')]) * 100
                summary.loc[key, (sum_col, 'R squared')] = np.power(
                    linregress(model[1], experiment[1])[2], 2)
                #  The third value returned by linregress is the
                #  R value, so the [2]
        average = {}
        for var in var_list:
            sum_col = var.name + ' [{}]'.format(var.units)
            average[(sum_col, 'Max model')] = ''
            average[(sum_col, 'Max reference')] = ''
            average[(sum_col, 'Min model')] = ''
            average[(sum_col, 'Min reference')] = ''
            average[(sum_col, 'abs error')] = summary[
                (sum_col, 'abs error')].mean()
            average[(sum_col, 'SMAPE %')] = summary[
                (sum_col, 'SMAPE %')].mean()
            average[(sum_col, 'std')] = summary[(sum_col, 'std')].mean()
            average[(sum_col, 'std rel %')] = summary[
                (sum_col, 'std rel %')].mean()
            average[(sum_col, 'R squared')] = summary[
                (sum_col, 'R squared')].mean()
        summary.loc['AVERAGE', :] = average
        writer = pd.ExcelWriter(os.path.join(path, 'summary_cyl_ins.xlsx'))
        summary.to_excel(writer, 'Sheet1')
        writer.save()

    elif args.mode == 'transient':
        columns = ['units', 'accum. ' + exp_label, 'accum. ' + mod_label,
                   'abs error whole transient',
                   'SMAPE (%) whole transient', 'std whole transient',
                   'std rel (%) whole transient', 'R squared whole transient']
        for div in profile['divisions']:
            columns.append('accum. {} {} division'.format(exp_label,
                                                          div['@name']))
            columns.append('accum. {} {} division'.format(mod_label,
                                                          div['@name']))
            columns.append('abs error {} division'.format(div['@name']))
            columns.append('SMAPE (%) {} division'.format(div['@name']))
            columns.append('std {} division'.format(div['@name']))
            columns.append('std rel (%) {} division'.format(div['@name']))
            columns.append('R squared {} division'.format(div['@name']))
        for error in profile['errors']:
            columns.append('max abs error at {}% of points'.format(
                error['@pointsPercentage']))
            columns.append('max abs SMAPE (%) at {}% of points'.format(
                error['@pointsPercentage']))

        summary_dict = OrderedDict()
        if args.nomultiprocessing is False:
            summary_list = Parallel(n_jobs=num_cores, verbose=8)(delayed(
                transient_variable_summary)(var, mod_label, exp_label,
                                            profile) for var in var_list)
            for summary_var in summary_list:
                for var, values in summary_var.items():
                    summary_dict[var] = values
        else:
            last_percent = 10
            for i, var in enumerate(var_list):
                summary_var = transient_variable_summary(var, mod_label,
                                                         exp_label, profile)
                if np.floor(i * 100 / len(var_list)) >= last_percent:
                    log.info("{} %".format(last_percent))
                    last_percent += 10

                for var, values in summary_var.items():
                    summary_dict[var] = values
            log.info('100 %')

        summary = pd.DataFrame.from_dict(summary_dict, orient='index',
                                         columns=columns)
        writer = pd.ExcelWriter(os.path.join(path, 'summary_transient.xlsx'))
        summary.to_excel(writer, 'Sheet1')
        writer.save()


def transient_variable_summary(var, mod_label=None, exp_label=None,
                               profile=None) -> dict:
    """Create a dictionary with relevant data of a transient variable

    Parameters
    ----------
    var: Variable class, required
        The engine variable
    mod_label: str, optional, default: None
        The model label
    exp_label: str, optional, default: None
        The reference label
    profile: dict, optional, default: None
        The dictionary with the transient vehicle speed, the transient,
        duration, the model signal delay and the transient divisions

    Returns
    -------
    dict: a dictionary with the statistical data of the engine variable
    """

    try:
        np.warnings.filterwarnings('ignore')
        summary_var = {str(var.name): {}}
        summary_var[var.name]['units'] = var.units
        # Ignore the first 2% of the points. For some variables it is
        #  useful because measured values start at a very low value
        start = int(len(var.values['exp']['y']) / 50)
        # Resize the model data in order to compare with the model
        (xdm, ydm), (xde, yde) = resize(var.values['model']['x'] +
                                        profile['delay'],
                                        var.values['model']['y'],
                                        var.values['exp']['x'],
                                        var.values['exp']['y'],
                                        limits=(0, profile['max_time'])
                                        )
        xdm, xde = xdm[start:], xde[start:]
        ydm, yde = ydm[start:], yde[start:]

        # Mask to remove nans in order to calculate the coefficient of
        # determination
        mask = ~np.isnan(ydm) & ~np.isnan(yde)
        ydm = ydm[mask]
        yde = yde[mask]
        try:
            slope, intercept, r_value, p_value, std_err = linregress(
                ydm, yde)
        except Exception:
            r_value = 0
            pass
        # Smooth signals
        try:
            if var.extra['tau'] != 0:
                ydm = np.convolve(ydm, np.ones(
                    (var.extra['tau'] * 20,)) /
                    var.extra['tau'] / 20, mode='same')
                ydm = np.concatenate((np.ones(var.extra['tau'] * 10) *
                                      var.values['model']['y'][0],
                                      ydm[:(-var.extra['tau'] * 10)]))
                # Trim the first values
                ydm = ydm[(var.extra['tau'] * 10):]
                yde = yde[(var.extra['tau'] * 10):]
                xde = xde[(var.extra['tau'] * 10):]
            else:
                ydm = gaussian_filter(ydm, sigma=2)
        except Exception:
            # When var.extra is None
            ydm = gaussian_filter(ydm, sigma=2)
        yde = gaussian_filter(yde, sigma=2)
        # Accumulated values
        if ('acc' in var.name.lower()) or (var.name.lower().endswith('_g')):
            summary_var[var.name]['accum. ' + exp_label] = yde[-1]
            summary_var[var.name]['accum. ' + mod_label] = ydm[-1]
        # Errors
        abs_error = [x - y for x, y in zip(ydm, yde)]
        summary_var[var.name]['abs error whole transient'] = np.mean(abs_error)
        summary_var[var.name]['SMAPE (%) whole transient'] = smape(ydm, yde)
        summary_var[var.name]['std whole transient'] = np.std(abs_error)
        summary_var[var.name]['std rel (%) whole transient'] = np.std(
            abs_error) / abs(np.mean(abs_error)) * 100
        summary_var[var.name]['R squared whole transient'] = r_value**2

        for div in profile['divisions']:
            ye_div = np.take(yde, np.where((xde >= div['@start']) &
                                           (xde <= div['@end']))[0])
            ym_div = np.take(ydm, np.where((xde >= div['@start']) &
                                           (xde <= div['@end']))[0])
            try:
                _, _, r_value, _, _ = linregress(ym_div, ye_div)
            except Exception:
                r_value = 0
                pass
                # Accumulated values
            if ('acc' in var.name.lower()) or\
                    (var.name.lower().endswith('_g')):
                summary_var[var.name]['accum. {} {} division'.format(
                        exp_label, div['@name'])] = ye_div[-1]
                summary_var[var.name]['accum. {} {} division'.format(
                        mod_label, div['@name'])] = ym_div[-1]
            # Errors
            abs_error_div = [x - y for x, y in zip(ym_div, ye_div)]
            summary_var[var.name]['abs error {} division'.format(div['@name'])
                                  ] = np.mean(abs_error_div)
            summary_var[var.name]['SMAPE (%) {} division'.format(div['@name'])
                                  ] = smape(ym_div, ye_div)
            summary_var[var.name]['std {} division'.format(div['@name'])
                                  ] = np.std(abs_error_div)
            summary_var[var.name]['std rel (%) {} division'.format(
                div['@name'])] = np.std(abs_error_div) /\
                abs(np.mean(abs_error_div)) * 100
            summary_var[var.name]['R squared {} division'.format(div['@name'])
                                  ] = r_value**2
        for error in profile['errors']:
            index = int(len(xde) * error['@pointsPercentage'] / 100)
            summary_var[var.name]['max abs error at {}% of points'.format(
                error['@pointsPercentage'])
                ] = sorted(map(abs, abs_error))[index]
            summary_var[var.name]['max abs SMAPE (%) at {}% of points'.format(
                    error['@pointsPercentage'])
                ] = sorted(map(abs, [smape([m], [e]) for m, e in zip(
                    ydm, yde)]))[index]
        return summary_var
    except Exception:
        pass
