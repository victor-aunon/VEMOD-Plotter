# -*- coding: utf-8 -*-
"""
VEMOD plotter. This program allows plotting simulation results (at transient
or steady state operating conditions) from VEMOD and compare them with the
experimental measurements.
"""

__author__ = "Angel Auñón"
__copyright__ = 'Copyright 2020, VEMOD plotter'
__credits__ = ["Angel Auñón"]
__license__ = 'GPL'
__version__ = '2.2.1'
__date__ = '2020-02-05'
__maintainer__ = "Angel Auñón"
__email__ = 'ngeaugar@mot.upv.es'
__status__ = 'Development'

# Imports
import os
import time
import sys
import argparse
import xmlschema
import multiprocessing
import pickle
import xml.etree.ElementTree as etree
import logging as log
import tkinter as tk
import tkinter.messagebox
from collections import OrderedDict
from shutil import copyfile, rmtree

# Libs
import pandas as pd
import numpy as np
from natsort import natsorted
from joblib import Parallel, delayed
from scipy.io import savemat
from scipy.integrate import cumtrapz

# Own modules
from resources.process_exp import process_experimental_steady,\
    process_experimental_emissions
from resources.engine_variable import Variable
from resources.plotting_functions import plot_combustion, plot_bars,\
    plot_stacked_bars, plot_trends, plot_evolution, plot_cylinder_var,\
    plot_transient_var, plot_transient_trends
from resources.stats import create_summary, get_cp, get_delay

# Globals
num_cores = multiprocessing.cpu_count()
"""The number of logical CPU cores of the machine"""

room_cond_dict = {'warm': 'h', 'cold': 'c', '': ''}
"""A dictionary that pairs steady-state room temperature with the
proper identifier letter in the case name"""

experimental_transient = {
    'warm': "../INPUT_dataset/medidas_WLTC/20170328_CicloWLTP_c1_"
            "Full_data_flow_corr V2 kvar DEF.xlsx",
    'ambient': "../INPUT_dataset/medidas_WLTC/20170911_CicloWLTP_ata3"
            "_Full_data_flow_corr V2 kvar DEF.xlsx",
    'cold': "../INPUT_dataset/medidas_WLTC/20170626_CicloWLTP_af4"
            "_Full_data_flow_corr V2 kvar DEF.xlsx"
}
"""A dictionary that pairs test bench room temperature with the
proper experimental transient data file"""


def parse_args(program) -> argparse.ArgumentParser.parse_args:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""
        VEMOD plotter -\n
        This program allows plotting simulation results (at transient\n
        or steady state operating conditions) from VEMOD and compare them\n
        with the experimental measurements.
        """,
        prog=program)

    parser.add_argument('config_file', type=str,
                        help="The configuration XML file.")
    parser.add_argument('mode', type=str, choices={'experimental',
                        'steady-avg', 'steady-ins', 'transient'},
                        metavar='mode',
                        help="What is going to be processed. Options: "
                             "experimental, steady-avg, steady-ins, "
                             "transient")
    parser.add_argument('-v', '--verbosity', required=False, action='count',
                        help="Increase output verbosity\
                             (e.g., -vv is more than -v)")
    parser.add_argument('--nomultiprocessing', required=False,
                        action='store_true',
                        help="Disables the multiprocessing when reading "
                             "files and plotting")
    parser.add_argument('-V', '--version', action='version',
                        version="VEMOD Plotter {}".format(__version__),
                        help="Show VEMOD plotter current version.")
    return parser.parse_args()


def create_folder(path, folder):
    """Create a folder in the given path"""

    try:
        os.makedirs(os.path.join(path, folder))
    except OSError:
        log.warning("Folder {}/{} already exists".format(path, folder))


def average_variable(array, percentage) -> float:
    """Average the last percentage of values in a given array.

    Parameters
    ----------
    array: pandas.Series, required
        The numeric array
    percentage: float, required
        The last percentage of values to be considered in the array

    Returns
    -------
    Float: the mean value of the selected values in the array
    """

    return array[-round(len(array) * percentage / 100):].mean()


def read_model_data(case_name, cases_dict, path) -> dict:
    """Read the results file corresponding steady-state case and store it
    in a pandas DataFrame.

    Parameters
    ----------
    case_name: str, required
        The name of the case
    cases_dict: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date, model file...
    path: str, required
        The processing path

    Returns
    -------
    dict: a dictionary with a single key (the case name) and a unique value,
        the DataFrame
    """

    return {case_name: pd.read_table(os.path.join(path,
                                     cases_dict[case_name]['model_file']),
                                     low_memory=False)}


def read_experimental_transient(filename, path) -> pd.DataFrame:
    """Read the transient experimental data file. First, try to load a pickle
    (binary) version of this file, if it is not available, then it loads the
    Excel version and saves its pickle version. Every time the original Excel
    file is modified, a new pickle version of this is saved.

    Parameters
    ----------
    filename: str, required
        The name of the experimental data file
    path: str, required
        The path where the experimental data file is stored

    Returns
    -------
    pandas.Dataframe: a DataFrame with all the transient experimental data
    """

    fileslog = '../INPUT_dataset/medidas_WLTC/fileslog.txt'
    if os.path.exists(fileslog):
        # Get the files and the last time they were modified
        content = [line.rstrip('\n') for line in open(fileslog, 'r')]
        files = [line.split(',')[0] for line in content]
        dates = [line.split(',')[1] for line in content]
    else:
        files = []
        dates = []

    if filename not in files:
        # Read the excel file
        exp_data = pd.read_excel(os.path.join(path, filename), skiprows=[1])
        # Save data as pickle to be loaded in the future
        pickle.dump(exp_data, open(os.path.join(path, os.path.splitext(
            filename)[0] + '.pickle'), 'wb'))
        # Write the filename and last modified time in fileslog
        with open(fileslog, 'a+') as flog:
            flog.write("{},{}\n".format(filename, time.ctime(os.path.getmtime(
                os.path.join(path, filename)))))
    else:
        # The file has been saved as pickle in the past
        last_time_modified = time.ctime(os.path.getmtime(os.path.join(
            path, filename)))
        if [date for file, date in zip(files, dates)
                if file == filename][0] == last_time_modified:
            # It has not been modified
            exp_data = pickle.load(open(os.path.join(path, os.path.splitext(
                filename)[0] + '.pickle'), 'rb'))
        else:
            # It has been modified, so read the excel file
            exp_data = pd.read_excel(os.path.join(path, filename),
                                     skiprows=[1])
            # Save data as pickle to be loaded in the future
            pickle.dump(exp_data, open(os.path.join(path, os.path.splitext(
                filename)[0] + '.pickle'), 'wb'))
            # Change the last time modified in fileslog
            with open(fileslog, 'w+') as flog:
                line_to_change = [line for line in content if filename in
                                  line][0]
                for line in content:
                    if line is line_to_change:
                        flog.write("{},{}\n".format(filename,
                                                    last_time_modified))
                    else:
                        flog.write("{}\n".format(line))

    return exp_data


def plot_variables(var, cases_dict, path, styles_dict, mode,
                   transient_dict=None) -> str:
    """Call the proper plotting function for each variable.

    Parameters
    ----------
    var: Variable, required
        The Variable class to be plotted
    cases_dict: dict, required
        The dictionary with the case names and some of their parameters:
        speed, load, torque, date...
    path: str, required
        The processing path
    styles_dict: dict, required
        The graphic styles: font size, font family, matplotlib style...
    mode: str, required
        The simulation mode: steady-avg, steady-ins, transient
    transient_dict: dict, optional, default: None
        The dictionary with the transient vehicle speed, the transient,
        duration, the model signal delay and the transient divisions

    Returns
    -------
    str: a string with the name of the variable and a message saying
        wether it could be plotted or not
    """

    try:
        if mode == 'steady-avg':
            if var.name == 'Injected fuel':
                plot_combustion(var.values, cases_dict, styles_dict,
                                path, units=var.units, limits=var.limits)
            elif var.name == 'Turbocharger Power':
                plot_stacked_bars(var.values, ['turbine'], ['compressor',
                                  'mech_losses'], cases_dict, styles_dict,
                                  path, var.name, units=var.units,
                                  limits=var.limits)
            elif var.name == 'Turbocharger Heat Power':
                plot_stacked_bars(var.values, ['turbine', 'compressor-'],
                                  ['ambient', 'oil', 'compressor+'],
                                  cases_dict, styles_dict, path, var.name,
                                  units=var.units, limits=var.limits)
            elif var.name == 'Turbocharger efficiency':
                plot_stacked_bars(var.values, ['turbine'], ['compressor'],
                                  cases_dict, styles_dict, path, var.name,
                                  units=var.units, limits=var.limits)
            else:
                if var.time_evolution is False:
                    plot_bars(var.values, cases_dict, styles_dict, path,
                              var.name, var.units, limits=var.limits)
                    if var.plot_trends:
                        plot_trends(var.values, cases_dict, styles_dict, path,
                                    var.name, var.units)
                else:
                    plot_evolution(var, cases_dict, styles_dict, path,
                                   var.name, var.units)
        elif mode == 'steady-ins':
            plot_cylinder_var(var, cases_dict, styles_dict, path, var.name,
                              var.units)
        elif mode == 'transient':
            plot_transient_var(var, styles_dict, path, var.name, var.units,
                               transient_dict)
            if var.plot_trends:
                if ('acc' not in var.name.lower()) and (var.name.endswith('_g')
                                                        is False):
                    plot_transient_trends(var, styles_dict, path, var.name,
                                          var.units, transient_dict)
        message = "{} plotted".format(var.name)
    except Exception as e:
        message = "{}:{} - {} could not be plotted".format(
            e.__class__.__name__, e, var.name)
    return message


def main(mode=None):
    # Read arguments
    args = parse_args(os.path.basename(sys.argv[0]))
    if args.verbosity is not None and args.verbosity == 2:
        log.getLogger().setLevel(log.DEBUG)
    elif args.verbosity is not None and args.verbosity == 1:
        log.getLogger().setLevel(log.INFO)

    # Validate config XML
    try:
        xmlschema.validate(args.config_file, 'config.xsd')
        log.info("{} config file read and valid".format(args.config_file))
    except xmlschema.validators.exceptions.XMLSchemaDecodeError as e:
        log.error("Error reading {}:\nReason: {}\nInstance: {}\n"
                  "Node path: {}".format(
                      args.config_file, e.reason,
                      etree.tostring(e.elem, encoding="unicode").strip(),
                      e.path),
                  exc_info=False)
        exit()
    if mode is not None:
        args.mode = mode

    # Convert the XML file into a dictionary, awesome!!
    xml_dict = xmlschema.XMLSchema('config.xsd').to_dict(args.config_file)

    # Create required subfolders
    processing_path = ''
    room_conditions = xml_dict['simulationProcessing']['settings'][
        '@roomTempConditions'] if '@roomTempConditions' in xml_dict[
            'simulationProcessing']['settings'].keys() else ''
    model_prefix = xml_dict['simulationProcessing']['settings'][
        '@modelNamePrefix']
    sim_mode = xml_dict['simulationProcessing']['settings'][
        '@simulationMode'] if '@simulationMode' in xml_dict[
            'simulationProcessing']['settings'].keys() else ''
    log.info("Room temperature conditions: {}".format(room_conditions))
    log.info("Model prefix: {}".format(model_prefix))
    log.info("Simulation mode: {}".format(sim_mode))
    if 'steady' in args.mode:
        room_conditions = 'warm' if room_conditions == 'ambient' \
            else room_conditions
        log.info("Creating subfolders")
        output_path = xml_dict['simulationProcessing']['settings'][
            '@processingPath'] if '@processingPath' in xml_dict[
                'simulationProcessing']['settings'].keys() else\
            os.path.abspath(os.path.dirname(__file__))
        processing_path = os.path.join(
            output_path,
            "{}{}{}".format(time.strftime("%Y-%b-%d", time.localtime()),
                            ('_' + room_conditions) if room_conditions != ''
                            else room_conditions,
                            ('_' + sim_mode) if sim_mode != '' else ''),
            xml_dict['simulationProcessing']['settings']['@xmlCase'])
        create_folder(processing_path, 'mat')
        create_folder(processing_path, 'xml')
        if args.mode == 'steady-avg':
            create_folder(processing_path, 'cycles')
            create_folder(processing_path, 'img/no_xticklabels')
            create_folder(processing_path, 'img/no_legend')
            if xml_dict['plottingOptions']['timeEvolution']['@plot']:
                create_folder(processing_path, 'time_evolution')
            if xml_dict['plottingOptions']['trends']['@plot']:
                create_folder(processing_path, 'trends/no_legend')
        elif args.mode == 'steady-ins':
            create_folder(processing_path, 'plots')
            create_folder(processing_path, 'img_ins')
    elif args.mode == 'transient':
        log.info("Creating subfolders")
        output_path = xml_dict['simulationProcessing']['settings'][
            '@processingPath'] if '@processingPath' in xml_dict[
                'simulationProcessing']['settings'].keys() else\
            os.path.abspath(os.path.dirname(__file__))
        processing_path = os.path.join(
            output_path,
            "{}{}{}".format(time.strftime("%Y-%b-%d", time.localtime()),
                            ('_' + room_conditions) if room_conditions != ''
                            else room_conditions,
                            ('_transient_' + sim_mode) if sim_mode != ''
                            else '_transient'),
            xml_dict['simulationProcessing']['settings']['@xmlCase'])
        create_folder(processing_path, 'mat')
        create_folder(processing_path, 'xml')
        create_folder(processing_path, 'cycles')
        create_folder(processing_path, 'plots')
        create_folder(processing_path, 'img/no_xticklabels')
        create_folder(processing_path, 'img/no_legend')
        if xml_dict['plottingOptions']['trends']['@plot']:
            create_folder(processing_path, 'trends')

    log.info("Processing mode: {}".format(args.mode))
    if args.mode == 'experimental':
        log.info("Reading engine data")
        engine = {
            'cylinders': xml_dict['engine']['@cylinders'],
            'total_displacement': xml_dict['engine']['@totalDisplacement'],
            'injection_pulses': xml_dict['engine']['@injectionPulses'],
            'PCI': xml_dict['engine']['@fuelPCI']}
        log.debug("Cylinders: {}, Fuel PCI: {} MJ/kg, Inj. pulses: {}, "
                  "Total displacement: {} m^3".format(
                      engine['cylinders'], engine['PCI'],
                      engine['injection_pulses'], engine['total_displacement'])
                  )
        log.info("Reading cases")
        cases = [c['@name'] for c in xml_dict['simulationProcessing'][
            'steady']['cases']['case']]
        log.debug(cases)
        calmec_avg_file = [f['$'] for f in xml_dict['experimentalProcessing'][
            'inputFile'] if f['@category'] == 'calmecAverages'][0].strip()
        exp_raw_file = [f['$'] for f in xml_dict['experimentalProcessing'][
            'inputFile'] if f['@category'] == 'experimentalRaw'][0].strip()
        log.debug("CALMEC average data file: {}".format(calmec_avg_file))
        log.debug("Experimental raw data file: {}".format(exp_raw_file))
        log.info("Processing experimental data")
        process_experimental_steady(
            calmec_avg_file, exp_raw_file, engine, cases,
            xml_dict['experimentalProcessing']['outputFile'].strip())
        exit()

    if 'steady' in args.mode:
        log.info("Reading case names")
        cases_dict = OrderedDict()
        for c in xml_dict['simulationProcessing']['steady']['cases']['case']:
            cases_dict[c['@name']] = {}
            for i, n in enumerate(c['nomenclature']['labelPart']):
                cases_dict[c['@name']][n['@property']] = c['@name'].split(
                    '_')[i]
            if 'speed' not in cases_dict[c['@name']].keys():
                cases_dict[c['@name']]['speed'] = ''
        # Only take those cases which match the current room temperature
        try:
            for case, props in cases_dict.copy().items():
                if room_cond_dict[room_conditions] not in\
                        props['roomTemperature']:
                    cases_dict.pop(case)
        except Exception:
            log.warning('Could not found any room temperature in tests '
                        'names. Ignoring room temperature conditions.')
            room_conditions = ''
        if args.verbosity is not None and args.verbosity == 2:
            for case, props in cases_dict.items():
                log.debug(case)
                log.debug(props)

    log.info("Copying simulation results from: {}\nto: {}".format(
        xml_dict['simulationProcessing']['settings']['@simulationPath'],
        processing_path))
    for root, dirs, files in os.walk(os.path.join(xml_dict[
        'simulationProcessing']['settings']['@simulationPath'].strip(),
            xml_dict['simulationProcessing']['settings']['@xmlCase'])):
        for fname in files:
            if (model_prefix in fname) and (room_conditions in fname)\
                    and (sim_mode in fname):
                if 'cycles' in fname and args.mode != 'steady-ins':
                    try:
                        copyfile(os.path.join(root, fname),
                                 os.path.join(processing_path,
                                 'cycles', fname))
                        log.debug(fname)
                    except OSError as e:
                        log.error(e)
                        pass
                elif '.xml' in fname:
                    try:
                        copyfile(os.path.join(root, fname),
                                 os.path.join(processing_path, 'xml', fname))
                        log.debug(fname)
                    except OSError as e:
                        log.error(e)
                        pass
                elif '.mat' in fname:
                    try:
                        copyfile(os.path.join(root, fname),
                                 os.path.join(processing_path, 'mat', fname))
                        log.debug(fname)
                    except OSError as e:
                        log.error(e)
                        pass
                elif 'plots' in fname and args.mode == 'steady-ins':
                    try:
                        copyfile(os.path.join(root, fname),
                                 os.path.join(processing_path, 'plots', fname))
                        log.debug(fname)
                    except OSError as e:
                        log.error(e)
                        pass

    log.debug("Assigning simulation results to cases")
    if args.mode == "steady-avg":
        files_mod = natsorted([f for f in os.listdir(os.path.join(
            processing_path, 'cycles'))])
    elif args.mode == "steady-ins":
        files_mod = natsorted([f for f in os.listdir(os.path.join(
            processing_path, 'plots'))])
    elif args.mode == "transient":
        files_mod = natsorted([f for f in os.listdir(os.path.join(
            processing_path, 'cycles'))])
    if args.verbosity is not None and args.verbosity == 2:
        for f in files_mod:
            log.debug(f)
    if len(files_mod) == 0:
        log.error("NO RESULTS FILES WERE FOUND\n"
                  "Check that {} is not empty. "
                  'Also check that "modelNamePrefix", "roomTempConditions"'
                  ' and "simulationMode" fields in {} are right.'.format(
                      xml_dict['simulationProcessing']['settings'][
                            '@simulationPath'], args.config_file))
        exit()
    if 'steady' in args.mode:
        # Sort also the cases_dict dictionary to prevent mismatching
        cases_dict = OrderedDict(natsorted(cases_dict.items()))
        # Assigning model file name to each case in case_dict
        files_assigned = []
        for case, props in cases_dict.items():
            for f in natsorted(set(files_mod) - set(files_assigned)):
                if 'model_file' in props.keys():
                    break
                else:
                    if props['speed'].lstrip('0') in f:
                        cases_dict[case]['model_file'] = f
                        files_assigned.append(f)
                    elif 'undefined' in props.keys():
                        log.warning("Speed value not found in case name. "
                                    "Sorting is not fully guaranteed.")
                        cases_dict[case]['model_file'] = f
                        files_assigned.append(f)
                # Also get the apparent combustion efficiency for that case
                if args.mode == "steady-avg":
                    try:
                        with open(os.path.join(processing_path, 'xml',
                                  f.split('_cycles.dat')[0] + '.xml')) as xml:
                            lines = xml.readlines()
                            for l in lines:
                                if 'AparentCombEff=' in l:
                                    cases_dict[case]['app_comb_eff'] = float(
                                        l.split('AparentCombEff="')[1].split(
                                            '"')[0]) * 100
                                    break
                                elif 'ApparentCombEff=' in l:
                                    cases_dict[case]['app_comb_eff'] = float(
                                        l.split('ApparentCombEff="')[1].split(
                                            '"')[0]) * 100
                                    break
                            else:
                                cases_dict[case]['app_comb_eff'] = 100
                    except OSError as e:
                        log.warning(e)
                        log.warning("Taking an apparent combustion efficiency "
                                    "of 100%, does not affect the calculus.")
                        cases_dict[case]['app_comb_eff'] = 100
                        pass

    # ===================  DECLARE MEAN CYCLE VARIABLES  ======================
    if "steady" in args.mode:
        exp_file = xml_dict['simulationProcessing']['steady'][
            '@experimentalFile'].strip()
        log.info("Reading experimental file: {}".format(exp_file))
        exp_data = pd.read_excel(exp_file)
        # Just take the rows matching the selected cases
        exp_data = exp_data.loc[exp_data['TEST'].isin(
            [t for t in exp_data['TEST'] if room_cond_dict[room_conditions] in
             t and t in cases_dict.keys()])]
        exp_data.sort_values('TEST', ascending=1, inplace=True)
    elif args.mode == 'transient':
        if room_conditions == '':
            log.error("Define the room temperature conditions in the config "
                      "XML file.")
            exit()
        exp_file = experimental_transient[room_conditions]
        log.info("Reading experimental file: {}".format(exp_file))
        exp_data = read_experimental_transient(
            os.path.basename(exp_file), '../INPUT_dataset/medidas_WLTC')

    if args.mode == "steady-avg":
        var_table_file = xml_dict['simulationProcessing']['steady'][
            '@variablesAverageFile'].strip()
        log.info("Reading variables file {}".format(var_table_file))
        try:
            var_table = pd.read_table(var_table_file)
        except Exception as e:
            log.error(e)
            exit()
        log.info("Found {} variables, {} printable".format(
            var_table.shape[0], (var_table['Print'] == 'Yes').sum()))
        cycles = float(xml_dict['simulationProcessing']['steady'][
            '@averagingPercentage']) if '@averagingPercentage' in\
            xml_dict['simulationProcessing']['steady'].keys() else 3.0
        log.info("{} % of the last cycles taken for mean values.".format(
            cycles))

        log.info("Declaring variables")
        var_list = []
        for i in var_table.index:
            if var_table.at[i, 'Print'].lower() == 'yes':
                try:
                    var_list.append(Variable(
                        {'name': var_table.at[i, 'Name'],
                         'model_col': var_table.at[i, 'Mod_column'],
                         'exp_col': var_table.at[i, 'Ref_column'],
                         'units': var_table.at[i, 'Units'],
                         'conv_factor':
                            var_table.at[i, 'ModToRef_factor'],
                         'limits':
                            (var_table.at[i, 'Ymin'],
                             var_table.at[i, 'Ymax']),
                         'plot_trends':
                            xml_dict['plottingOptions']['trends']['@plot'],
                         'plot_time_evolution': False,
                         'is_pollutant': False,
                         }))
                except Exception as e:
                    log.error(e)

        if xml_dict['plottingOptions']['emissions']['@plot']:
            log.info("Declaring pollutant variables")
            location = xml_dict['plottingOptions']['emissions'][
                '@location'].strip()
            for p, n in zip(['O2', 'CO2', 'NOx', 'Soot', 'CO', 'FUEL'],
                            ['O2', 'CO2', 'NOx', 'Soot', 'CO', 'THC']):
                var_list.append(Variable(
                    {'name': '{}_Y'.format(n),
                     'model_col': '{}/MassFraction[-]/{}'.format(location, p),
                     'exp_col': '{}_Y'.format(n),
                     'plot_trends':
                        xml_dict['plottingOptions']['trends']['@plot'],
                     'plot_time_evolution': False,
                     'is_pollutant': True,
                     }))
                var_list.append(Variable(
                    {'name': '{}_g_s'.format(n),
                     'model_col': '{}/MassFraction[-]/{}'.format(location, p),
                     'exp_col': '{}_g_s'.format(n),
                     'conv_factor': 1000,
                     'units': 'g/s',
                     'plot_trends':
                        xml_dict['plottingOptions']['trends']['@plot'],
                     'plot_time_evolution': False,
                     'is_pollutant': True,
                     }))
            process_experimental_emissions(exp_data, mode='steady-avg')

        if xml_dict['plottingOptions']['timeEvolution']['@plot']:
            log.info("Declaring time evolution variables")
            evolution_table_file = xml_dict['plottingOptions'][
                'timeEvolution']['@steadyVariablesFile'].strip()
            log.info("Reading time evolution variables file {}".format(
                evolution_table_file))
            try:
                evolution_table = pd.read_table(evolution_table_file)
                for i in range(evolution_table.shape[0]):
                    if evolution_table.at[i, 'Print'].lower() == 'yes':
                        if evolution_table.at[i, 'Name'] in\
                                [v.name for v in var_list]:
                            continue
                        else:
                            var_list.append(Variable(
                                {'name': evolution_table.at[i, 'Name'],
                                 'model_col':
                                    evolution_table.loc[evolution_table[
                                        'Name'] == evolution_table.at[
                                        i, 'Name'], 'Mod_column'].values,
                                 'units': evolution_table.at[i, 'Units'],
                                 'conv_factor':
                                    evolution_table.at[i, 'Conv_factor'],
                                 'plot_trends': False,
                                 'plot_time_evolution': True,
                                 'is_pollutant': False,
                                 'extra': {
                                    'legend': evolution_table.loc[
                                        evolution_table[
                                            'Name'] == evolution_table.at[
                                            i, 'Name'], 'Legend'].values}
                                 }))
            except Exception as e:
                log.error(e)
                pass

        path_model = os.path.join(processing_path, 'cycles')

    # ==============  DECLARE CYLINDER INSTANTANEOUS VARIABLES  ===============
    elif args.mode == "steady-ins":
        log.info("Reading CALMEC cylinder files.")
        # Retrieve Calmec steady files
        calmec_data = {}
        calmec_path = xml_dict['simulationProcessing']['steady'][
            '@calmecInstantaneousPath'].strip()
        try:
            for test in exp_data['TEST']:
                calmec_data[test] = {}
                calmec_data[test]['ins'] = []
                calmec_data[test]['cyl. pressure cycle'] = []
                calmec_data[test]['angle cycle'] = []
                for cil in range(1, xml_dict['engine']['@cylinders'] + 1):
                    file1 = "{}_cil{}.dat".format(test, cil)
                    calmec1 = pd.read_csv(os.path.join(calmec_path, file1),
                                          encoding='latin1', engine='python',
                                          sep=None)
                    file2 = "{}_cil{}_ciclo.dat".format(test, cil)
                    calmec2 = pd.read_csv(os.path.join(calmec_path, file2),
                                          encoding='latin1', engine='python',
                                          sep=None)
                    try:
                        file3 = "p_{} Cil {} (PCIL) .dat".format(test, cil)
                        calmec3 = pd.read_csv(os.path.join(calmec_path, file3),
                                              encoding='latin1',
                                              engine='python', sep=None)
                        calmec_data[test]['angle cycle'].append(
                            calmec3.iloc[:, 0])
                        calmec_data[test]['cyl. pressure cycle'].append(
                            calmec3.iloc[:, 1])
                    except Exception:
                        pass
                    calmec2['Cv(J/kgK)'] = calmec2['CVa(J/kgK)'] *\
                        calmec2['Ya(-)'] + calmec2['CVf(J/kgK)'] *\
                        calmec2['Yf(-)'] + calmec2['CVq(J/kgK)'] *\
                        calmec2['Yq(-)']
                    calmec2['Gamma(-)'] = (calmec2['Cv(J/kgK)'] +
                                           calmec2['R(J/kgK)']) /\
                        calmec2['Cv(J/kgK)']
                    common_keys = list(set(calmec1.columns) & set(
                        calmec2.columns))
                    calmec_data[test]['ins'].append(
                        pd.merge(calmec1, calmec2, on=common_keys))
                for col in exp_data.columns:
                    calmec_data[test]['avg'] = exp_data.loc[exp_data[
                        'TEST'] == test]
            exp_data = calmec_data
            del calmec_data
        except Exception as e:
            log.error(e)
            exit()

        var_table_file = xml_dict['simulationProcessing']['steady'][
            '@variablesInstantaneousFile'].strip()
        log.info("Reading variables file {}".format(var_table_file))
        try:
            var_table = pd.read_table(var_table_file)
        except Exception as e:
            log.error(e)
            exit()
        log.info("Found {} variables, {} printable".format(
            var_table.shape[0], (var_table['Print'] == 'Yes').sum()))

        log.info("Declaring variables")
        var_list = []
        for i in var_table.index:
            if var_table.at[i, 'Print'].lower() == 'yes':
                try:
                    var_list.append(Variable(
                        {'name': var_table.at[i, 'Name'],
                         'model_col': var_table.at[i, 'Mod_column'],
                         'exp_col': var_table.at[i, 'Ref_column'],
                         'units': var_table.at[i, 'Units'],
                         'conv_factor':
                            var_table.at[i, 'K_exp'],
                         'limits':
                            (var_table.at[i, 'Xmin'],
                             var_table.at[i, 'Xmax']),
                         'plot_trends': False,
                         'plot_time_evolution': False,
                         'is_pollutant': False,
                         }))
                except Exception as e:
                    log.error(e)

        path_model = os.path.join(processing_path, 'plots')

    # ===================  DECLARE TRANSIENT VARIABLES  =======================
    elif args.mode == 'transient':
        var_table_file = xml_dict['simulationProcessing']['transient'][
            '@variablesFile'].strip()
        log.info("Reading variables file {}".format(var_table_file))
        try:
            var_table = pd.read_table(var_table_file)
        except Exception as e:
            log.error(e)
            exit()
        log.info("Found {} variables, {} printable".format(
            var_table.shape[0], (var_table['Print'] == 'Yes').sum()))

        log.info("Declaring variables")
        var_list = []
        for i in var_table.index:
            if var_table.at[i, 'Print'].lower() == 'yes':
                try:
                    is_pollutant = True if var_table.at[i, 'Name'] in \
                         ['COH', 'CO2', 'THC', 'O2', 'NOx'] else False
                    if xml_dict['plottingOptions']['emissions']['@plot'] is\
                            False and is_pollutant is True:
                        continue
                    else:
                        var_list.append(Variable(
                            {'name': var_table.at[i, 'Name'],
                             'model_col': var_table.at[i, 'Mod_column'],
                             'exp_col': var_table.at[i, 'Ref_column'],
                             'units': var_table.at[i, 'Units'],
                             'conv_factor':
                                var_table.at[i, 'ModToRef_factor'],
                             'limits':
                                (var_table.at[i, 'Ymin'],
                                 var_table.at[i, 'Ymax']),
                             'plot_trends':
                                xml_dict['plottingOptions']['trends']['@plot'],
                             'plot_time_evolution': False,
                             'is_pollutant': is_pollutant,
                             'extra': {'tau': var_table.at[i, 'Tau']}
                             }))
                except Exception as e:
                    log.error(e)

        if xml_dict['plottingOptions']['emissions']['@plot']:
            log.info("Declaring pollutant variables")
            location = xml_dict['plottingOptions']['emissions'][
                '@location'].strip()
            for p, n in zip(['O2', 'CO2', 'NOx', 'Soot', 'CO', 'FUEL'],
                            ['O2', 'CO2', 'NOx', 'Soot', 'CO', 'THC']):
                var_list.append(Variable(
                    {'name': '{}_g_s'.format(n),
                     'model_col': '{}/MassFraction[-]/{}'.format(location, p),
                     'exp_col': '{}_g_s'.format(n),
                     'conv_factor': 1000,
                     'units': 'g/s',
                     'plot_trends':
                        xml_dict['plottingOptions']['trends']['@plot'],
                     'plot_time_evolution': False,
                     'is_pollutant': True,
                     }))
                var_list.append(Variable(
                    {'name': '{}_g'.format(n),
                     'model_col': '{}/MassFraction[-]/{}'.format(location, p),
                     'exp_col': '{}_g'.format(n),
                     'conv_factor': 1000,
                     'units': 'g',
                     'plot_trends':
                        xml_dict['plottingOptions']['trends']['@plot'],
                     'plot_time_evolution': False,
                     'is_pollutant': True,
                     }))

        if xml_dict['plottingOptions']['timeEvolution']['@plot']:
            log.info("Declaring only model variables")
            model_var_table_file = xml_dict['plottingOptions'][
                'timeEvolution']['@transientVariablesFile'].strip()
            log.info("Reading only model variables file {}".format(
                model_var_table_file))
            try:
                table = pd.read_table(model_var_table_file)
                for i in range(table.shape[0]):
                    if table.at[i, 'Print'].lower() == 'yes':
                        if table.at[i, 'Name'] in [v.name for v in var_list]:
                            continue
                        else:
                            var_list.append(Variable(
                                {'name': table.at[i, 'Name'],
                                 'model_col':
                                    table.loc[table['Name'] == table.at[
                                        i, 'Name'], 'Mod_column'].values,
                                 'units': table.at[i, 'Units'],
                                 'conv_factor':
                                    table.loc[table['Name'] == table.at[
                                        i, 'Name'], 'Conv_factor'].values,
                                 'plot_trends': False,
                                 'plot_time_evolution': True,
                                 'is_pollutant': False,
                                 'extra': {
                                    'legend': table.loc[table[
                                        'Name'] == table.at[i, 'Name'],
                                        'Legend'].values}
                                 }))
            except Exception as e:
                log.error(e)
                pass

        path_model = os.path.join(processing_path, 'cycles')

    log.info("Assigning values to variables")
    if 'steady' in args.mode:
        if args.nomultiprocessing is False:
            model_data = Parallel(n_jobs=num_cores, verbose=8)(delayed(
                read_model_data)(case, cases_dict, path_model) for case in
                cases_dict.keys())
        else:
            t = time.time()
            model_data = []
            for case in cases_dict.keys():
                model_data.append(read_model_data(
                    case, cases_dict, path_model))
    else:
        t = time.time()

    # ===================  PROCESS MEAN CYCLE VARIABLES  ======================
    if args.mode == "steady-avg":
        # Model data at IVC
        IVC_data = pd.DataFrame(
            index=var_list[0].values.keys(),
            columns=['Speed [rpm]', 'Mass flow (cycle) [kg/s]',
                     'Pressure [bar]', 'Temperature [degC]', 'Y O2', 'Y N2',
                     'Y CO', 'Y CO2', 'Y H2O', 'Y NOx', 'Y soot', 'Y THC',
                     'Fuel (cycle) [mg]', 'Ignition delay [s]', 'SOC [rad]',
                     'EOC [rad]', 'Real phi', 'Air (cycle) [mg]',
                     'Density (cycle) [kg/m3]'])
        for case in model_data:
            key = list(case.keys())[0]
            mod_case = case[key]
            exp_case = exp_data.loc[exp_data['TEST'] == key]

            # IVC data
            try:
                IVC_data.loc[key, 'Speed [rpm]'] = average_variable(mod_case[
                    'Engine/EngineSpeed[rpm]'], cycles)
                IVC_data.loc[key, 'Mass flow (cycle) [kg/s]'] =\
                    average_variable(mod_case['Engine/AirFlowRate[kg/s]'],
                                     cycles)
                IVC_data.loc[key, 'Fuel (cycle) [mg]'] = average_variable(
                    mod_case['Engine/FuelMassCylinder&Cycle[mg]'], cycles)
                IVC_data.loc[key, 'Air (cycle) [mg]'] = IVC_data.at[
                    key, 'Mass flow (cycle) [kg/s]'] * 1e6 * 2 /\
                    xml_dict['engine']['@cylinders'] * 60 /\
                    IVC_data.at[key, 'Speed [rpm]']
                IVC_data.loc[key, 'Density (cycle) [kg/m3]'] = IVC_data.at[
                    key, 'Air (cycle) [mg]'] / 1e6 / 0.350058 * 1e3
                IVC_data.loc[key, 'Y O2'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/O2'], cycles)
                IVC_data.loc[key, 'Y N2'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/N2'], cycles)
                IVC_data.loc[key, 'Y CO'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/CO'], cycles)
                IVC_data.loc[key, 'Y CO2'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/CO2'], cycles)
                IVC_data.loc[key, 'Y H2O'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/H2Ov'], cycles)
                IVC_data.loc[key, 'Y NOx'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/NOx'], cycles)
                IVC_data.loc[key, 'Y soot'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/Soot'], cycles)
                IVC_data.loc[key, 'Y THC'] = average_variable(
                    mod_case['Engine/MassFractionAtIVC[-]/FUEL'], cycles)
                IVC_data.loc[key, 'Pressure [bar]'] = average_variable(
                    mod_case['Cylinder/Cylinder-1/PressureAtIVC[bar]'],
                    cycles)
                IVC_data.loc[key, 'Temperature [degC]'] = average_variable(
                    mod_case['Cylinder/Cylinder-1/TemperatureAtIVC[degC]'],
                    cycles)
                IVC_data.loc[key, 'Ignition delay [s]'] = average_variable(
                    mod_case['Cylinder/Cylinder-1/IgnitionDelay[s]'], cycles)
                IVC_data.loc[key, 'SOC [rad]'] = average_variable(
                    mod_case['Cylinder/Cylinder-1/StartOfCombustion[rad]'],
                    cycles)
                IVC_data.loc[key, 'EOC [rad]'] = average_variable(
                    mod_case['Cylinder/Cylinder-1/EndOfCombustion[rad]'],
                    cycles)
                IVC_data.loc[key, 'Real phi'] = average_variable(
                    mod_case['Cylinder/Cylinder-1/InCylinderRichness[-]'],
                    cycles)
                writer = pd.ExcelWriter(os.path.join(processing_path,
                                                     'IVC_data.xlsx'))
            except Exception as e:
                log.error(e)
                pass

            var_fail_index_list = []
            for i, var in enumerate(var_list):
                try:
                    if var.name == 'Turbocharger Power':
                        var.set_values({key: {
                            'model':
                            {'turbine': average_variable(
                                mod_case[var.model_col.split(',')[0].strip()],
                                cycles),
                             'compressor': average_variable(
                                mod_case[var.model_col.split(',')[1].strip()],
                                cycles),
                             'mech_losses': average_variable(
                                mod_case[var.model_col.split(',')[2].strip()],
                                cycles),
                             }
                        }})
                        var.exp_col = None
                    elif var.name == 'Turbocharger Heat Power':
                        var.set_values({key: {
                            'model':
                            {'turbine': average_variable(mod_case[
                                var.model_col.split(',')[0].strip()] * -1,
                                cycles),
                             'compressor+': average_variable(
                                mod_case[var.model_col.split(',')[1].strip()],
                                cycles) if average_variable(
                                mod_case[var.model_col.split(',')[1].strip()],
                                cycles) >= 0.
                                else 0.,
                             'compressor-': average_variable(
                                mod_case[var.model_col.split(',')[1].strip()],
                                cycles) if average_variable(
                                mod_case[var.model_col.split(',')[1].strip()],
                                cycles) < 0.
                                else 0.,
                             'oil': average_variable(mod_case[
                                var.model_col.split(',')[2].strip()], cycles),
                             'ambient': average_variable(mod_case[
                                var.model_col.split(',')[3].strip()], cycles),
                             }
                        }})
                        var.exp_col = None
                    elif var.name == 'Turbocharger efficiency':
                        var.set_values({key: {
                            'model':
                            {'turbine': average_variable(mod_case[
                                var.model_col.split(',')[0].strip()],
                                cycles),
                             'compressor': average_variable(mod_case[
                                var.model_col.split(',')[1].strip()],
                                cycles),
                             }
                        }})
                        var.exp_col = None
                    elif var.name == 'Rings+Piston Friction':
                        var.set_values({key: {
                            'model': (average_variable(mod_case[
                                'Cylinder/Cylinder-1/S1FrictionPower[kW]'],
                                cycles) + average_variable(mod_case[
                                    'Cylinder/Cylinder-1/S2FrictionPower[kW]'],
                                cycles) + average_variable(mod_case[
                                    'Cylinder/Cylinder-1/S3FrictionPower[kW]'],
                                cycles) + average_variable(mod_case[
                                    'Cylinder/Cylinder-1/'
                                    'SkirtFrictionPower[kW]'],
                                cycles)) * var.conv_factor,
                            'exp': exp_case[var.exp_col].values[0]
                        }})
                    elif var.name == 'Friction + Auxiliaries':
                        var.set_values({key: {
                            'model': (average_variable(mod_case[
                                'Engine/NMEP[bar]'], cycles) -
                                average_variable(mod_case[
                                    'Engine/BMEP[bar]'], cycles)) *
                            var.conv_factor,
                            'exp': exp_case[var.exp_col].values[0]
                        }})
                    elif var.name == 'Injected fuel':
                        var.set_values({key: {
                            'model':
                            {'injected': average_variable(
                                mod_case['Cylinder/Cylinder-2/Combustion/'
                                         'InjectedFuel[mg]'], cycles),
                             'burned': average_variable(
                                mod_case['Cylinder/Cylinder-2/Combustion/'
                                         'BurnedFuel[mg]'], cycles),
                             'premix_burned': average_variable(
                                mod_case['Cylinder/Cylinder-2/Combustion/'
                                         'PremixBurnedFuel[mg]'], cycles),
                             'premix_unburned': average_variable(
                                mod_case['Cylinder/Cylinder-2/Combustion/'
                                         'PremixUnburnedFuel[mg]'],
                                cycles),
                             }
                        }})
                        var.exp_col = None
                    elif var.name == 'HP EGR Rate':
                        amf_col = [v.model_col for v in var_list
                                   if v.name == 'Mass Flow'][0]
                        amf = average_variable(mod_case[amf_col], cycles)
                        int_ports_amf_cols = var.model_col.split(',')
                        int_ports_amf = 0.0
                        for col in int_ports_amf_cols:
                            int_ports_amf += average_variable(mod_case[
                                col.strip()], cycles)
                        var.set_values({key: {
                            'model': 100 * (int_ports_amf - amf) /
                            (int_ports_amf) * var.conv_factor,
                            'exp': exp_case[var.exp_col].values[0]
                        }})
                    elif var.name == 'EGR Rate':
                        amf_col = [v.model_col for v in var_list
                                   if v.name == 'Mass Flow'][0]
                        amf = average_variable(mod_case[amf_col], cycles)
                        lpegr_amf = average_variable(mod_case[var.model_col],
                                                     cycles)
                        var.set_values({key: {
                            'model': 100 * lpegr_amf /
                            (lpegr_amf + amf) * var.conv_factor,
                            'exp': exp_case[var.exp_col].values[0]
                        }})
                    elif var.name == 'VGT Position':
                        var.set_values({key: {
                            'model': average_variable(mod_case[
                                var.model_col], cycles) * var.conv_factor,
                            'exp': 2.1909 * (100 - exp_case[
                                var.exp_col].values[0]) - 78.548,
                        }})
                    # Pollutants mass flow
                    elif '_g_s' in var.name:
                        model_col_amf = var.model_col.rsplit('/', 2)[0] +\
                            '/MassFlow[kg/s]'
                        var.set_values({key: {
                            'model': (average_variable(mod_case[
                                var.model_col], cycles) *
                                average_variable(mod_case[model_col_amf],
                                                 cycles)) * var.conv_factor,
                            'exp': exp_case[var.exp_col].values[0]
                        }})
                    else:
                        if var.time_evolution is True:
                            for col in var.model_col:
                                if 'hydrcircuit' in col.lower():
                                    if col not in mod_case.columns:
                                        col = col.replace('CoolantCircuit-1',
                                                          'MainCoolantCircuit')
                                        col = col.replace('OilCircuit-1',
                                                          'LubricationCircuit')
                            var.set_values({key: {
                                'model': [mod_case[val] * var.conv_factor
                                          for val in var.model_col],
                                'time': mod_case['Time[s]']
                            }})
                        else:
                            if 'hydrcircuit' in var.model_col.lower():
                                if var.model_col not in mod_case.columns:
                                    var.model_col = var.model_col.replace(
                                        'CoolantCircuit-1',
                                        'MainCoolantCircuit')
                                    var.model_col = var.model_col.replace(
                                        'OilCircuit-1', 'LubricationCircuit')
                            var.set_values({key: {
                                'model': average_variable(mod_case[
                                    var.model_col], cycles) * var.conv_factor,
                                'exp': exp_case[var.exp_col].values[0]
                            }})
                except Exception as e:
                    log.error(e)
                    var_fail_index_list.append(i)

            # Remove failed variables from var_list
            for ind in sorted(var_fail_index_list, reverse=True):
                var_list.pop(ind)

        # Save IVC data
        IVC_data.to_excel(writer, 'Sheet1')
        writer.save()
        log.info("Saved IVC data file at {}".format(processing_path))

    # ===============  PROCESS CYLINDER INSTANTANEOUS VARIABLES  ==============
    elif args.mode == "steady-ins":
        for case in model_data:
            key = list(case.keys())[0]
            mod_case = case[key]
            # exp_data now is a huge dictionary, not a dataFrame
            exp_case = exp_data[key]

            var_fail_index_list = []
            for i, var in enumerate(var_list):
                try:
                    if (var.name == 'Heat') or var.name == 'BlowBy':
                        var.set_values({key: {
                            'model':
                            {'x': np.linspace(
                                -360, 360, len(mod_case[var.model_col])),
                             'y': mod_case[var.model_col].values},
                            'exp':
                            {'x': [cdata['angulo(cad)'] for cdata in
                                   exp_case['ins']],
                             'y': [cdata[var.exp_col].values * exp_case['avg'][
                                   'DynoSpeed'].values * var.conv_factor for
                                   cdata in exp_case['ins']]}
                        }})
                    elif var.name == 'Heat Release':
                        var.set_values({key: {
                            'model':
                            {'x': np.linspace(
                                -360, 360, len(mod_case[var.model_col])),
                             'y': mod_case[var.model_col].values *
                                exp_case['avg']['TotalHRL'].values},
                            'exp':
                            {'x': [cdata['angulo(cad)'] for cdata in
                                   exp_case['ins']],
                             'y': [cdata[var.exp_col].values * var.conv_factor
                                   for cdata in exp_case['ins']]}
                        }})
                    elif var.name == 'Fuel':
                        var.set_values({key: {
                            'model':
                            {'x': np.linspace(
                                -360, 360, len(mod_case[var.model_col])),
                             'y':
                                {'injected': mod_case[var.model_col],
                                 'burned': mod_case[
                                    'Cylinder/Cylinder-2/Combustion/'
                                    'BurnedFuel[mg]'],
                                 'premix_burned': mod_case[
                                    'Cylinder/Cylinder-2/Combustion/'
                                    'PremixBurnedFuel[mg]'],
                                 'premix_unburned': mod_case[
                                    'Cylinder/Cylinder-2/'
                                    'Combustion/PremixUnburnedFuel[mg]']
                                 }
                             }}})
                    elif var.name == 'Fuel injected - burned norm. vs O2':
                        var.set_values({key: {
                            'model':
                            {'y':
                                {'injected': mod_case[var.model_col],
                                 'burned': mod_case[
                                    'Cylinder/Cylinder-2/Combustion/'
                                    'BurnedFuel[mg]'],
                                 'O2': mod_case[
                                    'Cylinder/Cylinder-2/MassFraction[-]/O2'],
                                 }
                             }}})
                    elif var.name == 'Cyl. Pressure':
                        y_exp = []
                        # Centering cylinder pressure
                        for i in range(xml_dict['engine']['@cylinders']):
                            cyl_press = exp_case['cyl. pressure cycle'][i]
                            cyl_angle = exp_case['angle cycle'][i]
                            if cyl_angle[0] <= -360:
                                start_index = np.where(
                                    cyl_angle >= -360)[0][0]
                                tail = np.take(cyl_press, np.where(
                                    cyl_angle <= -360)[0])
                                y_exp.append(np.concatenate(
                                    (cyl_press[start_index:], tail)))
                            else:
                                end_index = np.where(
                                    cyl_angle >= 360)[0][0]
                                head = np.take(cyl_press, np.where(
                                    cyl_angle >= 360)[0])
                                y_exp.append(np.concatenate(
                                    (head, cyl_press[0:end_index])))
                        var.set_values({key: {
                            'model':
                            {'x': np.linspace(
                                -360, 360, len(mod_case[var.model_col])),
                             'y': mod_case[var.model_col].values},
                            'exp':
                            {'x': [np.linspace(
                                -360, 360, len(y_cil)) for y_cil in y_exp],
                             'y': [y_cil for y_cil in y_exp]}
                        }})
                    else:
                        var.set_values({key: {
                            'model':
                            {'x': np.linspace(
                                -360, 360, len(mod_case[var.model_col])),
                             'y': mod_case[var.model_col].values},
                            'exp':
                            {'x': [cdata['angulo(cad)'] for cdata in
                                   exp_case['ins']],
                             'y': [cdata[var.exp_col].values * var.conv_factor
                                   for cdata in exp_case['ins']]}
                        }})
                except Exception as e:
                    log.error(e)
                    var_fail_index_list.append(i)

            # Remove failed variables from var_list
            for ind in sorted(var_fail_index_list, reverse=True):
                var_list.pop(ind)

    # ===================  PROCESS TRANSIENT VARIABLES  =======================
    elif args.mode == 'transient':
        mod_case = pd.read_table(os.path.join(processing_path, 'cycles',
                                              files_mod[0]))
        delay = get_delay(mod_case['Time[s]'].values,
                          mod_case['Engine/EngineSpeed[rpm]'].values,
                          exp_data['Time'].values,
                          exp_data['DynoSpeed'].values,
                          limits=(0, xml_dict['simulationProcessing'][
                              'transient']['@simTime']))
        log.debug("Model signals are {} {:.2f} seconds with respect to the "
                  "experimental ones".format('delayed' if delay < 0
                                             else 'advanced', abs(delay)))

        veh_speed = exp_data['ActVelocity']

        # Some experimental tranformations
        # Experimental EGR, blow-by, air mass flow and fuel mass flow
        exp_data['Air_mass_inca_corr_g_s'] = exp_data['ma_inca_corr'] * 1e-3 *\
            exp_data['DynoSpeed'] / 60 * xml_dict['engine']['@cylinders'] / 2
        exp_data['EGR_Rate'] = exp_data['EGRCO2'] / exp_data['CO2'] * 100
        exp_data.loc[exp_data['EGR_Rate'] > 60., 'EGR_Rate'] = 60.
        exp_data['EGR_mass_g_s'] = exp_data['Air_mass_inca_corr_g_s'] *\
            exp_data['EGR_Rate'] / (100 - exp_data['EGR_Rate'])
        exp_data['EGR_mass_LP_g_s'] = exp_data['EGR_mass_g_s'] *\
            exp_data['Vbx_egr_lp_mod_sta_1']
        exp_data['BlowBy_g_s'] = exp_data['Blow by'] / 60 * 1.16
        exp_data['Fuel_mass_g_s'] = exp_data['mftotal_inca_corr_kvar'] *\
            1e-3 * exp_data['DynoSpeed'] / 60 *\
            xml_dict['engine']['@cylinders'] / 2
        exp_data['Total_mass_g_s'] = exp_data['EGR_mass_LP_g_s'] +\
            exp_data['Air_mass_inca_corr_g_s'] +\
            exp_data['Fuel_mass_g_s'] + exp_data['BlowBy_g_s']

        # Some pollutants calculations.
        try:
            if xml_dict['plottingOptions']['emissions']['@plot']:
                location = xml_dict['plottingOptions']['emissions'][
                                    '@location'].strip()
                process_experimental_emissions(
                    exp_data, mode='transient', delay=delay,
                    model_amf=mod_case[location + '/MassFlow[kg/s]'],
                    model_time=mod_case['Time[s]'])

                # Convert model pollutants to experimental units (%vol)
                for name, specie, density in zip(
                    ['CO2', 'O2', 'CO'], ['CO2', 'O2', 'CO'],
                        [1842, 1331, 1165]):
                    model_col = '{}/MassFraction[-]/{}'.format(location,
                                                               specie)
                    model_col_amf = model_col.rsplit('/', 2)[0] +\
                        '/MassFlow[kg/s]'
                    mod_case[name] = mod_case[model_col] *\
                        mod_case[model_col_amf] * 118400 /\
                        mod_case[model_col_amf] / density
                y = 1.7843
                z = 0
                B_est = 1 + y / 4 - z / 2
                Y_O2_air = 0.23
                A = (1 - Y_O2_air) / 28 * 32 / Y_O2_air
                B = ((100 / mod_case['CO2']) - 1 + B_est) / (1 + A)
                mod_case['Fs_h'] = (1 + (B - B_est) + B * A) /\
                    (1 + y / 2 + (B - B_est) + B * A)
                # Model THC emissions (ppm)
                mod_case['THC'] = mod_case[
                    location + '/MassFraction[-]/FUEL'] *\
                    (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28) /\
                    (1e-6 * (12 + y) * (1 + y / 2 + (B - B_est) + B * A))
                # exp_data['NO2'], exp_data['NO'] and exp_data['NOx_total'] are
                # available since process_experimental_emissions() was called
                mNOx = 30 * (1 - exp_data['NO2'] / exp_data['NOx_total']) +\
                    46 * (exp_data['NO2'] / exp_data['NOx_total'])
                # Since the model only provides total NOx, Here I am taking the
                # same NOx molecular weight as in the experiment (based on NO +
                # NO2)
                # Interpolate the molecular weight array to fit the model
                # values size
                mNOx_interp = np.interp(mod_case['Time[s]'].values,
                                        exp_data['Time'].values, mNOx)
                # Model NOx emissions (ppm)
                mod_case['NOx'] = mod_case[
                    location + '/MassFraction[-]/NOx'] *\
                    (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28) /\
                    (1e-6 * mod_case['Fs_h'] * mNOx_interp *
                        (1 + y / 2 + (B - B_est) + B * A))
        except Exception as e:
            log.error("Impossible to process pollutant "
                      "emissions: {}".format(e))

        for i, var in enumerate(var_list):
            try:
                if var.is_pollutant is True:
                    if '_g_s' in var.name:
                        model_col_amf = model_col.rsplit('/', 2)[0] +\
                            '/MassFlow[kg/s]'
                        var.set_values({
                            'model': {
                                'x': mod_case['Time[s]'].values,
                                'y': (mod_case[var.model_col].values *
                                      mod_case[model_col_amf].values) *
                                var.conv_factor},
                            'exp': {
                                'x': exp_data['Time'].values,
                                'y': exp_data[var.exp_col].values}
                        })
                    elif '_g' in var.name:
                        model_col_amf = model_col.rsplit('/', 2)[0] +\
                            '/MassFlow[kg/s]'
                        var.set_values({
                            'model': {
                                'x': mod_case['Time[s]'].values,
                                'y': cumtrapz((
                                     mod_case[var.model_col].values * mod_case[
                                        model_col_amf].values) *
                                        var.conv_factor,
                                        mod_case['Time[s]'].values, initial=0)
                                      },
                            'exp': {
                                'x':  exp_data['Time'].values,
                                'y':  exp_data[var.exp_col].values}
                        })
                    else:
                        var.set_values({
                            'model': {
                                'x': mod_case['Time[s]'].values,
                                'y': mod_case[var.model_col].values *
                                var.conv_factor},
                            'exp': {
                                'x': exp_data['Time'].values,
                                'y': exp_data[var.exp_col].values}
                        })
                elif var.name == 'VGT Position':
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': mod_case[var.model_col].values *
                            var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': 2.1909 * (100 - exp_data[
                                var.exp_col].values) - 78.548}
                    })
                elif var.name == 'Exh. Throttle':
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': mod_case[var.model_col].values *
                            var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': 100 * exp_data[var.exp_col].values / 80.41}
                    })
                elif var.name == 'LP-EGR Valve':
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': mod_case[var.model_col].values *
                            var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': 100 * exp_data[var.exp_col].values / 39.97}
                    })
                elif var.name == 'Accum. Brake Energy':
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': cumtrapz(mod_case['Engine/BrakeTorque[Nm]'] *
                                          2 * np.pi * mod_case[
                                          'Engine/EngineSpeed[rpm]'] / 60000,
                                          mod_case['Time[s]'].values,
                                          initial=0) * var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': cumtrapz(exp_data['DynoTorque'] * 2 * np.pi *
                                          exp_data['DynoSpeed'] / 60000,
                                          exp_data['Time'].values, initial=0)}
                    })
                elif var.name == 'Accum. Injected Energy':
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': cumtrapz(mod_case[
                                'Engine/FuelMassCylinder&Cycle[mg]'] * 1e-3 *
                                mod_case['Engine/EngineSpeed[rpm]'] / 60 *
                                xml_dict['engine']['@cylinders'] /
                                2 * xml_dict['engine']['@fuelPCI'],
                                mod_case['Time[s]'].values, initial=0) *
                            var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': cumtrapz(exp_data['Fuel_mass_g_s'] *
                                          xml_dict['engine']['@fuelPCI'],
                                          exp_data['Time'].values, initial=0)}
                    })
                elif var.name == 'LP-EGR Accum. Mass':
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': cumtrapz(mod_case[var.model_col],
                                          mod_case['Time[s]'].values,
                                          initial=0) * var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': cumtrapz(exp_data[var.exp_col],
                                          exp_data['Time'].values, initial=0)}
                    })
                elif var.name == 'LP-EGR Cooler Heat':
                    amf, m_in, m_out = var.model_col.split(',')
                    model_in = m_in.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_in else m_in.strip() + '/Temperature[degC]'
                    model_out = m_out.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_out else m_out.strip() +\
                        '/Temperature[degC]'
                    exp_in = var.exp_col.split(',')[0].strip()
                    exp_out = var.exp_col.split(',')[1].strip()
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y':
                            mod_case[amf.strip() + '/MassFlow[kg/s]'].values *
                            get_cp(np.mean([mod_case[model_in],
                                            mod_case[model_out]], axis=0)) *
                            (mod_case[model_in].values -
                             mod_case[model_out].values)},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': exp_data['EGR_mass_LP_g_s'].values *
                            get_cp(np.mean([exp_data[exp_in],
                                            exp_data[exp_out]], axis=0) +
                                   273.15) *
                            (exp_data[exp_in].values -
                             exp_data[exp_out].values) / 1000}
                    })
                elif var.name == 'LP-EGR Cooler Accum. Energy':
                    amf, m_in, m_out = var.model_col.split(',')
                    model_in = m_in.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_in else m_in.strip() + '/Temperature[degC]'
                    model_out = m_out.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_out else m_out.strip() +\
                        '/Temperature[degC]'
                    exp_in = var.exp_col.split(',')[0].strip()
                    exp_out = var.exp_col.split(',')[1].strip()
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': cumtrapz(
                                mod_case[amf.strip() + '/MassFlow[kg/s]'] *
                                get_cp(np.mean([mod_case[model_in],
                                                mod_case[model_out]], axis=0)
                                       ) *
                                (mod_case[model_in] - mod_case[model_out]),
                                mod_case['Time[s]'].values, initial=0)},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': cumtrapz(
                                exp_data['EGR_mass_LP_g_s'] *
                                get_cp(np.mean([exp_data[exp_in],
                                                exp_data[exp_out]], axis=0) +
                                       273.15) *
                                (exp_data[exp_in] - exp_data[exp_out]) / 1000,
                                exp_data['Time'].values, initial=0)}
                    })
                elif var.name == 'WCAC Heat':
                    amf, m_in, m_out = var.model_col.split(',')
                    model_in = m_in.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_in else m_in.strip() + '/Temperature[degC]'
                    model_out = m_out.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_out else m_out.strip() +\
                        '/Temperature[degC]'
                    exp_in = var.exp_col.split(',')[0].strip()
                    exp_out = var.exp_col.split(',')[1].strip()
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y':
                            mod_case[amf.strip() + '/MassFlow[kg/s]'].values *
                            get_cp(np.mean([mod_case[model_in],
                                            mod_case[model_out]], axis=0)) *
                            (mod_case[model_in].values -
                             mod_case[model_out].values)},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': exp_data['Air_mass_inca_corr_g_s'].values *
                            get_cp(np.mean([exp_data[exp_in],
                                            exp_data[exp_out]], axis=0) +
                                   273.15) *
                            (exp_data[exp_in].values -
                             exp_data[exp_out].values) / 1000}
                    })
                elif var.name == 'WCAC Accum. Energy':
                    amf, m_in, m_out = var.model_col.split(',')
                    model_in = m_in.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_in else m_in.strip() + '/Temperature[degC]'
                    model_out = m_out.strip() + '/Temperature[degC]/Static' if\
                        'Pipe' in m_out else m_out.strip() +\
                        '/Temperature[degC]'
                    exp_in = var.exp_col.split(',')[0].strip()
                    exp_out = var.exp_col.split(',')[1].strip()
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y': cumtrapz(
                                mod_case[amf.strip() + '/MassFlow[kg/s]'] *
                                get_cp(np.mean([mod_case[model_in],
                                                mod_case[model_out]], axis=0)
                                       ) *
                                (mod_case[model_in] - mod_case[model_out]),
                                mod_case['Time[s]'].values, initial=0)},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': cumtrapz(
                                exp_data['Air_mass_inca_corr_g_s'] *
                                get_cp(np.mean([exp_data[exp_in],
                                                exp_data[exp_out]], axis=0) +
                                       273.15) *
                                (exp_data[exp_in] - exp_data[exp_out]) / 1000,
                                exp_data['Time'].values, initial=0)}
                    })
                elif var.name == 'Exh-Int Diff. Pressure':
                    p_exh_mod, p_int_mod = var.model_col.split(',')
                    p_exh_exp, p_int_exp = var.exp_col.split(',')
                    var.set_values({
                        'model': {
                            'x': mod_case['Time[s]'].values,
                            'y':
                            (mod_case[p_exh_mod.strip()].values -
                             mod_case[p_int_mod.strip()].values) *
                            var.conv_factor},
                        'exp': {
                            'x': exp_data['Time'].values,
                            'y': 
                            (exp_data[p_exh_exp.strip()].values -
                             exp_data[p_int_exp.strip()].values) *
                            var.conv_factor},
                    })
                else:
                    if var.time_evolution is True:
                        for col in var.model_col:
                            if 'hydrcircuit' in col.lower():
                                if col not in mod_case.columns:
                                    col = col.replace('CoolantCircuit-1',
                                                      'MainCoolantCircuit')
                                    col = col.replace('OilCircuit-1',
                                                      'LubricationCircuit')
                        if 'acc' in var.name.lower():
                            var.set_values({
                                'model': {
                                    'x': mod_case['Time[s]'].values,
                                    'y': [cumtrapz(
                                        mod_case[val] * factor,
                                        mod_case['Time[s]'],
                                        initial=0) for val, factor in zip(
                                            var.model_col, var.conv_factor)]}
                            })
                        elif ('acc' in var.name.lower()) and\
                                ('fuel' in var.name.lower()):
                            var.set_values({
                                'model': {
                                    'x': mod_case['Time[s]'].values,
                                    'y': [cumtrapz(
                                        mod_case[val] * factor * mod_case[
                                            'Engine/EngineSpeed[rpm]'] / 60 *
                                        4 / 2, mod_case['Time[s]'],
                                        initial=0) for val, factor in zip(
                                            var.model_col, var.conv_factor)]}
                            })
                        else:
                            var.set_values({
                                'model': {
                                    'x': mod_case['Time[s]'].values,
                                    'y': [mod_case[val].values *
                                          factor for val, factor in zip(
                                          var.model_col, var.conv_factor)]}
                            })
                    else:
                        if 'hydrcircuit' in var.model_col.lower():
                            if var.model_col not in mod_case.columns:
                                var.model_col = var.model_col.replace(
                                    'CoolantCircuit-1', 'MainCoolantCircuit')
                                var.model_col = var.model_col.replace(
                                    'OilCircuit-1', 'LubricationCircuit')
                        var.set_values({
                            'model': {
                                'x': mod_case['Time[s]'].values,
                                'y': mod_case[var.model_col].values *
                                var.conv_factor},
                            'exp': {
                                'x': exp_data['Time'].values,
                                'y': exp_data[var.exp_col].values}
                        })
            except Exception as e:
                log.error(e)

    ###########################################################################

    if args.nomultiprocessing is True:
        elapsed = time.time() - t
        log.info("{:.1f} seconds reading and processing "
                 "simulation data".format(elapsed))

    # Free memory by removing model and experimental data
    transient_dict = {}
    if args.mode == 'transient':
        transient_dict['sim_time'] = mod_case['Time[s]'].values
        transient_dict['sim_eng_speed'] = mod_case[
            'Engine/EngineSpeed[rpm]'].values
        transient_dict['sim_veh_speed'] = mod_case['ECU/Actuator-14[-]'].values
        transient_dict['sim_torque'] = mod_case[
            'Engine/BrakeTorque[Nm]'].values
        transient_dict['sim_fuel'] = mod_case[
            'Engine/FuelMassCylinder&Cycle[mg]'].values
    try:
        del model_data
    except Exception:
        del mod_case
    del exp_data

    # Print variables values in debug mode
    if args.verbosity is not None and args.verbosity == 2:
        for var in var_list:
            if var.exp_col is not None:
                log.debug(var.get_values())

    # Save average values to file
    if args.mode == 'transient':
        cases_dict = None
        if 'divisions' in xml_dict['simulationProcessing']['transient'].keys():
            divisions = [div for div in xml_dict['simulationProcessing'][
                'transient']['divisions']['division']]
        else:
            divisions = []
        if 'errorEvaluation' in xml_dict['simulationProcessing'][
                'transient'].keys():
            errors = [error for error in xml_dict['simulationProcessing'][
                'transient']['errorEvaluation']['error']]
        else:
            errors = []
        transient_dict['veh_speed'] = veh_speed
        transient_dict['delay'] = delay
        transient_dict['max_time'] = xml_dict['simulationProcessing'][
            'transient']['@simTime'] if '@simTime' in xml_dict[
                'simulationProcessing']['transient'].keys() else None
        transient_dict['start'] = xml_dict['simulationProcessing'][
            'transient']['@sectionStart'] if '@sectionStart' in xml_dict[
                'simulationProcessing']['transient'].keys() else 0
        transient_dict['end'] = xml_dict['simulationProcessing'][
            'transient']['@sectionEnd'] if '@sectionEnd' in xml_dict[
                'simulationProcessing']['transient'].keys()\
            else transient_dict['max_time']
        transient_dict['filter'] = xml_dict['simulationProcessing'][
            'transient']['@filterModel'] if '@filterModel' in xml_dict[
                'simulationProcessing']['transient'].keys() else True
        if transient_dict['filter'] is False:
            log.warning("Model signals are not filtered. You should expect "
                        "some noise in the graphs and a poor correlation "
                        "coefficient in trend graphs")
        transient_dict['divisions'] = divisions
        transient_dict['errors'] = errors
    else:
        transient_dict = None

    log.info("Creating summary file")
    try:
        try:
            mod_label = xml_dict['plotStyles']['@labelModel']
            exp_label = xml_dict['plotStyles']['@labelExp']
        except Exception:
            mod_label = 'Model'
            exp_label = 'Reference'
        vars_for_summary = [v for v in var_list if (v.time_evolution is False)
                            and (v.exp_col is not None)]
        create_summary(args, vars_for_summary, processing_path, mod_label,
                       exp_label, profile=transient_dict, num_cores=num_cores)
        log.info("Saved summary file at {}".format(processing_path))
    except Exception as e:
        log.error("Could not save summary.xlsx\n{}".format(e))

    log.info("Plotting variables")
    if args.nomultiprocessing is True:
        t = time.time()
        for var in var_list:
            print(plot_variables(var, cases_dict, processing_path,
                                 xml_dict['plotStyles'], args.mode,
                                 transient_dict=transient_dict))
        elapsed = time.time() - t
        log.info("{:.1f} seconds plotting variables".format(elapsed))
    else:
        outputs = Parallel(n_jobs=num_cores, verbose=8)(delayed(
            plot_variables)(var, cases_dict, processing_path,
                            xml_dict['plotStyles'], args.mode,
                            transient_dict=transient_dict) for var in
                            var_list)
        for output in outputs:
            print(output)

    if sim_mode == 'testbench' and args.mode == 'steady-avg':
        create_folder(processing_path, 'structs_for_Matlab')
        log.info("Saving Matlab structs to impose simulation data.")
        try:
            try:
                speed = [int(p['speed']) for c, p in cases_dict.items()]
            except Exception:
                speed_var = [var for var in var_list
                             if var.name == 'Engine speed'][0]
                speed = [round(val['model'][0][-1]/25) * 25 for
                         k, val in speed_var.values.items()]
            torque_var = [var for var in var_list
                          if var.name.lower() == 'engine torque'][0]
            fuel_var = [var for var in var_list
                        if var.name.lower() == 'fuel mass'][0]
            dictNT = {
                'SS_N_T_' + room_conditions: {
                    'speed': speed,
                    'torque': [val['model'] for k, val in
                               torque_var.values.items()],
                    'paramcalmec': [p['app_comb_eff'] for
                                    c, p in cases_dict.items()]
                }}
            savemat(os.path.join(
                processing_path, 'structs_for_Matlab',
                'SS_N_T_' + room_conditions + '.mat'), dictNT)
            dictNmf = {
                'SS_N_mf_' + room_conditions: {
                    'speed': speed,
                    'mfuel': [val['model'] for k, val in
                              fuel_var.values.items()],
                    'paramcalmec': [p['app_comb_eff'] for
                                    c, p in cases_dict.items()]
                }}
            savemat(os.path.join(
                processing_path, 'structs_for_Matlab',
                'SS_N_mf_' + room_conditions + '.mat'), dictNmf)
        except Exception as e:
            log.error(e)
            pass
    elif sim_mode == 'testbench' and args.mode == 'transient':
        log.info("Saving transient files to impose simulation data.")
        df_N_T = pd.DataFrame()
        df_N_mf = pd.DataFrame()
        df_N_T['Time[s]'] = transient_dict['sim_time']
        df_N_mf['Time[s]'] = transient_dict['sim_time']
        df_N_T['Engine Speed[rpm]'] = transient_dict['sim_eng_speed']
        df_N_mf['Engine Speed[rpm]'] = transient_dict['sim_eng_speed']
        df_N_T['Vehicle Speed[m/s]'] = transient_dict['sim_veh_speed']
        df_N_mf['Vehicle Speed[m/s]'] = transient_dict['sim_veh_speed']
        df_N_T['Torque[Nm]'] = transient_dict['sim_torque']
        # Ensure torque is zero at idle
        if df_N_T['Vehicle Speed[m/s]'][0] == 0:
            df_N_T['Torque[Nm]'][0] = 0.0
        for i in range(1, len(df_N_T['Torque[Nm]'])):
            delta_vel = df_N_T['Vehicle Speed[m/s]'][i-1] -\
                df_N_T['Vehicle Speed[m/s]'][i]
            if delta_vel > 0:
                if df_N_T['Torque[Nm]'][i] > 0 and\
                        df_N_T['Vehicle Speed[m/s]'][i] < 1.6:
                    df_N_T['Torque[Nm]'][i] = 0.0
            else:
                if df_N_T['Torque[Nm]'][i] > 0 and\
                        df_N_T['Vehicle Speed[m/s]'][i] < 0.01:
                    df_N_T['Torque[Nm]'][i] = 0.0
        df_N_mf['Fuel Mass[mg]'] = transient_dict['sim_fuel']
        df_N_T.to_excel('{}/transient_for_N-T_mode_{}.xlsx'.format(
            processing_path, room_conditions, index=False))
        df_N_mf.to_excel('{}/transient_for_N-mf_mode_{}.xlsx'.format(
            processing_path, room_conditions, index=False))

    # Ask user to process instantaneous files
    if args.mode == 'steady-avg':
        # Create a tkinter app. It is necessary for the messagebox dialog
        tkroot = tk.Tk()
        tkroot.withdraw()
        tkroot.minsize(1, 1)
        answer = tkinter.messagebox.askquestion(
            "", "Dou you want to process cylinder instantaneous variables "
            "now?")
        if answer == 'yes':
            tkroot.destroy()
            main(mode='steady-ins')
        else:
            tkroot.destroy()

    # Ask user to delete simulation files
    # Create a tkinter app. It is necessary for the messagebox dialog
    tkroot = tk.Tk()
    tkroot.withdraw()
    tkroot.minsize(1, 1)
    answer = tkinter.messagebox.askquestion(
            "", "Dou you want to remove simulation results files from origin?")
    if answer == 'yes':
        for root, dirs, files in os.walk(os.path.join(xml_dict[
            'simulationProcessing']['settings']['@simulationPath'].strip(),
                xml_dict['simulationProcessing']['settings']['@xmlCase'])):
            for fname in files:
                if ((model_prefix in fname) and (room_conditions in fname)
                        and (sim_mode in fname)) or ('PollutantNN' in fname):
                    try:
                        os.remove(os.path.join(root, fname))
                    except OSError:
                        log.error("{} could not be removed.".format(fname))
                        pass
            # Remove empty folders
            for dir in dirs:
                if 'slprj' in dir:
                    rmtree(os.path.join(root, dir))
                elif len(os.listdir(os.path.join(root, dir))) == 0:
                    rmtree(os.path.join(root, dir))
    tkroot.destroy()

    # Save a file with the processing timestamp
    with open(os.path.join(processing_path, 'Processed on {}.txt'.format(
        time.strftime("%Y-%b-%d at %Hh %Mm %Ss", time.localtime()))), 'w')\
            as ts:
        ts.write("Figures processed on {}".format(time.strftime(
            "%Y-%b-%d at %H:%M:%S", time.localtime())))

    exit()


if __name__ == '__main__':
    main()
