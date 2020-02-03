# -*- coding: utf-8 -*-
"""
This file is part of VEMOD plotter
"""

# Imports
import logging as log

# Libs
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz


def process_experimental_steady(calmec_avg_file, experimental_raw_file,
                                engine, cases, output_file):
    """Generate a table file with all the experimental tests processed

    Parameters
    ----------
    calmec_avg_file: str, required
        path of the Calmec file which contains the averages values from
        the steady state tests.
    experimental_raw_file: str, required
        path of the file which contains the testbench averages values
        from the steady state tests.
    engine: dict, required
        dictionary with engine information.
    cases: list, required
        a list with the selected steady cases.
    output_file: str, required
        path of the output processed table file.
    """

    calmec_data = pd.read_table(calmec_avg_file, engine='python', sep=None)
    data = pd.read_excel(experimental_raw_file)

    # Filter by the selected cases
    data = data.loc[data['TEST'].isin(cases)]
    data = data.sort_values(by='TEST').reset_index()

    calmec_selected = pd.DataFrame(columns=calmec_data.keys())
    for test in data['TEST']:
        average = pd.DataFrame(columns=calmec_data.keys())
        cases_calmec = [dataframe for dataframe in calmec_data['Nombre_del_Ensayo']
                        if test in dataframe][0:engine['cylinders']]
        cases_index = [calmec_data.index[calmec_data['Nombre_del_Ensayo'] == dataframe]
                       for dataframe in cases_calmec]
        cases_index = [dataframe[0] for dataframe in cases_index]
        rows = calmec_data.loc[cases_index]
        average = rows.mean().to_frame().T
        average['Nombre_del_Ensayo'] = test
        calmec_selected = calmec_selected.append(average.iloc[0])

    calmec_selected = calmec_selected.set_index(data.index)

    # Concatenate both dataframes
    data = pd.concat([data, calmec_selected], axis=1)

    # Calculate intake mean temperature based on intake pipes temperature
    try:
        data['T_adm_media'] = np.mean([data[c] for c in data.columns if 'T_adm' in c], axis=0)
    except Exception as e:
        log.error(e)
        exit()

    # Calculate EGR mass flow
    amf = data['M_Aire']
    egr_percent = data['EGR']
    data['MEGR'] = [(x * y)/(100 - y) for x, y in zip(amf, egr_percent)]

    # Calculate total heat transferred to walls
    coef_open_loop = 0.5
    q_cylhead_o = data[' Calor_transmitido_por_convección_a_la_Culata_en_ciclo_abierto(J/cc)'] * coef_open_loop
    q_cylhead_c = data[' Calor_transmitido_por_convección_a_la_Culata_en_ciclo_cerrado(J/cc)']
    q_piston_o = data[' Calor_transmitido_por_convección_al_Pistón_en_ciclo_abierto(J/cc)'] * coef_open_loop
    q_piston_c = data[' Calor_transmitido_por_convección_al_Pistón_en_ciclo_cerrado(J/cc)']
    q_cyl_o = data[' Calor_transmitido_por_convección_al_Cilindro_en_ciclo_abierto(J/cc)'] * coef_open_loop
    q_cyl_c = data[' Calor_transmitido_por_convección_al_Cilindro_en_ciclo_cerrado(J/cc)']
    engine_speed = data['DynoSpeed']

    data['CALMEC/HeatTransfered[kW]'] = [(q1 + q2 + q3 + q4 + q5 + q6) * n /
                                         60 / 1000 * 2 for q1, q2, q3, q4, q5,
                                         q6, n in zip(q_cylhead_o, q_cylhead_c,
                                         q_piston_o, q_piston_c, q_cyl_o,
                                         q_cyl_c, engine_speed)]

    # Calculate volumetric efficiency
    T_intake = data['T_adm_media']
    try:
        P_intake = data['P_S_EGR']
    except Exception:
        try:
            P_intake = data['P_S_WCAC']
        except Exception:
            log.error('Could not find a reference for intake pressure.'
                      'Define a new one in process_exp.py on line 98')

    egr_mass_flow = data['MEGR']

    data['vol_eff'] = [(m + m_egr) / 3600 / (p*1e5 / 287 / (T + 273) * 0.5 *
                       engine['total_displacement'] * n / 60) for m, m_egr, p,
                       T, n in zip(amf, egr_mass_flow, P_intake, T_intake,
                       engine_speed)]

    # Calculate engine power
    data['DynoPower'] = data['DynoSpeed'] * data['DynoTorque'] * 2 * np.pi / 1000 / 60

    # Convert turbocharger speed
    data['rpm_turbo'] = data['rpm_turbo'] / 1000

    # Calculate fuel mass
    # Not a requirement: replace '*' by zeros in data dataframe mf columns
    for pulse in range(1, engine['injection_pulses'] + 1):
        data.loc[data['mf ' + str(pulse)] == '*', 'mf ' + str(pulse)] = 0

    try:
        data['MF'] = data[['mf' + str(p) + '_corr' for p in range(1, engine['injection_pulses'] + 1)]].sum(axis=1)
    except Exception as e:
        log.error(e)
        log.warning("Trying with not corrected fuel instead.")
        try:
            data['MF'] = data[['mf' + str(p) + '_corr' for p in range(1, engine['injection_pulses'] + 1)]].sum(axis=1)
        except Exception as e:
            log.error(e)
            exit()

    # Calculate effective efficiency
    data['effective_eff'] = [M * np.pi / mf / engine['PCI'] for mf, M in zip(data['MF'], data['DynoTorque'])]

    # Calculate total heat released
    YO2_AIR = 0.23
    PM_N2, PM_O2, PM_O, PM_CO2, PM_H2O = 28.01, 32, 16, 44.01, 18.02
    PM_NO, PM_NO2, PM_C = 30.01, 46.01, 12.01
    N2_O2_AIR = (1 - YO2_AIR) / PM_N2 * PM_O2 / YO2_AIR
    HC_RATIO, OC_RATIO = 1.7843, 0
    B_PARAM = 1 + HC_RATIO / 4 - OC_RATIO / 2

    data['Yair_exh'] = data['O2'] / YO2_AIR / 100
    data['Mol_air_exh'] = data['Yair_exh'] / (PM_O2 * 1 + PM_N2 * N2_O2_AIR) * (1 + N2_O2_AIR)
    data['Mol_quem_exh'] = (1 - data['Yair_exh']) /\
        (PM_CO2 + HC_RATIO / 2 * PM_H2O + B_PARAM * N2_O2_AIR * PM_N2) * (1 + HC_RATIO / 2 + B_PARAM * N2_O2_AIR)

    # The measurement is in dry conditions
    data['Mol_NO'] = data['NO'] / 1e6 * (data['Mol_air_exh'] + data[
        'Mol_quem_exh'] * (1 - HC_RATIO / 2 / (1 + HC_RATIO / 2 + B_PARAM * N2_O2_AIR)))
    # The measurement is in dry conditions
    data['Mol_NO2'] = data['NO2'] / 1e6 * (data['Mol_air_exh'] + data[
        'Mol_quem_exh'] * (1 - HC_RATIO / 2 / (1 + HC_RATIO / 2 + B_PARAM * N2_O2_AIR)))
    data['YNO_exh'] = data['Mol_NO'] * PM_NO / 1.
    data['YNO2_exh'] = data['Mol_NO2'] * PM_NO2 / 1.
    data['YNOx_exh'] = data['YNO_exh'] + data['YNO2_exh']

    # The measurement is in wet conditions
    data['Mol_HC'] = data['THC'] / 1e6 * (data['Mol_air_exh'] + data['Mol_quem_exh'])
    data['YHC_exh'] = data['Mol_HC'] * (PM_C + HC_RATIO) / 1.

    # The measurement is in dry conditions
    data['Mol_CO'] = data['COL'] / 1e6 * (data['Mol_air_exh'] + data[
        'Mol_quem_exh'] * (1 - HC_RATIO / 2 / (1 + HC_RATIO / 2 + B_PARAM * N2_O2_AIR)))
    data['YCO_exh'] = data['Mol_CO'] * (PM_C + PM_O) / 1.

    data['YNOxint'] = data['EGR'] / 100 * (data['YNO_exh'] + data['YNO2_exh'])
    data['YO2_int'] = data[' Fracción_másica_de_oxígeno_en_la_admisión']

    data['Yair_int'] = data['YO2_int'] / YO2_AIR
    data['YCO2int'] = (1. - data['Yair_int']) * PM_CO2 /\
        (PM_CO2 + HC_RATIO / 2 * PM_H2O + B_PARAM * N2_O2_AIR * PM_N2)
    data['YH2Oint'] = (1. - data['Yair_int']) * HC_RATIO / 2 * PM_H2O /\
        (PM_CO2 + HC_RATIO / 2 * PM_H2O + B_PARAM * N2_O2_AIR * PM_N2)
    data['YO2int'] = YO2_AIR * data['Yair_int']
    data['YN2int'] = 1. - data['YO2int'] - data['YH2Oint'] - data['YCO2int'] - data['YNOxint']

    # Soot (es la última versión, negociada con Pedro Piqueras)
    R = 287  # For the Exhaust Gases: the same R as the air is considered
    # These two values are the ones used in the AVL439 for correction
    P_exh = 1.013e5
    T_exh = 373
    rho_exh = P_exh / (R * T_exh)

    # From Opacity to mg/m^3
    if any(data['Opacidad'] < 0):
        log.warning('Negative values in experimental opacity, '
                    'taking absolute values')
    data['XSoot'] = 0.917 * 3.1963 * np.abs(data['Opacidad']**1.0584)
    # From XSoot to YSoot
    data['YSoot'] = data['XSoot'] / rho_exh / 1e6  # From mg to kg

    data['mbb'] = data['Blow by'] * data['P_sala'] * 1e5 / (R * (data['T_sala_1'] + 273)) * 60 / 1000

    # Exhaust mass
    # data['Mexh'] = data['M_Aire']+data['CNS_Combus']-data['mbb']
    data['Mexh'] = (data['M_Aire'] + data['CNS_Combus']) * 0.99

    data['EFF_HC'] = 1 - data['YHC_exh'] * data['Mexh'] / data['CNS_Combus']
    data['EFF_CO'] = 1 - data['YCO_exh'] * data['Mexh'] / data['CNS_Combus'] / 4
    data['EFF_SOOT'] = 1 - data['YSoot'] * data['Mexh'] / data['CNS_Combus'] / 1.25

    data['EFF_COMB'] = data['EFF_HC'] * data['EFF_CO'] * data['EFF_SOOT']

    data['TotalHRL'] = data[' Parámetro_CALMEC(%)'] / 100 / data['EFF_COMB'] * data['MF'] * engine['PCI']

    # Save data dataframe into an Excel file
    writer = pd.ExcelWriter(output_file)
    data.to_excel(writer, 'Sheet1', index=False)
    writer.save()
    print("Output file saved at {}".format(output_file))


def process_experimental_emissions(exp_df, mode, model_time=None, model_amf=None, delay=None) -> pd.DataFrame:
    """Process experimental pollutant emissions

    Parameters
    ----------
    exp_df: pandas.DataFrame, required
        A pandas DataDrame with experimental data
    mode: str, required
        The processing mode: steady-avg or transient
    model_time: pandas.Serie, optional, default: None
        The simulation time array
    model_amf: pandas.Serie, optional, default: None
        The simulation air mass flow array at the pollutants measuring point
    delay: float, optional, default: None
        The delay between model and experimental signals

    Returns
    -------
    pandas.DataFrame: A pandas DataDrame with experimental data

    Notes
    -----
    https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A31999L0096
    Some corrections must be applied to the experimental pollutant measurements.

        - K_W : The Horiba gas analyzer performs the measurements by drying up the gas sample.
                Since VEMOD simulations are performed with dry air, K_W correction must not be applied.

        - K_HD : This correction must be applied to NOx emissions. Firstly, a K_HD coefficient
                 corrects the emissions due to the different ambient humidity compared to the reference
                 of the STANDARD. Later, another K_HD coefficient corrects the emissions due to the
                 different ambient humidity compared to the MODEL, which is zero.
                 The B term in the denominator of K_HD formula must not be taken into account because
                 experimental ambient temperature and model ambient temperature are the same.

    Equations
    ---------
        Pa (Saturation vapor pressure [kPa])
        Pa = 0.4625 + 7.99e-2 * Tamb[°C] - 9.593e-4 * Tamb[°C]^2 + 8.2646e-5 * Tamb[°C]^3

        Ha (Specific humidity [g_water/g_dry_air])
        Ha = (6.22 * Rel.Humidity(%) / 100 * Pa[kPa]) / (Pamb[kPa] - Rel.Humidity(%) / 100 * Pa[kPa])

        A = 0.309 * (m_fuel/m_air) * (1 + Ha * 0.001) - 0.0266

        K_HD = 1 / (1 + A * (Ha - 10.71)) (without B term)

    """

    # Fuel composition CHyOz
    y = 1.7843
    z = 0
    B_est = 1 + y / 4 - z / 2
    Y_O2_AIR = 0.23
    A = (1 - Y_O2_AIR) / 28 * 32 / Y_O2_AIR

    # Humidity
    exp_df['P_a'] = 0.4625 + 0.0799 * exp_df['T_sala_1'] - 0.0009593 *\
        exp_df['T_sala_1'] ** 2 + 0.000082646 * exp_df['T_sala_1'] ** 3
    exp_df['H_a'] = 6.22 * exp_df['Humedad']/100 * exp_df['P_a'] /\
        (exp_df['P_sala']*100 - exp_df['Humedad']/100 * exp_df['P_a'])

    if mode == 'transient':
        # Use the model air mass flow at the measured point instead of the
        # experimental mass flow, as Javi López proposed. Doing this, less
        # differences will be seen when comparing the cumulative values
        starting_index = [i for i in model_amf.index if model_time[i] >= delay][0]
        exp_df['Total_mass_g_s'] = np.interp(exp_df['Time'], model_time[starting_index:] + delay,
                                             model_amf[starting_index:] * 1000)
        exp_df['_A'] = 0.309 * exp_df['Fuel_mass_g_s'] / exp_df['Air_mass_inca_corr_g_s'] *\
            (1 + exp_df['H_a'] * 0.001) - 0.0266
        exp_df['_A_dry'] = 0.309 * exp_df['Fuel_mass_g_s'] / exp_df['Air_mass_inca_corr_g_s'] - 0.0266
        exp_df.loc[exp_df['Air_mass_inca_corr_g_s'] <= 0., '_A'] = 0.
        exp_df.loc[exp_df['Air_mass_inca_corr_g_s'] <= 0., '_A_dry'] = 0.
        exp_df['Air_mass_kg'] = cumtrapz(exp_df['Air_mass_inca_corr_g_s'] / 1000, exp_df['Time'], initial=0)
        exp_df['Fuel_mass_g'] = cumtrapz(exp_df['Fuel_mass_g_s'], exp_df['Time'], initial=0)

    elif mode == 'steady-avg':
        try:
            exp_df['EGR_mass_g_s'] = exp_df['M_Aire'] * exp_df['EGR'] / (100 - exp_df['EGR']) / 3.6
        except Exception as e:
            log.error(e)
            log.warning('Trying with EGRCO2')
            try:
                exp_df['EGR_Rate'] = exp_df['EGRCO2'] / exp_df['CO2'] * 100
                exp_df.loc[exp_df['EGR_Rate'] > 60., 'EGR_Rate'] = 60.
                exp_df['EGR_mass_g_s'] = exp_df['M_Aire'] * exp_df['EGR_Rate'] / (100 - exp_df['EGR_Rate']) / 3.6
            except Exception as e:
                log.error(e)
                exit()
        try:
            exp_df['EGR_mass_LP_g_s'] = exp_df['EGR_mass_g_s'] * exp_df['Vbx_egr_lp_mod_sta_1']
        except Exception as e:
            log.error(e)
            log.warning('Taking EGR column as LP-EGR')
            exp_df['EGR_mass_LP_g_s'] = exp_df['EGR_mass_g_s']
            pass
        exp_df['Total_mass_g_s'] = exp_df['EGR_mass_LP_g_s'] + (exp_df['M_Aire'] + exp_df['CNS_Combus']) / 3.6
        exp_df['_A'] = 0.309 * exp_df['CNS_Combus'] / exp_df['M_Aire'] * (1 + exp_df['H_a'] * 0.001) - 0.0266
        exp_df['_A_dry'] = 0.309 * exp_df['CNS_Combus'] / exp_df['M_Aire'] - 0.0266
        exp_df.loc[exp_df['M_Aire'] <= 0., '_A'] = 0.
        exp_df.loc[exp_df['M_Aire'] <= 0., '_A_dry'] = 0.

    exp_df['K_H'] = 1 / (1 + exp_df['_A'] * (exp_df['H_a'] - 10.71))
    exp_df['K_H_dry'] = 1 / (1 - 10.71 * exp_df['_A_dry'])
    if any(exp_df['Opacidad'] < 0):
        log.warning('Negative values in experimental opacity, '
                    'taking absolute values')
    exp_df['ADS'] = -np.log(1 - np.abs(exp_df['Opacidad'] / 100)) / 0.43

    # ---- O2 ----
    B = (B_est + (exp_df['O2'] / 100) * (1 - B_est)) / (1 - (exp_df['O2'] / 100) * (1 + A))
    exp_df['O2_Y'] = 32 * (B - B_est) / (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28)
    exp_df['O2_g_s'] = exp_df['O2_Y'] * exp_df['Total_mass_g_s']
    if mode == 'transient':
        exp_df['O2_g'] = cumtrapz(exp_df['O2_g_s'], exp_df['Time'], initial=0)

    # ---- CO2 ----
    B = ((100 / exp_df['CO2']) - 1 + B_est) / (1 + A)
    exp_df['CO2_Y'] = 44 / (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28)
    exp_df['CO2_g_s'] = exp_df['CO2_Y'] * exp_df['Total_mass_g_s']
    if mode == 'transient':
        exp_df['CO2_g'] = cumtrapz(exp_df['CO2_g_s'], exp_df['Time'], initial=0)

    exp_df['Fs_h'] = (1 + (B - B_est) + B * A) / (1 + y / 2 + (B - B_est) + B * A)

    # ---- CO ----
    exp_df['CO_Y'] = (exp_df['COH'] / 100) * exp_df['Fs_h'] * 28 *\
        (1 + y / 2 + (B - B_est) + B * A) / (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28)
    exp_df['CO_g_s'] = exp_df['CO_Y'] * exp_df['Total_mass_g_s']
    if mode == 'transient':
        exp_df['CO_g'] = cumtrapz(exp_df['CO_g_s'], exp_df['Time'], initial=0)

    # ---- NOx ----
    exp_df.loc[exp_df['NO2'] < 0., 'NO2'] = 0.
    exp_df['NOx_total'] = exp_df['NO'] + exp_df['NO2']
    mNOx = 30 * (1 - exp_df['NO2'] / exp_df['NOx_total']) + 46 * (exp_df['NO2'] / exp_df['NOx_total'])
    exp_df['NOx_Y'] = exp_df['NOx_total'] * exp_df['K_H'] / exp_df['K_H_dry'] * 1e-6 * exp_df['Fs_h'] *\
        mNOx * (1 + y / 2 + (B - B_est) + B * A) / (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28)
    exp_df['NOx_g_s'] = exp_df['NOx_Y'] * exp_df['Total_mass_g_s']
    if mode == 'transient':
        exp_df['NOx_g'] = cumtrapz(exp_df['NOx_g_s'], exp_df['Time'], initial=0)

    # ---- UHC ----
    exp_df['THC_Y'] = exp_df['THC'] * 1e-6 * (12 + y) * (1 + y / 2 + (B - B_est) + B * A) /\
        (44 + y / 2 * 18 + (B - B_est) * 32 + B * A * 28)
    exp_df['THC_g_s'] = exp_df['THC_Y'] * exp_df['Total_mass_g_s']
    if mode == 'transient':
        exp_df['THC_g'] = cumtrapz(exp_df['THC_g_s'], exp_df['Time'], initial=0)

    # ---- Soot ----
    K = 1.025 * 100000 / 287 / (100 + 273)
    exp_df['Soot_Y'] = (exp_df['ADS'] * (0.6**1.345) / 6.3) / 1000 / 1000 / K * 1650
    exp_df['Soot_g_s'] = (exp_df['ADS'] * (0.6 ** 1.345) / 6.3) * exp_df['Total_mass_g_s'] / 1000 / 1000 / K * 1650
    if mode == 'transient':
        exp_df['Soot_g'] = cumtrapz(exp_df['Soot_g_s'], exp_df['Time'], initial=0)

    return exp_df
