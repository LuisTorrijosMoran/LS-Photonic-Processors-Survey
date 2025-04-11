##________________________________________________________##

# Plotting evolution (comparison electronic ICs)

# Graph 1: actuators vs year (IC and PICs)
# Graph 2: actuators vs year (based on PIC architecture)
# Graph 3: actuators vs year (based on actuator type)

##________________________________________________________##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.optimize import curve_fit

filepath = 'data/LS_Photonic_Processors_rev20250411.xlsx'

plt.rcParams['font.family'] = 'Arial'  
plt.rcParams['font.size'] = 11  
Nat_blue = '#3c5488'
Nat_red = '#e64b35'
Nat_green = '#00a087'
Nat_lightblue = '#4dbbd5'
Nat_orange = '#f28e2b'
colors_nat = [Nat_blue, Nat_red, Nat_green, Nat_orange, '#9d7660', '#b07aa1']

df_EIC = pd.read_excel (filepath, sheet_name='Transistors')
df_EIC = df_EIC[df_EIC['Transistors'].notna()]
df_Other = pd.read_excel (filepath, sheet_name='Other Reconfigurable devices')
df_Other = df_Other[df_Other['Phase shifters'].notna()]
df_MultiportInterferometers = pd.read_excel (filepath, sheet_name='Feedfoward')
df_MultiportInterferometers = df_MultiportInterferometers[df_MultiportInterferometers['Phase shifters'].notna()]
df_WaveguideMeshes = pd.read_excel (filepath, sheet_name='Waveguide meshes')
df_WaveguideMeshes = df_WaveguideMeshes[df_WaveguideMeshes['Phase shifters'].notna()]
df_Switch = pd.read_excel (filepath, sheet_name='OpticalSwitches')
df_Switch = df_Switch[df_Switch['Phase shifters'].notna()]

year_EIC = df_EIC ['Year']
ps_EIC = df_EIC['Transistors']
year_Other = df_Other['Year']
ps_Other = df_Other['Phase shifters']
year_WM= df_WaveguideMeshes['Year']
ps_WM= df_WaveguideMeshes['Phase shifters']
year_MI = df_MultiportInterferometers['Year']
ps_MI = df_MultiportInterferometers['Phase shifters']
year_OS = df_Switch ['Year']
ps_OS = df_Switch['Phase shifters']

# Formating figure:
####################################################
plt.rcParams['font.family'] = 'Arial'  
plt.rcParams['font.size'] = 11  
Nat_blue = '#3c5488'
Nat_red = '#e64b35'
Nat_green = '#00a087'
Nat_lightblue = '#4dbbd5'
Nat_orange = '#f28e2b'
Nat_blue_rgb = mcolors.to_rgba(Nat_blue)
Nat_red_rgb = mcolors.to_rgba(Nat_red)
Nat_green_rgb = mcolors.to_rgba(Nat_green)
Nat_Nat_lightblue_rgb = mcolors.to_rgba(Nat_lightblue)
df_Full = df_Other.merge(df_WaveguideMeshes,how='outer')
df_Full = df_Full.merge(df_MultiportInterferometers,how='outer')
df_Full = df_Full.merge(df_Switch,how='outer')

years = [2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025, 2027, 2029, 2031]
actuators = [1250/512, 1250/256, 1250/128, 1250/64, 1250/32, 1250/16, 1250/8, 1250/4, 1250/2, 1250, 2500, 5000, 10000, 20000, 40000, 80000] # x2/24M 

def exponential_func(x, a, b):
    return a * np.exp(b * (x - years[0])) + actuators[0]

params, covariance = curve_fit(exponential_func, years, actuators)
a, b = params
interp_years = np.linspace(2005, 2035, 100)
interp_actuators = exponential_func(interp_years, a, b)
years2 = np.array([1972, 1991, 2010])
actuators2 = np.array([3000, 1350000, 1200000000])
def exponential_function(x2, a2, b2):
    return a2 * np.exp(b2 * (x2 - years2[0]))
popt, pcov = curve_fit(exponential_function, years2, actuators2, maxfev=10000)
a_fit, b_fit = popt
years_fit = np.linspace(years2[0], years2[-1], 100)
actuators_fit = exponential_function(years_fit, *popt)

rang_1 = 90 # percentage
rang_2 = 50
rang_3 = 25
alpha_1 = 1
alpha_2 = 0.5
alpha_3 = 0.25
alpha_4 = 0.1
size_1 = 25
size_2 = 20
size_3 = 18
size_4 = 15

df_3peryear=pd.DataFrame(df_Full.groupby('Year')['Phase shifters'].nlargest(3).reset_index())
df_1peryear=pd.DataFrame(df_Full.groupby('Year')['Phase shifters'].nlargest(1).reset_index())

df_Full['differences'] = np.abs(np.array(df_Full['Phase shifters']) - exponential_func(df_Full['Year'], a, b))
for index, row in df_Full.iterrows():
    a=1
    # print(index, row)

alphas = np.where((df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) >= rang_1/100) & (df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) <= 10), alpha_1, 
                  np.where((df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) >= rang_2/100) & (df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) <= 20), alpha_2,
                           np.where((df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) >= rang_3/100) & (df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) <= 100), alpha_3, alpha_4)))
sizes = np.where((df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) >= rang_1/100) & (df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) <= 10), size_1, 
                  np.where((df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) >= rang_2/100) & (df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) <= 20), size_2,
                           np.where((df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) >= rang_3/100) & (df_Full['Phase shifters']/(exponential_func(df_Full['Year'], a, b)) <= 100), size_3, size_4)))


df_EIC['Transistors'] = pd.to_numeric(df_EIC['Transistors'], errors='coerce')
alphas2 = np.where((df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) >= rang_1/100) & (df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) <= 10), alpha_1, 
                  np.where((df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) >= rang_2/100) & (df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) <= 20), alpha_2,
                           np.where((df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) >= rang_3/100) & (df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) <= 100), alpha_3, alpha_4)))
sizes2 = np.where((df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) >= rang_1/100) & (df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) <= 10), size_1, 
                  np.where((df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) >= rang_2/100) & (df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) <= 20), size_2,
                           np.where((df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) >= rang_3/100) & (df_EIC['Transistors']/(exponential_function(df_EIC['Year'], a_fit, b_fit)) <= 100), size_3, size_4)))

plt.figure(figsize=(18/2.54, 16.3/2.54)) # 18x14 cm @ Nature

plt.subplot2grid((5, 2), (0, 0), rowspan=3, colspan=2)
plt.scatter(df_EIC['Year'], df_EIC['Transistors'], alpha=alphas2, s=sizes2, color=Nat_blue)
plt.plot(years_fit, actuators_fit, color=Nat_blue, linewidth=1.5, label='Electronics')
plt.scatter(df_Full['Year'], df_Full['Phase shifters'], alpha=alphas, s=sizes, color=Nat_red)
plt.plot(interp_years, interp_actuators, color=Nat_red, linewidth=1.5, label='Photonics')

plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Number of transistors/actuators')
plt.ylim(2, 1e11)
plt.xlim(1970, 2030)
plt.legend(fontsize=8, loc='upper left', frameon=False)

max_per_year = 1
df_filtered_RASPIC = df_Other.query('Year != 2017 and Year != 2023 and Year != 2024')
df_limited_RASPIC = df_filtered_RASPIC.groupby('Year').apply(lambda group: group.nlargest(max_per_year, 'Phase shifters')).reset_index(drop=True)
df_limited_FF = df_MultiportInterferometers.groupby('Year').apply(lambda group: group.nlargest(max_per_year, 'Phase shifters')).reset_index(drop=True)
df_filtered_Switch = df_Switch.query('Year != 2024')
df_limited_Switch = df_filtered_Switch.groupby('Year').apply(lambda group: group.nlargest(max_per_year, 'Phase shifters')).reset_index(drop=True)
df_filtered_GP_drop = df_WaveguideMeshes
df_limited_GP = df_filtered_GP_drop.groupby('Year').apply(lambda group: group.nlargest(max_per_year, 'Phase shifters')).reset_index(drop=True)

plt.subplot2grid((5, 2), (3, 0), rowspan=2)
plt.plot(df_limited_RASPIC['Year'], df_limited_RASPIC['Phase shifters'], marker='o', color=Nat_blue, markersize=5, label='ASPICs')
plt.plot(df_limited_Switch['Year'], df_limited_Switch['Phase shifters'], marker='s', color=Nat_red, markersize=5, label='Switching')
plt.plot(df_limited_FF['Year'].iloc[:-1], df_limited_FF['Phase shifters'].iloc[:-1], color=Nat_green, markersize=5, marker='^', label='Feedforward')
plt.plot(df_limited_GP['Year'], df_limited_GP['Phase shifters'], marker='d', color=Nat_orange, markersize=5, label='General-purpose')

plt.xlabel('Year')
plt.ylabel('Number of photonic actuators')
plt.yscale('log')
plt.ylim(2, 1e5)
plt.xlim(2005, 2025)
plt.xticks([2005, 2010, 2015, 2020, 2025])
plt.legend(fontsize=8, loc='upper left', frameon=False)
colors = [Nat_blue, Nat_red, Nat_green, Nat_orange, Nat_lightblue, '#AF5BBA']
plt.subplot2grid((5, 2), (3, 1), rowspan=2)

# Continous
# chart = sns.scatterplot(x='Year', y='Phase shifters', data=df_Full, hue='Tuning Effect', palette=colors, s=size_1, edgecolor=None)
# plt.xlabel('Year')
# plt.ylabel('Number of photonic actuators')
# plt.yscale('log')
# plt.ylim(2, 1e5)
# plt.xlim(2005, 2025)
# plt.xticks([2005, 2010, 2015, 2020, 2025])
# plt.legend(fontsize=8, loc='upper left', frameon=False)
# plt.tight_layout()
# plt.savefig('plots/PIC_evolution.svg', format='svg')
# plt.show()

# bloxplot
chart = sns.boxplot(x='Year', y='Phase shifters', data=df_Full)
plt.xlabel('Year')
plt.ylabel('Number of photonic actuators')
plt.yscale('log')
xticklabels = ['2005', '', '', '', '2010', '', '', '', '', '2015', '', '', '', '', '2020', '', '', '', '2025']
chart.set_xticklabels(xticklabels)
plt.legend(fontsize=8, loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('plots/PIC_evolution.svg', format='svg')
plt.show()