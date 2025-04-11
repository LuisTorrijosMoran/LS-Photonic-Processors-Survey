##________________________________________________________##

# Plotting statistics

# Graph 1: boxplot and probability density (based on platform)
# Graph 2: boxplot and probability density (packaging)

##________________________________________________________##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import seaborn as sns

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

df_Full = df_Other.merge(df_WaveguideMeshes,how='outer')
df_Full = df_Full.merge(df_MultiportInterferometers,how='outer')
df_Full = df_Full.merge(df_Switch,how='outer')
df_Full['differences'] = np.abs(np.array(df_Full['Phase shifters']) - exponential_func(df_Full['Year'], a, b))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, gamma, beta
from scipy.optimize import curve_fit


fig, axs = plt.subplots(2, 2, figsize=(24/2.54, 8/2.54)) # 18x14 cm @ Nature
df = df_Full
distribution_functions = {
    'SOI': norm,
    'SiN': norm,
    'InP': norm,
    'Silica': norm,
}

order = ['SOI', 'SiN', 'InP', 'Silica']
ind=0
for material in order:
    group = df[df['Foundry'] == material]
    data = group['Phase shifters']
    x = np.linspace(data.min() - 10, data.max() + 10, 1000)
    
    if material == 'Silica':
        param = distribution_functions[material].fit(data)
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.5)
    elif material == 'InP':
        param = distribution_functions[material].fit(data)        
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.5)
    elif material == 'SOI':
        param = distribution_functions[material].fit(data)
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.75)
    else:
        param = distribution_functions[material].fit(data)
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*1.0)
    
    pdf_fitted /= 1.02*pdf_fitted.max()
    
    axs[1,0].plot(x, pdf_fitted, color=colors_nat[ind], label=material)
    axs[1,0].fill_between(x, pdf_fitted, color=colors_nat[ind], alpha=0.15)  # Color del sombreado
    ind=ind+1
    

axs[1,0].set_xlabel('Number of photonic actuators')
axs[1,0].set_ylabel('Probability density')
axs[1,0].legend(loc='upper right',fontsize=8, frameon=False)
axs[1,0].set_xscale('log')
axs[1,0].set_xlim(10,100000)
axs[1,0].set_ylim(0,1)
axs[1,0].set_yticks([0.0, 0.5, 1.0])
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False)

sns.boxplot(x='Phase shifters', y='Foundry', data=df, palette=colors_nat, order=order, ax=axs[0,0], fliersize=4)
axs[0,0].set_ylabel('')
axs[0,0].set_xscale('log')
axs[0,0].set_xlabel('')
axs[0,0].set_xlim(10,100000)
axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['right'].set_visible(False)
df_clean = df_Full.dropna(subset=['Electrical Packaging'])
df_clean = df_clean[df_clean['Electrical Packaging'] != 'ND']
order = ['Probing', 'Wire bonding', 'Flip-chip', 'Monolithic']
df_reordered = df_clean[df_clean['Electrical Packaging'].isin(order)]
df_reordered['Electrical Packaging'] = pd.Categorical(df_reordered['Electrical Packaging'], categories=order, ordered=True)
df_reordered = df_reordered.sort_values('Electrical Packaging')
sns.boxplot(x='Phase shifters', y='Electrical Packaging', data=df_reordered, palette=colors_nat, ax=axs[0,1], fliersize=4)
axs[0,1].set_ylabel('')
axs[0,1].set_xscale('log')
axs[0,1].set_xlabel('')
axs[0,1].set_xlim(10,100000)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)

distribution_functions = {
    'Probing': norm,
    'Wire bonding': norm,
    'Flip-chip': norm,
    'Monolithic': norm,
}
order = ['Probing', 'Wire bonding', 'Flip-chip', 'Monolithic']
ind=0
for material in order:
    group = df_reordered[df_reordered['Electrical Packaging'] == material]
    data = group['Phase shifters']
    x = np.linspace(data.min() - 10, data.max() + 10, 1000)
    if material == 'Probing':
        param = distribution_functions[material].fit(data)
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.75)
    elif material == 'Wire bonding':
        param = distribution_functions[material].fit(data)        
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.75)
    elif material == 'Flip-chip':
        param = distribution_functions[material].fit(data)
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.5)
    else:
        param = distribution_functions[material].fit(data)
        pdf_fitted = distribution_functions[material].pdf(x, param[0], param[1]*0.5)

    pdf_fitted /= 1.02*pdf_fitted.max()
    axs[1,1].plot(x, pdf_fitted, color=colors_nat[ind], label=material)
    axs[1,1].fill_between(x, pdf_fitted, color=colors_nat[ind], alpha=0.15)  # Color del sombreado
    ind=ind+1
axs[1,1].legend(loc='upper right',fontsize=8, frameon=False)
axs[1,1].set_xscale('log')
axs[1,1].set_xlim(10,100000)
axs[1,1].set_ylim(0,1)
axs[1,1].set_yticks([0.0, 0.5, 1.0])
axs[1,1].set_xlabel('Number of photonic actuators')
axs[1,1].set_ylabel('Probability density')
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False)
fig.tight_layout()
plt.savefig('plots/PIC_boxplots.svg', format='svg')
plt.show()