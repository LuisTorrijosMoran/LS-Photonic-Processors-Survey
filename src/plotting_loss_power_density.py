##________________________________________________________##

# Plotting performance

# Graph 1: Loss per puc (based on platform)
# Graph 2: Power per area (based on platform)
# Graph 3: Density (based onPUC type)

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

df_Other['PS density (1/mm2)'] = df_Other['PS density (1/mm2)'].replace('NE', pd.NA)
df_Other['PS density (1/mm2)'] = pd.to_numeric(df_Other['PS density (1/mm2)'], errors='coerce')
df_Other_density = df_Other['PS density (1/mm2)']
df_WaveguideMeshes['PS density (1/mm2)'] = df_WaveguideMeshes['PS density (1/mm2)'].replace('NE', pd.NA)
df_WaveguideMeshes['PS density (1/mm2)'] = pd.to_numeric(df_WaveguideMeshes['PS density (1/mm2)'], errors='coerce')
df_WaveguideMeshes_density = df_WaveguideMeshes['PS density (1/mm2)']
df_MultiportInterferometers['PS density (1/mm2)'] = df_MultiportInterferometers['PS density (1/mm2)'].replace('NE', pd.NA)
df_MultiportInterferometers['PS density (1/mm2)'] = pd.to_numeric(df_MultiportInterferometers['PS density (1/mm2)'], errors='coerce')
df_MultiportInterferometers_density = df_MultiportInterferometers['PS density (1/mm2)']
df_Switch['PS density (1/mm2)'] = df_Switch['PS density (1/mm2)'].replace('NE', pd.NA)
df_Switch['PS density (1/mm2)'] = pd.to_numeric(df_Switch['PS density (1/mm2)'], errors='coerce')
df_Switch_density = df_Switch['PS density (1/mm2)']

# WE FILTER LARGE PHASE ARRAYS
df_Other_cleaned = df_Other_density.dropna()
df_Other_filtered = df_Other_cleaned[df_Other_cleaned <= 150]
Other_indices = df_Other_filtered.index
Other_years_filtered = year_Other.loc[Other_indices]
# ONLY MZI
df_ONLY_MZI_Other_cleaned = df_Other.dropna()
df_ONLY_MZI_Other = df_ONLY_MZI_Other_cleaned[df_ONLY_MZI_Other_cleaned['TBU'].str.contains('MZI')]
df_ONLY_MZI_Other_density = df_ONLY_MZI_Other['PS density (1/mm2)']
ONLY_MZI_Other_years = df_ONLY_MZI_Other['Year']

# WE FILTER RRs GP MESHES
df_WaveguideMeshes_cleaned = df_WaveguideMeshes_density.dropna()
df_WaveguideMeshes_filtered = df_WaveguideMeshes_cleaned[df_WaveguideMeshes_cleaned <= 150]
WaveguideMeshes_indices = df_WaveguideMeshes_filtered.index
WaveguideMeshes_years_filtered = year_WM.loc[WaveguideMeshes_indices]

# WE FILTER BiM MATMUL MESHES
df_MultiportInterferometers_cleaned = df_MultiportInterferometers_density.dropna()
df_MultiportInterferometers_filtered = df_MultiportInterferometers_cleaned[df_MultiportInterferometers_cleaned <= 150]
MultiportInterferometers_indices = df_MultiportInterferometers_filtered.index
MultiportInterferometers_years_filtered = year_MI.loc[MultiportInterferometers_indices]

# WE FILTER MEMS SWITCHES
df_Switch_cleaned = df_Switch_density.dropna()
df_Switch_filtered = df_Switch_cleaned[df_Switch_cleaned <= 40]

Switch_indices = df_Switch_filtered.index
Switch_years_filtered = year_OS.loc[Switch_indices]

df_Switch_filtered = df_Switch_filtered.iloc[1:]
Switch_years_filtered = Switch_years_filtered.iloc[1:]

years = [1992, 1997, 2002, 2007, 2012, 2017, 2022, 2027, 2032]
actuator_density = [0, 0, 1, 2, 3, 6, 12, 24, 48] # x2/5Y 

x1, y1 = 2009, 0
x2, y2 = 2026, 20
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1
def exponential_func(x):
    return m * x + b

interp_years = np.linspace(2005, 2035, 100)
interp_density = exponential_func(interp_years)
example_density = exponential_func(np.linspace(2025, 2030, 2))
interp_density_OTHER = exponential_func(ONLY_MZI_Other_years)

rang_1 = 1
rang_2 = 3
rang_3 = 10
alpha_1 = 1
alpha_2 = 0.75
alpha_3 = 0.5
alpha_4 = 0.25
size_1 = 25
size_2 = 20
size_3 = 18
size_4 = 15

differences_other = np.abs(np.array(df_ONLY_MZI_Other_density) - exponential_func(ONLY_MZI_Other_years))
alphas_other = np.where(differences_other <= rang_1, alpha_1, 
                  np.where(differences_other <= rang_2, alpha_2,
                           np.where(differences_other <= rang_3, alpha_3, alpha_4)))
size_Other = np.where(differences_other <= rang_1, size_1, 
                  np.where(differences_other <= rang_2, size_2,
                           np.where(differences_other <= rang_3, size_3, size_4)))

differences_MI = np.abs(np.array(df_MultiportInterferometers_filtered) - exponential_func(MultiportInterferometers_years_filtered))
alphas_MI = np.where(differences_MI <= rang_1, alpha_1, 
                  np.where(differences_MI <= rang_2, alpha_2,
                           np.where(differences_MI <= rang_3, alpha_3, alpha_4)))
size_MI = np.where(differences_MI <= rang_1, size_1, 
                  np.where(differences_MI <= rang_2, size_2,
                           np.where(differences_MI <= rang_3, size_3, size_4)))

differences_WM = np.abs(np.array(df_WaveguideMeshes_filtered) - exponential_func(WaveguideMeshes_years_filtered))
alphas_WM = np.where(differences_WM <= rang_1, alpha_1, 
                  np.where(differences_WM <= rang_2, alpha_2,
                           np.where(differences_WM <= rang_3, alpha_3, alpha_4)))
size_WM = np.where(differences_WM <= rang_1, size_1, 
                  np.where(differences_WM <= rang_2, size_2,
                           np.where(differences_WM <= rang_3, size_3, size_4)))

differences_SW = np.abs(np.array(df_Switch_filtered) - exponential_func(Switch_years_filtered))
alphas_SW = np.where(differences_SW <= rang_1, alpha_1, 
                  np.where(differences_SW <= rang_2, alpha_2,
                           np.where(differences_SW <= rang_3, alpha_3, alpha_4)))
size_SW = np.where(differences_SW <= rang_1, size_1, 
                  np.where(differences_SW <= rang_2, size_2,
                           np.where(differences_SW <= rang_3, size_3, size_4)))



fig, axs = plt.subplots(1, 3, figsize=(25/2.54, 7/2.54)) # 18x14 cm @ Nature
probing_data = df_Full[df_Full['Foundry'] == 'SOI']
wirebonding_data = df_Full[df_Full['Foundry'] == 'SiN']
monolithic_data = df_Full[df_Full['Foundry'] == 'InP']
num_actuators_probing = probing_data['PS density (1/mm2)'].tolist()
num_actuators_wirebonding = wirebonding_data['PS density (1/mm2)'].tolist()
num_actuators_monolithic = monolithic_data['PS density (1/mm2)'].tolist()
df_clean = df_Full.dropna(subset=['Foundry'])
df_clean = df_clean[df_clean['Foundry'] != 'ND']
order = ['SOI', 'SiN', 'InP', 'Silica']
df_reordered = df_clean[df_clean['Foundry'].isin(order)]
df_reordered['Foundry'] = pd.Categorical(df_reordered['Foundry'], categories=order, ordered=True)
df_reordered = df_reordered.sort_values('Foundry')
colors = [Nat_blue, Nat_red, Nat_green, Nat_orange]
ind=0

dF_phase_array = df_Full[df_Full['Field'] == 'Phase Array']
dF_MEMS = df_Full[df_Full['Tuning Effect'] == 'MEMS']
dF_MZI = df_Full[df_Full['TBU'] == 'MZI']
dF_RR = df_Full[df_Full['TBU'].isin(['ORR', 'MRR', 'Disk Array'])]
df_mix_a = df_Full[df_Full['TBU'].isin(['BiM'])]
df_mix_b = df_Full[df_Full['Tuning Effect'].isin(['PCM', 'MEMS'])]
df_mix = pd.concat([df_mix_a, df_mix_b], ignore_index=True)

chart = sns.scatterplot(x='Year', y='PS density (1/mm2)', data=dF_MZI, color=Nat_green, s=size_1, marker='o', edgecolor=None, legend=False, ax=axs[2], label='2x2 MZI')
chart = sns.scatterplot(x='Year', y='PS density (1/mm2)', data=dF_RR, color=Nat_orange, s=size_1, marker='o', edgecolor=None, legend=False, ax=axs[2], label='Resonators')
chart = sns.scatterplot(x='Year', y='PS density (1/mm2)', data=dF_phase_array, color=Nat_blue, s=size_1, marker='o', edgecolor=None, legend=False, ax=axs[2], label='PA doped heaters')
chart = sns.scatterplot(x='Year', y='PS density (1/mm2)', data=df_mix, color=Nat_red, s=size_1, marker='o', edgecolor=None, legend=False, ax=axs[2], label='Crossings and other')

axs[2].set_ylabel("Density (actuator/mm2)")
axs[2].set_xlabel("Year")

axs[2].set_xlim(2005, 2025)
axs[2].set_xticks([2005, 2010, 2015, 2020, 2025])
axs[2].set_ylim(1e-2, 1e3)
axs[2].set_yscale("log", base=10)

df_other_filtered=df_Other[(df_Other['dB/TBU(dB)'] != 'NE')]
df_other_filtered = df_other_filtered.dropna()

n_pa = np.array([1, 10, 100, 1000, 10000, 100000, 1000000])
loss_puc = np.array([12.97, 5.76, 2.56, 1.14, 0.50, 0.225, 0.1])

coefficients = np.polyfit(np.log(n_pa), np.log(loss_puc), 3)
p = np.poly1d(coefficients)
x_fit = np.linspace(min(np.log(n_pa)), max(np.log(n_pa)), 100)
y_fit = np.exp(p(x_fit))

alphas_other2 = np.zeros(len(df_other_filtered))
size_other2 = np.zeros(len(df_other_filtered))
alphas_MI2 = np.zeros(len(df_MultiportInterferometers))
size_MI2 = np.zeros(len(df_MultiportInterferometers))
alphas_SW2 = np.zeros(len(df_Switch))
size_SW2 = np.zeros(len(df_Switch))
alphas_GP2 = np.zeros(len(df_WaveguideMeshes))
size_GP2 = np.zeros(len(df_WaveguideMeshes))
ind=0

rang_1 = 0.75
rang_2 = 1.5
rang_3 = 3
alpha_1 = 1
alpha_2 = 0.75
alpha_3 = 0.5
alpha_4 = 0.25
size_1 = 25
size_2 = 20
size_3 = 18
size_4 = 15

for index, row in df_other_filtered.iterrows():
    value_loss = float(row['dB/TBU(dB)'])
    PA_loss = float(row['Phase shifters'])
    value_loss_func = np.exp(p(np.log(PA_loss)))
    diff_other=abs(value_loss-value_loss_func)
    # print(diff_other)
    value_alpha = np.where(diff_other <= rang_1, alpha_1, 
                  np.where(diff_other <= rang_2, alpha_2,
                           np.where(diff_other <= rang_3, alpha_3, alpha_4)))
    value_size = np.where(diff_other <= rang_1, size_1, 
                  np.where(diff_other <= rang_2, size_2,
                           np.where(diff_other <= rang_3, size_3, size_4)))
    
    alphas_other2[ind] = value_alpha
    size_other2[ind] = value_size
    ind=ind+1
ind=0

for index, row in df_MultiportInterferometers.iterrows():
    value_loss = float(row['dB/TBU'])
    PA_loss = float(row['Phase shifters'])
    value_loss_func = np.exp(p(np.log(PA_loss)))
    diff_other=abs(value_loss-value_loss_func)
    # print(diff_other)
    value_alpha = np.where(diff_other <= rang_1, alpha_1, 
                  np.where(diff_other <= rang_2, alpha_2,
                           np.where(diff_other <= rang_3, alpha_3, alpha_4)))
    value_size = np.where(diff_other <= rang_1, size_1, 
                  np.where(diff_other <= rang_2, size_2,
                           np.where(diff_other <= rang_3, size_3, size_4)))
    
    alphas_MI2[ind] = value_alpha
    size_MI2[ind] = value_size
    ind=ind+1

ind=0

for index, row in df_Switch.iterrows():
    value_loss = float(row['dB/TBU(dB)'])
    PA_loss = float(row['Phase shifters'])
    value_loss_func = np.exp(p(np.log(PA_loss)))
    diff_other=abs(value_loss-value_loss_func)
    # print(diff_other)
    value_alpha = np.where(diff_other <= rang_1, alpha_1, 
                  np.where(diff_other <= rang_2, alpha_2,
                           np.where(diff_other <= rang_3, alpha_3, alpha_4)))
    value_size = np.where(diff_other <= rang_1, size_1, 
                  np.where(diff_other <= rang_2, size_2,
                           np.where(diff_other <= rang_3, size_3, size_4)))
    
    alphas_SW2[ind] = value_alpha
    size_SW2[ind] = value_size
    ind=ind+1

ind=0
for index, row in df_WaveguideMeshes.iterrows():
    value_loss = float(row['dB/TBU'])
    PA_loss = float(row['Phase shifters'])
    value_loss_func = np.exp(p(np.log(PA_loss)))
    diff_other=abs(value_loss-value_loss_func)
    # print(diff_other)
    value_alpha = np.where(diff_other <= rang_1, alpha_1, 
                  np.where(diff_other <= rang_2, alpha_2,
                           np.where(diff_other <= rang_3, alpha_3, alpha_4)))
    value_size = np.where(diff_other <= rang_1, size_1, 
                  np.where(diff_other <= rang_2, size_2,
                           np.where(diff_other <= rang_3, size_3, size_4)))
    
    alphas_GP2[ind] = value_alpha
    size_GP2[ind] = value_size
    # print(value_size)
    ind=ind+1
ind=0


hue_order = ['SOI', 'SiN', 'InP', 'Silica']
chart = sns.scatterplot(x='Phase shifters', y='dB/TBU(dB)', data=df_other_filtered, hue='Foundry', hue_order=hue_order, palette=colors, s=size_1, edgecolor=None, ax=axs[0], legend=False)
chart = sns.scatterplot(x='Phase shifters', y='dB/TBU', data=df_WaveguideMeshes, hue='Foundry', hue_order=hue_order, palette=colors, s=size_1, edgecolor=None, ax=axs[0], legend=False)
chart = sns.scatterplot(x='Phase shifters', y='dB/TBU', data=df_MultiportInterferometers, hue='Foundry', hue_order=hue_order, palette=colors, s=size_1, edgecolor=None, ax=axs[0], legend=False)
chart = sns.scatterplot(x='Phase shifters', y='dB/TBU(dB)', data=df_Switch, hue='Foundry', hue_order=hue_order, palette=colors, s=size_1, edgecolor=None, ax=axs[0], legend=False)
axs[0].set_xscale('log')
axs[0].set_xlabel('Number of photonic actuators')
axs[0].set_ylabel('Loss (dB/PUC)')
axs[0].set_xlim(2, 1e4)
axs[0].set_xticks([1e1, 1e2, 1e3, 1e4])
axs[0].set_yticks([0, 1, 2, 3, 4, 5, 6])
axs[0].set_ylim(0, 6)


# POWER PER AREA
####################################################
df_Full['Power consumption(mW) per 2pi'] = df_Full['Power consumption(mW) per 2pi'].astype(float)
df_Full['Area (mm2)'] = pd.to_numeric(df_Full['Area (mm2)'], errors='coerce')
df_Full['Power_per_area'] = (df_Full['Power consumption(mW) per 2pi'] * df_Full['Phase shifters']) / (df_Full['Area (mm2)']*1000)
max_per_year = 2
df_Full_mpy = df_Full.groupby('Year').apply(lambda group: group.nlargest(max_per_year, 'Power_per_area')).reset_index(drop=True)
x1, y1 = 2008, 1000
x2, y2 = 2025, 50
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1
def interp(x):
    return m * x + b
x_valor = 2010
y_valor = interp(x_valor)
val_x = np.linspace(2005, 2025, 100)
val_y = interp(val_x)
alphas_PPA = np.zeros(len(df_Full['Power_per_area']))
sizes_PPA = np.zeros(len(df_Full['Power_per_area']))
rang_1 = 100
rang_2 = 250
rang_3 = 500
alpha_1 = 1
alpha_2 = 0.75
alpha_3 = 0.5
alpha_4 = 0.25
size_1 = 25
size_2 = 20
size_3 = 18
size_4 = 15

for index, row in df_Full.iterrows():
    value_PPA = float(row['Power_per_area'])
    year_PPA = float(row['Year'])
    value_PPA_func = interp(year_PPA)
    diff_PPA=abs(value_PPA-value_PPA_func)
    value_alpha = np.where(diff_PPA <= rang_1, alpha_1, 
                  np.where(diff_PPA <= rang_2, alpha_2,
                           np.where(diff_PPA <= rang_3, alpha_3, alpha_4)))
    value_size = np.where(diff_PPA <= rang_1, size_1,
                  np.where(diff_PPA <= rang_2, size_2,
                           np.where(diff_other <= rang_3, size_3, size_4)))
    
    alphas_PPA[ind] = value_alpha
    sizes_PPA[ind] = value_size
    ind=ind+1
chart = sns.scatterplot(x='Year', y='Power_per_area', data=df_Full, hue='Foundry', hue_order=hue_order, palette=colors, s=size_1, edgecolor=None, ax=axs[1], legend=False)
axs[1].set_ylim(0,1)
axs[1].set_xlim(2005, 2025)
axs[1].set_ylabel("Power per area (W/mm2)")
axs[1].set_xlabel("Year")


# ELECTRICAL PACKAGING
####################################################
rang_1 = 100
rang_2 = 250
rang_3 = 500
alpha_1 = 1
alpha_2 = 0.75
alpha_3 = 0.5
alpha_4 = 0.25
size_1 = 25
size_2 = 20
size_3 = 18
size_4 = 15

probing_data = df_Full[df_Full['Electrical Packaging'] == 'Probing']
wirebonding_data = df_Full[df_Full['Electrical Packaging'] == 'Wire bonding']
monolithic_data = df_Full[df_Full['Electrical Packaging'] == 'Monolithic']
num_actuators_probing = probing_data['Phase shifters'].tolist()
num_actuators_wirebonding = wirebonding_data['Phase shifters'].tolist()
num_actuators_monolithic = monolithic_data['Phase shifters'].tolist()
df_clean = df_Full.dropna(subset=['Electrical Packaging'])
df_clean = df_clean[df_clean['Electrical Packaging'] != 'ND']
order = ['Probing', 'Wire bonding', 'Flip-chip', 'Monolithic']
df_reordered = df_clean[df_clean['Electrical Packaging'].isin(order)]
df_reordered['Electrical Packaging'] = pd.Categorical(df_reordered['Electrical Packaging'], categories=order, ordered=True)
df_reordered = df_reordered.sort_values('Electrical Packaging')
colors = [Nat_blue, Nat_red, Nat_green, Nat_orange]
ind=0
fig.tight_layout()
plt.savefig('plots/PIC_performance.svg', format='svg')
plt.show()