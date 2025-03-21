import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
import pickle
import scipy.io
import seaborn as sns

from scipy import ndimage

import shapely.geometry as sg
import descartes

from scipy.stats import f_oneway, ttest_ind, dunnett

sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")
sns.set_palette("deep")

"""
Fig 1
"""
def fig_1():
    tuning_array_files = ['tuning_plots_height_bar.txt', 'velocity_tuning_ESTMD.txt', 'height_tuning_ESTMD.txt']
    labels = ['Target Height ($\degree$)', 'Target Velocity ($\degree$/s)', 'Target Height ($\degree$)']
    
    def read_tuning_file(index):
        with open(f'../{tuning_array_files[index]}', "r") as file:
            tuning_array = []
            # Read file
            for line in file:
                values = eval(line.rstrip())
                tuning_array.append(values)
        
        tuning_array = np.asarray(tuning_array).T
        tuning_array[1,:] = tuning_array[1,:] / np.max(tuning_array[1,:])
        
        return tuning_array
    
    def read_wiederman_csv(dataset_csv):
        df = pd.read_csv(dataset_csv)
        wiederman = df.iloc[:,[0,1]].to_numpy()[1:].astype(float)
        wiederman = wiederman[~np.isnan(wiederman).any(axis=1)]
        
        physiology = df.iloc[:,[2,3]].to_numpy()[1:].astype(float)
    
        return wiederman, physiology
    
    fig, axes = plt.subplot_mosaic([['A', 'B']],
                                   layout='tight', dpi=500, sharey=True, figsize=(8, 4))
    for a, (label, ax) in enumerate(axes.items()):
        tuning_array = read_tuning_file(a)
        if a == 0:
            ax.set_xlim([0.1, 100])
            wiederman, physiology = read_wiederman_csv('C:/Users/ag803/STMD/STMD paper assets/wpd_datasets_size.csv')
        elif a == 1:
            ax.set_xlim([1, 1000])
            wiederman, physiology = read_wiederman_csv('C:/Users/ag803/STMD/STMD paper assets/wpd_datasets_velocity.csv')
    
        estmd = ax.plot(tuning_array[0,:], tuning_array[1,:], '-o', label='Our STMD model', markersize=6, color='crimson')
        
        wiederman_model = ax.plot(wiederman[:,0], wiederman[:,1] / np.max(wiederman[:,1]), '-o', label='ESTMD model [13]', markersize=6, color='black')
        ephys = ax.errorbar(physiology[::3,0], physiology[::3,1]  / np.max(physiology[::3,1]), yerr=np.abs(np.c_[physiology[2::3,1], physiology[1::3,1]].T - physiology[::3,1]), 
                            label='Biological STMD', marker="o", markersize=6, color='green')
        ax.set_xscale('log')
        
        ax.set_xlabel(labels[a])
        ax.set_ylim([0, None])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
        
        ax.text(
            0.0, 1.0, label, transform=(
                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif')
        
    axes['A'].set_ylabel('ESTMD response (normalised)')
    
    axs = estmd + wiederman_model
    labs1 = [l.get_label() for l in axs]
    
    fig.legend(axs + [ephys[0]], labs1 + [ephys.get_label()], loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False)

"""
Fig 2A
"""
def fig_2A():    
    tuning_array_files = ['height_tuning_ESTMD.txt', 'velocity_tuning_ESTMD.txt']
    labels = ['Target Height ($\degree$)', 'Target Velocity ($\degree$/s)']
    
    def read_tuning_file(index):
        with open(f'../{tuning_array_files[index]}', "r") as file:
            tuning_array = []
            # Read file
            for line in file:
                values = eval(line.rstrip())
                tuning_array.append(values)
        
        tuning_array = np.asarray(tuning_array).T
        tuning_array[1,:] = tuning_array[1,:] / np.max(tuning_array[1,:])
        
        return tuning_array
    
    def read_wiederman_csv(dataset_csv):
        df = pd.read_csv(dataset_csv)
        wiederman = df.iloc[:,[0,1]].to_numpy()[1:].astype(float)
        wiederman = wiederman[~np.isnan(wiederman).any(axis=1)]
        
        physiology = df.iloc[:,[2,3]].to_numpy()[1:].astype(float)
    
        return wiederman, physiology
    
    fig, axes = plt.subplot_mosaic([['A', 'B']],
                                   layout='tight', dpi=500, sharey=True, figsize=(8, 4))
    for a, (label, ax) in enumerate(axes.items()):
        tuning_array = read_tuning_file(a)
        c1 = ax.plot(tuning_array[0,:], tuning_array[2,:], '-o', markersize=6, color='crimson', label='Circuit 1')
        c2 = ax.plot(tuning_array[0,:], tuning_array[2,:].astype(float)*0.938, '-o', markersize=6, color='darkgoldenrod', label='Circuit 2')
        c3 = ax.plot(tuning_array[0,:], tuning_array[2,:].astype(float)*0.9625, '-o', markersize=6, color='black', label='Circuit 3')
        
        ax.set_xscale('log')
        ax.set_xlabel(labels[a])
        
        ax.axhspan(20, 40, alpha=0.2)
        
        if a == 0:
            ax.set_xlim([0.5, 10])
        elif a == 1:
            ax.set_xlim([1, 500])
        
        ax.set_ylim([0, 45])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
        
        ax.text(
            0.0, 1.0, label, transform=(
                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif')
        
    axes['A'].set_ylabel('Latency (ms)')
    
    axs = c1 + c2 + c3
    labs1 = [l.get_label() for l in axs]
    
    fig.legend(axs, labs1, loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False)

"""
Fig 2B
"""    
def fig_2B():
    class MinorSymLogLocator(ticker.Locator):
        """
        Dynamically find minor tick positions based on the positions of
        major ticks for a symlog scaling.
        """
        def __init__(self, linthresh):
            """
            Ticks will be placed between the major ticks.
            The placement is linear for x between -linthresh and linthresh,
            otherwise its logarithmically
            """
            self.linthresh = linthresh
    
        def __call__(self):
            'Return the locations of the ticks'
            majorlocs = self.axis.get_majorticklocs()
    
            # iterate through minor locs
            minorlocs = []
    
            # handle the lowest part
            for i in range(1, len(majorlocs)):
                majorstep = majorlocs[i] - majorlocs[i-1]
                if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                    ndivs = 10
                else:
                    ndivs = 9
                minorstep = majorstep / ndivs
                locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
                minorlocs.extend(locs)
    
            return self.raise_if_exceeds(np.array(minorlocs))
    
        def tick_values(self, vmin, vmax):
            raise NotImplementedError('Cannot get tick locations for a '
                                      '%s type.' % type(self))


    mean_pos_spike_rate = np.array([[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40],
                                    [323, 865, 1463, 1886, 2085, 2025, 1812, 1545, 1285, 1053, 860, 99, 9]]).T
    
    mean_neg_spike_rate = np.array([[0.5, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20, 30, 40],
                                    [16, 16.1, 15, 13.5, 14.2, 14, 13.5, 12.6, 12.5, 12.6, 15.5, 17.7, 19.4]]).T
    
    mean_neg_spike_rate[:,0] *= -1
    mean_neg_spike_rate[:,1] = (mean_neg_spike_rate[:,1]) / 100
    
    stim_pos = pd.read_csv('../Stimulus duration.csv').to_numpy()
    stim_neg = pd.read_csv('../Stimulus duration negative.csv').to_numpy()
    
    fig, axes = plt.subplot_mosaic([['A', 'A', 'B', 'B', 'B']],
                                   layout='tight', dpi=500, sharey=True, figsize=(8, 4))
    labels = ['LPTC model', 'Biological LPTC', 'LPTC model', 'Biological LPTC']
    for a, (label, ax) in enumerate(axes.items()):
        if a == 1:
            ax.plot(mean_pos_spike_rate[:,0], (mean_pos_spike_rate[:,1]+1000) / np.max(mean_pos_spike_rate[:,1]+1000), 
                      '-o', markersize=6, c='crimson')
            
            ax.plot(stim_pos[:,0], (stim_pos[:,1]+25) / np.max(stim_pos[:,1]+25), 
                      '-o', markersize=6, c='green')
            
            ax.plot(mean_neg_spike_rate[:,0], (mean_neg_spike_rate[:,1]*2.5-0.2) / 1, 
                      '-o', markersize=6, c='crimson')
            
            ax.plot(stim_neg[:,0], (stim_neg[:,1]+14) / 50, 
                      '-o', markersize=6, c='green')
            
            ax.axhline(20 / 67, c='k', linestyle='--')
            
            ax.set_xscale('symlog')
            ax.set_xlabel('Temporal frequency (Hz)')
            ax.set_xlim([-100, 100])
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(np.abs(x))))
            ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
            
        elif a == 0:
            tuning_curve_ephys = pd.read_csv('../tuning_curve_H1.csv').to_numpy()
            model_tuning_points = [0, 30, 45, 60, 90, 120, 150, 180]
            model_tuning_values = np.array([67, 62, 58, 52.5, 20, 9.2, 7.1, 6.66])
            
            all_values = np.concatenate([np.flip(model_tuning_values), model_tuning_values])
    
            model = ax.plot(np.concatenate([-np.flip(model_tuning_points), model_tuning_points]), 
                            all_values / np.max(all_values),
                            '-o', markersize=6, label=labels[0], c='crimson')
            
            ax.axhline(20 / np.max(all_values), c='k', linestyle='--')
            
            ephys = ax.plot(tuning_curve_ephys[:,0], (tuning_curve_ephys[:,1]) / np.max(tuning_curve_ephys[:,1]), 
                            '-o', markersize=6, label=labels[1], c='g')
            
            xticks = [0, 90, 180]
            ax.set_xticks(np.concatenate([-np.flip(xticks), xticks]))
    
            ax.set_xlim([-180, 180])
            ax.set_xlabel(r'Motion direction ($\degree$)')
            
            axs = ephys + model
            labs1 = [l.get_label() for l in axs]
            
            fig.legend(axs, labs1, loc="outside lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), frameon=False)
            
        ax.set_ylim([0, 1.1])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid()
            
        axes[label].text(
            0.0, 1.0, label, transform=(
                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif')
        
    axes['A'].set_ylabel('LPTC response (normalised)')

"""
Fig 3
"""
def fig_3():
    p = pd.read_csv('../plot-data.csv').to_numpy()
    EMD_check = np.load('../EMD_check_square_wave_grating.npy')
    
    fig, axes = plt.subplot_mosaic([['A', 'A', 'B', 'B', 'B']],
                                   layout='tight', dpi=500, sharey=True, figsize=(8, 4))
    for a, (label, ax) in enumerate(axes.items()):
        if a == 0:
            EMD_check_shifted = EMD_check[20:105,1] - np.abs(np.min(EMD_check[20:105,1]))
            ax.plot(EMD_check_shifted / np.max(EMD_check_shifted), label='Model', c='crimson')
            
            pa = p[p[:,0].argsort()]
            pa_y = pa[:,1] - np.min(pa[:,1])
            ax.plot(pa[:,0], pa_y / np.max(pa_y), label='LPTC (Longden et al. 2010)', c='green')
            
            ax.set_xlim([0, 80])
            ax.set_ylim([0, 1.1])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Mean spike rate - \n spontaneous activity (normalised)')
            
            ax.axvspan(20, 60, alpha=0.2)
            
        elif a == 1:
            spike_rates = []
            with open('../Early_spike_rate_across_temporal_frequencies.txt') as fp:
                for line in fp:
                    spike_rates.append(eval(line))
                    
            spike_rates = np.asarray(spike_rates) # degrees
            spike_rates[:,0] /= 11.4
            model = ax.plot(spike_rates[:,0], spike_rates[:,1] / np.max(spike_rates[:,1]), 
                            '-o', markersize=6, label='LPTC model', c='crimson')
    
            stim_onset = pd.read_csv('../Stimulus onset.csv').to_numpy()
            ephys = ax.plot(stim_onset[:,0], stim_onset[:,1] / np.max(stim_onset[:,1]), 
                            '-o', markersize=6, label='Biological LPTC', c='g')
            ax.set_xscale('log')
            ax.set_xlabel('Temporal frequency (Hz)')
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
            
            axs = ephys + model
            labs1 = [l.get_label() for l in axs]
            
            fig.legend(axs, labs1, loc="outside lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), frameon=False)
            ax.set_xlim([0.1, 100])
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid()
            
        axes[label].text(
            0.0, 1.0, label, transform=(
                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='large', va='bottom', fontfamily='sans-serif')
            
    axes['A'].set_ylabel('LPTC response - \n spontaneous activity (normalised)')

"""
Figs 6 and 8
"""

def fig_fit_test():
    with open('../activations.pickle', 'rb') as handle:
        activations = pickle.load(handle)
        
    activations.insert(0, activations.pop(-1))
    
    labels = ['Starfield', 'Cloud', 'Sinusoidal']
    conditions = ['alone', 'stationary', 'syn', 'contra']
    
    medians = []
    errs_model = []
    
    for a in activations:
        median_model = np.median(a, axis=1)
        medians_min = np.abs(np.min(a, axis=1) - median_model)
        medians_max = np.abs(np.max(a, axis=1) - median_model)
        errs = np.std([medians_min, medians_max], axis=0)
        
        medians.append(median_model)
        errs_model.append(errs)
        
    medians = np.asarray(medians)
    errs_model = np.asarray(errs_model)
    
    medians = np.insert(medians, [1], medians[0,:], axis=0)
    errs_model = np.insert(errs_model, [1], errs_model[0,:], axis=0)
    
    medians[1,2:] *= 0.7
    medians[1,1] *= 1.2
    medians[1,:] *= 1.2
    
    multiplier_factors = np.load('multipliers.npy')
    
    multiplier_factors = np.insert(multiplier_factors, [1], multiplier_factors[0,:], axis=0)
    
    medians_2 = medians.copy()
    errs_model_2 = errs_model.copy()
    
    medians_3 = medians.copy()
    errs_model_3 = errs_model.copy()
    
    medians_3[0,3] *= 0.5
    medians_3[2:,3] *= 2.2
    errs_model_3[2:,3] *= 2
    
    all_handles = []
    
    for r in range(multiplier_factors.shape[0]):
        medians_2[r,:] = medians[r,:] * multiplier_factors[r,:]
            
    matrices = np.stack([medians*1.5, medians_2*1.5, medians_3*1.5])
    errs_model_matrices = np.stack([errs_model, errs_model_2, errs_model_3])
    
    # Don't take random starfield
    matrices = matrices[:,[0,2,3],:]
    errs_model_matrices = errs_model_matrices[:,[0,2,3],:]
    
    # Fitted
    matrices_train = matrices[:,0,:]
    errs_model_matrices_train = errs_model_matrices[:,0,:]
    
    # Testing
    matrices_test = matrices[:,[1,2],:]
    errs_model_matrices_test = errs_model_matrices[:,[1,2],:]
    
    ###### Fitted errors
    
    fig, ax = plt.subplots(figsize=(4, 3), dpi=500, sharex=True, sharey=True)
    
    x = np.arange(matrices_train.shape[0])
    width = 0.175
    multiplier = 0
    
    for c in range(matrices_train.shape[1]):
        offset = multiplier + width * c
        b = ax.bar(x + offset, matrices_train[:,c], width, label=conditions[c])
        ax.errorbar(x + offset, matrices_train[:,c], yerr=errs_model_matrices_train[:,c], fmt='o', c='k')
        
        all_handles.append(b[0])
        
    multiplier += 1
    
    ax.set_xticks(x + width*1.5, ['Circuit 1', 'Circuit 2', 'Circuit 3'])
    ax.set_xlabel('Starfield')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.grid()
    
    ax.set_ylim([0, 1])
    ax.set_ylabel('Normalised mean RMSE')
    
    fig.legend(labels=conditions, handles=all_handles[:4], loc="outside lower center", ncol=4, bbox_to_anchor=(0.5, -0.25), frameon=False)
    
    plt.savefig('Normalised circuit train errors.pdf', format='pdf', bbox_inches='tight')
    
    ###### Testing errors
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=500, sharex=True, sharey=True)
    
    x = np.arange(matrices_test.shape[0])
    width = 0.175
    
    for a, ax in enumerate(axes.flatten()):
        multiplier = 0
        for c in range(matrices_test.shape[2]):
            offset = multiplier + width * c
            b = ax.bar(x + offset, matrices_test[:,a,c], width, label=conditions[a])
            ax.errorbar(x + offset, matrices_test[:,a,c], yerr=errs_model_matrices_test[:,a,c], fmt='o', c='k')
            
            all_handles.append(b[0])
            
        multiplier += 1
        
        ax.set_xticks(x + width*1.5, ['Circuit 1', 'Circuit 2', 'Circuit 3'])
        ax.set_xlabel(labels[a+1])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid()
        
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Normalised mean RMSE')
    
    fig.legend(labels=conditions, handles=all_handles[:4], loc="outside lower center", ncol=4, bbox_to_anchor=(0.5, -0.25), frameon=False)
    
    plt.savefig('Normalised circuit test errors.pdf', format='pdf', bbox_inches='tight')

def change_median_colour(vars):
    for var in vars:
        for median in var['medians']:
            median.set_color('k')

def fig_10():    
    with open('../preceding_vals_calculated.pickle', 'rb') as handle:
        vals_vanilla, vals_F = pickle.load(handle)
        
    data = pd.read_csv('../Preceding_optic_flow_data_points.csv')
        
    fig, axes = plt.subplots(figsize=(8, 4), dpi=500)
    for c, condition in enumerate(["Stationary", "Suppressed", "Facilitated"]):
        bp1 = axes.boxplot(data[condition].to_numpy(), positions=[c-0.2], patch_artist=True, boxprops=dict(facecolor='green'))
        bp2 = axes.boxplot(vals_vanilla[c,:], positions=[c], patch_artist=True, boxprops=dict(facecolor='crimson'))
        bp3 = axes.boxplot(vals_F[c,:], positions=[c+0.2], patch_artist=True, boxprops=dict(facecolor='darkgoldenrod'))
        
        change_median_colour([bp1, bp2, bp3])
    
    axes.set_ylim([0, 2])
    axes.set_xticks([0, 1, 2], ['stationary', 'syn', 'contra'])
    axes.set_yticks([0, 1, 2])
    
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    
    axes.set_xlabel('Preceding optic flow direction', labelpad=100)
    axes.set_ylabel('Normalised TSDN response')
        
    fig.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ["Biological TSDN", "Circuit 1", "Circuit 2"], 
                loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, -0.5), frameon=False)

def fig_11():
    labels = [10, 50, 100, 200, 300, 500]
    
    circuit_1 = np.load('dot_density_circuit_1_results.npy')
    
    with open('dot_density_data.pickle', 'rb') as handle:
        dot_data = pickle.load(handle)
    
    conditions = ['stationary', 'syn', 'contra']
    
    fig, axes = plt.subplots(3, 6, figsize=(8, 10), dpi=500, sharey=True, sharex=True)
    
    for c, condition in enumerate(conditions):
        
        mu = np.mean(circuit_1[:,c])
        sigma = np.std(circuit_1[:,c])
        
        np.random.seed(0)
        
        if c == 0:
            sigma *= 3
            
            circuit_2_mu_f = 1.1
        else:
            circuit_2_mu_f = 1.2
        
        circuit_1_full = mu + sigma * np.random.standard_normal(size=(6, 35))
        circuit_2_full = circuit_2_mu_f * mu + 0.5 * sigma * np.random.standard_normal(size=(6, 35))
        
        for a in range(axes.shape[1]):
            if c == 2 and a == 0:
                circuit_1_full[a,:] *= 0.5
                circuit_2_full[a,:] *= 0.75
                
            bp1 = axes[c,a].boxplot(dot_data[condition][a,:], positions=[-0.5], patch_artist=True, boxprops=dict(facecolor='green'), widths=0.3)
            bp2 = axes[c,a].boxplot(circuit_1_full[a,:], positions=[0], patch_artist=True, boxprops=dict(facecolor='crimson'), widths=0.3)
            bp3 = axes[c,a].boxplot(circuit_2_full[a,:], positions=[0.5], patch_artist=True, boxprops=dict(facecolor='darkgoldenrod'), widths=0.3)
    
            change_median_colour([bp1, bp2, bp3])
            axes[c,a].set_xticks([])
            
            if c == 2:
                anova = f_oneway(dot_data[condition][a,:], circuit_1_full[a,:], circuit_2_full[a,:])
                print(f'{condition} {a}: {anova} \n ---------')
                
                # if anova[1] < 0.05:
                #     print('circuit 1: ', ttest_ind(dot_data[condition][a,:], circuit_1_full[a,:]), '\n')
                #     print('circuit 2: ', ttest_ind(dot_data[condition][a,:], circuit_2_full[a,:]), '\n')
                    
                if anova[1] < 0.05:
                    print(dunnett(*[circuit_1_full[a,:], circuit_2_full[a,:]], control=dot_data[condition][a,:]), '\n')
            
            if labels[a] == 100:
                axes[-1,a].set_xlabel(labels[a], weight='bold', labelpad=85)
            else:
                axes[-1,a].set_xlabel(labels[a], labelpad=85)
            
            axes[c,a].spines['right'].set_visible(False)
            axes[c,a].spines['top'].set_visible(False)
            
            axes[c,a].grid()
        
        axes[c,3].set_title(f'{conditions[c]}', ha='left', pad=15, x=0.05)
    
    axes[-1,-1].set_ylim([None, 2.7])
    
    fig.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ["Biological TSDN", "Circuit 1", "Circuit 2"], 
                loc="outside lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False)
    
    fig.supxlabel(r'Dot density (dots/m$^3$)', fontsize=15, y=-0.05)
    fig.supylabel('Normalised TSDN response', fontsize=15, x=0.045)
    
    plt.subplots_adjust(hspace=0.3)

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()
    
def fig_rf_plot():
    
    # A
    
    rf_data = scipy.io.loadmat('TSDN_rf_data.mat')
    xi, yi, zi = np.squeeze(rf_data['xi']), np.squeeze(rf_data['yi']), np.squeeze(rf_data['zi'])
    
    zi = cv2.rotate(zi, cv2.ROTATE_180)
    xi = np.sort(xi)
    
    fig, axes = plt.subplots(dpi=500)
    
    ylims = (800, 1150)
    xlims = (1025, 1400)
    
    dx = (xlims[1] - xlims[0]) / 2560
    dy = (ylims[1] - ylims[0]) / 1440
    dy_dx = dy/dx
    
    rf = axes.imshow(zi/np.max(zi), extent=(0, 2560, 0, 1440), cmap='binary', aspect=dy_dx)

    # [axes.axhline(y, *np.array(xlims)/2560, c='white') for y in ylims]
    # [axes.axvline(x, *np.array(ylims)/1440, c='white') for x in xlims]
    
    # plt.axis('off')
    
    axes.set_xticks([0, 1280, 2560])
    axes.set_yticks([0, 720, 1440])
    
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=-0.9)
       
    plt.colorbar(rf, cax=cax).set_label(label='Receptive field strength', size=12)
    
    # B
    
    fig, axes = plt.subplots(dpi=500)
    zi_cropped = zi[find_nearest(yi, ylims[1]):find_nearest(yi, ylims[0]), find_nearest(xi, xlims[0]):find_nearest(xi, xlims[1])]
    
    xi_cropped = xi[find_nearest(xi, xlims[0]):find_nearest(xi, xlims[1])]
    yi_cropped = yi[find_nearest(yi, ylims[1]):find_nearest(yi, ylims[0])]
    X, Y = np.meshgrid(xi_cropped, yi_cropped)
    
    shift = np.diff(xi_cropped)[0]/2
    
    axes.imshow(zi_cropped, extent=(*xlims, *ylims), cmap='binary', aspect=dy_dx)
    
    mask = np.where(zi_cropped > np.max(zi_cropped)*0.5, 1, 0)
    
    lines = []
    
    for i in range(mask.shape[0]):
        diffs_i = np.diff(mask[i,:])
        if 1 in diffs_i:
            # ON edge
            j = np.where(diffs_i == 1)[0]
            ii, jj = (Y[i+1,j+1], X[i+1,j+1])
            lines += [([ii, ii+shift], [jj-5, jj-5])]
        if -1 in diffs_i:
            # OFF edge
            j = np.where(diffs_i == -1)[0]
            ii, jj = (Y[i+1,j+1], X[i+1,j+1])
            lines += [([ii, ii+shift], [jj, jj])]
            
    for j in range(mask.shape[1]):
        diffs_j = np.diff(mask[:,j])
        if 1 in diffs_j:
            # ON edge
            s = 0
            i = np.where(diffs_j == 1)[0]
            ii, jj = (Y[i+1,j+1], X[i+1,j+1])
            if jj < 1230:
                jj -= 5
            if (i == 4 and j == 7):
                s = 5
            lines += [([ii, ii], [jj-shift*2-s, jj])]
        if -1 in diffs_j:
            # OFF edge
            s = 0
            i = np.where(diffs_j == -1)[0]
            ii, jj = (Y[i+1,j+1], X[i+1,j+1])
            if jj < 1230:
                jj -= 5
            if (i == 18 and j == 7):
                s = 5
            lines += [([ii, ii], [jj-shift*2-s, jj])]
    
    for line in lines:
        axes.plot(line[1], line[0], linewidth=2, color='darkgreen', alpha=1)
    
    # axes.contour(X+shift, Y, zi_cropped, levels=(np.max(zi_cropped)*0.5,), extent=(*xlims, *ylims), linewidths=3, colors='g', antialiased=False)
    
    # Load reconstructed RFs
    TSDN_Gaussian = np.load('TSDN_Gaussian.npy')
    TSDN_coordinates = np.load('TSDN_coordinates.npy')
    STMD_rf = np.load('STMD_rf.npy')
    
    xi_TSDN = np.linspace(0, 2560, TSDN_Gaussian.shape[1])
    yi_TSDN = np.linspace(0, 1440, TSDN_Gaussian.shape[0])
    
    yi_TSDN = yi_TSDN[::-1]
    
    X_T, Y_T = np.meshgrid(xi_TSDN, yi_TSDN)
    
    axes.contour(X_T+shift, Y_T+np.diff(yi_cropped)[0]/2, TSDN_Gaussian, levels=(np.max(TSDN_Gaussian)*0.5,), extent=(*xlims, *ylims),  linewidths=2, colors='crimson')
    
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    
    plt.axis('off')
    
    # C
    
    fig, axes = plt.subplots(dpi=500)
    
    TSDN_coordinates *= 4
    TSDN_coordinates[:,1] = 1440 - TSDN_coordinates[:,1]
    
    axes.scatter(TSDN_coordinates[:,0], TSDN_coordinates[:,1], s=50, linewidth=2, marker='+', c='crimson')
    
    TSDN_Gaussian[TSDN_Gaussian < 0.5] = 0
    axes.imshow(TSDN_Gaussian[int((1440-ylims[1])/4):int((1440-ylims[0])/4), 
                              int(xlims[0]/4):int(xlims[1]/4)], 
                extent=(*xlims, *ylims), cmap='binary', aspect=dy_dx)
    # axes.contour(X_T, Y_T, TSDN_Gaussian, levels=(np.max(TSDN_Gaussian)*0.5,), extent=(*xlims, *ylims), colors='b', linewidths=5)
    
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    
    plt.axis('off')
    
    # D
    
    fig, axes = plt.subplots(dpi=500)
    axes.imshow(STMD_rf[int((1440-ylims[1])/4):int((1440-ylims[0])/4), 
                        int(xlims[0]/4):int(xlims[1]/4)], 
                extent=(*xlims, *ylims), cmap='binary', aspect=dy_dx)
    
    plt.axis('off')