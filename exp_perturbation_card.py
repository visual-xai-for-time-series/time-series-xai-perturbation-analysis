import os
import time

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec


def create_perturbation_analysis_card(dataset, attribution, name, data, results, cur_dir):
    changed = results['changed']
    unchanged = results['unchanged']
    
    data_changed = data[[x['index'] for x in changed]]
    data_unchanged = data[[x['index'] for x in unchanged]]
    
    skewness_changed = np.nan_to_num([x['stats']['skew'] for x in changed])
    skewness_mean_changed = np.round(np.mean(skewness_changed), decimals=4)
    
    skewness_unchanged = np.nan_to_num([x['stats']['skew'] for x in unchanged])
    skewness_mean_unchanged =np.round(np.mean(skewness_unchanged), decimals=4)
    
    mean_changed = np.nan_to_num([x['stats']['mean'] for x in changed])
    mean_unchanged = np.nan_to_num([x['stats']['mean'] for x in unchanged])
    
    euc_dist_changed = np.nan_to_num([x['dist']['euclidean'] for x in changed])
    euc_dist_mean_changed = np.round(np.mean(euc_dist_changed), decimals=4)

    euc_dist_unchanged = np.nan_to_num([x['dist']['euclidean'] for x in unchanged])
    euc_dist_mean_unchanged = np.round(np.mean(euc_dist_unchanged), decimals=4)
    
    cos_dist_changed = np.nan_to_num([x['dist']['cosine'] for x in changed])
    cos_dist_mean_changed = np.round(np.mean(cos_dist_changed), decimals=4)
    
    cos_dist_unchanged = np.nan_to_num([x['dist']['cosine'] for x in unchanged])
    cos_dist_mean_unchanged = np.round(np.mean(cos_dist_unchanged), decimals=4)
    
    title = f'Dataset: {dataset:<25s} Attribution: {attribution:<25s} Perturbation: {name:<25s} Size: {len(data):<20d}\n'
    title += f'Changed: {len(changed):>20d} / {len(data):<20d} {" ":<100s} Unchanged: {len(unchanged):>20d} / {len(data):<20d}\n'

    cmap = mpl.colormaps['tab10']
    alpha = 0.5

    fig = plt.figure(layout='constrained')

    changed_text = '\u2588' * int(len(changed) / len(data) * 50)
    unchanged_text = '\u2588' * int(len(unchanged) / len(data) * 50)
    changed_text_length = int(len(changed) / len(data) * 100)
    fig.text(0.39, 0.9555, changed_text, size="medium", color=cmap(0), alpha=alpha)
    fig.text(0.39 + changed_text_length * 0.0018, 0.9555, unchanged_text, size="medium", color=cmap(1), alpha=alpha)

    fig.suptitle(title)
    fig.set_size_inches(30, 15)

    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])
    
    # Change bar chart
    
    labels, _ = np.unique([x['label'] for x in unchanged], return_counts=True)
    
    unique, counts = np.unique([x['old_label'] for x in changed], return_counts=True)
    counts = np.array([counts[np.where(x == unique)][0] if x in unique else 0 for x in range(len(labels))])
    ax1.bar(np.arange(len(labels)) * 0.5, counts, label='old changed', width = 0.45, alpha=alpha)
    
    unique, counts = np.unique([x['new_label'] for x in changed], return_counts=True)
    counts = np.array([counts[np.where(x == unique)][0] if x in unique else 0 for x in range(len(labels))])
    ax1.bar((np.arange(len(labels)) + len(labels)) * 0.5, counts, label='new changed', width = 0.45, alpha=alpha)

    unique, counts = np.unique([x['new_label'] for x in unchanged], return_counts=True)
    counts = np.array([counts[np.where(x == unique)][0] if x in unique else 0 for x in range(len(labels))])
    ax1.bar(np.arange(len(labels)) + len(labels) + 0.5, counts, label='unchanged', alpha=alpha)
    
    ax1.set_xticks((np.arange(len(labels) * 2) * 0.5).tolist() +  (0.5 + len(labels) * 2 * 0.5 + np.arange(len(labels))).tolist())
    ax1.set_xticklabels([*labels, *labels, *labels])
    ax1.legend(loc='center right')
    ax1.set_title('Labels of changed and unchanged')
    
    # Perturbed amount
    
    ax2.hist([x['perturbed_values'] for x in changed], bins=50, alpha=alpha)
    ax2.set_title('Histogram of amount of perturbed values for changed')
    
    # Distance Histogram
    
    ax3.hist(euc_dist_changed, bins=50, alpha=alpha, label='changed')
    ax3.hist(euc_dist_unchanged, bins=50, alpha=alpha, label='unchanged')
    ax3.legend()
    ax3.set_title('Histogram of euclidean distances of the perturbed to the original instance')
    
    # Skewness distribution
    
    ax4.hist(skewness_changed, bins=50, edgecolor='None', alpha=alpha, color=cmap(0), label='changed')
    ax4.hist(skewness_unchanged, bins=50, edgecolor='None', alpha=alpha, color=cmap(1), label='unchanged')
    ax4.legend()
    ax4.set_title('Skewness of the attributions')

    # 
    ax5.plot(np.mean(data_changed, axis=0), label='changed', alpha=alpha, color=cmap(0))
    ax5.plot(np.mean(data_unchanged, axis=0), label='unchanged', alpha=alpha, color=cmap(1))
    ax5.legend()
    ax5.set_title('Means of the data samples')

    # Distance Histogram
    
    ax6.hist(cos_dist_changed, bins=50, alpha=alpha, label='changed')
    ax6.hist(cos_dist_unchanged, bins=50, alpha=alpha, label='unchanged')
    ax6.legend()
    ax6.set_title('Histogram of cosine distances of the perturbed to the original instance')
    
    # Mean distribution
    
    ax7.hist(mean_changed, bins=50, edgecolor='None', alpha=alpha, color=cmap(0), label='changed')
    ax7.hist(mean_unchanged, bins=50, edgecolor='None', alpha=alpha, color=cmap(1), label='unchanged')
    ax7.legend()
    ax7.set_title('Mean of the attributions')
    

    plt.tight_layout()
    os.makedirs(cur_dir, exist_ok=True)
    plt.savefig(f'{cur_dir}/{attribution}-{name}.png', dpi=300)
#     plt.show()
    plt.close()