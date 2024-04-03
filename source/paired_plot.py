import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np
import pandas as pd

def paired_plot(filename, type_ai=None, group_user=None, sub=None, sub_vals=[], ai_level=None, savename="plot", palette=None, measure="Accuracy"):
    data = pd.read_csv(filename)

    if group_user is None:
        accs = data.groupby("id")["HD1"].mean()
        q25 = np.quantile(accs, 0.25)
        q75 = np.quantile(accs, 0.75)
        data["Type_H"] = "Others"
        data.loc[data["id"].isin(accs[accs <= q25].index), "Type_H"] = "Low performer"
        data.loc[data["id"].isin(accs[accs >= q75].index), "Type_H"] = "High performer"
        group_user = "Type_H"
        
    if type_ai is not None:
        data = data[data["Type_AI"] == type_ai]

    if sub is not None:
        data = data[data[sub].isin(sub_vals)]

    if group_user is not None:
        data = data.loc[:,["id","HD1","FHD", group_user]]
    else:
        data = data.loc[:,["id","HD1","FHD"]]
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [3, 1]})

    
    acc_pre = data.groupby("id")["HD1"].mean().values
    acc_post = data.groupby("id")["FHD"].mean().values
    diff = acc_post - acc_pre
    mean_diff = np.mean(diff)
    ci = 1.96*np.std(mean_diff)/np.sqrt(len(diff))

    x1 = 0.5 + 0.1*np.random.rand(len(data["id"].unique()))
    x2 = 1.5+ 0.1*np.random.rand(len(data["id"].unique()))

    if ai_level is not None:
        staircase = diff[(diff > 0) & (acc_post < ai_level)].shape[0]/diff.shape[0]
        repulsion = diff[(diff < 0) & (acc_pre < ai_level)].shape[0]/diff.shape[0]
        outperformance = diff[(diff > 0) & (acc_post >= ai_level) & (acc_pre < ai_level)].shape[0]/diff.shape[0]
        ballast = diff[(diff < 0) & (acc_pre >= ai_level)].shape[0]/diff.shape[0]
        spur = diff[(diff > 0) & (acc_pre >= ai_level)].shape[0]/diff.shape[0]
        unchanged = diff[(diff == 0)].shape[0]/diff.shape[0]
    
        annot = f'Lifted: {staircase:.2f},   Repulsed: {repulsion:.2f}, Unaffected: {unchanged:.2f},\nOutperformers: {outperformance:.2f}, Ballasted: {ballast:.2f}, Spurred: {spur:.2f}'
    else:
        empowered = diff[(diff > 0)].shape[0]/diff.shape[0]
        undermined = diff[(diff < 0)].shape[0]/diff.shape[0]
        unaffected = diff[(diff == 0)].shape[0]/diff.shape[0]
        annot = f'Empowered: {empowered:.2f}, Unaffected: {unaffected:.2f}, Undermined: {undermined:.2f}'

    # Paired Dot Plot
    if group_user is not None:
        if palette is None:
            palette = dict(zip([i for i, _ in enumerate(data[group_user].unique())],
                               sns.color_palette('colorblind', len(data[group_user].unique()))))
        for i, v in enumerate(data[group_user].unique()):
            acc_pre_v = data[data[group_user] == v].groupby("id")["HD1"].mean().values
            acc_post_v = data[data[group_user] == v].groupby("id")["FHD"].mean().values
            x1 = 0.5 + 0.1*np.random.rand(len(data[data[group_user] == v]["id"].unique()))
            x2 = 1.5+ 0.1*np.random.rand(len(data[data[group_user] == v]["id"].unique()))
            ax1.plot([x1,x2], [acc_pre_v, acc_post_v], color="gray", alpha=0.5)
            ax1.scatter(x1, acc_pre_v, color=palette[v], alpha=0.7, marker='o', label=v)
            ax1.scatter(x2, acc_post_v, color=palette[v], alpha=0.7, marker='o')
            ax1.set_xticks([0.5, 1.5])
            ax1.set_xticklabels(['HD1', 'FHD'])
            ax1.set_ylabel(measure)
            ax1.set_title('Paired Dot Plot')

            diff_v = acc_post_v - acc_pre_v
            mean_diff_v = np.mean(diff_v)
            ci_v = 1.96*np.std(mean_diff_v)/np.sqrt(len(diff_v))
            ax2.errorbar(x=np.random.normal(loc=0.0, scale=0.1), y=mean_diff_v, xerr=ci_v, fmt='o', color=palette[v], capsize=5)
        
        if ai_level is not None:
            ax1.axhline(y=ai_level, color='grey', linestyle='--', label="AI")
        ax1.legend()

    else:
        ax1.plot([x1,x2], [acc_pre, acc_post], color="gray", alpha=0.5)
        ax1.scatter(x1, acc_pre, color='black', alpha=0.7, marker='o')
        ax1.scatter(x2, acc_post, color='black', alpha=0.7, marker='o')
        ax1.set_xticks([0.5, 1.5])
        ax1.set_xticklabels(['HD1', 'FHD'])
        ax1.set_ylabel(measure)
        ax1.set_title('Paired Dot Plot')

        ax2.errorbar(x=0, y=mean_diff, xerr=ci, fmt='o', color='black', capsize=5)
        if ai_level is not None:
            ax1.axhline(y=ai_level, color='grey', linestyle='--', label="AI")
        ax1.legend()
    
    ax2.axhline(y=0, color='grey', linestyle='--')  # Nil difference line
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title('Mean Difference')

    
    # Move the vertical axis to the right side
    ax2.set_xlabel(None)
    ax2.set_ylabel(None)
    ax2.set_xticks([])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.spines[['right', 'top']].set_visible(False)
    ax2.spines[['left', 'top', 'bottom']].set_visible(False)

    ax1.annotate(annot,
                 xy=(0.5, -0.125), ha='center', va='center', xycoords='axes fraction',
                 bbox=dict(facecolor='none', edgecolor='black',boxstyle='round', alpha=0.2))

    plt.tight_layout()
    plt.savefig(savename + (sub_vals[0] if sub is not None else "") + "_paired.png", dpi=300)