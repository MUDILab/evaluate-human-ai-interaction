import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

def compute_benefits(filename="file.csv", type_ai=None, group_var=None, group_vals=[], savename="plot", palette=None, measure="Accuracy"):
    data = pd.read_csv(filename)
    
    if ("HD1" not in data.columns):
        return "Mandatory field HD1 is missing!"
    if ("FHD" not in data.columns):
        return "Mandatory field FHD is missing!"
    
    if "Type_AI" not in data.columns:
        data["Type_AI"] = ""

    if group_var is not None:
        data = data[data[group_var].isin(group_vals)]

    
    if type_ai == "all":
        data["Type_AI"] = "AI"
    elif type_ai is not None:
        data = data[data["Type_AI"] == type_ai]
        
    if "Study" not in data.columns:
        data["Study"] = ""
        
    
        
    if ("id" not in data.columns):
        data["id"] = data.index

    data = data.loc[:, ["id","HD1", "FHD", "Type_AI", "Type_H", "Study"]] if "Type_H" in data.columns else data.loc[:, ["id","HD1", "FHD", "Type_AI", "Study"]]
    grouped = data.groupby(["id","Type_AI", "Type_H", "Study"]).mean().reset_index() if "Type_H" in data.columns else data.groupby(["id","Type_AI", "Study"]).mean().reset_index()
    
    for v in grouped["Type_AI"].unique():
        for s in grouped["Study"].unique():
            temp = grouped[(grouped["Type_AI"] == v) & (grouped["Study"] == s)].copy()   
            baseline = temp["HD1"]
            difference = temp["FHD"] - temp["HD1"]
            q25 = np.quantile(baseline, 0.25)
            q75 = np.quantile(baseline, 0.75)
            
            if "Type_H" not in temp.columns:
                temp.loc[:,"Type_H"] = "Others"
                temp.loc[temp["id"].isin(temp.loc[temp["HD1"] <= q25, "id"]), "Type_H"] = "Low performer"
                temp.loc[temp["id"].isin(temp.loc[temp["HD1"] >= q75, "id"]), "Type_H"] = "High performer"
            else:
                temp.loc[:,"Type_H2"] = "Others"
                temp.loc[temp["id"].isin(temp.loc[temp["HD1"] <= q25, "id"]), "Type_H2"] = "Low performer"
                temp.loc[temp["id"].isin(temp.loc[temp["HD1"] >= q75, "id"]), "Type_H2"] = "High performer"
                
            name = savename + str(v) + " (" + str(s) + ")"
            benefit_diagram(baseline, difference, temp["Type_H"] if "Type_H" in temp.columns else temp["Type_H2"], name, palette, measure)

#baseline is a baseline accuracy (e.g., accuracy without AI)
#difference is the difference between two accuracies (e.g., accuracy with AI - accuracy without AI)
def benefit_diagram(baseline, difference, groups, ai_group="", palette=None, measure="Accuracy"):

    if palette is None:
        palette = dict(zip([v for i, v in enumerate(np.unique(groups))],
                            sns.color_palette('colorblind', len(np.unique(groups)))))
    
    plt.figure(figsize=(10,10))
    for i, v in enumerate(np.unique(groups)):
        group_size = groups[groups == v].shape[0]
        plt.scatter(baseline[groups == v], difference[groups == v], color=palette[v], label=v + " (n=" + str(group_size) + ")")

    #sns.scatterplot(x=baseline, y=difference, hue=groups, palette="muted")
    
    #plt.title(ai_group)

    vals = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.fill_between(vals, 1 - vals, alpha=0.3, color="blue" )
    plt.fill_between(vals, - vals, alpha=0.3, color="red" )

    max = np.max(difference)
    min = np.min(difference)
    ax_lim = np.max([max, np.abs(min)])
    
    plt.axhline(np.mean(difference), color="black")
    reg = LinearRegression()
    reg.fit(baseline.values.reshape(-1, 1),difference)
    plt.plot(np.linspace(0,1,10),reg.predict(np.linspace(0,1,10).reshape(-1,1)), 'k--', alpha=0.5, label="All")
    stat = stats.pearsonr(baseline, difference).statistic
    pval = stats.pearsonr(baseline, difference).pvalue
    upp = np.mean(difference) + 1.96*np.std(difference)/np.sqrt(len(difference))
    low = np.mean(difference) - 1.96*np.std(difference)/np.sqrt(len(difference))
    plt.axhline(low, color="black", alpha=0.25)
    plt.axhline(upp, color="black", alpha=0.25)

    if "Others" in np.unique(groups):
        reg_e = LinearRegression()
        reg_e.fit(baseline[groups != "Others"].values.reshape(-1, 1), difference[groups != "Others"])
        plt.plot(np.linspace(0,1,10),reg_e.predict(np.linspace(0,1,10).reshape(-1,1)), 'r--', alpha=0.5, label="Low & High performers")
        stat_e = stats.pearsonr(baseline[groups != "Others"], difference[groups != "Others"]).statistic
        pval_e = stats.pearsonr(baseline[groups != "Others"], difference[groups != "Others"]).pvalue

    
        annot = f'Avg. {measure:s} Diff.: {np.mean(difference):.2f} ({low: .2f}, {upp: .2f})\n\n r: {stat:.2f} (p: {pval:.3f}),   tan(α): {reg.coef_[0]:.2f},   α: {(180 - np.abs(np.rad2deg(np.arctan(reg.coef_[0])))):.2f}\n\n r (Low & High Perf.): {stat_e:.2f} (p: {pval_e:.3f}),   tan(α) (Low & High Perf.): {reg_e.coef_[0]:.2f},   α (Low & High Perf.): {(180 - np.abs(np.rad2deg(np.arctan(reg_e.coef_[0])))):.2f}'
    else:
        annot = f'Avg. {measure:s} Diff.: {np.mean(difference):.2f} ({low: .2f}, {upp: .2f})\n\n r: {stat:.2f} (p: {pval:.3f}),   tan(α): {reg.coef_[0]:.2f},   α: {(180 - np.abs(np.rad2deg(np.arctan(reg.coef_[0])))):.2f}\n\n'

        
    plt.annotate(annot,
                 xy=(0.5, -0.125), ha='center', va='center', xycoords='axes fraction',
                 bbox=dict(facecolor='none', edgecolor='black',boxstyle='round', alpha=0.2))
    plt.xlabel("HD1")
    plt.ylabel("FHD - HD1")
    plt.xlim(0,1)
    plt.ylim(-1.1*ax_lim, 1.1*ax_lim)
    plt.legend(title="User Group")
    plt.savefig("benefit_" + ai_group + ".png", dpi=300, bbox_inches="tight")
