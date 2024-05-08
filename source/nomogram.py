import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def protocols_real(filename):
    data = pd.read_csv(filename)
    displaced, displaced_std = displacement(data)
    inhibited = data.groupby("id")["FHD"].mean().mean()
    inhibited_std = data.groupby("id")["FHD"].mean().std()/len(data["id"].unique())
    foreclosed = data.groupby("id")["HD1"].mean().mean()
    foreclosed_std = data.groupby("id")["HD1"].mean().std()/len(data["id"].unique())
    replaced = data.groupby("id")["AI"].mean().mean()
    replaced_std = data.groupby("AI")["HD1"].mean().std()/len(data["id"].unique())

    if ("Type_AI" in data.columns) and data[data["Type_AI"] == "AI-first"].shape[0] != 0:
        traditional = data[data["Type_AI"] == "AI-first"].groupby("id")["FHD"].mean().mean()
        traditional_std = data[data["Type_AI"] == "AI-first"].groupby("id")["FHD"].mean().std()/len(data[data["Type_AI"] == "AI-first"]["id"].unique())
        return displaced, displaced_std, inhibited, inhibited_std, foreclosed, foreclosed_std, replaced, replaced_std, traditional, traditional_std

    return displaced, displaced_std, inhibited, inhibited_std, foreclosed, foreclosed_std, replaced, replaced_std, "NA", "NA"

def displacement(data_in, n_boots=100):
    data = data_in.copy()
    accs = []
    for _ in range(n_boots):
        iters = data.shape[0]
        acc = 0
        for i in range(iters):
            id = np.random.choice(data["id"].unique())
            data_tmp = data[data["id"] == id]
            
            r = np.random.randint(low=0, high=data_tmp.shape[0])
            if data_tmp.iloc[r]["HD1"] != data_tmp.iloc[r]["AI"]:
                ids = np.random.choice(data["id"].unique())
                if ids == id:
                    acc +=  data_tmp.iloc[r]["FHD"]
                else:
                    data_tmps = data[data["id"] == ids]
                    acc +=  data_tmps.iloc[r]["HD1"]
            else:
                acc += data_tmp.iloc[r]["HD1"]
        
        accs.append(acc/iters)
    return np.mean(accs), np.std(accs)/np.sqrt(len(accs))

def protocols(x,y,z):
    foreclosure = x
    replacement = y
    traditional = np.max([y,z])
    displacement = x*(2*y + x - 2*x*y)
    inhibition = np.min([1,x*y + x*(1-y)*z + (1-x)*y*z])
    l = zip(["Replacement", "Foreclosure", "Traditional", "Displacement", "Inhibition"],
            [replacement, foreclosure, traditional, displacement, inhibition])
    return l

def best_protocol(X,Y,z=0.5, palette= sns.color_palette("colorblind")):
    vals = np.zeros((len(X), len(Y),3))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            foreclosure = x
            replacement = y
            traditional = np.max([y,z])
            displacement = x*(2*y + x - 2*x*y)
            inhibition = np.min([1,x*y + x*(1-y)*z + (1-x)*y*z])
            idx = np.argmax([replacement, foreclosure, traditional, displacement, inhibition])
            vals[j, i] = palette[int(idx)]
    return vals



def plot_nomogram(filename, ar=None):
    _, _, _, _, foreclosed, _, replaced, _, _, _ = protocols_real(filename)
    data = pd.read_csv(filename)
    
    X = Y = np.linspace(0,1,1000)

    if ar is None:
        num = data[(data["HD1"] == 0) & (data["AI"] == 1) & (data["FHD"] == 1)].shape[0]
        num += data[(data["HD1"] == 1) & (data["AI"] == 0) & (data["FHD"] == 1)].shape[0]
        num += data[(data["HD1"] == 0) & (data["AI"] == 0) & (data["FHD"] == 1)].shape[0]
    
        if "C1" in data.columns and "FC" in data.columns:
            num += data[(data["HD1"] == 0) & (data["AI"] == 0) & (data["FHD"] == 0) & (data["C1"] > data["FC"])].shape[0]
            num += data[(data["HD1"] == 1) & (data["AI"] == 1) & (data["FHD"] == 1) & (data["C1"] < data["FC"])].shape[0]
        else:
            num += data[(data["HD1"] == 0) & (data["AI"] == 0) & (data["FHD"] == 0)].shape[0]*0.5
            num += data[(data["HD1"] == 1) & (data["AI"] == 1) & (data["FHD"] == 1)].shape[0]*0.5
        
        ar = num/data.shape[0]

    plt.figure(figsize=(5,5))
    plt.imshow(best_protocol(X, Y, z=ar), origin="lower", extent=(0,1,0,1))
    plt.scatter([foreclosed], [replaced])
    plt.xlabel("(Baseline) Average Human Accuracy")
    plt.ylabel("AI Accuracy")
    plt.title("Appropriate Reliance = %.2f" % (ar))

    palette= sns.color_palette("colorblind")
    legend_elems = []
    for i, n in enumerate(["Replacement", "Foreclosure", "Traditional", "Displacement", "Inhibition"]):
        legend_elems.append(Patch(facecolor=palette[i], label=n))

    plt.legend(handles=legend_elems, ncols=5, loc='lower center')
    plt.savefig("nomograms.png", dpi=300, bbox_inches="tight")

