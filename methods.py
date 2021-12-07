import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
import seaborn as sns
import glob
from random import *
from sys import *
import random
import re
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import time
from sklearn.manifold import TSNE
from pyclustertend import vat
from pyclustertend import hopkins
from pyclustertend import ivat
from sklearn.preprocessing import scale
from sklearn import tree
import graphviz
from mpl_toolkits.mplot3d import Axes3D
from hmmlearn import hmm
import networkx as nx
import pygraphviz as pgv
import pomegranate as pg
from subprocess import call
import graphviz
import os.path
import warnings
warnings.filterwarnings('ignore')


def create_vector(data, min_round = 1, max_round = 100):
    data_prev = pd.pivot_table(data.loc[(data.prev.isin(["CC","CD", "DC","DD"])) & (data["round"] >= min_round)& (data["round"] <= max_round)], values='round', index=['player'],
                    columns=['prev'], aggfunc=np.count_nonzero)
    data_prev.reset_index(inplace=True)
    data_prev = data_prev.fillna(0)

    data_context = pd.pivot_table(data.loc[(data["round"] >= min_round)& (data["round"] <= max_round)], values='round', index=['player'], columns=['context'], aggfunc=np.count_nonzero).fillna(0)
    data_context.reset_index(inplace = True)

    data_vector = pd.concat([data_prev, data_context], axis = 1)
    data_vector = data_vector[["player","CC","CD","DC","DD", "CCC","CDC","DCC", "DDC"]]
    data_vector = data_vector.loc[:,~data_vector.columns.duplicated()]
    data_vector["p_CCC"] = data_vector["CCC"]/data_vector["CC"]
    data_vector["p_CDC"] = data_vector["CDC"]/data_vector["CD"]
    data_vector["p_DCC"] = data_vector["DCC"]/data_vector["DC"]
    data_vector["p_DDC"] = data_vector["DDC"]/data_vector["DD"]
    data_vector["se_CCC"] = 1.96 * np.sqrt((data_vector["p_CCC"] * (1 - data_vector["p_CCC"])) / data_vector["CC"])
    data_vector["se_CDC"] = 1.96 * np.sqrt((data_vector["p_CDC"] * (1 - data_vector["p_CDC"])) / data_vector["CD"])
    data_vector["se_DCC"] = 1.96 * np.sqrt((data_vector["p_DCC"] * (1 - data_vector["p_DCC"])) / data_vector["DC"])
    data_vector["se_DDC"] = 1.96 * np.sqrt((data_vector["p_DDC"] * (1 - data_vector["p_DDC"])) / data_vector["DD"])
    return data_vector

def plot_kmeans_curve(vector, variables, name = "k_means_"):
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(vector[variables])
        distortions.append(km.inertia_)

    # plot
    kneedle = KneeLocator(range(1, 11), distortions, S=1.0, curve="convex", direction="decreasing")
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.vlines(kneedle.knee, 0, 1000)
    plt.savefig('./images/clustering/%s.png' % name, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_dendogram(vector, variables, name = "dend_"):
    X = vector.loc[:, variables]
    dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
    plt.xlabel('Players')
    plt.xticks([])
    plt.ylabel('Euclidean distances')
    plt.savefig('./images/clustering/%s.png' % name, dpi=300, bbox_inches='tight')
    plt.show()

def perform_kmeans(vector, n_clusters, variables):
    km = KMeans(
        n_clusters=n_clusters, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    X = vector[variables]
    y_km = km.fit_predict(X)
    return y_km

def perform_hc(vector, n_clusters, variables):
    X = vector[variables]
    hc = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'euclidean', linkage ='ward')
    yhat=hc.fit_predict(X)
    return yhat

def get_silhouette(x, y):
    return metrics.silhouette_score(x, y)

def plot_cluster_prob(vector, cluster_var, palette = None, cluster_label = "A"):
    if palette is None:
         palette = {0:"#f94144", 1:"#f3722c", 2:"#f8961e", 3:"#f9844a", 4: "#f9c74f", 5: "#90be6d", 6: "#43aa8b", 7: "#4d908e", 8:"#577590", 9:"#277da1", 10:"#5e60ce"}
    symbols = {0:"s", 1:"d", 2:"o", 3:"D", 4: "v", 5: "^", 6: "P", 7: "X", 8:"<", 9:"H", 10:"*"}
    fix_m = vector.melt(id_vars=[cluster_var], value_vars=["p_CCC", "p_CDC", "p_DCC", "p_DDC"])
    fix_se = vector.melt(id_vars=[cluster_var], value_vars=["se_CCC", "se_CDC", "se_DCC", "se_DDC"])
    cluster_labels = sorted(list(vector[cluster_var].unique()))
    plt.figure(figsize = (6,6))
    clust_count = 0
    for c in cluster_labels:
        clu0 = fix_m.loc[fix_m[cluster_var] == c].groupby("variable").value.mean()
        se0 = fix_se.loc[fix_se[cluster_var] == c].groupby("variable").value.mean()
        se0.index = clu0.index
        plt.plot(clu0, symbols[c], label = "%s.%d, n = %d" % (cluster_label, clust_count, len(vector.loc[vector[cluster_var] == c])),
                 markersize =15, alpha = 0.8, color = palette[c])
        plt.vlines(range(0,4), clu0, se0 + clu0, alpha = 0.5, color = palette[c])
        plt.vlines(range(0,4), clu0 - se0, clu0, alpha = 0.5, color = palette[c])
        plt.hlines(se0 + clu0, [-0.1, 0.9, 1.9, 2.9], [0.1, 1.1, 2.1, 3.1], alpha = 0.5, color = palette[c])
        plt.hlines(clu0-se0, [-0.1, 0.9, 1.9, 2.9], [0.1, 1.1, 2.1, 3.1], alpha = 0.5, color = palette[c])
        plt.xticks(["p_CCC", "p_CDC", "p_DCC", "p_DDC"], ["p(C|CC)", "p(C|CD)","p(C|DC)","p(C|DD)"])
        clust_count += 1
    plt.legend() 
    plt.ylim(-0.1, 1.1) 
    plt.ylabel("Probability of cooperation")
    
def plot_cluster_context(vector, cluster_var):
    fix_m = vector.melt(id_vars=[cluster_var], value_vars=["p_CCC", "p_CDC", "p_DCC", "p_DDC"])
    g = sns.FacetGrid(fix_m, col=cluster_var, height = 4)
    g.map(sns.barplot, "variable", "value")

def get_prev_actions(data, no_actions = 2):
    data = data.sort_values(["player", "round"], ascending = (True, True))
    data["prev_p%d" % no_actions] = ""
    data["prev_o%d" % no_actions] = ""
    for p in data.player.unique():
        prev_actions = data.loc[data.player == p, "action_player"]
        prev_opp = data.loc[data.player == p, "action_opponent"]
        prev_actions = pd.concat([pd.Series([-1] * no_actions), prev_actions], ignore_index=True)
        prev_opp = pd.concat([pd.Series([-1] * no_actions), prev_opp], ignore_index=True)
        prev_actions = prev_actions.iloc[0:100]
        prev_opp = prev_opp.iloc[0:100]
        data.loc[data["player"] == p, "prev_p%d" % no_actions] = prev_actions.values
        data.loc[data["player"] == p, "prev_o%d" % no_actions] = prev_opp.values
    return data

def get_first_actions(data):
    first_act = data.loc[data["round"] == 1, ["player","action_player"]]
    first_act.columns = ['player', 'first_round']
    return first_act

def add_clustering_vector(vector, name, clusters):
    vector[name] = clusters
    return vector

def add_clustering_data(data, vector, name):
    if name in data.columns:
        del data[name]
    data_m = data.merge(vector[["player", name]], on="player")
    return data_m

def draw_graph(hidden_states, start_probs, trans, emission, min_r, max_r, color_cluster):
    g = graphviz.Digraph(format='png')
    g.attr('node', fontsize="25", shape = "rectangle", color = color_cluster)
    context = {"(CC)C":0,"(CC)D":1,"(CD)C":2,"(CD)D":3, "(DC)C":4, "(DC)D":5, "(DD)C":6, "(DD)D":7}
    states = []
    #for e in emission:
    #    states.append("D %.3f, C %.3f" % tuple(e))
    for h in range(0,hidden_states):
        hidden_lab = ""
        for e in range(0, len(emission[h])):
            if np.round(emission[h][e], 2) >= 0.05: 
                hidden_lab = hidden_lab + "%s = %.3f \n" % (list(context.keys())[e], emission[h][e])
        states.append(hidden_lab)
    for s in states:
        g.node(s)
    for i in range(0, len(trans)):
            for j in range(i, len(trans)):
                if trans[i][j] > 0:
                    g.edge(states[i], states[j], str(trans[i][j]))
    g.node(states[0], style = "setlinewidth(5)")
    g.render('./images/graphs/%d-%d/sc_%d_hs_%d_%d-%d' % (min_r, max_r, subcluster, hidden_states, min_r, max_r))
