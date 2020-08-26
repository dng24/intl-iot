import os
import sys
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import time
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from collections import Counter
global fowlkes_mallows_dict,min_clusters_dict,result_df,column_names
fowlkes_mallows_dict = {}
min_clusters_dict = {}


def group_eval(actual,labelled):
    grouped_actual = {}
    for index, elem in enumerate(actual):
        key = elem
        grouped_actual.setdefault(key, []).append(index)
    grouped_actual = grouped_actual.values()

    print(len(grouped_actual))

    grouped_labelled = {}
    for index, elem in enumerate(labelled):
        key = elem
        grouped_labelled.setdefault(key, []).append(index)
    grouped_labelled = grouped_labelled.values()

    print(len(grouped_labelled))
    for ind_cluster in grouped_actual:
        arr = []
        for index in ind_cluster:
            arr.append(labelled[index])
            most_common, num_most_common = Counter(arr).most_common(1)[0]

        #print(arr)

def evaluate(new_data):
    new_data['Device'] = pd.Categorical(new_data.Device)
    new_data['Actual'] = new_data.Device.cat.codes
    labels_true = new_data['Actual']
    labels_pred = new_data['clusters']
    fowlkes_mallows = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    #group_eval(np.array(labels_true),np.array(labels_pred))
    return fowlkes_mallows


def gmm_cluster(new_data):
    n_components = np.arange(2, 20)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(new_data.drop(['Device'], axis=1))
              for n in n_components]
    min_arr_bic = [m.bic(new_data.drop(['Device'], axis=1)) for m in models]
    delta = []
    for i in range(len(min_arr_bic)):
        try:
            delta.append((min_arr_bic[i + 1] - min_arr_bic[i]))
        except IndexError:
            pass
    min_index = delta.index(max(delta))
    min_cluster = n_components[min_index]

    gmm = GaussianMixture(n_components=min_cluster, covariance_type='full', random_state=0)
    gmm_clusters = gmm.fit_predict(new_data.drop(['Device'], axis=1))
    new_data['clusters'] = gmm_clusters
    return new_data,min_cluster


def feature_engineering(file):
    data_collected = pd.read_csv(file)
    num_actions = len(set(data_collected.state))
    device = list(set(data_collected.device))[0]
    hosts = data_collected['hosts']
    data = data_collected.drop(
        ['start_time', 'end_time', 'network_from', 'network_to_external', 'network_local', 'network_both',
         'anonymous_source_destination', 'device', 'state', 'network_to', 'hosts'], axis=1)

    ###############PCA GENERATION #####################
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    pca = PCA(n_components=8)
    pc = pca.fit_transform(data_scaled)

    pc_df = pd.DataFrame(data=pc,
                         columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
    pc_df['Device'] = list(data_collected['state'])
    df = pd.DataFrame({'var': pca.explained_variance_ratio_,
                       'PC': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']})
    pc_filtered = pc_df[['PC1', 'PC2', 'Device']]
    pc_filtered['hosts'] = hosts
    pc_filtered['id'] = pc_filtered.index + 1
    hosts_column = pc_filtered.set_index('id').hosts.str.split(';', expand=True).stack().reset_index(1,
                                                                                                     drop=True).reset_index(
        name='hosts_split')
    hosts_column = hosts_column.set_index('id').hosts_split.str.split(',', expand=True).stack().reset_index(1,
                                                                                                            drop=True).reset_index(
        name='hosts_split')
    hosts_column.drop(hosts_column.index[hosts_column['hosts_split'].str.contains('192.168')], inplace=True)
    merged = pd.merge(hosts_column, pc_filtered, on='id', how='left')
    pivoted = merged[['id', 'hosts_split']].pivot_table(index=['id'], columns=['hosts_split'], aggfunc=[len],
                                                        fill_value=0)

    ###############One Hot Encoding #####################
    one_hot_encoded = pivoted['len'].reset_index()
    new_data = pd.merge(one_hot_encoded, pc_filtered, on='id', how='left')
    new_data = new_data.drop(['hosts'], axis=1)
    try:
        new_data,min_clusters = gmm_cluster(new_data)
        fowlkes_mallows = evaluate(new_data)
        return device, fowlkes_mallows, num_actions, min_clusters
    except:
        return device,None,num_actions,None


def main():
    """
    Usage: python3 unsupervised_check.py /path_to_labelled_features_file
    Returns: Output file with evaluation of the unsupervised techniques on all devices in the path.
    """
    features_file = sys.argv[1]
    column_names = ['Device', 'FMI', 'Actual Num Clusters', 'Pred Num Clusters']
    results_df = pd.DataFrame(columns=column_names)
    for files in os.listdir(features_file):
        if '.csv' in files:
            print(files)
            device,fowlkes_mallows,num_actions,min_clusters= feature_engineering(f'{features_file}/{files}')
            new_row = {'Device':device,'FMI':fowlkes_mallows,'Actual Num Clusters':num_actions,'Pred Num Clusters':min_clusters}
            results_df= results_df.append(new_row,ignore_index=True)

        else:
            pass
    results_df.to_csv('results.csv')



if __name__ == "__main__":
    main()


