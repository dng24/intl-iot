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
from sklearn.mixture import GaussianMixture
import time
from sklearn import metrics

def unsupervised_classification(data,device,hosts,dev_result_dir):
    if not os.path.exists(f'{dev_result_dir}untagged_results'):
        os.makedirs(f'{dev_result_dir}untagged_results')
    data_collected = data
    num_actions = len(set(data_collected.state))
    #hosts = data_collected['hosts']
    data = data_collected.drop(
        ['start_time', 'end_time', 'network_from', 'network_to_external', 'network_local', 'network_both',
         'anonymous_source_destination', 'predictions', 'state', 'network_to'], axis=1)
    ###############PCA GENERATION #####################
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    pca = PCA(n_components=8)
    pc = pca.fit_transform(data_scaled)

    pc_df = pd.DataFrame(data=pc,
                         columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
    df = pd.DataFrame({'var': pca.explained_variance_ratio_,
                       'PC': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']})
    pc_filtered = pc_df[['PC1', 'PC2']]
    pc_filtered['hosts'] = hosts
    pc_filtered['id'] = pc_filtered.index + 1
    hosts_column = pc_filtered.set_index('id').hosts.str.split(';', expand=True).stack().reset_index(1,
                                                                                                     drop=True).reset_index(
        name='hosts_split')
    merged = pd.merge(hosts_column, pc_filtered, on='id', how='left')
    pivoted = merged[['id', 'hosts_split']].pivot_table(index=['id'], columns=['hosts_split'], aggfunc=[len],
                                                        fill_value=0)
    ############### One Hot Encoding #####################
    one_hot_encoded = pivoted['len'].reset_index()
    new_data = pd.merge(one_hot_encoded, pc_filtered, on='id', how='left')
    new_data = new_data.drop(['hosts'], axis=1)
    ############### GMM Clustering ########################
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(new_data)
              for n in n_components]
    min_arr_bic = [m.bic(new_data) for m in models]
    min_arr_aic = [m.aic(new_data) for m in models]
    min_index = min_arr_bic.index(min(min_arr_bic))
    min_cluster = n_components[min_index]
    plt.clf()
    plt.plot(n_components, min_arr_bic, label='BIC')
    plt.plot(n_components, min_arr_aic, label='AIC')
    plt.legend(loc='best')
    plt.xlabel(f'n_components_{device}_{num_actions}_min_{min_cluster}')
    plt.savefig(f'{dev_result_dir}untagged_results/aic_bic.png')
    plt.clf()
    ############### KMeans Clustering #####################
    gmm = GaussianMixture(n_components=min_cluster)
    gmm_clusters = gmm.fit_predict(new_data)
    new_data['clusters'] = gmm_clusters
    sns_plot = sns.lmplot(x="PC1", y="PC2",
                          data=new_data,
                          fit_reg=False,
                          hue='clusters',  # color by cluster
                          legend=True,
                          scatter_kws={"s": 80},
                          height=20)
    ax = plt.gca()
    ax.set_title("GMM")
    sns_plot.savefig(f"{dev_result_dir}untagged_results/GMM_clusters.png")
    # new_data['Device'] = pd.Categorical(pc_filtered.Device)
    # new_data['Actual'] = new_data.Device.cat.codes
    ############### Evaluation #####################
    # labels_true = new_data['Actual']
    # labels_pred = new_data['kmean_clusters']
    # fowlkes_mallows = metrics.fowlkes_mallows_score(labels_pred, labels_true)
    predictions = ['cluster' + str(i) for i in new_data['clusters']]
    return predictions


