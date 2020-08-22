#TODO: Replace retrained model with the orignal model instead of creating a new folder.
import warnings
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from multiprocessing import Pool
import Constants as c


def tsne_plot(X, y, figfile, pp=30):
    tsne = TSNE(n_components=2, perplexity=pp, n_iter=5000)
    """
    This is independent from any clustering/classification algorithm
    Just to visualize the data
    """
    t1 = time.time()
    X_2d = tsne.fit_transform(X)
    # list_clusters = set(y_predicted)
    t2 = time.time()
    print('\tTime to perform tSNE: %.2fs' % (t2 - t1))
    plot_data = pd.DataFrame(X_2d, columns=['x', 'y'])
    plot_data['cluster_label'] = y
    # print(plot_data.head())
    fig = plt.figure()
    ax = plt.subplot(111)
    for yi, g in plot_data.groupby('cluster_label'):
        g2 = g.drop('cluster_label', axis=1)
        if yi == -1:
            plt.scatter(g.x, g.y, label='cluster_%s' % yi, marker='*')
        else:
            plt.scatter(g.x, g.y, label='cluster_%s' % yi)
    ax.legend(bbox_to_anchor=(1.1, 1.1))

    print('\tSaved the tSNE plot to %s' % figfile)
    plt.savefig(figfile, bbox_inches="tight")


def retrain_model(data, model_dir,trained_features_file):
    print('Retraining Models!')
    print(model_dir)
    global root_feature, root_model, root_output, dir_tsne_plots
    device = data['device'][0]
    new_data = data[data['state'].str.contains("cluster")]
    old_labelled_data = pd.read_csv(trained_features_file)
    train_data = pd.concat([old_labelled_data,new_data])

    num_data_points = len(train_data)
    if num_data_points < 1:
        print('  Not enough data points for %s' % device)
        return
    print('\t#Total data points: %d ' % num_data_points)
    X_feature = train_data.drop(['device', 'state', 'hosts'], axis=1).fillna(-1)
    ss = StandardScaler()
    pca = PCA(n_components=20)
    X_std = ss.fit_transform(X_feature)
    # Create a PCA instance: pca
    X_std = pca.fit_transform(X_std)
    # Save components to a DataFrame
    X_std = pd.DataFrame(X_std)
    X_feature = X_std.iloc[:, :4]
    y_labels = np.array(train_data.state)
    # y_labels, example: on, off, change_color
    """
    Split data set into train & test, default fraction is 30% test
    """
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y_labels, test_size=.3, random_state=42)
    print('Train: %s' % len(X_train))
    print('Test: %s' % len(X_test))

    num_lables = len(set(y_labels))
    if num_lables < 2:
        print('\tNo enough labels for %s' % device)
        return

    """
    One hot encoding y labels
    """
    lb = LabelBinarizer()
    lb.fit(y_labels)  # collect all possible labels
    y_train_bin = lb.transform(y_train)
    y_test_bin = lb.transform(y_test)
    y_test_bin_1d = np.argmax(y_test_bin, axis=1)

    """
    Train through the list of interested ML algorithms
    """
    ret_results = []
    root_output = os.path.join(model_dir, 'output')
    print(f'{model_dir}/knn')
    if not os.path.exists(f'{model_dir}/knn'):
        os.system('mkdir -pv %s' % f'{model_dir}/knn')

    model_file = os.path.join(model_dir, "knn", device + "knn.model")
    label_file = os.path.join(model_dir, "knn", device + ".label.txt")
    single_outfile = os.path.join(model_dir, "knn", device + "result.csv")
    output_file = os.path.join(root_output, "result_" + 'knn' + ".txt")

    _acc_score = -1
    _noise = -1
    _silhouette = -1

    """
    Two steps
        1. Train (70%)
        2. Test 
        3. Evaluate 
    """

    print('  knn: n_neighbors=%s' % num_lables)
    trained_model = KNeighborsClassifier(n_neighbors=num_lables)
    trained_model.fit(X_train, y_train_bin)

    y_predicted = trained_model.predict(X_test)
    y_predicted_1d = np.argmax(y_predicted, axis=1)
    if len(set(y_predicted_1d)) > 1: _silhouette = silhouette_score(X_test, y_predicted_1d)

    """
        Eval clustering based metrics
    """
    _acc_score = accuracy_score(y_test_bin_1d, y_predicted_1d)
    _homogeneity = homogeneity_score(y_test_bin_1d, y_predicted_1d)
    _complete = completeness_score(y_test_bin_1d, y_predicted_1d)
    _vmeasure = v_measure_score(y_test_bin_1d, y_predicted_1d)
    _ari = adjusted_rand_score(y_test_bin_1d, y_predicted_1d)
    """
    Plot tSNE graph
    """
    root_model = os.path.join(model_dir, "knn")
    figfile = '%s/%s-%s.png' % (root_model,'knn', device)
    pp = 30  # perplexity
    if num_data_points > 200:
        pp = 50
    tsne_plot(X_feature, y_labels, figfile, pp)

    """
    Save the model 
    """
    model_dictionary = dict({'standard_scaler': ss, 'pca': pca, 'trained_model': trained_model})
    pickle.dump(model_dictionary, open(model_file, 'wb'))
    """
    Save the label for onehot encoding 
    """
    # unique_labels = label_encoder.classes_.tolist()
    unique_labels = lb.classes_.tolist()
    open(label_file, 'w').write('%s\n' % '\n'.join(unique_labels))

    """
    Save eval results
    """
    # TODO: due to the multi-thread, needs to change the settings
    with open(single_outfile, 'a+') as off:
        off.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (device, _acc_score, _homogeneity, _complete,
                                                        _vmeasure, _ari, _noise, _silhouette))
        # y_test_bin_1d, y_predicted_1d
        off.write('%s\n' % ','.join(map(str, y_test_bin_1d)))
        off.write('%s\n' % ','.join(map(str, y_predicted_1d)))

    ret_results.append([output_file, device, _acc_score, _homogeneity, _complete, _vmeasure,
                        _ari, _noise, _silhouette])
    """
    Print to Terminal 
    """
    print('    model -> %s' % (model_file))
    print('    labels -> %s' % label_file)
    print('\t' + '\n\t'.join(unique_labels) + '\n')
    print('    _homogeneity: %.3f' % _homogeneity)
    print('    _completeness: %.3f' % _complete)
    print('    _vmeausre: %.3f' % _vmeasure)
    print('    _ari: %.3f' % _ari)
    print('    _silhouette: %.3f' % _silhouette)
    print('    _acc_score: %.3f' % _acc_score)
    print('    measures saved to: %s' % single_outfile)

    list_results = ret_results
    for ret in list_results:
        if ret is None or len(ret) == 0:
            continue

        tmp_outfile = ret[0]
        tmp_res = ret[1:]
        with open(tmp_outfile, 'a+') as off:
            off.write('%s\n' % '\t'.join(map(str, tmp_res)))
            print('Agg saved to %s' % tmp_outfile)

    return ret_results