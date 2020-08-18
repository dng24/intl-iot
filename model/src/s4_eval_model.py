import warnings
import matplotlib

matplotlib.use("Agg")
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
import warnings
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from multiprocessing import Pool
import Constants as c

"""
Output: root_model/{alg}/*.models
root_model/output/result_{alg}.txt
"""
warnings.simplefilter("ignore", category=DeprecationWarning)

root_feature = '/Volumes/Abhijit-Seagate/Data_iot/Features/required-features-split'
root_model = '/Volumes/Abhijit-Seagate/Data_iot/models/new-tagged-models'

root_output = root_model + '/output'
dir_tsne_plots = root_model + '/tsne-plots'

num_pools = 12

# default_models = ['rf']
# default_models = ['rf', 'knn']
#default_models = ['dbscan', 'kmeans', 'knn', 'rf', 'spectral']
default_models = ['knn']
model_list = []

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.EVAL_MOD_USAGE, file=sys.stderr) if is_error else print(c.EVAL_MOD_USAGE)
    exit(is_error)


def main():
    # test()
    global root_feature, root_model, root_output, dir_tsne_plots, model_list

    # Parse Arguments
    parser = argparse.ArgumentParser(usage=c.EVAL_MOD_USAGE, add_help=False)
    parser.add_argument("-i", dest="root_feature", default="")
    parser.add_argument("-o", dest="root_model", default="")
    parser.add_argument("-d", dest="dbscan", action="store_true", default=False)
    parser.add_argument("-k", dest="kmeans", action="store_true", default=False)
    parser.add_argument("-n", dest="knn", action="store_true", default=False)
    parser.add_argument("-r", dest="rf", action="store_true", default=False)
    parser.add_argument("-s", dest="spectral", action="store_true", default=False)
    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    if args.help:
        print_usage(0)

    print("Running %s..." % c.PATH)

    # Error checking command line args
    root_feature = args.root_feature
    root_model = args.root_model
    errors = False
    #check -i in features
    if root_feature == "":
        errors = True
        print(c.NO_FEAT_DIR, file=sys.stderr)
    elif not os.path.isdir(root_feature):
        errors = True
        print(c.INVAL % ("Features directory", root_feature, "directory"), file=sys.stderr)
    else:
        if not os.access(root_feature, os.R_OK):
            errors = True
            print(c.NO_PERM % ("features directory", root_feature, "read"), file=sys.stderr)
        if not os.access(root_feature, os.X_OK):
            errors = True
            print(c.NO_PERM % ("features directory", root_feature, "execute"), file=sys.stderr)

    #check -o out models
    if root_model == "":
        errors = True
        print(c.NO_MOD_DIR, file=sys.stderr)
    elif os.path.isdir(root_model):
        if not os.access(root_model, os.W_OK):
            errors = True
            print(c.NO_PERM % ("model directory", root_model, "write"), file=sys.stderr)
        if not os.access(root_model, os.X_OK):
            errors = True
            print(c.NO_PERM % ("model directory", root_model, "execute"), file=sys.stderr)

    #add requested models for generation
    if args.dbscan:
        model_list.append("dbscan")

    if args.kmeans:
        model_list.append("kmeans")

    if args.knn:
        model_list.append("knn")

    if args.rf:
        model_list.append("rf")

    if args.spectral:
        model_list.append("spectral")

    if not model_list:
        model_list = default_models.copy()

    if errors:
        print_usage(1)
    #end error checking

    print("Input files located in: %s\nOutput files placed in: %s" % (root_feature, root_model))
    root_output = os.path.join(root_model, 'output')
    if not os.path.exists(root_output):
        os.system('mkdir -pv %s' % root_output)
        for model_alg in model_list:
            model_dir = '%s/%s' % (root_model, model_alg)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

    train_models()


def train_models():
    global root_feature, root_model, root_output, dir_tsne_plots
    """
    Scan feature folder for each device
    """
    print('root_feature: %s' % root_feature)
    print('root_model: %s' % root_model)
    print('root_output: %s' % root_output)
    lfiles = []
    lparas = []
    ldnames = []
    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            print(csv_file)
            train_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]
            lfiles.append(train_data_file)
            ldnames.append(dname)
            lparas.append((train_data_file, dname))
    p = Pool(num_pools)
    t0 = time.time()
    print(dname)
    list_results = p.map(eid_wrapper, lparas)
    for ret in list_results:
        if ret is None or len(ret) == 0: continue
        for res in ret:
            tmp_outfile = res[0]
            tmp_res = res[1:]
            with open(tmp_outfile, 'a+') as off:
                off.write('%s\n' % '\t'.join(map(str, tmp_res)))
                print('Agg saved to %s' % tmp_outfile)
    t1 = time.time()
    print('Time to train all models for %s devices using %s threads: %.2f' % (len(lparas),num_pools, (t1 - t0)))
    # p.map(target=eval_individual_device, args=(lfiles, ldnames))


def eid_wrapper(a):
    return eval_individual_device(a[0], a[1])


def eval_individual_device(train_data_file, dname, specified_models=None):
    global root_feature, root_model, root_output, dir_tsne_plots
    """
    Assumptions: the train_data_file contains only 1 device, all possible states(tags); the models can only be 
    one of the implementated: knn, kmeans, dbscan, random forest classifier  
    """
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    """
    Skip trained models, return if there is no model to train. 
    """
    list_all_models = model_list
    if specified_models is not None:
        list_all_models = specified_models

    list_models_todo = []
    for model_alg in list_all_models:
        """
        Prepare the directories and add only models that have not been trained yet 
        """
        model_dir = '%s/%s' % (root_model, model_alg)
        model_file = '%s/%s%s.model' % (model_dir, dname, model_alg)
        label_file = '%s/%s.label.txt' % (model_dir, dname)
        if not os.path.exists(model_file) and os.path.exists(train_data_file):
            # check .model
            # check if training data set is available
            list_models_todo.append(model_alg)
    
    if len(list_models_todo) < 1:
        print('skip %s, all models trained for alg: %s' % (dname, str(list_all_models)))
        return
    print('Training %s using algorithm(s): %s' % (dname, str(list_models_todo)))
    train_data = pd.read_csv(train_data_file)

    num_data_points = len(train_data)
    if num_data_points < 1:
        print('  Not enough data points for %s' % dname)
        return
    print('\t#Total data points: %d ' % num_data_points)
    X_feature = train_data.drop(['device', 'state','hosts'], axis=1).fillna(-1)
    ss= StandardScaler()
    pca = PCA(n_components=5)
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
        print('\tNo enough labels for %s' % dname)
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
    for model_alg in list_models_todo:
        model_dir = os.path.join(root_model, model_alg)
        if not os.path.exists(model_dir):
            os.system('mkdir -pv %s' % model_dir)
        model_file = os.path.join(model_dir, dname + model_alg + ".model")
        label_file = os.path.join(model_dir, dname + ".label.txt")
        single_outfile = os.path.join(model_dir, dname + ".result.csv")
        output_file = os.path.join(root_output, "result_" + model_alg + ".txt")
        _acc_score = -1
        _noise = -1
        _silhouette = -1
        """
        Two steps
            1. Train (70%)
            2. Test 
            3. Evaluate 
        """
        if model_alg == 'knn':
            print('  knn: n_neighbors=%s' % num_lables)
            trained_model = KNeighborsClassifier(n_neighbors=num_lables)
            trained_model.fit(X_train, y_train_bin)

            y_predicted = trained_model.predict(X_test)
            try:
                y_predicted_1d = np.argmax(y_predicted, axis=1)
            except:
                y_predicted_1d = y_predicted


            if len(set(y_predicted_1d)) > 1: _silhouette = silhouette_score(X_test, y_predicted_1d)


        elif model_alg == 'kmeans':
            print('  kmeans: n_clusters=%s' % num_lables)
            trained_model = MiniBatchKMeans(n_clusters=num_lables, random_state=0, batch_size=6)
            trained_model.fit(X_train)

            y_predicted_1d = trained_model.predict(X_test).round()
            if len(set(y_predicted_1d)) > 1: _silhouette = silhouette_score(X_test, y_predicted_1d)

        elif model_alg == 'spectral':
            print('  Spectral Clustering: n_clusters=%s' % num_lables)
            trained_model = SpectralClustering(n_clusters=num_lables, affinity='nearest_neighbors', random_state=0)
            trained_model.fit(X_train)

            y_predicted_1d = trained_model.fit_predict(X_test).round()
            if len(set(y_predicted_1d)) > 1: _silhouette = silhouette_score(X_test, y_predicted_1d)

        elif model_alg == 'dbscan':
            print('  eps=%s' % 300)
            trained_model = DBSCAN(eps=200, min_samples=5)
            trained_model.fit(X_train)
            y_predicted_1d = trained_model.fit_predict(X_test).round()
            if len(set(y_predicted_1d)) > 1: _silhouette = silhouette_score(X_test, y_predicted_1d)
            _noise = list(y_predicted_1d).count(-1) * 1. / num_data_points

        elif model_alg == 'rf':
            trained_model = RandomForestClassifier(n_estimators=1000, random_state=42)
            trained_model.fit(X_train, y_train_bin)
            y_predicted = trained_model.predict(X_test).round()
            # print(y_predicted)
            if y_predicted.ndim == 1:
                y_predicted_1d = y_predicted
            else:
                y_predicted_1d = np.argmax(y_predicted, axis=1)

        try:
            _acc_score = accuracy_score(y_test_bin_1d, y_predicted_1d)
        except:
            print(y_predicted_1d[0])
            _acc_score = accuracy_score(y_test_bin_1d, y_predicted_1d[0])
        """
        Eval clustering based metrics
        """

        _homogeneity = -1
        _complete = -1
        _vmeasure = -1
        _ari = -1
        _f1_score = -1
        if model_alg not in ['rf']:
            """
            Metrics for clustering algorithms 
            """
            # print('y_test_bin: %s' % y_test_bin_1d)
            # print('y_predicted_1d: %s' % y_predicted_1d)
            _homogeneity = homogeneity_score(y_test_bin_1d, y_predicted_1d)
            _complete = completeness_score(y_test_bin_1d, y_predicted_1d)
            _vmeasure = v_measure_score(y_test_bin_1d, y_predicted_1d)
            _ari = adjusted_rand_score(y_test_bin_1d, y_predicted_1d)
        """
        Plot tSNE graph
        """
        figfile = '%s/%s/%s-%s.png' % (root_model, model_alg, model_alg, dname)
        pp = 30  # perplexity
        if num_data_points > 200:
            pp = 50
        tsne_plot(X_feature, y_labels, figfile, pp)

        """
        Save the model 
        """
        model_dictionary = dict({'standard_scaler':ss, 'pca':pca, 'trained_model':trained_model})
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
            off.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (dname, _acc_score, _homogeneity, _complete,
                                                            _vmeasure, _ari, _noise, _silhouette))
            # y_test_bin_1d, y_predicted_1d
            off.write('%s\n' % ','.join(map(str, y_test_bin_1d)))
            off.write('%s\n' % ','.join(map(str, y_predicted_1d)))

        ret_results.append([output_file, dname, _acc_score, _homogeneity, _complete, _vmeasure,
                            _ari, _noise, _silhouette])
        """
        Print to Terminal 
        """
        print('    model -> %s' % (model_file))
        print('    labels -> %s' % label_file)
        print('\t' + '\n\t'.join(unique_labels) + '\n')
        if model_alg not in ['rf']:
            print('    _homogeneity: %.3f' % _homogeneity)
            print('    _completeness: %.3f' % _complete)
            print('    _vmeausre: %.3f' % _vmeasure)
            print('    _ari: %.3f' % _ari)
            print('    _silhouette: %.3f' % _silhouette)
        print('    _acc_score: %.3f' % _acc_score)
        print('    measures saved to: %s' % single_outfile)
    return ret_results
    # return score, _homogeneity, _complete, _vmeausre, _ari


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



if __name__ == '__main__':
    main()
    num_pools = 12

