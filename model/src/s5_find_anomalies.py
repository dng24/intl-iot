import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix,f1_score
import itertools
import pickle
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt,style
from multiprocessing import Pool
import sys

import Constants as c

style.use('ggplot')
np.random.seed(42)

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.FIND_ANOM_USAGE, file=sys.stderr) if is_error else print(c.FIND_ANOM_USAGE)
    exit(is_error)


def plot_confusion_matrix(cm, classes, recall, precision, f2, f1, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.text(0,2.3, f" Recall:{recall},\n Precision:{precision},\n F2 Score:{f2},\n F1 Score:{f1}", fontsize=12)
    plt.show()


def main():
    warnings.simplefilter("ignore", category=DeprecationWarning)

    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % c.PATH)

    root_feature = '/Users/abhijit/Desktop/GIT_Projects/intl-iot/model/features-testing1.1/us'
    root_model='/Users/abhijit/Desktop/GIT_Projects/intl-iot/models_final/features-testing1.1/us'

    #error checking
    #check that there are 2 args
    if len(sys.argv) != 3:
        print(c.WRONG_NUM_ARGS % (2, (len(sys.argv) - 1)), file=sys.stderr)
        print_usage(1)

    root_feature = sys.argv[1]
    root_model = sys.argv[2]
    root_output = os.path.join(root_model, 'anomaly_model')

    #check root_feature
    errors = False
    if not os.path.isdir(root_feature):
        errors = True
        print(c.INVAL % ("Features directory", root_feature, "directory"), file=sys.stderr)
    else:
        if not os.access(root_feature, os.R_OK):
            errors = True
            print(c.NO_PERM % ("features directory", root_feature, "read"), file=sys.stderr)
        if not os.access(root_feature, os.X_OK):
            errors = True
            print(c.NO_PERM % ("features directory", root_feature, "execute"), file=sys.stderr)

    #check root_model
    errors = False
    if not os.path.isdir(root_model):
        errors = True
        print(c.INVAL % ("Model directory", root_model, "directory"), file=sys.stderr)
    else:
        if not os.access(root_model, os.R_OK):
            errors = True
            print(c.NO_PERM % ("model directory", root_model, "read"), file=sys.stderr)
        if not os.access(root_feature, os.X_OK):
            errors = True
            print(c.NO_PERM % ("model directory", root_model, "execute"), file=sys.stderr)

    #check root_output, if exists
    if os.path.isdir(root_output):
        if not os.access(root_output, os.W_OK):
            errors = True
            print(c.NO_PERM % ("output directory", root_output, "write"), file=sys.stderr)
        if not os.access(root_output, os.X_OK):
            errors = True
            print(c.NO_PERM % ("output directory", root_output, "execute"), file=sys.stderr)
    
    if errors:
        print_usage(1)

    print("Input training features: %s\nInput models: %s\nOutput anomaly model: %s\n"
          % (root_feature, root_model, root_output))

    lfiles = []
    lparas= []
    ldnames = []
    device_names = []

    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            device_name = csv_file.replace('csv', '')
            device_names.append(device_name)
            train_data_file = os.path.join(root_feature, csv_file)
            dname = csv_file[:-4]
            lfiles.append(train_data_file)
            ldnames.append(dname)
            lparas.append((train_data_file, dname))


    for i, j in enumerate(lparas):
        # Data Loading
        print(f"Loading Normal data from --> {lparas[i][1]}")
        data = pd.read_csv(lparas[i][0])
        anomaly_data = pd.read_csv('./sample-anomaly/cloudcam.csv')
        anomaly_data = anomaly_data.sample(round(data.shape[0] * 0.10))
        anomaly_data['state'] = 'anomaly'
        data_features = data.drop(['device','hosts'], axis=1).fillna(-1)

        anomaly_features = anomaly_data.drop(['device'], axis=1).fillna(-1)

        train, normal_test, _, _ = train_test_split(data_features, data_features,
                                                    test_size=.2, random_state=42)

        normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test,
                                                           test_size=.25, random_state=42)
        anormal_valid, anormal_test, _, _ = train_test_split(anomaly_features, anomaly_features,
                                                             test_size=.25, random_state=42)

        train = train.reset_index(drop=True)
        valid = normal_valid.append(anormal_valid).sample(frac=1).reset_index(drop=True)
        test = normal_test.append(anormal_test).sample(frac=1).reset_index(drop=True)

        train['state'] = train['state'].apply(lambda x: 1 if x == 'anomaly' else 0)
        valid['state'] = valid['state'].apply(lambda x: 1 if x == 'anomaly' else 0)
        test['state'] = test['state'].apply(lambda x: 1 if x == 'anomaly' else 0)

        # Training the model
        mu = train.drop('state', axis=1).mean(axis=0).values
        sigma = train.drop('state', axis=1).cov().values
        model = multivariate_normal(cov=sigma, mean=mu, allow_singular=True)


        # Validation and Testing
        tresholds = np.linspace(-100, -10, 300)
        scores = []
        for treshold in tresholds:
            y_hat = (model.logpdf(valid.drop('state', axis=1).values) < treshold).astype(int)
            scores.append([recall_score(y_pred=y_hat, y_true=valid['state'].values),
                           precision_score(y_pred=y_hat, y_true=valid['state'].values),
                           fbeta_score(y_pred=y_hat, y_true=valid['state'].values, beta=2)])

        scores = np.array(scores)

        final_tresh = tresholds[scores[:, 2].argmax()]
        y_hat = (model.logpdf(valid.drop('state', axis=1).values) < final_tresh).astype(int)

        cm = confusion_matrix(valid['state'].values, y_hat)
        print(cm)
        print(f"Final threshold --> {final_tresh}")
        d = dict({'mvmodel': model, 'treshold': final_tresh})
        if not os.path.isdir("%s/model" % root_output):
            os.system("mkdir -pv %s" % root_output)

        model_filepath = f"{root_output}/multivariate_model_{lparas[i][1]}.pkl"
        with open(model_filepath, "wb") as f:
            pickle.dump(d, f)
        print("Model written to \"%s\"" % model_filepath)

if __name__ == "__main__":
    main()
