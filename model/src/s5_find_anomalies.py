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
style.use('ggplot')
np.random.seed(42)

def plot_confusion_matrix(cm, classes,
                          recall,precision,f2,f1,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.text(0,2.3, f" Recall:{recall},\n Precision:{precision},\n F2 Score:{f2},\n F1 Score:{f1}", fontsize=12)
    plt.show()

warnings.simplefilter("ignore", category=DeprecationWarning)

root_feature = '/Users/abhijit/Desktop/GIT_Projects/intl-iot/model/features-testing1.1/us'
root_model='/Users/abhijit/Desktop/GIT_Projects/intl-iot/models_final/features-testing1.1/us'
root_feature = sys.argv[1]
root_model = sys.argv[2]

root_output=root_model+'/anomaly_model'


lfiles = []
lparas= []
ldnames = []
device_names = []

for csv_file in os.listdir(root_feature):
    if csv_file.endswith('.csv'):
        device_name = csv_file.replace('csv','')
        device_names.append(device_name)
        train_data_file = '%s/%s' % (root_feature, csv_file)
        dname = csv_file[:-4]
        lfiles.append(train_data_file)
        ldnames.append(dname)
        lparas.append((train_data_file, dname))

recall_dict = {}
precision_dict = {}
f1_dict = {}
f2_dict = {}

for i, j in enumerate(lparas):
    # Data Loading
    print(f"Loading Normal data from --> {lparas[i][1]}")
    data = pd.read_csv(lparas[i][0])
    try:
        print(f"Loading Anomaly data from --> {lparas[i + 1][1]}")
        anomaly_data = pd.read_csv(lparas[i + 1][0])
    except IndexError:
        print(f"Loading Anomaly data from --> {lparas[0][1]}")
        anomaly_data = pd.read_csv(lparas[0][0])
    try:
        anomaly_data = anomaly_data.sample(round(data.shape[0] * 0.10))
    except ValueError:
        anomaly_data = anomaly_data
    anomaly_data['state'] = 'anomaly'
    data_features = data.drop(['device'], axis=1).fillna(-1)

    # Data Processing
    anomaly_features = anomaly_data.drop(['device'], axis=1).fillna(-1)

    train, normal_test, _, _ = train_test_split(data_features, data_features, test_size=.2, random_state=42)

    normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test, test_size=.25, random_state=42)
    anormal_valid, anormal_test, _, _ = train_test_split(anomaly_features, anomaly_features, test_size=.25,
                                                         random_state=42)

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

    print(np.median(model.logpdf(valid[valid['state'] == 0].drop('state', axis=1).values)))
    print(np.median(model.logpdf(valid[valid['state'] == 1].drop('state', axis=1).values)))

    # Validation and Testing
    tresholds = np.linspace(-100, -10, 300)
    scores = []
    try:
        for treshold in tresholds:
            y_hat = (model.logpdf(valid.drop('state', axis=1).values) < treshold).astype(int)
            scores.append([recall_score(y_pred=y_hat, y_true=valid['state'].values),
                           precision_score(y_pred=y_hat, y_true=valid['state'].values),
                           fbeta_score(y_pred=y_hat, y_true=valid['state'].values, beta=2)])

        scores = np.array(scores)
        print(scores[:, 2].max(), scores[:, 2].argmax())

        plt.plot(tresholds, scores[:, 0], label='$Recall$')
        plt.plot(tresholds, scores[:, 1], label='$Precision$')
        plt.plot(tresholds, scores[:, 2], label='$F_2$')
        plt.ylabel('Score')
        # plt.xticks(np.logspace(-10, -200, 3))
        plt.xlabel('Threshold')
        plt.legend(loc='best')
        plt.show()

        final_tresh = tresholds[scores[:, 2].argmax()]
        d = dict({'mvmodel': model, 'treshold': final_tresh})
        if not os.path.isdir("%s/model" % root_output):
            os.system("mkdir -pv %s/model" % root_output)
        f = open(f"{root_output}/multivariate_model_{lparas[i][1]}.pkl", "wb")
        pickle.dump(d, f)
        f.close()

    except:
        if i <= 45:
            print(f"Error Calculating Outputs for {lparas[i][1]} and {lparas[i + 1][1]}")
            recall_dict[f'{lparas[i][1]}-{lparas[i + 1][1]}'] = 0
            precision_dict[f'{lparas[i][1]}-{lparas[i + 1][1]}'] = 0
            f1_dict[f'{lparas[i][1]}-{lparas[i + 1][1]}'] = 0
            f2_dict[f'{lparas[i][1]}-{lparas[i + 1][1]}'] = 0
            continue
        else:
            print(f"Error Calculating Outputs for {lparas[i][1]} and {lparas[0][1]}")
            recall_dict[f'{lparas[i][1]}-{lparas[0][1]}'] = 0
            precision_dict[f'{lparas[i][1]}-{lparas[i + 1][1]}'] = 0
            f1_dict[f'{lparas[i][1]}-{lparas[0][1]}'] = 0
            f2_dict[f'{lparas[i][1]}-{lparas[0][1]}'] = 0
            continue
