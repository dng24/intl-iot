import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix,f1_score
import itertools
import pickle
from matplotlib import pyplot as plt,style
from multiprocessing import Pool
import json
import os
import sys

import Constants as c

warnings.simplefilter("ignore", category=DeprecationWarning)
style.use('ggplot')
np.random.seed(42)

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.PREDICT_USAGE, file=sys.stderr) if is_error else print(c.PREICT_USAGE)
    exit(is_error)


def label(label_file):
    labels = []
    with open(label_file) as ff:
        for line in ff.readlines():
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            labels.append(line)
    return labels


def dictionary_create(labels):
    # TODO: Do not hardcode dictionary. Labels need to be taken from the device.
    di ={}
    reverse_di = {}
    for i in range(len(labels)):
        di.update({labels[i]:i})
        reverse_di.update({i:labels[i]})
    
    di.update({'normal':len(labels)})
    return di,reverse_di
    

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    plt.xticks(rotation=90)
    plt.text(12,0, f" Recall:{recall},\n Precision:{precision},\n F2 Score:{f2},\n F1 Score:{f1}", fontsize=12)
    return plt
    #plt.show()


def load_data(path):
    anomaly_data = pd.read_csv(path)
   # anomaly_data = anomaly_data.drop(anomaly_data.columns[0], axis=1)
    return anomaly_data


def filter_anomaly(ss,anomaly_data,multivariate_model_dict,dev_result_dir):
    mv_model = multivariate_model_dict['mvmodel']
    treshold = multivariate_model_dict['treshold']
    y_test = anomaly_data['state'].apply(lambda x: 1 if x == 'anomaly' else 0)
    y_predict = (mv_model.logpdf(anomaly_data.drop(['state'], axis=1).values) < treshold).astype(int)
    anomaly_data['anomalous'] = y_predict
    normal_data = anomaly_data[anomaly_data['anomalous'] == 0]
    anomalous_data = anomaly_data[anomaly_data['anomalous'] == 1]
    output_dict = {'predictions': y_predict}
    if not os.path.isdir(dev_result_dir):
        os.system("mkdir -pv %s" % dev_result_dir)

    with open(dev_result_dir+'/anomaly_output.txt','w+') as f:
        f.write(json.dumps(output_dict,cls=NumpyEncoder))

    return normal_data,anomalous_data


def action_classification_model(normal_data,action_class_dict):
    ss = action_class_dict['standard_scaler']
    pca = action_class_dict['pca']
    trained_model = action_class_dict['trained_model']
    transformed_data = ss.transform(normal_data.drop(['state','anomalous'], axis=1))
    transformed_data = pca.transform(transformed_data)
    transformed_data = pd.DataFrame(transformed_data)
    transformed_data = transformed_data.iloc[:, :4]
    y_predict = trained_model.predict(transformed_data)
    y_predicted_1d = np.argmax(y_predict, axis=1)
    normal_data['predictions'] = y_predicted_1d
    return normal_data


def final_accuracy(final_data,dev_result_dir):
    global di
    y_test = final_data['state'].map(di)
    y_predict = final_data['predictions']
    return y_predict


def run_process(features_file,dev_result_dir,base_model_file,anomaly_model_file):
    anomaly_data = load_data(features_file)
    start_time = anomaly_data['start_time']
    end_time = anomaly_data['end_time']
    anomaly_data = anomaly_data.drop(['device'], axis=1)
    action_classification_model_dict = pickle.load(open(base_model_file, 'rb'))
    ss = action_classification_model_dict['standard_scaler']
    anomaly_model = pickle.load(open(anomaly_model_file, 'rb'))
    normal_data, anomalous_data = filter_anomaly(ss, anomaly_data, anomaly_model, dev_result_dir)
    print("Normal")
    print(normal_data.head())
    print("Abnormal")
    print(anomalous_data.head())
    # TODO: The label for anomalous data will be according to the dictionary count for the device.
    normal_data['predictions'] = di['normal']
    anomalous_data = action_classification_model(anomalous_data, action_classification_model_dict)
    final_data = normal_data.append(anomalous_data).sort_index()
    y_predict = final_accuracy(final_data, dev_result_dir)
    arr = list(range(0, len(y_predict)))
    out_dict = {'start_time': start_time, 'end_time': end_time, 'tagged': final_data['state'], 'prediction': y_predict}
    out_df = pd.DataFrame(out_dict)
    out_df['prediction'] = out_df['prediction'].map(reverse_di).fillna("normal")
    out_df.to_csv(dev_result_dir + '/model_results.csv', index=False)


def main():
    global di, reverse_di, labels

    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % c.PATH)

    #error checking
    #check for 3 args
    if len(sys.argv) != 4:
        print(c.WRONG_NUM_ARGS % (3, (len(sys.argv) - 1)), file=sys.stderr)
        print_usage(1)

    features_dir = sys.argv[1]
    model_dir = sys.argv[2]
    results_dir = sys.argv[3]
    
    #check features dir
    errors = False
    if not os.path.isdir(features_dir):
        errors = True
        print(c.INVAL % ("Features directory", features_dir, "directory"), file=sys.stderr)
    else:
        if not os.access(features_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("features directory", features_dir, "read"), file=sys.stderr)
        if not os.access(features_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("features directory", features_dir, "execute"), file=sys.stderr)

    #check model dir
    if not os.path.isdir(model_dir):
        errors = True
        print(c.INVAL % ("Model directory", model_dir, "directory"), file=sys.stderr)
    else:
        if not os.access(model_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("model directory", model_dir, "read"), file=sys.stderr)
        if not os.access(model_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("model directory", model_dir, "execute"), file=sys.stderr)

    #check results_dir
    if os.path.isdir(results_dir):
        if not os.access(results_dir, os.W_OK):
            errors = True
            print(c.NO_PERM % ("results directory", results_dir, "write"), file=sys.stderr)
        if not os.access(results_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("results directory", results_dir, "execute"), file=sys.stderr)

    if errors:
        print_usage(1)
    #end error checking

    for path in os.listdir(features_dir):
        base_model_name = ''
        anomaly_model_file = ''
        device_label = ''
        if path.endswith(".csv"):
            errors = False
            features_file = features_dir + '/' + path
            device = path.replace('.csv','')
            base_model_file = os.path.join(model_dir, "knn", device + "knn.model")
            if not os.path.isfile(base_model_file):
                print(c.MISSING_MOD % ("model", device, base_model_file), file=sys.stderr)
                errors = True

            device_label = os.path.join(model_dir, "knn", device + ".label.txt")
            if not os.path.isfile(device_label):
                print(c.MISSING_MOD % ("labels", device, device_label), file=sys.stderr)
                errors = True

            anomaly_model_file = os.path.join(model_dir, "anomaly_model",
                                              "multivariate_model_" + device + ".pkl")
            if not os.path.isfile(anomaly_model_file):
                print(c.MISSING_MOD % ("anomaly model", device, anomaly_model_file), file=sys.stderr)
                errors = True

            if errors:
                break

            labels = label(device_label)
            di,reverse_di = dictionary_create(labels)
            dev_result_dir = os.path.join(results_dir, device + '_results/')
            print(f"Running process for {device}")
            run_process(features_file, dev_result_dir,base_model_file,anomaly_model_file )
            print("Results for %s written to \"%s\"" % (device, dev_result_dir))


if __name__ == '__main__':
    main()
