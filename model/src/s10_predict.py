import pandas as pd
import numpy as np
import warnings
import itertools
import pickle
from matplotlib import pyplot as plt,style
import json
import os
import sys
from unsupervised_classification import unsupervised_classification
from retrain import retrain_model
import Constants as c

warnings.simplefilter("ignore", category=DeprecationWarning)
style.use('ggplot')
np.random.seed(42)

#In: train_features_dir untagged_features_dir tagged_models_dir [idle_models_dir] out_results_dir
#Out: Directory containing predictions

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
    normal_data['predictions'] = 'unknown'
    anomalous_data['predictions'] = 'anomalous'
    anomalous_data = anomalous_data.drop(['anomalous'],axis=1)
    normal_data = normal_data.drop(['anomalous'],axis=1)
    output_dict = {'predictions': y_predict}
    if not os.path.isdir(dev_result_dir):
        os.system("mkdir -pv %s" % dev_result_dir)

    with open(dev_result_dir+'/anomaly_output.txt','w+') as f:
        f.write(json.dumps(output_dict,cls=NumpyEncoder))

    return normal_data,anomalous_data



def filter_idle(ss,data,multivariate_model_dict,dev_result_dir):
    mv_model = multivariate_model_dict['mvmodel']
    treshold = multivariate_model_dict['treshold']
    y_test = data['state'].apply(lambda x: 1 if x == 'anomaly' else 0)
    y_predict = (mv_model.logpdf(data.drop(['state','predictions'], axis=1).values) < treshold).astype(int)
    data['idle'] = y_predict
    unknown_data = data[data['idle'] == 0]
    idle_data = data[data['idle'] == 1]
    unknown_data['predictions'] = 'unknown'
    idle_data['predictions'] = 'idle'
    output_dict = {'predictions': y_predict}
    unknown_data = unknown_data.drop(['idle'],axis=1)
    idle_data = idle_data.drop(['idle'],axis=1)
    if not os.path.isdir(dev_result_dir):
        os.system("mkdir -pv %s" % dev_result_dir)

    with open(dev_result_dir+'/idle_output.txt','w+') as f:
        f.write(json.dumps(output_dict,cls=NumpyEncoder))
    return unknown_data,idle_data


def action_classification_model(data,action_class_dict):
    ss = action_class_dict['standard_scaler']
    pca = action_class_dict['pca']
    trained_model = action_class_dict['trained_model']
    transformed_data = ss.transform(data.drop(['state','predictions'], axis=1))
    transformed_data = pca.transform(transformed_data)
    transformed_data = pd.DataFrame(transformed_data)
    transformed_data = transformed_data.iloc[:, :4]
    y_predict = trained_model.predict(transformed_data)
    y_predicted_1d = np.argmax(y_predict, axis=1)
    data['predictions'] = y_predicted_1d
    return data


def final_accuracy(final_data,dev_result_dir):
    global di
    y_test = final_data['state'].map(di)
    y_predict = final_data['predictions']
    return y_predict


def run_process(features_file,dev_result_dir,base_model_file,anomaly_model_file,idle_model_file,model_dir,
                trained_features_file):
    anomaly_data = load_data(features_file)
    original_data = anomaly_data
    #print(original_data.head())
    hosts = anomaly_data['hosts']
    start_time = anomaly_data['start_time']
    end_time = anomaly_data['end_time']
    device = list(set(anomaly_data.device))[0]
    anomaly_data = anomaly_data.drop(['device','hosts'], axis=1)
    action_classification_model_dict = pickle.load(open(base_model_file, 'rb'))
    ss = action_classification_model_dict['standard_scaler']
    anomaly_model = pickle.load(open(anomaly_model_file, 'rb'))
    if not idle_model_file is None:
        idle_model = pickle.load(open(idle_model_file, 'rb'))

    normal_data, anomalous_data = filter_anomaly(ss, anomaly_data, anomaly_model, dev_result_dir)
    # TODO: The normal data is further classified into idle vs the rest. The rest can be passed through an unsup model.
    #normal_data['predictions'] = di['normal']
    hosts_normal = [hosts[i] for i in normal_data.index]
    self_labelled_data = normal_data
    afaf = unsupervised_classification(normal_data,device,hosts_normal,dev_result_dir)
    self_labelled_data['predictions'] = afaf #unsupervised_classification(normal_data,device,hosts_normal,dev_result_dir)


    if anomalous_data.shape[0] == 0:
        print("No Labelled Anomalous data.")
        final_data = self_labelled_data
        y_predict = final_accuracy(final_data, dev_result_dir)
        out_dict = {'start_time': start_time, 'end_time': end_time, 'tagged': final_data['state'],
                    'prediction': y_predict}
        out_df = pd.DataFrame(out_dict)
        out_df['prediction'] = out_df['prediction'].map(reverse_di).fillna("normal")
        out_df.to_csv(dev_result_dir + '/model_results.csv', index=False)
        original_data['state'] = y_predict
        retrain_model(original_data, model_dir,trained_features_file)
    else:
        anomalous_data = action_classification_model(anomalous_data, action_classification_model_dict)
        anomalous_data['predictions'] = anomalous_data['predictions'].map(reverse_di).fillna("anomaly")
        final_data = self_labelled_data.append(anomalous_data).sort_index()
        y_predict = final_accuracy(final_data, dev_result_dir)
        out_dict = {'start_time': start_time, 'end_time': end_time,
                    'tagged': final_data['state'], 'prediction': y_predict}
        out_df = pd.DataFrame(out_dict)
        out_df.to_csv(dev_result_dir + '/model_results.csv', index=False)
        original_data['state'] = y_predict
        retrain_model(original_data,model_dir,trained_features_file)


#Check dir to make sure it exists and has read/execute permission
def check_dir(dir_description, dir_path):
    errors = False
    if not os.path.isdir(dir_path):
        errors = True
        print(c.INVAL % (dir_description, dir_path, "directory"), file=sys.stderr)
    else:
        if not os.access(dir_path, os.R_OK):
            errors = True
            print(c.NO_PERM % (dir_description.lower(), dir_path, "read"), file=sys.stderr)
        if not os.access(dir_path, os.X_OK):
            errors = True
            print(c.NO_PERM % (dir_description.lower(), dir_path, "execute"), file=sys.stderr)
    return errors


#Check that a file exists
def check_file(description, device, filepath):
    if not os.path.isfile(filepath):
        print(c.MISSING_MOD % (description, device, filepath), file=sys.stderr)
        return True
    else:
        return False


def main():
    global di, reverse_di, labels

    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % c.PATH)

    #error checking
    #check for 3 args
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print(c.WRONG_NUM_ARGS % (4, (len(sys.argv) - 1)), file=sys.stderr)
        print_usage(1)

    has_idle = len(sys.argv) == 6
    train_features_dir = sys.argv[1]
    untagged_features_dir = sys.argv[2]
    model_dir = sys.argv[3]
    if has_idle:
        idle_dir = sys.argv[4]
        results_dir = sys.argv[5]
    else:
        results_dir = sys.argv[4]

    
    #check training features dir
    errors = check_dir("Training features directory", train_features_dir)
    #check untagged features dir
    errors = check_dir("Untagged features directory", untagged_features_dir)
    #check tagged model dir
    errors = check_dir("Tagged model directory", model_dir)
    if has_idle:
        #check idle model dir
        errors = check_dir("Idle model directory", idle_dir)

    #check results_dir if it exists
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

    for path in os.listdir(untagged_features_dir):
        # base_model_name = ''
        # anomaly_model_file = ''
        # device_label = ''
        if path.endswith(".csv"):
            features_file = untagged_features_dir + '/' + path
            device = path.replace('.csv','')
            base_model_file = os.path.join(model_dir, "knn", device + "knn.model")
            errors = check_file("model", device, base_model_file)

            device_label = os.path.join(model_dir, "knn", device + ".label.txt")
            errors = check_file("labels", device, device_label) or errors

            anomaly_model_file = os.path.join(model_dir, "anomaly_model",
                                              "multivariate_model_" + device + ".pkl")
            errors = check_file("anomaly model", device, anomaly_model_file) or errors
            
            if has_idle:
                idle_model_file = os.path.join(idle_dir,
                                               "multivariate_model_" + device + "_idle.pkl")
                errors = check_file("idle model", device, idle_model_file) or errors
            else:
                idle_model_file = None
            
            for feat_path in os.listdir(train_features_dir):
                if feat_path == f'{device}.csv':
                    trained_features_file = f'{train_features_dir}/{feat_path}'
            print(trained_features_file)

            if errors:
                continue

            labels = label(device_label)
            di,reverse_di = dictionary_create(labels)
            dev_result_dir = os.path.join(results_dir, device + '_results/')
            print(f"Running process for {device}")
            run_process(features_file, dev_result_dir, base_model_file, anomaly_model_file,
                        idle_model_file, model_dir,trained_features_file)
            print("Results for %s written to \"%s\"" % (device, dev_result_dir))


if __name__ == '__main__':
    main()
