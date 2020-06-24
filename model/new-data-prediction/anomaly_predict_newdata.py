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
warnings.simplefilter("ignore", category=DeprecationWarning)
style.use('ggplot')
np.random.seed(42)

usage_stm = """
Usage: python {prog_name} data_path root_model base_model_path anomaly_model_path

data_path = Feature file of the device for which you want predictions. 
root_model = Path where you want the model results to be saved. 
base_model_path = Path to the knn model of the device. 
anomaly_model_path = Path to the anomaly model of the device. 
""".format(prog_name=sys.argv[0])


def print_usage():
    print(usage_stm, file=sys.stderr)
    exit(1)

label_file = sys.argv[5]
labels = []
with open(label_file) as ff:
    for line in ff.readlines():
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        labels.append(line)


# TODO: Do not hardcode dictionary. Labels need to be taken from the device.
di ={}
reverse_di = {}
for i in range(len(labels)):
    di.update({labels[i]:i})
    reverse_di.update({i:labels[i]})

di.update({'normal':len(labels)})

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

def filter_anomaly(ss,anomaly_data,multivariate_model_dict,model_path):
    mv_model = multivariate_model_dict['mvmodel']
    treshold = multivariate_model_dict['treshold']
    y_test = anomaly_data['state'].apply(lambda x: 1 if x == 'anomaly' else 0)
    y_predict = (mv_model.logpdf(anomaly_data.drop(['state'], axis=1).values) < treshold).astype(int)
    recall = recall_score(y_pred=y_predict, y_true=y_test, average='weighted')
    precision = precision_score(y_pred=y_predict, y_true=y_test, average='weighted')
    f2 = fbeta_score(y_pred=y_predict, y_true=y_test, average='weighted', beta=2)
    f1 = f1_score(y_pred=y_predict, y_true=y_test, average='weighted')
    _acc_score = accuracy_score(y_test, y_predict)
    cm = confusion_matrix(y_test, y_predict)
    plt = plot_confusion_matrix(cm, classes=['Normal', 'Anomalous'],
                          recall=recall, precision=precision, f2=f2, f1=f1, title='Confusion matrix')
    if not os.path.exists(model_path+'/plots'):
        os.makedirs(model_path+'/plots')
    plt.savefig(model_path+'/plots/anomalous_cm.png',bbox_inches='tight')

    anomaly_data['anomalous'] = y_predict
    normal_data = anomaly_data[anomaly_data['anomalous'] == 0]
    anomalous_data = anomaly_data[anomaly_data['anomalous'] == 1]
    output_dict = {'predictions': y_predict, 'recall': recall, 'precision': precision, 'f1': f1, 'f2': f2}
    if not os.path.exists(model_path+'/results'):
        os.makedirs(model_path+'/results')
    with open(model_path+'/results/anomaly_output.txt','w') as file:
        file.write(json.dumps(output_dict,cls=NumpyEncoder))

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

def final_accuracy(final_data,model_path):
    global di
    y_test = final_data['state'].map(di)
    y_predict = final_data['predictions']


    return y_predict




def main():
    global di,reverse_di,labels
    data_path = '/Volumes/Abhijit-Seagate/Data_iot/Features/required-features-split/google-home-mini-req-april.csv'
    root_model = '/Volumes/Abhijit-Seagate/Data_iot/results/results_google_mini_req'
    base_model_path = '/Volumes/Abhijit-Seagate/Data_iot/models/new-tagged-models/knn/google-home-miniknn.model'
    anomaly_model_path = '/Volumes/Abhijit-Seagate/Data_iot/tagged-models/anomaly_models/multivariate_model_google_home_mini_req_tagged.pkl'

    data_path = sys.argv[1]
    root_model = sys.argv[2]
    base_model_path = sys.argv[3]
    anomaly_model_path = sys.argv[4]

    num_pools = 12
    p = Pool(num_pools)

    anomaly_data = load_data(data_path)
    start_time = anomaly_data['start_time']
    end_time = anomaly_data['end_time']
    anomaly_data = anomaly_data.drop(['device'], axis=1)
    action_classification_model_dict = pickle.load(open(base_model_path,'rb'))
    ss = action_classification_model_dict['standard_scaler']
    anomaly_model = pickle.load(open(anomaly_model_path,'rb'))
    normal_data,anomalous_data = filter_anomaly(ss,anomaly_data, anomaly_model,root_model)
    print("Normal")
    print(normal_data.head())
    print("Abnormal")
    print(anomalous_data.head())
    #TODO: The label for anomalous data will be according to the dictionary count for the device.
    normal_data['predictions'] = di['normal']
    anomalous_data = action_classification_model(anomalous_data,action_classification_model_dict)
    final_data = normal_data.append(anomalous_data).sort_index()
    y_predict = final_accuracy(final_data,root_model)
    arr = list(range(0,len(y_predict)))
    out_dict = {'start_time':start_time,'end_time':end_time,'tagged':final_data['state'],'prediction':y_predict}
    out_df = pd.DataFrame(out_dict)
    out_df['prediction'] = out_df['prediction'].map(reverse_di).fillna("normal")
    out_df.to_csv(root_model+'/results/model_results.csv', index=False)

if __name__ == '__main__':
    main()
