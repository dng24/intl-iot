import pandas as pd
import os
import multiprocessing as mp
import numpy as np
import json
import time

global column_names
column_names = ['frame_no', 'ts', 'ts_delta', 'protocols', 'frame_len', 'eth_src',
                'eth_dst', 'ip_src', 'ip_dst', 'tcp_srcport', 'tcp_dstport',
                'http_host', 'sni', 'udp_srcport', 'udp_dstport']


def create_time_dict(root,idle=False):
    if idle:
        idle_time_dict = {}
        max_arr = []
        min_arr = []
        idle_time_dict['idle'] = []
        for files in os.listdir(root):
            tmp = pd.read_csv(f'{root}/{files}', sep='\t', names=column_names)
            idle_time_dict['idle'].extend(tmp['ts'])
        for keys, values in idle_time_dict.items():
            max_arr.append(max(values))
            min_arr.append(min(values))
        max_arr = max(max_arr)
        min_arr = min(min_arr)
        return idle_time_dict,min_arr,max_arr
    else:
        time_dict = {}
        max_arr = []
        min_arr = []
        for file in os.listdir(root):
            if '.DS' not in file:
                print(f"Generating Dictionary for --> {file}")
                time_dict[file] = []
                for files in os.listdir(f'{root}/{file}'):
                    tmp = pd.read_csv(f'{root}/{file}/{files}', sep='\t', names=column_names)
                    time_dict[file].extend(tmp['ts'])
        for keys, values in time_dict.items():
            max_arr.append(max(values))
            min_arr.append(min(values))
        max_arr = max(max_arr)
        min_arr = min(min_arr)
        return time_dict, min_arr, max_arr


def label_tagged(split_df):
    num_rows = 0
    for index, row in split_df.iterrows():
        num_rows+=1
        print(f"Completed labelling of {num_rows}/{split_df.shape[0]} rows \n")
        for label, time_stamps in labelled_time_dict.items():
            for time_step in time_stamps:
                if (time_step<= row['end_time']) and (time_step>= row['start_time']):
                        if row['labelled_data']=='unknown':
                          #  row['labelled_data']=f"{label}"
                            split_df.at[index,'labelled_data']= f"{label}"
                            print(f'Row {index} has been labelled as a {label}')
                            break
                        else:
                           # row['labelled_data']=f"{row['labelled_data']}|{label}"
                            split_df.at[index,'labelled_data']= f"{row['labelled_data']}|{label}"
                            print(f'Row {index} has been labelled as a {label}')
                            break
                        print(f'Row {index} has been labelled as a {label}')

    return split_df


def idle_tagged(split_df):
    num_rows = 0
    for index, row in split_df.iterrows():
        num_rows+=1
        print(f"Completed labelling of {num_rows}/{split_df.shape[0]} rows \n")
        for label, time_stamps in idle_time_dict.items():
            for time_step in time_stamps:
                if (time_step<= row['end_time']) and (time_step>= row['start_time']):
                    if row['labelled_data']=='unknown':
                        #row['labelled_data']=f"{label}"
                        split_df.at[index,'labelled_data']= f"{label}"
                        print(f'Row {index} has been labelled as a {label}')
                        break
                    else:
                        #row['labelled_data']=f"{row['labelled_data']}|{label}"
                        split_df.at[index,'labelled_data']= f"{row['labelled_data']}|{label}"
                        print(f'Row {index} has been labelled as a {label}')
                        break
                    print(f'Row {index} has been labelled as a {label}')
    return split_df


def parallelize_dataframe(cores,df, func):
    num_cores = cores
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = mp.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def result_calc(test_df, idle=False):
    if idle:
        idle_filtered = test_df[test_df['labelled_data'] == 'idle']
        idle_filtered['prediction'] = idle_filtered['prediction'].map(lambda x: 'idle' if x == 'anomaly' else x)
        idle_filtered['Accurate-anomaly'] = 0
        for index, rows in idle_filtered.iterrows():
            if rows['prediction'] in rows['labelled_data']:
                idle_filtered.at[index, 'Accurate-anomaly'] = 1
            else:
                pass

        idle_labelling = sum(idle_filtered['Accurate-anomaly']) / idle_filtered.shape[0]
        false_positive = 1 - idle_labelling
        return idle_labelling, false_positive

    else:
        results_table = test_df
        results_table['Accurate-label'] = 0
        results_table['Accurate-anomaly'] = 1
        print(results_table)
        for index, rows in results_table.iterrows():
            if rows['prediction'] in rows['labelled_data']:
                results_table.at[index, 'Accurate-label'] = 1
            else:
                pass
        for index, rows in results_table.iterrows():
            if rows['labelled_data'] == 'unknown':
                results_table.at[index, 'Accurate-anomaly'] = 0
            else:
                pass
        Accuracy_labelling = sum(results_table['Accurate-label']) / results_table.shape[0]
        Accuracy_anomaly = sum(results_table['Accurate-anomaly']) / results_table.shape[0]
        print(f'Labelling -> {Accuracy_labelling}',
              f'Anomaly -->{Accuracy_anomaly}')
        return Accuracy_labelling, Accuracy_anomaly


def main():
    global idle_time_dict, min_arr_idle, max_arr_idle,labelled_time_dict, min_arr_label, max_arr_label
    start_time = time.time()
    cores = 8
    labelled_intermediate = '/Volumes/Abhijit-Seagate/Data_iot/Intermediate/tagged_intermediate/yi-camera'
    idle_intermediate = '/Volumes/Abhijit-Seagate/Data_iot/Intermediate/idle-intermediate/yi-camera/unctrl'
    results = pd.read_csv('/Volumes/Abhijit-Seagate/Data_iot/results/results_yi_camera/results/model_results.csv')
    results['labelled_data'] = 'unknown'
    idle_time_dict, min_arr_idle, max_arr_idle = create_time_dict(root=idle_intermediate,idle=True)
    labelled_time_dict, min_arr_label, max_arr_label = create_time_dict(root=labelled_intermediate,idle=False)
    test_label = results[(results['start_time'] >= min_arr_label) & (results['end_time'] <= max_arr_label)]
    test_label = test_label[test_label['prediction'] != 'anomaly']
    label_time = time.time()
    print("Labelling Tagged Data")
    test_label = parallelize_dataframe(cores,test_label, label_tagged)
    label_end_time = time.time() - label_time
    test_label.to_csv('parallel_labelled_results.csv', index=False)
    print("Labelling Idle Data")
    idle_time = time.time()
    test_idle = results[(results['start_time'] >= min_arr_idle) & (results['end_time'] <= max_arr_idle)]
    test_idle = parallelize_dataframe(cores,test_idle, idle_tagged)
    idle_end_time = time.time() - idle_time
    test_idle.to_csv('parallel_idle_labelled_results.csv',index=False)
    labelling_accuracy, true_positive = result_calc(test_label)
    true_negative,false_positive = result_calc(test_idle, idle=True)
    print(f'True Positives --> {true_positive} \n'
          f'True Negatives --> {true_negative} \n'
          f'Labelling Accuracy --> {labelling_accuracy} \n'
          f'False Positives --> {false_positive}'
          )
    results_dict = {'tp':anomaly_accuracy,'tn':idle_labelling_accuracy,'la':labelling_accuracy}
    with open('eval_results.txt', 'w') as file:
        file.write(json.dumps(results_dict))
    end_time = time.time() - start_time
    print(f'Total Time --> {end_time} \n'
          f'Labelled Data Time --> {label_end_time} \n'
          f'Idle Data Time --> {idle_end_time}')


if __name__ == '__main__':
    main()