import pandas as pd
import numpy as np


def result_calc(test_df,idle=False):
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
        for index,rows in results_table.iterrows():
            if rows['prediction'] in rows['labelled_data']:
                results_table.at[index,'Accurate-label'] = 1
            else:
                pass
        for index,rows in results_table.iterrows():
            if rows['labelled_data'] =='unknown':
                results_table.at[index,'Accurate-anomaly'] = 0
            else:
                pass
        Accuracy_labelling = sum(results_table['Accurate-label']) / results_table.shape[0]
        Accuracy_anomaly = sum(results_table['Accurate-anomaly']) / results_table.shape[0]
        print(f'Labelling -> {Accuracy_labelling}',
              f'Anomaly -->{Accuracy_anomaly}')
        return Accuracy_labelling,Accuracy_anomaly


def main():
    test_label = pd.read_csv('parallel_labelled_results.csv')
    test_idle = pd.read_csv('parallel_idle_labelled_results.csv')
    labelling_accuracy, anomaly_accuracy = result_calc(test_label)
    idle_labelling_accuracy = result_calc(test_idle, idle=True)
    print(labelling_accuracy, anomaly_accuracy,idle_labelling_accuracy)

if __name__ == '__main__':
    main()