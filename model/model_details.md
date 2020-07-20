# Detailed Descriptions for Content Analysis Models and Scripts

Below is a detailed description about the machine learning models, the files, and the directories in this section.

## Machine Learning Models to Detect Device Activity

### Problem Statement

For a specified device, given a sequence of network frames, what is the device activity?

Examples:
- device: amcrest-cam-wired
- network traffic: 10 minutes of network traffic
- device activity: one of
    - movement
    - power
    - watch_android
    - watch_cloud_android
    - watch_cloud_ios
    - watch_ios

**++ Cases**: the 10 minutes of traffic could have more than one activity.

### Machine Learning

During evaluation, we use following algorithms:
- RF:  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) (supervised)
- KNN: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) (supervised)
- *k*-means: [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) (unsupervised)
- DBSCAN: [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) (unsupervised)

For the purpose of IMC submission, we do not consider unsupervised approaches (i.e. *k*-means, DBSCAN).

### Variables in sklearn:

N samples of M features of L classes
- X_features: features of N samples, N * M,
- y_labels: labels of N samples
- X_train: default 70% of N samples (shuffled)
- X_test: default 30% of N samples (shuffled)
- y_train: original encoded values, e.g. "watch_ios_on"
    - y_train_bin: onehot encoded, e.g. [0, 1, 0, 0] as watch_ios_on is the second in the .classes_
- y_test: original encoded values
    - y_test_bin: onehot encoded
    - y_test_bin_1d: encoded values
    - y_predicted: onehot encoded prediction of X_test
    - y_predicted_1d: encoded values
    - y_predicted_label: original values
- _acc_score: Trained with X_train,y_train; eval with X_test, y_test; refer to [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
-  _complete: refer to [completeness_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score)
    > This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
-  _silhouette: [silhouetee_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)

## Steps

The table below lists the steps needed to complete the model pipeline.

| Step | Phase         | Script               | Description                                      | Inputs                                   | Outputs                               |
| ---- | ------------- | -------------------- | ------------------------------------------------ | ---------------------------------------- | ------------------------------------- |
| 1    | Preprocessing | s1_split_data.py     | Make training and testing sets from tagged pcaps | Tagged pcap directory                    | Training and testing pcaps text files |
| 2.1  | Preprocessing | s2_7_decode_raw.py   | Decode tagged training pcaps                     | Training pcaps text file                 | Decoded training pcaps directory      |
| 2.2  | Preprocessing | s2_7_decode_raw.py   | Decode tagged testing pcaps                      | Testing pcaps text file                  | Decoded testing pcaps directory       |
| 3.1  | Preprocessing | s3_9_get_features.py | Statistically analyze decoded training pcaps     | Decoded training pcaps directory         | Training features directory           |
| 3.2  | Preprocessing | s3_9_get_features.py | Statistically analyze decoded testing pcaps      | Decoded testing pcaps directory          | Testing features directory            |
| 4    | Modeling      | s4_eval_model.py     | Generate base model                              | Training features directory              | Models directory - base model         |
| 5    | Modeling      | s5_find_anomalies.py | Generate anomaly model                           | Training features directory              | Models directory - anomaly model      |
| 6    | Prediction    | `find` command       | Make text file from untagged pcaps               | Untagged pcap directory                  | Untagged pcaps text file              |
| 7    | Prediction    | s2_7_decode_raw.py   | Decode untagged pcaps                            | Untagged pcaps text file                 | Decoded untagged pcaps directory      |
| 8    | Prediction    | s8_slide_split.py    | Split decoded untagged pcaps by timestamp        | Decoded untagged pcaps directory         | Split decoded pcaps directory         |
| 9    | Prediction    | s3_9_get_features.py | Statistically analyze split decoded pcaps        | Split decoded pcaps directory            | Untagged features directory           |
| 10   | Prediction    | s10_predict.py       | Predict activity using base and anomaly models   | Untagged features and models directories | Results directory                     |

## Scripts

### main.py

This is the main script of the content analysis pipeline. The scripts listed below are parts of this pipeline. The main goal of this pipeline is to use machine learning to predict the device activity given the network traffic of that device. First, the pipeline needs to create a model using machine learning. The raw network traffic (provided in several pcap files with known device activity) is decoded into human-readable raw data. This raw data is then statistically analyzed and sent into an algorithm for training. Once trained, one or more models will be generated for each device. A pcap file with unknown device activity can be put through the same process of decoding it into human-readable data and statistically analyzing it. The analyzed data can then be used to predict the device activity based on the network traffic.

#### Usage

Usage: `python3 main.py -i TAGGED_DIR [OPTION]...`

Example: `python3 main.py -i traffic/us/ -u sample-untagged/ -n -o output/ -p 4`

#### Input

`-i TAGGED_DIR` - The path to the directory containing pcap files with known device activity to generate the machine learning models. See the [traffic/](#traffic) section below for the required structure of this directory. This option is required.

`-u UNTAGGED_DIR` - The path to the directory containing pcap files with unknown device activity for prediction. See the [traffic/](#traffic) section below for the required structure of this directory.

`-d` - Generate a model using the DBSCAN algorithm.

`-k` - Generate a model using the *k*-means algorithm.

`-n` - Generate a model using the *k*-nearest neighbors (KNN) algorithm.

`-r` - Generate a model using the random forest (RF) algorithm.

`-s` - Generate a model using the spectral clustering algorithm.

`-o OUT_DIR` - The path to a directory to place all intermediate and final prediction output. This directory will be generated if it currently does not exist.

`-p NUM_PROC` - The number of processes to use to run parts of this pipeline. Default is `1`.

`-h` - Display the usage statement and exit.

Note: If no models are specified to be generated, all five models will be created.

#### Output

This script places all output in `OUT_DIR`:

- `s1_test_paths.txt` - The paths to pcap files used to test the trained models.
- `s1_train_paths.txt` - The paths to pcap files used to create the machine learning models.
- `s2.1-train-decoded/` - The directory containing the decoded training pcap files.
- `s2.2-test-decoded/` - The directory containing the decoded testing pcap files.
- `s3.1-train-features/` - The directory containing the statistically-analyzed training files.
- `s3.2-test-features/` - The directory containing the statistically-analyzed testing files.
- `s4-5-models/` - The directory containing the base and anomaly machine learning models.
- `s6_untagged_paths.txt` - The paths to pcap files with unknown device activity.
- `s7-untagged-decoded/` - The directory containing the decoded untagged pcap files.
- `s8-untagged-decoded-split/` - The directory containing the decoded untagged pcap files split into different files by timestamp.
- `s9-untagged-features/` - The directory containing the statistically-analyzed untagged files.
- `s10-results/` - The directory containing the device activity prediction results of the untagged files.

Steps 6-10 are run only if `-u` is specified.

Information about the contents of each of these files and directories can be found below.

### src/s1_split_data.py

This script is the first step of the model pipeline and the first step in the preprocessing phase. The script takes a directory of pcaps recursively and randomly splits the pcaps into a training set and a testing set.

#### Usage

Usage: `python3 s1_split_data.py in_pcap_dir out_train_file out_test_file`

Example: `python3 s1_split_data.py traffic/ s1_train_paths.txt s1_test_paths.txt`

#### Input

`in_pcap_dir` - The path to a directory containing input pcap files. Pcap files found in nested directories will also be processed.

`out_train_file` - The path to a text (.txt) file to write the filenames of training files. This file will be generated if it does not already exist.

`out_test_file` - The path to a text (.txt) file to write the filenames of testing files. This file will be generated if it does not already exist.

#### Output

Two newline-delimited text files are produced, one of which contains file paths to pcaps for training models, while the other one contains file paths to pcaps for validating the models. Two-thirds of the pcap paths will be randomly selected and written to the training text file, while the rest will be written to the testing text file. If an existing text file is passed in, none of the paths in that text file will be written again to either text file.

### src/s2_7_decode_raw.py

#### Usage

This script is the second and seventh steps of the model pipeline and the second step in the preprocessing and prediction phases. The script decodes data in pcap files (whose filenames are listed in a text file) into human-readable text files using TShark.

Usage: `python3 s2_7_decode_raw.py exp_list out_imd_dir [num_proc]`

Example: `python3 s2_7_decode_raw.py exp_list.txt tagged-decoded/us/ 4`

#### Input

`exp_list` - The text file that contains paths to input pcap files to generate the models. To see the format of this text file, please see the [traffic/](#traffic) section below.

`out_imd_dir` - The path to the directory where the script will create and put decoded pcap files. If this directory current does not exist, it will be generated.

`num_proc` - The number of processes to use to decode the pcaps. Default is 1.

#### Output

A plain-text file will be produced for every input pcap (.pcap) file. Each output file contains a translation of some of the fields in the raw input file into human-readable form. The raw data output is tab-delimited and is stored in a text (.txt) file at `{out_imd_dir}/{device}/{activity}/{filename}.txt` (see the [traffic/](#traffic) section below for an explanation of `device` and `activity`.

If an output file already exists, TShark will not run with its corresponding input file, and the existing output file will remain. If TShark cannot read the input file, no output file will be produced.

Output files contain the following columns:

- `frame_num` - The frame number.
- `ts` - The Unix time of when the frame was captured.
- `ts_delta` - The time difference in seconds between the current and previous frame.
- `frame_len` - The number of bytes in the frame.
- `ip_src` - The source IP address.
- `ip_dst` - The destination IP address.
- `host` - The hostname of the destination IP address, if one exists.

### src/3_9_get_features.py

#### Usage

This script is the third and ninth steps of the model pipeline, the third step of the data preprocessing phase, and the fourth step of the prediction phase. The script uses the decoded pcap data output from `s2_7_decoded_raw.py` to perform data analysis to get features.

Usage: `python3 s3_9_get_features.py in_dec_dir out_features_dir [num_proc]`

Example: `python3 extract_features.py tagged-decoded/us/ features/us/ 4`

#### Input

`in_dec_dir` - The path to a directory containing text files of human-readable raw pcap data.

`out_features_dir` - The path to the directory to write the analyzed CSV files. If this directory current does not exist, it will be generated.

`num_proc` - The number of processes to use to generate the feature files.

#### Output

Each valid input text (.txt) file in the input directory will be analyzed, and a CSV file containing statistical analysis will be produced in a `cache/` directory in `out_features_dir`. The name of this file is a sanitized version of the input file path. After each input file is processed, all the CSV files of each device will be concatenated together in a separate CSV file, named {device}.csv, which will be placed in `out_features_dir`.

If a device already has a concatenated CSV file located in `out_features_dir`, no analysis will occur for that device, and the existing file will remain. If a device does not have a concatenated CSV file, the script will regenerate any cache files, as necessary, and the cache files of the device will be concatenated. If an input file is not a text (.txt) file, no output will be produced for that file.

Each CSV file contains ten rows. Each row is generated by performing the same statistical analysis on a random eighty percent of the input data.

An output CSV has the following columns (all regarding the random eighty percent of the data):

- `start_time` - The lowest Unix timestamp.
- `end_time` - The highest Unix timestamp.
- `spanOfGroup` - The difference between `start_time` and `end_time` in number of seconds.
- `meanBytes` - The mean number of bytes in a frame.
- `minBytes` - The lowest number of bytes in a frame.
- `maxBytes` - The highest number of bytes in a frame.
- `medAbsDev` - The median absolute deviation of the number of bytes in a frame.
- `skewLength` - The skewness of the number of bytes in a frame.
- `kurtosisLength` - The kurtosis of the number of bytes in a frame.
- `q10` - The tenth percentile of the number of bytes in a frame.
- `q20` - The twentieth percentile of the number of bytes in a frame.
- `q30` - The thirtieth percentile of the number of bytes in a frame.
- `q40` - The fortieth percentile of the number of bytes in a frame.
- `q50` - The fiftieth percentile of the number of bytes in a frame.
- `q60` - The sixtieth percentile of the number of bytes in a frame.
- `q70` - The seventieth percentile of the number of bytes in a frame.
- `q80` - The eightieth percentile of the number of bytes in a frame.
- `q90` - The ninetieth percentile of the number of bytes in a frame.
- `meanTBP` - The mean number of seconds of the time differences between consecutive frame.
- `varTBP` - The variance of the time differences between consecutive frames.
- `medianTBP` - The median of the time differences between consecutive frames.
- `kurtosisTBP` - The kurtosis of the time differences between consecutive frames.
- `skewTBP` - The skew of the time differences between consecutive frames.
- `network_to` - The number of frames whose destination IP address is `192.168.10.204`.
- `network_from` - The number of frames whose source IP address is `192.168.10.204`.
- `network_both` - The number of frames whose source IP addresses are `192.168.10.248,192.168.10.204`.
- `network_to_external` - The number of frames whose destination IP address is not `192.168.10.204` and whose source IP address is not `192.168.10.204`.
- `network_local` - The number of frames whose destination IP addresses are `192.168.10.204,129.10.227.248`.
- `anonymous_source_destination` - The number of frames which do not fit into `network_to`, `network_from`, `network_both`, `network_to_external`, or `network_local`.
- `device` - The device which the data was recorded on.
- `state` - The activity of the device when the data was recorded.
- `host` - A semicolon-delimited list of hostnames of the destination IP addresses in the random sample. If a packet's destination IP address does not have a hostname, the destination IP address is used instead.

### src/s4_eval_model.py

This script is the fourth step of the model pipeline and the first step of the model development phase. The script trains analyzed pcap data and generates one or more models that can be used to predict device activity.

#### Usage

Usage: `python3 s4_eval_model.py -i IN_FEATURES_DIR -o OUT_MODELS_DIR [-dknrs]`

Example: `python3 s4_eval_model.py -i features/us/ -o tagged-models/us/ -kn`

#### Input

`-i IN_FEATURES_DIR` - The path to a directory containing CSV files that have analyzed pcap data. This option is required.

`-o OUT_MODELS_DIR` - The path to the directory to place the generated model. If this directory currently does not exist, it will be generated. This option is required.

`-d` - Generate a model using the DBSCAN algorithm.

`-k` - Generate a model using the *k*-means algorithm.

`-n` - Generate a model using the *k*-nearest neighbors (KNN) algorithm.

`-r` - Generate a model using the random forest (RF) algorithm.

`-s` - Generate a model using the spectral clustering algorithm.

Note: If no model is chosen, all the models will be produced.

#### Output

The script will generate five files for each model specified:

- `{OUT_MODELS_DIR}/{model}/{model}-{device}.png` - An image of the activity clusters during model training.
- `{OUT_MODELS_DIR}/{model}/{device}{model}.model` - The model. If this file exists, then this model will not be regenerated.
- `{OUT_MODELS_DIR}/{model}/{device}.label.txt` - A newline-delimited file of the possible activities of a device that the model can predict.
- `{OUT_MODELS_DIR}/{model}/{device}.result.csv` - A file containing the results of model generation. The first line is a tab-delimited, with the following columns:
 - First column - The device name.
 - Second column - The accuracy classification score.
 - Third column - The homogeneity score.
 - Fourth column - The completeness score.
 - Fifth column - The V-measure score.
 - Sixth column - The adjusted Rand index.
 - Seventh column - The noise if the model is DBSCAN. Otherwise, this column is `-1`.
 - Eighth column - The mean Silhouette Coefficient. This column is `-1` for RF models.
 The second and third lines contain the predictions. 
- `{OUT_MODELS_DIR}/output/result_{model}.txt` - This file is a copy of `{OUT_MODELS_DIR}/{model}/{device}.result.csv` without the second and third lines.

### src/s5_find_anomalies.py

This script is the fifth step of the model pipeline and the second step of the model development phase. The script finds anomalies in the models generated in `s4_eval_model.py`.

#### Usage

Usage: `python3 s5_find_anomalies.py in_features_dir out_models_dir

Example: `python3 s5_find_anomalies.py features/us/ tagged-models/us/

#### Input

`in_features_dir` - The path to a directory containing CSV files that have analyzed pcap data.

`out_models_dir` - The path to a directory containing the models generated by `s4_eval_model.py`.

#### Output

This script produces an anomaly model to `{out_models_dir}/anomaly_model/multivariate_model_{device}.pkl`.

### src/s8_slide_split.py

This script is the eighth step of the model pipeline and the second step of the prediction phase. The script takes decoded pcap text files generated from `s2_7_decode_raw.py` and splits the contents of each file into separate files based on a time window and slide interval.

#### Usage

Usage: `python3 slide_split.py -i IN_DEC_DIR -o OUT_DIR [OPTION]...`

Example: `python3 slide_split.py -i decoded/us/ -o decoded-split/us/ -t 20 -s 15 -p 4`

#### Input

`-i IN_DEC_DIR` - The path to a directory containing decoded pcap text files, generated by `s2_7_decode_raw.py`. This option is required.

`-o OUT_DIR` - The path to the directory to place the split decoded files. This directory will be created if it does not exist. This option is required.

`-t TIME_WIN` - The maximum number of seconds of traffic that each split file will contain. Defualt is `30`.

`-s SLIDE_INT` - The minimum number of seconds between the first timestamp of each file. Default is `5`.

`-p NUM_PROC` - The number of processes to use to split the decoded files. Default is `1`.

`-h` - Print the usage statement and exit.

#### Output

The script will recursive take each text file in `IN_DEC_DIR` and split it based on the time window and slide interval. The resulting text files will be placed in `OUT_DIR` where the filename will be appended by `_part_#`, where `#` is the file number in the split. If an input file's Part 0 exists, then the script will assume all the other parts exist, and the script will move on to the next input file. Delete Part 0 to regenerate.

Each text file is tab-delimited and contains the same columns as those described in the [output](#output-2) section of `src/s2_7_decode_raw.py`.

### src/s10_predict.py

This script is the tenth step of the model pipeline and the fifth step of the prediction phase. The script takes untagged pcap files and predicts their device activity using the base and anomaly models.

#### Usage

Usage: `python3 s10_predict.py in_features_dir in_models_dir out_results_dir`

Example: `python3 predict.py features/us/ tagged-models/us/ results/`

#### Input

`in_features_dir` - The path to a directory containing CSV files, generated by `s3_9_get_features.py`, of statistically-analyzed untagged pcap files.

`model_dir` - The path to a directory containing machine-learning models to predict device activity, generated by `s4_eval_model.py`.

`out_result_dir` - The path to a directory to place prediction results. This directory will be generated by the script if it currently does not exist.

#### Output

The script takes the features in `in_features_dir/{device}.csv` and uses the models in `model_dir` to predict device activity. A directory named `{device}_results/` is created in `out_result_dir`. In each device result directory, two files are created:

- `anomaly_output.txt` - contains prediction results from the anomaly model
- `model_results.csv` - contains the prediction results

`model_results.csv` contains four columns:

- `end_time` - the end time of the packets in the specific prediction
- `prediction` - the activity prediction of the device between `start_time` and `end_time`
- `start_time` - the start time of the packets in the specific prediction
- `tagged` - the actual activity of the device in the specified time frame

## Non-scripts

### model_sample.ipynb

A Jupyter Notebook that contains runnable code with the same commands as `model.sh`. However, there are explanations as to what each command does in the Jupyter Notebook.

### model_details.md

This file. Provides detailed information about the models and the files in the content analysis pipeline.

### README.md

The README.

### requirements.txt

The software that should be installed before running this pipeline. To install, run:

```
pip install -r requirements.txt
```

### yi_camera_sample.pcap

A sample pcap file for demonstration that can be used to predict the device activity based on the traffic in the file.

### traffic/

An input directory with sample pcap files to generate machine learning models. The more pcap files in an input directory, the better the model. **Every input directory should have the following structure: `{root_experiment_director(r|ies)}/{device}/{activity}/{filename}.pcap`.** For example, `traffic/us/yi-camera/power/2019-04-25_19:28:58.154s.pcap`:

- `traffic/us/` is the root experiment directory.
- `yi-camera/` is the device directory.
- `power/` is the activity type directory.
- `2019-04-25_19:28:58.154s.pcap` is the input pcap file.

Each path should be on a new line.

Note: To obtain the sample files in `traffic/`, please follow the directions in the Download Datasets section in [Getting_Started.md](../Getting_Started.md#download-datasets). If you have your own pcap files, you do not need to obatain these files.

