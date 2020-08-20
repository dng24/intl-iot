import sys
import os

#script paths
PATH = sys.argv[0]
MODEL_DIR = os.path.dirname(PATH)
if MODEL_DIR == "":
    MODEL_DIR = "."
SRC_DIR = MODEL_DIR + "/src/"
SPLIT_DATA = SRC_DIR + "s1_split_data.py"
DEC_RAW = SRC_DIR + "s2_7_decode_raw.py"
GET_FEAT = SRC_DIR + "s3_9_get_features.py"
EVAL_MOD = SRC_DIR + "s4_eval_model.py"
FIND_IDLE = SRC_DIR + "s4i_find_idle.py"
FIND_ANOM = SRC_DIR + "s5_find_anomalies.py"
SLIDE_SPLIT = SRC_DIR + "s8_slide_split.py"
PREDICT = SRC_DIR + "s10_predict.py"

SCRIPTS = [SPLIT_DATA, DEC_RAW, GET_FEAT, EVAL_MOD, FIND_ANOM, FIND_IDLE, SLIDE_SPLIT, PREDICT]

#output paths
OUT_DIR = "results/"
for i, arg in enumerate(sys.argv):
    if arg == "-o" and i + 1 < len(sys.argv):
        OUT_DIR = sys.argv[i + 1]
        break

TAGGED_OUT = os.path.join(OUT_DIR, "tagged/")
IDLE_OUT = os.path.join(OUT_DIR, "idle/")

TRAIN_PATHS = os.path.join(TAGGED_OUT, "s1_train_paths.txt")
TEST_PATHS = os.path.join(TAGGED_OUT, "s1_test_paths.txt")
DEC_TRAIN_DIR = os.path.join(TAGGED_OUT, "s2.1-train-decoded/")
DEC_TEST_DIR = os.path.join(TAGGED_OUT, "s2.2-test-decoded/")
FEAT_TRAIN_DIR = os.path.join(TAGGED_OUT, "s3.1-train-features/")
FEAT_TEST_DIR = os.path.join(TAGGED_OUT, "s3.2-test-features/")
MODELS_DIR = os.path.join(TAGGED_OUT, "s4-5-models/")
NEW_PATHS = os.path.join(TAGGED_OUT, "s6_untagged_paths.txt")
NEW_DEC_DIR = os.path.join(TAGGED_OUT, "s7-untagged-decoded/")
NEW_DEC_SPLIT_DIR = os.path.join(TAGGED_OUT, "s8-untagged-decoded-split/")
NEW_FEAT_DIR = os.path.join(TAGGED_OUT, "s9-untagged-features/")
RESULTS_DIR = os.path.join(TAGGED_OUT, "s10-results/")

IDLE_PATHS = os.path.join(IDLE_OUT, "s1_idle_paths.txt")
DEC_IDLE_DIR = os.path.join(IDLE_OUT, "s2-idle-decoded/")
FEAT_IDLE_DIR = os.path.join(IDLE_OUT, "s3-idle-features/")
MODEL_IDLE_DIR = os.path.join(IDLE_OUT, "s4i-models/")

#basics
RED = "\033[31;1m"
BLUE = "\033[36;1m"
END = "\033[0m"
BEG = RED + PATH + ": Error: "

#basic errors
WRONG_NUM_ARGS = BEG + "%d arguments required. %d arguments found." + END
MISSING = BEG + "The \"%s\" %s is missing.\n"\
          "    Please make sure it is in the \"%s\" directory." + END
NO_PERM = BEG + "The %s \"%s\" does not have %s permission." + END
INVAL = BEG + "%s \"%s\" is not a %s." + END
WRONG_EXT = BEG + "%s must be a %s file. Received \"%s\"" + END

#main.py errors
NO_IN_DIR = BEG + "Tagged pcap input directory (-i) or idle directory (-l) required." + END
NO_TAGGED_DIR = BEG + "Tagged pcap input directory (-i) required when -u option is used." + END
NON_POS = BEG + "The number of processes must be a positive integer. Received \"%s\"." + END
SCRIPT_FAIL = BEG + "Something went wrong with \"%s\". Exit status \"%d\".\n"\
              "    Please make sure you have properly set up your environment and that all" \
              " your pcap files are placed in the correct file structure." + END

#eval_model.py errors
NO_FEAT_DIR = BEG + "Features directory (-i) required." + END
NO_MOD_DIR = BEG + "Model directory (-o) required." + END

#slide_split.py
NO_SRC_DIR = BEG + "Source directory (-i) required." + END
NO_DEST_DIR = BEG + "Destination directory (-o) required." + END
INT_GT_TIME_WIN = BEG + "The slide interval (%d) cannot be greater than the time window (%d)." + END 
NO_VAL_TS = BEG + "%s does not have valid timestamps, skipping..." + END

#predict.py
MISSING_MOD = BEG + "The %s for %s does not exist at \"%s\". Skipping device..." + END

#main.py usage
MAIN_USAGE = """
Usage: python3 {prog_name} (-i TAGGED_DIR [-u UNTAGGED_DIR] | -l IDLE_DIR) [OPTION]...

Predicts the device activity of pcap files using machine learning models
that is created using several input pcap files with known device activity.
To create the models, the input pcap files are decoded into human-readable
text files. Statistical analysis is performed on this data, which can then
be used to generate the machine learning models. There currently are five
algorithms available to generate the models.

Example: python3 {prog_name} -i traffic/ -u sample-untagged -n -p 4

At least one required:
  -i TAGGED_DIR   path to the directory containing pcap files with known device
                    activity to generate the models; see the traffic/ section
                    of model_details.md for the structure of this directory
  
  
  -l IDLE_DIR     path to the directory containing pcap files with idle device
                    activity to generate the idle activity detection models.

Optional arguments:
  -u UNTAGGED_DIR path to the directory containing pcap files with unknown
                    device activity for prediction; see the traffic/ section
                    of model_details.md for the structure of this directory;
                    the -i option must be used
  -d              generate a model using the DBSCAN algorithm
  -k              generate a model using the k-means algorithm
  -n              generate a model using the k-nearest neighbors (KNN) algorithm
  -r              generate a model using the random forest (RF) algorithm
  -s              generate a model using the spectral clustering algorithm
  -o OUT_DIR      path to an output directory to place all intermediate and
                    final prediction output; directory will be generated if it
                    currently does not exist (Default = results/)
  -p NUM_PROC     number of processes to use to run parts of this pipeline
                    (Default = 1)
  -h              display this usage statement and exit

Note: If no model is specified to be generated, all five models will be generated.

For more information, see the README and model_details.md.""".format(prog_name=PATH)

#split_data.py usage
IDLE_DAT_USAGE = """
Usage: python3 {prog_name} in_pcap_dir out_idle_file

Recursively splits the pcaps in a directory into a training set and a testing set.

Example: python3 {prog_name} traffic/ s0_idle_paths.txt

Arguments:
  in_pcap_dir:    path to a directory containing pcap files
  out_idle_file: path to a text file to write the filenames of idle files;
                    file will be generate if it does not already exist

For more information, see the README or model_details.md""".format(prog_name=PATH)

#split_data.py usage
SPLIT_DAT_USAGE = """
Usage: python3 {prog_name} in_pcap_dir out_train_file [out_test_file]

Recursively splits the pcaps in a directory into a training set and an optional testing set.
If a output testing filename is provided, 1/3 of the files will be placed in the testing file,
while 2/3 of the files will be placed in the training file. If only the training file is
provided, all pcap filenames will be placed in the training file.

Example: python3 {prog_name} traffic/ s1_train_paths.txt s1_test_paths.txt

Arguments:
  in_pcap_dir:    path to a directory containing pcap files
  out_train_file: path to a text file to write the filenames of training files;
                    file will be generate if it does not already exist
  out_test_file:  path to a text file to write the filenames of testing files;
                    file will be generated if it does not already exist

For more information, see the README or model_details.md""".format(prog_name=PATH)

#decocde_raw.py usage
DEC_RAW_USAGE = """
Usage: python3 {prog_name} exp_list out_dec_dir [num_proc]

Decodes raw pcap data into human-readable text files.

Example: python3 {prog_name} exp_list.txt decoded/us/ 4

Arguments:
  exp_list:    a text file containing the file paths to pcap files to decode; pcap
                 paths must be formatted as .../{{device}}/{{activity}}/{{filename}}.pcap
  out_dec_dir: path to the directory to place the decoded output; directory will be
                 generated if it does not already exist
  num_proc:    number of processes to use to decode the pcaps (Default = 1)

For more information, see model_details.md.""".format(prog_name=PATH)

#get_features.py usage
GET_FEAT_USAGE = """
Usage: python3 {prog_name} in_dec_dir out_features_dir [num_proc]

Performs statistical analysis on decoded pcap files to generate feature files.

Example: python3 {prog_name} decoded/us/ features/us/ 4

Arguments:
  in_dec_dir:   path to a directory containing text files of decoded pcap data
  out_feat_dir: path to the directory to write the analyzed CSV files;
                  directory will be generated if it does not already exist
  num_proc:     number of processes to use to generate feature files
                  (Default = 1)

For more information, see the README or model_details.md.""".format(prog_name=PATH)

#eval_models.py usage
EVAL_MOD_USAGE = """
Usage: python3 {prog_name} -i IN_FEATURES_DIR -o OUT_MODELS_DIR [-dknrs]

Trains anaylzed pcap files and produces one or more models using different algorithms
that can predict device activity.

Example: python3 {prog_name} -i features/us/ -o models/us/ -kn

Required arguments:
  -i IN_FEATURES_DIR path to a directory containing CSV files of statistically-analyzed
                       pcap files
  -o OUT_MODELS_DIR  path to the directory to put the generated models; this directory
                       will be created if it does not exist

Optional arguments:
  -d produce a model using the dbscan algorithm
  -k produce a model using the kmeans algorithm
  -n produce a model using the knn algorithm
  -r produce a model using the rf algorithm
  -s produce a model using the spectral algorithm
  -h print this usage statement and exit

Note: If no model is chosen, all of the models will be produced.

For more information, see the README or model_details.md.""".format(prog_name=PATH)

#find_anomalies.py
FIND_ANOM_USAGE = """
Usage: python3 {prog_name} in_features_dir out_models_dir

Finds anomalies.

Example: python3 {prog_name} features/us/ models/us/

Arguments:
  in_features_dir: path to a directory containing CSV files of statistically-analyzed
                     pcap files
  out_models_dir:  path to a directory containing machine-learning models generated by
                     s4_eval_model.py

For more information, see the README or model_details.md.""".format(prog_name=PATH)

#slide_split.py
SLIDE_SPLIT_USAGE = """
Usage: python3 {prog_name} -i IN_DEC_DIR -o OUT_DIR [OPTION]...

Splits a decoded pcap text file into multiple text files based on a time window and
slide interval.

Example: python3 {prog_name} -i decoded/us/ -o decoded-split/us/ -t 25 -s 4 -p 4

Required arguments:
  -i IN_DEC_DIR path to a directory containing decoded pcap text files
  -o OUT_DIR    path to the directory to place the split decoded files; this directory
                  will be created if it does not exist

Optional arguments:
  -t TIME_WIN  the maximum number of seconds of traffic that each file will contain
                 (Default = 30)
  -s SLIDE_INT the minimum number of seconds between the first timestamp of each file
                 (Default = 5)
  -p NUM_PROC  number of processes to use to split the decoded files (Default = 1)
  -h           print this usage statement and exit

Note: TIME_WIN must be greater or equal to SLIDE_INT. To have each packet appear only
  once, TIME_WIN should be equal to SLIDE_INT.

For more information, see the README or model_details.md.""".format(prog_name=PATH)

#predict.py
PREDICT_USAGE = """
Usage: python3 {prog_name} in_train_feat_dir in_untagged_feat_dir in_models_dir out_results_dir

Uses machine learning models to predict device activity of unknown traffic.

Example: python3 {prog_name} features/us/train features/us/untagged models/us/ results/

Arguments:
  in_train_feat_dir:    path to a directory containing CSV files of statistically-analyzed
                          tagged training pcap files
  in_untagged_feat_dir: path to a directory containing CSV files of statistically-analyzed
                          untagged pcap files
  in_models_dir:        path to a directory containing machine-learning models to predict
                          device activity
  out_results_dir:      path to the directory to place prediction results; directory will
                          be generated if it currently does not exist

For more information, see the README or model_details.md.""".format(prog_name=PATH)

