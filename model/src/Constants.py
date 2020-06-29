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
FIND_ANOM = SRC_DIR + "s5_find_anomalies.py"
SLIDE_SPLIT = SRC_DIR + "s8_slide_split.py"
PREDICT = SRC_DIR + "s10_predict.py"

SCRIPTS = [SPLIT_DATA, DEC_RAW, GET_FEAT, EVAL_MOD, FIND_ANOM, SLIDE_SPLIT, PREDICT]

#output paths
OUT_DIR = "results/"
for i, arg in enumerate(sys.argv):
    if arg == "-o" and i + 1 < len(sys.argv):
        OUT_DIR = sys.argv[i]
        break

TRAIN_PATHS = os.path.join(OUT_DIR, "s1_train_paths.txt")
TEST_PATHS = os.path.join(OUT_DIR, "s1_test_paths.txt")
DEC_TRAIN_DIR = os.path.join(OUT_DIR, "s2.1-train-decoded/")
DEC_TEST_DIR = os.path.join(OUT_DIR, "s2.2-test-decoded/")
FEAT_TRAIN_DIR = os.path.join(OUT_DIR, "s3.1-train-features/")
FEAT_TEST_DIR = os.path.join(OUT_DIR, "s3.2-test-features/")
MODELS_DIR = os.path.join(OUT_DIR, "s4-5-models/")
NEW_PATHS = os.path.join(OUT_DIR, "s6_untagged_paths.txt")
NEW_DEC_DIR = os.path.join(OUT_DIR, "s7-untagged-decoded/")
NEW_DEC_SPLIT_DIR = os.path.join(OUT_DIR, "s8-untagged-decoded-split/")
NEW_FEAT_DIR = os.path.join(OUT_DIR, "s9-untagged-features/")
RESULTS_DIR = os.path.join(OUT_DIR, "s10-results/")

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
NO_TAGGED_DIR = BEG + "Tagged pcap input directory (-i) required." + END
NON_POS = BEG + "The number of processes must be a positive integer. Received \"%s\"." + END
SCRIPT_FAIL = BEG + "Something went wrong with \"%s\". Exit status \"%d\".\n"\
              "    Please make sure you have properly set up your environment." + END

#main.py usage
MAIN_USAGE = """
Usage: {prog_name} [OPTION]...

Predicts the device activity of a pcap file using a machine learning model
that is created using several input pcap files with known device activity.
To create the models, the input pcap files are decoded into human-readable
text files. Statistical analysis is performed on this data, which can then
be used to generate the machine learning models. There currently are three
algorithms available to generate the models.

Example: {prog_name} -i exp_list.txt -rn -v yi-camera -l knn -p yi_camera_sample.pcap -o results.csv

Options:
  -i EXP_LIST   path to text file containing filepaths to the pcap files to be used
                     to generate machine learning models (Default = exp_list.txt)
  -t IMD_DIR    path to the directory to place the decoded pcap files
                     (Default = tagged-intermediate/us/)
  -f FEAT_DIR   path to the directory to place the statistically-analyzed files
                     (Default = features/us/)
  -m MODELS_DIR path to the directory to place the generated models
                     (Default = tagged-models/us/)
  -d            generate a model using the DBSCAN algorithm
  -k            generate a model using the k-means algorithm
  -n            generate a model using the k-nearest neighbors (KNN) algorithm
  -r            generate a model using the random forest (RF) algorithm
  -s            generate a model using the spectral clustering algorithm
  -p IN_PCAP    path to the pcap file with unknown device activity
                     (Default = yi_camera_sample.pcap)
  -v DEV_NAME   name of the device that generated the data in IN_PATH
                     (Default = yi-camera)
  -l MODEL_NAME name of the model to be used to predict the device activity in
                     IN_PATH; choose from kmeans, knn, or rf; DBSCAN and spectral
                     clustering algorithms cannot be used for prediction; specified
                     model must exist to be used for prediction (Default = rf)
  -o OUT_CSV    path to a CSV file to write the results of predicting the
                     device activity of IN_PATH (Default = results.csv)
  -h            display this usage statement and exit

Notes:
 - All directories and out_CSV will be generated if they currently do not exist.
 - If no model is specified to be generated, all five models will be generated.

For more information, see the README and model_details.md.""".format(prog_name=PATH)

#split_data.py usage
SPLIT_DAT_USAGE = """
Usage: python3 {prog_name} in_pcap_dir out_train_file out_test_file

Recursively splits the pcaps in a directory into a training set and a testing set.

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

Example: python3 {prog_name} exp_list.txt tagged-decoded/us/ 4

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

Example: python3 {prog_name} tagged-decoded/us/ features/us/ 4

Arguments:
  in_dec_dir:   path to a directory containing text files of decoded pcap data
  out_feat_dir: path to the directory to write the analyzed CSV files;
                      directory will be generated if it does not already exist
  num_proc:     number of processes to use to generate feature files
                      (Default = 1)

For more information, see the README or model_details.md.""".format(prog_name=PATH)

