import sys
import os

#script paths
PATH = sys.argv[0]
MODEL_DIR = os.path.dirname(PATH)
if MODEL_DIR == "":
    MODEL_DIR = "."
DATA_PREPROC_DIR = MODEL_DIR + "/data-preprocessing/"
SPLIT_DATA = DATA_PREPROC_DIR + "split_data.py"
RAW2INT = DATA_PREPROC_DIR + "raw2intermediate.sh"
EXT_FEAT = DATA_PREPROC_DIR + "extract_features.py"
MOD_DEV_DIR = MODEL_DIR + "/model-development/"
EVAL_MOD = MOD_DEV_DIR + "eval_models.py"
ANOM_DETECT = MOD_DEV_DIR + "anomaly_detection.py"
PREDICT_DIR = MODEL_DIR + "/new-data-prediction/"
SLIDE_SPLIT = PREDICT_DIR + "sliding_split.py"
ANOM_PREDICT = PREDICT_DIR + "anomaly_predict_newdata.py"

#output paths
OUT_DIR = "results/"
for i, arg in enumerate(sys.argv):
    if arg == "-o" and i + 1 < len(sys.argv):
        OUT_DIR = sys.argv[i]
        break

TRAIN_PATHS = os.path.join(OUT_DIR, "step1_training_paths.txt")
TEST_PATHS = os.path.join(OUT_DIR, "step1_testing_paths.txt")
IMD_TRAIN_DIR = os.path.join(OUT_DIR, "step2.1-tagged-intermediates-train/")
IMD_TEST_DIR = os.path.join(OUT_DIR, "step2.2-tagged-intermediates-test/")
FEAT_TRAIN_DIR = os.path.join(OUT_DIR, "step3.1-tagged-features-train/")
FEAT_TEST_DIR = os.path.join(OUT_DIR, "step3.2-tagged-features-test/")
MODELS_DIR = os.path.join(OUT_DIR, "step4-5-tagged-models/")
NEW_PATHS = os.path.join(OUT_DIR, "step6_untagged_paths.txt")
NEW_IMD_DIR = os.path.join(OUT_DIR, "step7-untagged-intermediates/")
NEW_IMD_SPLIT_DIR = os.path.join(OUT_DIR, "step8-untagged-intermediates-split/")
NEW_FEAT_DIR = os.path.join(OUT_DIR, "step9-untagged-features/")
RESULTS_DIR = os.path.join(OUT_DIR, "step10-results/")

#basics
RED = "\033[31;1m"
BLUE = "\033[36;1m"
END = "\033[0m"
BEG = RED + PATH + ": Error: "

#basic errors
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

USAGE_STM = """
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
