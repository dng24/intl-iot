import sys
import os
import time
import argparse

from src import Constants as c

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.MAIN_USAGE, file=sys.stderr) if is_error else print(c.MAIN_USAGE)
    exit(is_error)


def check_dir(direc, description=""):
    errors = False
    if direc == "":
        direc = "."
    if not os.path.isdir(direc):
        errors = True
        if description == "":
            print(c.MISSING % (direc, "directory", os.path.dirname(direc)), file=sys.stderr)
        else:
            print(c.INVAL % (description, direc, "directory"), file=sys.stderr)
    else:
        if not os.access(direc, os.R_OK):
            errors = True
            print(c.NO_PERM % ("directory", direc, "read"), file=sys.stderr)
        if not os.access(direc, os.X_OK):
            errors = True
            print(c.NO_PERM % ("directory", direc, "execute"), file=sys.stderr)

    return errors


def check_files(direc, files, description=""):
    errors = False
    if description == "":
        errors = check_dir(direc)
    if not errors:
        for f in files:
            if not os.path.isfile(f):
                errors = True
                if description == "":
                    print(c.MISSING % (f, "file", direc), file=sys.stderr)
                else:
                    print(c.INVAL % (description, f, "file"), file=sys.stderr)
            else:
                if not os.access(f, os.R_OK):
                    errors = True
                    print(c.NO_PERM % ("file", f, "read"), file=sys.stderr)

    return errors


def run_cmd(cmd, script):
    ret_code = os.system(cmd)
    if ret_code != 0:
        print(c.SCRIPT_FAIL % (os.path.basename(script), ret_code), file=sys.stderr)
        exit(ret_code)


def print_step(step):
    print("%s%s%s" % (c.BLUE, step, c.END))


def main():
    start_time = time.time()
    
    #Options
    parser = argparse.ArgumentParser(usage=c.MAIN_USAGE, add_help=False)
    parser.add_argument("-i", dest="tagged_dir", default="")
    parser.add_argument("-u", dest="untagged_dir", default="")
    parser.add_argument("-d", dest="models", action="append_const", const="d")
    parser.add_argument("-k", dest="models", action="append_const", const="k")
    parser.add_argument("-n", dest="models", action="append_const", const="n")
    parser.add_argument("-r", dest="models", action="append_const", const="r")
    parser.add_argument("-s", dest="models", action="append_const", const="s")
    parser.add_argument("-o", dest="out_dir", default="results/")
    parser.add_argument("-p", dest="num_proc", default="1")
    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.help:
        print_usage(0)

    print("Performing content analysis pipeline. \nRunning %s...\nStart time: %s\n"
          % (c.PATH, time.strftime("%A %d %B %Y %H:%M:%S %Z", time.localtime(start_time))))
    # Thursday 11 June 2020 11:37:02 EDT

    if args.models == None:
        args.models = ["d", "k", "n", "r", "s"]

    #Error checking and script checks
    #check scripts
    errors = check_files(c.SRC_DIR, c.SCRIPTS)

    #check -i tagged dir
    if args.tagged_dir == "":
        errors = True
        print(c.NO_TAGGED_DIR, file=sys.stderr)
    else:
        errors = check_dir(args.tagged_dir, "Tagged pcap input directory") or errors
    
    #check -u untagged dir
    if args.untagged_dir != "":
        errors = check_dir(args.untagged_dir, "Untagged pcap input directory") or errors

    #check -p number processes
    bad_proc = False
    try:
        num_proc = int(args.num_proc)
        if num_proc < 0:
            errors = bad_proc = True
    except ValueError:
        errors = bad_proc = True

    if bad_proc:
        print(c.NON_POS % args.num_proc, file=sys.stderr)
    
    if errors:
        print_usage(1)
    #End error checking

    mods = []
    if "d" in args.models:
        mods.append("DBSCAN")
    if "k" in args.models:
        mods.append("k-means")
    if "n" in args.models:
        mods.append("KNN")
    if "r" in args.models:
        mods.append("RF")
    if "s" in args.models:
        mods.append("spectral")

    mods_str = ", ".join(mods)
    untagged_print = "None" if args.untagged_dir == "" else args.untagged_dir
    print("Tagged directory:     %s\n"
          "Untagged directory:   %s\n"
          "Model(s) to generate: %s\n"
          "Output directory:     %s\n"
          "Number of processes:  %s\n"
          % (args.tagged_dir, untagged_print, mods_str, args.out_dir, num_proc))

    if not os.path.isdir(args.out_dir):
        os.system("mkdir -pv %s" % args.out_dir)

    #Run pipeline
    cmd = "python3 %s %s %s %s" % (c.SPLIT_DATA, args.tagged_dir, c.TRAIN_PATHS, c.TEST_PATHS)
    print_step("\nStep 1: Spliting pcaps into training and testing sets...\n$ %s" % cmd)
    run_cmd(cmd, c.SPLIT_DATA)

    cmd = "%s %s %s" % (c.DEC_RAW, c.TRAIN_PATHS, c.DEC_TRAIN_DIR)
    print_step("\nStep 2.1: Decoding training pcaps into human-readable form...\n$ %s" % cmd)
    run_cmd(cmd, c.DEC_RAW)

    cmd = "%s %s %s" % (c.DEC_RAW, c.TEST_PATHS, c.DEC_TEST_DIR)
    print_step("\nStep 2.2: Decoding testing pcaps into human-readable form...\n$ %s" % cmd)
    run_cmd(cmd, c.DEC_RAW)

    cmd = "python3 %s %s %s" % (c.GET_FEAT, c.DEC_TRAIN_DIR, c.FEAT_TRAIN_DIR)
    print_step("\nStep 3.1: Performing statistical analysis on training set...\n$ %s" % cmd)
    run_cmd(cmd, c.GET_FEAT)

    cmd = "python3 %s %s %s" % (c.GET_FEAT, c.DEC_TEST_DIR, c.FEAT_TEST_DIR)
    print_step("\nStep 3.2: Performing statistical analysis on testing set...\n$ %s" % cmd)
    run_cmd(cmd, c.GET_FEAT)

    models = "".join(args.models)
    cmd = "python3 %s -i %s -o %s -%s" % (c.EVAL_MOD, c.FEAT_TRAIN_DIR, c.MODELS_DIR, models)
    print_step("\nStep 4: Training data and creating model(s)...\n$ %s" % cmd)
    run_cmd(cmd, c.EVAL_MOD)

    cmd = "python3 %s %s %s" % (c.FIND_ANOM, c.FEAT_TRAIN_DIR, c.MODELS_DIR)
    print_step("\nStep 5: Detecting anomalies in the model(s)...\n$ %s" % cmd)
    run_cmd(cmd, c.FIND_ANOM)

    if args.untagged_dir != "":
        cmd = "find %s -name \"*.pcap\"" % args.untagged_dir
        print_step("\nStep 6: Gathering untagged pcaps\n$ %s > %s" % (cmd, c.NEW_PATHS))
        run_cmd("%s | tee %s" % (cmd, c.NEW_PATHS), "find command")
        print("Untagged pcap filenames written to %s" % c.NEW_PATHS)

        cmd = "%s %s %s" % (c.DEC_RAW, c.NEW_PATHS, c.NEW_DEC_DIR)
        print_step("\nStep 7: Decoding untagged pcaps into human-readable form...\n$ %s" % cmd)
        run_cmd(cmd, c.DEC_RAW)

        cmd = "python3 %s %s %s" % (c.SLIDE_SPLIT, c.NEW_DEC_DIR, c.NEW_DEC_SPLIT_DIR)
        print_step("\nStep 8: Organizing decoded pcaps...\n$ %s" % cmd)
        run_cmd(cmd, c.SLIDE_SPLIT)

        cmd = "python3 %s %s %s" % (c.GET_FEAT, c.NEW_DEC_SPLIT_DIR, c.NEW_FEAT_DIR)
        print_step("\nStep 9: Performing statistical analysis on untagged pcaps...\n$ %s" % cmd)
        run_cmd(cmd, c.GET_FEAT)
        
        cmd = "python3 %s %s %s %s " % (c.PREDICT, c.NEW_FEAT_DIR, c.MODELS_DIR, c.RESULTS_DIR)
        print_step("\nStep 10: Predicting device activity...\n$ %s" % cmd)
        print("Step 10 TBD......")
        #run_cmd(cmd, c.PREDICT)

    #Calculate elapsed time
    end_time = time.time()
    sec = round(end_time - start_time)
    hrs = sec // 3600
    if hrs != 0:
        sec = sec - hrs * 3600

    minute = sec // 60
    if minute != 0:
        sec = sec - minute * 60
    
    print("\nEnd time: %s\nElapsed time: %s hours %s minutes %s seconds\n\nContent analysis finished."
          % (time.strftime("%A %d %B %Y %H:%M:%S %Z", time.localtime(end_time)), hrs, minute, sec))

if __name__ == "__main__":
    main()

