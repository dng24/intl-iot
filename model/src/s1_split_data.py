import sys
import os
import random

import Constants as c

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.SPLIT_DAT_USAGE, file=sys.stderr) if is_error else print(c.SPLIT_DAT_USAGE)
    exit(is_error)


def main():
    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % c.PATH)

    #error checking
    if len(sys.argv) != 4:
        print(c.WRONG_NUM_ARGS % (3, (len(sys.argv) - 1)))
        print_usage(1)

    pcap_dir = sys.argv[1]    
    train_path = sys.argv[2]
    test_path = sys.argv[3]

    errors = False
    if not os.path.isdir(pcap_dir):
        errors = True
        print(c.INVAL % ("Input pcap directory", pcap_dir, "directory"))
    else:
        if not os.access(pcap_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("directory", pcap_dir, "read"))
        if not os.access(pcap_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("directory", pcap_dir, "execute"))

    for f in (train_path, test_path):
        if not f.endswith(".txt"):
            errors = True
            print(c.WRONG_EXT % ("Output file", "text (.txt)", f))
        elif os.path.isfile(f):
            if not os.access(f, os.R_OK):
                errors = True
                print(c.NO_PERM % ("file", f, "read"))
            if not os.access(f, os.W_OK):
                errors = True
                print(c.NO_PERM % ("file", f, "write"))

    if errors:
        print_usage(1)
    #end error checking

    #create output dirs if nonexistent
    if not os.path.isdir(os.path.dirname(train_path)):
        os.system("mkdir -pv %s" % os.path.dirname(train_path))

    if not os.path.isdir(os.path.dirname(test_path)):
        os.system("mkdir -pv %s" % os.path.dirname(test_path))
    
    existing_train = []
    existing_test = []

    #check for paths that already exist in output files
    if os.path.isfile(train_path):
        with open(train_path, "r") as f:
            existing_train = f.read().splitlines()
            [ print("  %s exists in %s" % (f, train_path)) for f in existing_train ]

    if os.path.isfile(test_path):
        with open(test_path, "r") as f:
            existing_test = f.read().splitlines()
            [ print("  %s exists in %s" % (f, test_path)) for f in existing_test ]

    test_files = []
    train_files = []
    for root, subdirs, files in os.walk(sys.argv[1]):
        pcaps = [ os.path.join(root, f) for f in files if f.endswith(".pcap") ]
        #remove pcaps that are already in the output file
        pcaps = list(set(pcaps) - set(existing_train) - set(existing_test))

        new_test = random.sample(pcaps, round(len(pcaps) / 3)) #test is 1/3 of pcaps
        new_train = list(set(pcaps) - set(new_test)) #train is 2/3 of pcaps

        for f in new_train:
            train_files.append(f)
            print("  Adding %s to %s" % (f, train_path))

        for f in new_test:
            test_files.append(f)
            print("  Adding %s to %s" % (f, test_path))

    #don't add newline to beginning of first line if starting a new file
    train_beg = "\n" if os.path.isfile(train_path) else ""
    test_beg = "\n" if os.path.isfile(test_path) else ""

    with open(train_path, "a+") as f:
        f.write(train_beg + "\n".join(train_files))
        print("Training filenames written to %s" % train_path)

    with open(test_path, "a+") as f:
        f.write(test_beg + "\n".join(test_files))
        print("Testing filenames written to %s" % test_path)


if __name__ == '__main__':
    main()
