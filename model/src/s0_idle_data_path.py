import sys
import os
import random

import Constants as c


# is_error is either 0 or 1
def print_usage(is_error):
    print(c.IDLE_DAT_USAGE, file=sys.stderr) if is_error else print(c.IDLE_DAT_USAGE)
    exit(is_error)


def main():
    [print_usage(0) for arg in sys.argv if arg in ("-h", "--help")]

    print("Running %s..." % c.PATH)

    # error checking
    # check for 3 args
    if len(sys.argv) != 3:
        print(c.WRONG_NUM_ARGS % (2, (len(sys.argv) - 1)), file=sys.stderr)
        print_usage(1)

    pcap_dir = sys.argv[1]
    file_path = sys.argv[2]
    print(file_path)

    errors = False
    # check input pcap directory
    if not os.path.isdir(pcap_dir):
        errors = True
        print(c.INVAL % ("Input pcap directory", pcap_dir, "directory"), file=sys.stderr)
    else:
        if not os.access(pcap_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("directory", pcap_dir, "read"), file=sys.stderr)
        if not os.access(pcap_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("directory", pcap_dir, "execute"), file=sys.stderr)

    # check output text file

    if not file_path.endswith(".txt"):
        errors = True
        print(c.WRONG_EXT % ("Output file", "text (.txt)", file_path), file=sys.stderr)
    elif os.path.isfile(file_path):
        if not os.access(file_path, os.R_OK):
            errors = True
            print(c.NO_PERM % ("file", file_path, "read"), file=sys.stderr)
        if not os.access(file_path, os.W_OK):
            errors = True
            print(c.NO_PERM % ("file", file_path, "write"), file=sys.stderr)

    if errors:
        print_usage(1)
    # end error checking

    # create output dirs if nonexistent
    if not os.path.isdir(os.path.dirname(file_path)):
        os.system("mkdir -pv %s" % os.path.dirname(file_path))

    existing_path = []

    # check for paths that already exist in output files
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            existing_path = f.read().splitlines()
            [print("  %s exists in %s" % (f, file_path)) for f in existing_path]

    train_files = []
    for root, subdirs, files in os.walk(sys.argv[1]):
        pcaps = [os.path.join(root, f) for f in files if f.endswith(".pcap")]
        # remove pcaps that are already in the output file
        pcaps = list(set(pcaps) - set(existing_path) )

        new_train = list(set(pcaps))

        for f in new_train:
            train_files.append(f)
            print("  Adding %s to %s" % (f, existing_path))

    print(existing_path)
    # don't add newline to beginning of first line if starting a new file
    train_beg = "\n" if os.path.isfile(file_path) else ""

    with open(file_path, "a+") as f:
        f.write(train_beg + "\n".join(train_files))
        print("Training filenames written to %s" %  file_path)


if __name__ == '__main__':
    main()

