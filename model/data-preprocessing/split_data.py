import sys
import os
import random

RED = "\033[31;m"
END = "\033[0m"
PATH = sys.argv[0]

usage_stm = """
Usage: python3 {prog_name} in_pcap_dir out_train_file out_test_file
""".format(prog_name=PATH)

def print_usage(is_error):
    print(usage_stm, file=sys.stderr) if is_error else print(usage_stm)
    exit(is_error)


def main():
    print("Running %s..." % PATH)

    for arg in sys.argv:
        if arg in ("-h", "--help"):
            print_usage(0)

    if len(sys.argv) != 4:
        print("wrong # args")

    test_files = []
    train_files = []
    for root, subdirs, files in os.walk(sys.argv[1]):
        pcaps = [ f for f in files if f.endswith(".pcap") ]
        new_test = random.sample(pcaps, round(len(pcaps) / 3))
        for f in new_test:
            test_files.append(os.path.join(root, f))

        new_train = list(set(pcaps) - set(new_test))
        for f in new_train:
            train_files.append(os.path.join(root, f))

    train_path = sys.argv[2]
    test_path = sys.argv[3]
    if not os.path.isdir(os.path.dirname(train_path)):
        os.system("mkdir -pv %s" % os.path.dirname(train_path))

    if not os.path.isdir(os.path.dirname(test_path)):
        os.system("mkdir -pv %s" % os.path.dirname(test_path))

    with open(train_path, "w+") as f:
        f.write("\n".join(train_files))
        print("Training filenames written to %s" % train_path)

    with open(test_path, "w+") as f:
        f.write("\n".join(test_files))
        print("Testing filenames written to %s" % test_path)


if __name__ == '__main__':
    main()
