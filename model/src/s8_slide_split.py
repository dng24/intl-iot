import sys
import os
import math
import argparse
from multiprocessing import Process

import Constants as c

#is_error is either 0 or 1
def print_usage(is_error): 
    print(c.SLIDE_SPLIT_USAGE, file=sys.stderr) if is_error else print(c.SLIDE_SPLIT_USAGE)
    exit(is_error)


def get_num(str_num, description):
    bad_num = False
    try:
        num = int(str_num)
        if num < 0:
            bad_num = True
    except ValueError:
        bad_num = True

    if bad_num:
        print(c.NON_POS % (description, str_num), file=sys.stderr)
        num = -1

    return num


def run(pid, files, src, dest, slide_int, time_window):
    for fpath in files:
        times = []
        with open(fpath, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                print("%s is empty, skipping..." % fpath)
                break
            times = []
            for l in lines:
                try:
                    times.append(float(l.split("\t")[1]))
                except (IndexError, ValueError) as e:
                    break

            if len(times) == 0:
                print("%s does not have valid timestamps, skipping..." % fpath)
                break

        print("P%s: IN: %s" % (pid, fpath)) 
        start_int = times[0]
        end_int = start_int + time_window
        last_poss_start = math.ceil((times[len(times) - 1] - time_window) / slide_int) * slide_int
        start_idxes = [0]
        idx = 0
        num = 0
        last_bucket = False
        for t in times:
            while t - start_int >= slide_int and not last_bucket:
                if t > last_poss_start and last_poss_start - start_int < slide_int:
                    last_bucket = True

                start_int += slide_int
                start_idxes.append(idx)

            num_pop = 0
            for i in start_idxes:
                if t > end_int:
                    dest_file = os.path.join(dest, fpath.replace(src, "", 1)[:-4]
                                                   + "_part_%d.txt" % num)
                    print("P%s: OUT: %s" % (pid, dest_file))
                    os.system("sed -n \"%d,%dp\" %s > %s" % (i + 1, idx, fpath, dest_file))
                    num_pop += 1
                    num += 1
                    end_int += slide_int

            [ start_idxes.pop(0) for _ in range(num_pop) ]
            idx += 1

        while len(start_idxes) > 0:
            dest_file = os.path.join(dest, fpath.replace(src, "", 1)[:-4] + "_part_%d.txt" % num)
            print("P%s: OUT: %s" % (pid, dest_file))
            if not os.path.isdir(os.path.dirname(dest_file)):
                os.system("mkdir -pv %s" % os.path.dirname(dest_file))
            os.system("sed -n \"%d,%dp\" %s > %s" % (start_idxes[0] + 1, idx, fpath, dest_file))
            start_idxes.pop(0)
            num += 1


def main():
    #parse arguments
    parser = argparse.ArgumentParser(usage=c.SLIDE_SPLIT_USAGE, add_help=False)
    parser.add_argument("-i", dest="dec_dir", default="")
    parser.add_argument("-o", dest="dest_dir", default="")
    parser.add_argument("-t", dest="time_window", default="30")
    parser.add_argument("-s", dest="slide_int", default="5")
    parser.add_argument("-p", dest="num_proc", default="1")
    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    if args.help:
        print_usage(0)

    print("Running %s..." % c.PATH)

    #error checking
    errors = False
    #check -i in source
    if args.dec_dir == "":
        errors = True
        print(c.NO_SRC_DIR, file=sys.stderr)
    elif not os.path.isdir(args.dec_dir):
        errors = True
        print(c.INVAL % ("Source directory", args.dec_dir, "directory"), file=sys.stderr)
    else:
        if not os.access(args.dec_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("source directory", args.dec_dir, "read"), file=sys.stderr)
        if not os.access(args.dec_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("source directory", args.dec_dir, "execute"), file=sys.stderr)

    #check -o out destination
    if args.dest_dir == "":
        errors = True
        print(c.NO_DEST_DIR, file=sys.stderr)
    elif os.path.isdir(args.dest_dir):
        if not os.access(args.dest_dir, os.W_OK):
            errors = True
            print(c.NO_PERM % ("destination directory", args.dest_dir, "write"), file=sys.stderr)
        if not os.access(args.dest_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("destination directory", args.dest_dir, "execute"), file=sys.stderr)

    #check -t time window
    time_window = get_num(args.time_window, "time window")
    if time_window == -1:
        errors = True

    #check -s slide interval
    slide_int = get_num(args.slide_int, "slide interval")
    if slide_int == -1:
        errors = True

    if slide_int > time_window:
        errors = True
        print(c.INT_GT_TIME_WIN % (slide_int, time_window), file=sys.stderr)

    #check -p number processes
    num_proc = get_num(args.num_proc, "number of processes")
    if num_proc == -1:
        errors = True

    if errors:
        print_usage(1)
    #end error checking

    if not os.path.isdir(args.dest_dir):
        os.system("mkdir -pv %s" % args.dest_dir)

    files = [ [] for _ in range(num_proc) ]

    index = 0
    for root, dirs, fs in os.walk(args.dec_dir):
        for fname in fs:
            if fname.endswith(".txt"):
                files[index].append(os.path.join(root, fname))
                index += 1
                if index >= num_proc:
                    index = 0
            else:
                print(c.WRONG_EXT % ("Decoded file", "text (.txt)", os.path.join(root, fname)),
                      file=sys.stderr)

    procs = []
    for pid, files in enumerate(files):
        p = Process(target=run, args=(pid, files, args.dec_dir, args.dest_dir, slide_int, time_window))
        procs.append(p)
        p.start()

if __name__ == "__main__":
    main()

