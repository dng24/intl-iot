import sys
import os
from multiprocessing import Process

import Constants as c

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.DEC_RAW_USAGE, file=sys.stderr) if is_error else print(c.DEC_RAW_USAGE)
    exit(is_error)


def extract_pcap(in_pcap, out_txt):
    #decode pcap file
    os.system("tshark -r %s -Y ip -Tfields -e frame.number -e frame.time_epoch"
              " -e frame.time_delta -e frame.protocols -e frame.len -e eth.src -e eth.dst"
              " -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e http.host -e udp.srcport"
              " -e udp.dstport -E separator=/t > %s 2>/dev/null" % (in_pcap, out_txt))

    #check if TShark worked
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        print("In pcap: %s\n  Out decoded: %s" % (in_pcap, out_txt))
    elif os.path.exists(out_txt):
        print("In pcap: %s\n  %s is empty, removing..." % (in_pcap, out_txt))
        os.remove(out_txt)


def run(files, out_dir):
    for f in files:
        #parse pcap filename
        dir_name = os.path.dirname(f)
        activity = os.path.basename(dir_name)
        dev_name = os.path.basename(os.path.dirname(dir_name))
        dir_target = os.path.join(out_dir, dev_name, activity)
        if not os.path.isdir(dir_target):
            os.system("mkdir -pv %s" % dir_target)

        out_txt = os.path.join(dir_target, os.path.basename(f)[:-4] + "txt")
        #nothing happens if output file exists
        if os.path.isfile(out_txt):
            print("%s exists" % out_txt)
        else:
            extract_pcap(f, out_txt)

def main():
    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % sys.argv[0])

    #error checking
    #check for 2 or 3 arguments
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(c.WRONG_NUM_ARGS % (2, (len(sys.argv) - 1)))
        print_usage(1)

    in_txt = sys.argv[1]
    out_dir = sys.argv[2]
    str_num_proc = sys.argv[3] if len(sys.argv) == 4 else "1"

    #check in_txt
    errors = False
    if not in_txt.endswith(".txt"):
        errors = True
        print(c.WRONG_EXT % ("Input text file", "text (.txt)"), file=sys.stderr)
    elif not os.path.isfile(in_txt):
        errors = True
        print(c.INVAL % ("Input text file", in_txt, "file"), file=sys.stderr)
    elif not os.access(in_txt, os.R_OK):
        errors = True
        print(c.NO_PERM % ("input text file", in_txt, "read"), file=sys.stderr)

    #check out_dir
    if os.path.isdir(out_dir):
        if not os.access(out_dir, os.W_OK):
            errors = True
            print(c.NO_PERM % ("output directory", out_dir, "write"), file=sys.stderr)
        if not os.access(out_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("output directory", out_dir, "execute"), file=sys.stderr)

    #check num_proc
    bad_proc = False
    num_proc = 1
    try:
        num_proc = int(str_num_proc)
        if num_proc < 0:
            errors = bad_proc = True
    except ValueError:
        errors = bad_proc = True

    if bad_proc:
        print(c.NON_POS % ("number of processes", str_num_proc), file=sys.stderr)

    if errors:
        print_usage(1)
    #end error checking

    print("Input file located in: %s\nOutput files placed in: %s\n" % (in_txt, out_dir))

    #create groups to run TShark with processes
    in_files = [ [] for _ in range(num_proc) ]

    #split pcaps into num_proc groups
    with open(in_txt, "r") as f:
        index = 0
        for pcap in f:
            pcap = pcap.strip()
            if not pcap.endswith(".pcap"):
                print(c.WRONG_EXT % ("Input pcaps", "pcap (.pcap)", pcap))
            elif not os.path.isfile(pcap):
                print(c.INVAL % ("Input pcap", pcap, "file"))
            elif not os.access(pcap, os.R_OK):
                print(c.NO_PERM % ("input pcap", pcap, "read"))
            else:
                in_files[index % num_proc].append(pcap)
                index += 1

    #decode the pcaps
    procs = []
    for files in in_files:
        p = Process(target=run, args=(files, out_dir))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

if __name__ == "__main__":
    main()

