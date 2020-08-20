import os
import sys
from multiprocessing import Process

import whois

import Constants as c

#In: in_pcap_txt out_decoded_dir [num_processes]
#Out: tab-delim txt w/ header: frame_num\tts\tts_delta\tframe_len\tip_src\tip_dst\thost


#is_error is either 0 or 1
def print_usage(is_error):
    print(c.DEC_RAW_USAGE, file=sys.stderr) if is_error else print(c.DEC_RAW_USAGE)
    exit(is_error)


def extract_pcap(in_pcap, out_txt):
    #Note: PcapReader from scapy and pyshark seems to be slower than using tshark

    #file contains hosts and ips in format [hostname]\t[ip,ip2,ip3...]
    hosts = str(os.popen("tshark -r %s -Y \"dns&&dns.a\" -T fields -e dns.qry.name -e dns.a"
                           % in_pcap).read()).splitlines()
    #make dictionary of ip to host from DNS requests
    ip_host = {} #dictionary of destination IP to hostname
    for line in hosts: #load ip_host
        #line[0] is host, line[1] contains IPs that resolve to host
        line = line.split("\t")
        ips = line[1].split(",")
        for ip in ips:
            ip_host[ip] = line[0]

    ip_host["8.8.8.8"] = "dns.google" #might remove, whois can't resolve this

    #csv output - host will be added to last column
    out = str(os.popen("tshark -r %s -Y ip -T fields -e frame.number -e frame.time_epoch"
                       " -e frame.time_delta -e frame.len -e ip.src -e ip.dst 2> /dev/null"
                       % in_pcap).read()).splitlines()
    #old command - removed fields that are not used in s3
    #out = str(os.popen("tshark -r %s -Y ip -T fields -e frame.number -e frame.time_epoch"
    #                   " -e frame.time_delta -e frame.protocols -e frame.len -e eth.src"
    #                   " -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport"
    #                   " -e udp.srcport -e udp.dstport -E separator=, 2>/dev/null"
    #                   % in_pcap).read()).splitlines()

    #add host to rest of output: 1) get host from tshark 2) get host from whois 3) host is ""
    for i, line in enumerate(out):
        ip_dst = line.split("\t")[5] #desintation host -> -e ip.dst
        host = ip_host[ip_dst] if ip_dst in ip_host else "N/A"
        if host == "N/A":
            ip_spl = ip_dst.split(".")
            #detect local address
            if (ip_spl[0] == "10" or (ip_spl[0] == "172" and 16 < int(ip_spl[1]) < 32)
                    or (ip_spl[0] == "192" and ip_spl[1] == "168")):
                host = ip_host[ip_dst] = ""
            else: #use whois if not local address
                try:
                    w = whois.whois(ip_dst)
                    if w.domain_name is None:
                        host = ip_host[ip_dst] = ""
                    elif isinstance(w.domain_name, (list,)):
                        host = ip_host[ip_dst] = w.domain_name[0].lower()
                    else:
                        host = ip_host[ip_dst] = w.domain_name.lower()
                except:
                    host = ip_host[ip_dst] = ""

        out[i] += "\t" + host #append host as last column of output

    #write output file
    header = "frame_num\tts\tts_delta\tframe_len\tip_src\tip_dst\thost\n"
    with open(out_txt, "w") as f:
        f.write(header + "\n".join(out))

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
    [print_usage(0) for arg in sys.argv if arg in ("-h", "--help")]

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
        print(c.WRONG_EXT % ("Input text file", "text (.txt)", in_txt), file=sys.stderr)
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

