""" Scripts processing pcap files and generating text output and figures """

import argparse
import os
import re
import sys
from multiprocessing import Process
import gc
import time

import pyshark

#from trafficAnalyzer import *  #Import statement below, after package files are checked

__author__ = "Roman Kolcun"
__copyright__ = "Copyright 2019"
__credits__ = ["Roman Kolcun"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Roman Kolcun"
__email__ = "roman.kolcun@imperial.ac.uk"
__status__ = "Development"

#Updated by Derek Ng in 2020

#File paths
PATH = sys.argv[0]
DEST_DIR = os.path.dirname(PATH)
if DEST_DIR == "":
    DEST_DIR = "."

TRAFFIC_ANA_DIR = DEST_DIR + "/trafficAnalyzer"
CONSTS = TRAFFIC_ANA_DIR + "/Constants.py"
DATA_PRES = TRAFFIC_ANA_DIR + "/DataPresentation.py"
DEV = TRAFFIC_ANA_DIR + "/Device.py"
DNS_TRACK = TRAFFIC_ANA_DIR + "/DNSTracker.py"
INIT = TRAFFIC_ANA_DIR + "/__init__.py"
IP = TRAFFIC_ANA_DIR + "/IP.py"
NODE = TRAFFIC_ANA_DIR + "/Node.py"
STAT = TRAFFIC_ANA_DIR + "/Stats.py"
UTIL = TRAFFIC_ANA_DIR + "/Utils.py"
GEO_DIR = DEST_DIR + "/geoipdb"
GEO_DB_CITY = GEO_DIR + "/GeoLite2-City.mmdb"
GEO_DB_COUNTRY = GEO_DIR + "/GeoLite2-Country.mmdb"
AUX_DIR = DEST_DIR + "/aux"
IP_TO_ORG = AUX_DIR + "/ipToOrg.csv"
IP_TO_COUNTRY = AUX_DIR + "/ipToCountry.csv"

SCRIPTS = [CONSTS, DATA_PRES, DEV, DNS_TRACK, INIT, IP, NODE, STAT, UTIL]

RED = "\033[31;1m"
END = "\033[0m"

#Check that traffic analyzer package has all files and correct permissions
errors = False
if not os.path.isdir(TRAFFIC_ANA_DIR):
    errors = True
    print("%s%s: Error: The \"%s/\" directory is missing.\n"
          "     Make sure it is in the same directory as %s.%s"
          % (RED, PATH, TRAFFIC_ANA_DIR, PATH, END), file=sys.stderr)
else:
    if not os.access(TRAFFIC_ANA_DIR, os.R_OK):
        errors = True
        print("%s%s: Error: The \"%s/\" directory does not have read permission.%s"
              % (RED, PATH, TRAFFIC_ANA_DIR, END), file=sys.stderr)
    if not os.access(TRAFFIC_ANA_DIR, os.X_OK):
        errors = True
        print("%s%s: Error: The \"%s/\" directory does not have execute permission.%s"
              % (RED, PATH, TRAFFIC_ANA_DIR, END), file=sys.stderr)
if errors:
    exit(1)

for f in SCRIPTS:
    if not os.path.isfile(f):
        errors = True
        print("%s%s: Error: The script \"%s\" cannot be found.\n"
              "     Please make sure it is in the same directory as \"%s\".%s"
              % (RED, PATH, f, PATH, END), file=sys.stderr)
    elif not os.access(f, os.R_OK):
        errors = True
        print("%s%s: Error: The script \"%s\" does not have read permission.%s"
              % (RED, PATH, f, END), file=sys.stderr)

if errors:
    exit(1)

from trafficAnalyzer import *
from trafficAnalyzer import Constants as c

args = [] #Main args
plots = [] #Graph args
devices = None


#isError is either 0 or 1
def print_usage(is_error):
    print(c.USAGE_STM, file=sys.stderr) if is_error else print(c.USAGE_STM)
    exit(is_error)


def check_dir(direc, description=""):
    errors = False
    if direc == "":
        direc = "."
    if not os.path.isdir(direc):
        errors = True
        if description == "":
            print(c.MISSING % (direc, "directory"), file=sys.stderr)
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


def check_files(direc, files, is_geo, description=""):
    errors = check_dir(direc)
    if not errors:
        missing_file = False
        for f in files:
            if not os.path.isfile(f):
                missing_file = errors = True
                if description == "":
                    print(c.MISSING % (f, "file"), file=sys.stderr)
                else:
                    print(c.INVAL % (description, f, "file"), file=sys.stderr)
            elif not os.access(f, os.R_OK):
                errors = True
                print(c.NO_PERM % ("file", f, "read"), file=sys.stderr)

        if missing_file and is_geo:
            print(c.DOWNLOAD_DB, file=sys.stderr)

    return errors


def main():
    global args, plots, devices

    start_time = time.time()

    print("Performing destination analysis...")
    print("Running %s..." % c.PATH)
    print("Start time: %s\n" % time.strftime("%A %d %B %Y %H:%M:%S %Z", time.localtime(start_time)))

    #Check that GeoLite2 databases and aux scripts exist and have proper permissions
    errors = check_files(GEO_DIR, [GEO_DB_CITY, GEO_DB_COUNTRY], True)
    errors = check_files(AUX_DIR, [IP_TO_ORG, IP_TO_COUNTRY], False) or errors
    if errors:
        exit(1)

    #Options
    parser = argparse.ArgumentParser(usage=c.USAGE_STM, add_help=False)
    parser.add_argument("-i", dest="in_dir", default="")
    parser.add_argument("-m", dest="mac_addr", default="")
    parser.add_argument("-d", dest="dev", default="")
    parser.add_argument("-c", dest="dev_list", default=DEST_DIR+"/aux/devices_us.txt")
    parser.add_argument("-a", dest="ip_addr")
    parser.add_argument("-s", dest="hosts_dir", default="")
    parser.add_argument("-b", dest="lab", default="")
    parser.add_argument("-e", dest="experiment", default="")
    parser.add_argument("-w", dest="network", default="")
    parser.add_argument("-t", dest="no_time_shift", action="store_true", default=False)
    parser.add_argument("-y", dest="find_diff", action="store_true", default=False)
    parser.add_argument("-f", dest="fig_dir", default="figures")
    parser.add_argument("-o", dest="out_file", default="results.csv")
    parser.add_argument("-n", dest="num_proc", default="1")
    parser.add_argument("-g", dest="plots")
    parser.add_argument("-p", dest="protocols", default="")
    parser.add_argument("-l", dest="ip_locs", default="")
    parser.add_argument("-r", dest="ip_attrs", default="")
    parser.add_argument("-h", dest="help", action="store_true", default=False)

    #Parse Arguments
    args = parser.parse_args()

    if args.help:
        print_usage(0)
   
    #Parse plot options
    if args.plots is not None:
        plots = [{"plt": val.strip().lower()} for val in args.plots.split(",")]

    headings = ["prot", "ip_loc", "ip_attr"]
    plot_len = len(plots)
    for header, attrs in zip(headings, [args.protocols, args.ip_locs, args.ip_attrs]):
        vals = [val.strip().lower() for val in attrs.split(",")]
        if len(vals) < plot_len:
            vals.extend([""] * (plot_len - len(vals)))

        for plt, val in zip(plots, vals):
            plt[header] = val
        
    for plt in plots:
        if "pieplot" == plt["plt"]:
            print(c.PIE_STM, file=sys.stderr)
            exit(1)

        if "ripecountry" == plt["ip_loc"]:
            print(c.RP_STM, file=sys.stderr)
            exit(1)

    #Error checking command line args and files
    errors = False
    #check -i input dir
    if args.in_dir == "":
        errors = True 
        print(c.NO_IN_DIR, file=sys.stderr)
    elif check_dir(args.in_dir, "Input pcap directory"):
        errors = True

    #check -m mac address
    no_mac_device = False
    valid_device_list = True
    if args.mac_addr == "" and args.dev == "":
        no_mac_devce = errors = True
        print(c.NO_MAC, file=sys.stderr)
    elif args.mac_addr != "":
        args.mac_addr = Device.Device.normaliseMac(args.mac_addr).lower()
        if not re.match("([0-9a-f]{2}[:]){5}[0-9a-f]{2}$", args.mac_addr):
            errors = True
            print(c.INVAL_MAC % args.mac_addr, file=sys.stderr)

    #check -c device list
    if not args.dev_list.endswith(".txt"):
        errors = True
        print(c.WRONG_EXT % ("Device list", "text (.txt)", args.dev_list), file=sys.stderr)
        valid_device_list = False
    elif check_files(os.path.dirname(args.dev_list), [args.dev_list], False, "Device list"):
        errors = True
        valid_device_list = False

    if valid_device_list:
        devices = Device.Devices(args.dev_list)

    #check -d device
    if (args.mac_addr != "" or args.dev != "") and valid_device_list:
        if args.mac_addr == "" and not no_mac_device:
            if not devices.deviceInList(args.dev):
                errors = True
                print(c.NO_DEV % (args.dev, args.dev_list), file=sys.stderr)
            else:
                args.mac_addr = devices.getDeviceMac(args.dev)

    #check -s hosts dir
    if args.hosts_dir != "" and check_dir(args.hosts_dir, "Hosts directory"):
        errors = True

    #check -o output csv
    if not args.out_file.endswith(".csv"):
        errors = True
        print(c.WRONG_EXT % ("Output file", "CSV (.csv)", args.out_file), file=sys.stderr)

    #check -n number processes
    bad_proc = True
    num_proc = 1
    try:
        if int(args.num_proc) > 0:
            bad_proc = False
            num_proc = int(args.num_proc)
    except ValueError:
        pass

    if bad_proc:
        errors = True
        print(c.NON_POS % args.num_proc, file=sys.stderr)

    plot_types = ["", "stackplot", "lineplot", "scatterplot", "barplot", "pieplot", "barhplot"]
    ip_loc_types = ["country", "host", "tsharkhost", "ripecountry", "ip"]
    ip_attr_types = ["addrpcktsize", "addrpcktnum"]
    for plt in plots:
        if plt["ip_loc"] == "":
            plt["ip_loc"] = "ip"

        if plt["ip_attr"] == "":
            plt["ip_attr"] = "addrpcktsize"

        #check -g plot type
        if plt["plt"] not in plot_types:
            errors = True
            print(c.INVAL_PLT % plt["plt"], file=sys.stderr)
        elif plt["plt"] != "":
            #check -p protocol
            if plt["prot"] == "":
                errors = True
                print(c.NO_PROT % plt["plt"], file=sys.stderr)
            else:
                try:
                    plt["prot_snd"], plt["prot_rcv"] = plt["prot"].split(".")
                    plt["prot_snd"] += "-snd"
                    plt["prot_rcv"] += "-rcv"
                    del plt["prot"]
                except ValueError:
                    errors = True
                    print(c.INVAL_PROT % (plt["prot"], plt["plt"]), file=sys.stderr)

            #check -l ip location
            if plt["ip_loc"] not in ip_loc_types:
                errors = True
                print(c.INVAL_LOC % (plt["ip_loc"], plt["plt"]), file=sys.stderr)
    
            #check -r ip attribute
            if plt["ip_attr"] not in ip_attr_types:
                errors = True
                print(c.INVAL_ATTR % (plt["ip_attr"], plt["plt"]), file=sys.stderr)

    if errors:
        print_usage(1)
    #End error checking

    #Create output file if it doesn't exist
    if not os.path.isfile(args.out_file):
        DataPresentation.DomainExport.create_csv(args.out_file)

    #Create the groups to run analysis with processes
    raw_files = [ [] for _ in range(num_proc) ]

    index = 0
    # Split the pcap files into num_proc groups
    for root, dirs, files in os.walk(args.in_dir):
        for filename in files:
            if filename.endswith(".pcap") and not filename.startswith("."):
                raw_files[index].append(root + "/" + filename)
                index += 1
                if index >= num_proc:
                    index = 0

    gc.collect()

    print("Analyzing input pcap files...")
    # run analysis with num_proc processes
    procs = []
    for pid, files in enumerate(raw_files):
        p = Process(target=run, args=(pid, files))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    DataPresentation.DomainExport.sort_csv(args.out_file)

    end_time = time.time()
    print("\nEnd time: %s" % time.strftime("%A %d %B %Y %H:%M:%S %Z", time.localtime(end_time)))

    #Calculate elapsed time
    sec = round(end_time - start_time)
    hrs = sec // 3600
    if hrs != 0:
        sec = sec - hrs * 3600

    minute = sec // 60
    if minute != 0:
        sec = sec - minute * 60

    print("Elapsed time: %s hours %s minutes %s seconds" % (hrs, minute, sec))
    print("\nDestintaion analysis finished.")


def run(pid, pcap_files):
    files_len = len(pcap_files)
    for idx, f in enumerate(pcap_files):
        perform_analysis(pid, idx + 1, files_len, f)
        gc.collect()


def perform_analysis(pid, idx, files_len, pcap_file):
    print("P%s (%s/%s): Processing pcap file \"%s\"..." % (pid, idx, files_len, pcap_file))
    cap = pyshark.FileCapture(pcap_file, use_json=True)
    Utils.sysUsage("PCAP file loading")
    cap.close()
    base_ts = 0
    try:
        if args.no_time_shift:
            cap[0]
        else:
            base_ts = float(cap[0].frame_info.time_epoch)
    except KeyError:
        print(c.NO_PCKT % pcap_file, file=sys.stderr)
        return

    node_id = Node.NodeId(args.mac_addr, args.ip_addr)
    node_stats = Node.NodeStats(node_id, base_ts, devices)

    print("  P%s: Processing packets..." % pid)
    try:
        for packet in cap:
            node_stats.processPacket(packet)
    except:
        print("  %sP%s: Error: There is something wrong with \"%s\". Skipping file.%s"
              % (RED, pid, pcap_file, END), file=sys.stderr)
        return

    del cap

    Utils.sysUsage("Packets processed")

    print("  P%s: Mapping IP to host..." % pid)
    ip_map = IP.IPMapping()
    if args.hosts_dir != "":
        host_file = args.hosts_dir + "/" + os.path.basename(pcap_file)[:-4] + "txt"
        ip_map.extractFromFile(pcap_file, host_file)
    else:
        ip_map.extractFromFile(pcap_file)

    ip_map.loadOrgMapping(IP_TO_ORG)
    ip_map.loadCountryMapping(IP_TO_COUNTRY)

    Utils.sysUsage("TShark hosts loaded")

    print("  P%s: Generating CSV output..." % pid)
    de = DataPresentation.DomainExport(node_stats.stats.stats, ip_map, GEO_DB_CITY, GEO_DB_COUNTRY)
    de.loadDiffIPFor("eth") if args.find_diff else de.loadIPFor("eth")
    de.loadDomains(args.dev, args.lab, args.experiment, args.network, pcap_file, str(base_ts))
    de.exportDataRows(args.out_file)

    print("  P%s: Analyzed data from \"%s\" successfully written to \"%s\""
          % (pid, pcap_file, args.out_file))

    Utils.sysUsage("Data exported")

    if len(plots) != 0:
        print("  P%s: Generating plots..." % pid)
        pm = DataPresentation.PlotManager(node_stats.stats.stats, plots)
        pm.ipMap = ip_map
        pm.generatePlot(pid, pcap_file, args.fig_dir, GEO_DB_CITY, GEO_DB_COUNTRY)
        Utils.sysUsage("Plots generated")


if __name__ == "__main__":
    main()

