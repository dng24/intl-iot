import sys

class Layer(object):
    ETH = 'eth'
    ARP = 'arp'
    IP = 'ip'
    ICMP = 'icmp'
    UDP = 'udp'
    TCP = 'tcp'
    BOOTP = 'bootp'
    NTP = 'ntp'
    HTTP = 'http'
    DHCP = 'dhcp'
    LLC = 'llc'
    DNS = 'dns'


class Direction(object):
    SND = 'snd'
    RCV = 'rcv'

RED = "\033[31;1m"
END = "\033[0m"
PATH = sys.argv[0]
BEG = RED + PATH + ": Error: "

USAGE_STM = """
Usage: python3 {prog_name} -i IN_DIR {{-m MAC_ADDR | -d DEV}} [OPTION]... [-g PLOTS -p PROTOS [GRAPH_OPTION]...]

Performs destination analysis on several pcap files. Produces a CSV file detailing
the organizations that traffic in the pcap files have been to and the number of
packets that were sent and received from those organizations. The program also
can produce plots of this data.

Example: python3 {prog_name} -i iot-data/us/appletv/ -m 7c:61:66:10:46:18 -g StackPlot,LinePlot -p eth.eth,eth.eth

Options:
  -i IN_DIR   path to the directory containing input pcap files to be analyzed;
                option required
  -m MAC_ADDR MAC address of the device that generated the data in IN_DIR;
                option required if DEV not specified
  -d DEV      name of the device used to generate the data in IN_DIR;
                option required if MAC_ADDR not specified
  -c DEV_LIST path to a text file containing the names of devices along with
                the devices' MAC addresses; each device is on its own line,
                with each line having the format: "[MAC_ADDR] [DEVICE]"
                (Default = aux/devices_us.txt)
  -a IP_ADDR  IP address of the device used to create the data in IN_DIR
  -s HOSTS    path to a directory containing text (.txt) files produced by TShark
                extracting hosts from IN_DIR; host filenames should match input
                pcap but with .txt extension; generate host file using "tshark -r
                [input_pcap] -q -z hosts > HOSTS/[input_pcap].txt"
  -b LAB      name of the lab that the pcap files in IN_DIR were generated in
  -e EXP      name of the experiment that the pcap files in IN_DIR are a part of
  -w NETWORK  name of the network
  -t          do not perform a time shift
  -y          find domains which do not reply
  -f FIG_DIR  path to a directory to place generated plots; will be generated
                if it does not currently exist (Default = figures/)
  -o OUT_CSV  path to the output CSV file; if it exists, results will be
                appended, else, it will be created (Default = results.csv)
  -n NUM_PROC number of processes to use to analyze the pcap files; must be a
                positive integer (Default = 1)
  -h          print this usage statement and exit

Graph options:
  -g PLOTS  comma-delimited list of graph types to plot; choose from StackPlot,
              LinePlot, ScatterPlot, BarPlot, PiePlot, or BarHPlot; PiePlot
              currently does not function properly
  -p PROTOS comma-delimited list of protocols that should be analyzed; for each
              plot specified in PLOTS, there should be two protocols in this
              period-delimited format: "[send_protocol].[receive_protocol]"
  -l IPLOCS comma-delimited list of methods to map an IP address to a host
              or country for each plot; choose from Country, Host, IP,
              RipeCountry, or TSharkHost; RipeCountry currently does not
              function properly (Default = IP)
  -r IPATTS comma-delimited list of IP packet attributes to display for each
              plot; choose from either addrPcktSize or addrPcktNum (Default =
              addrPcktSize)

Notes:
 - The position of an argument in the comma-delimited lists in the graph options
     determine which graph that argument will affect. For example, in the
     command: "-g LinePlot,StackPlot -p eth.eth,eth.eth -l ,IP -r addrPcktNum",
     two plots are produced:
     1) a line plot (-g) displaying the number of packets (-r) with Ethernet as
          both the send and receive protocols (-p)
     2) a horizontal bar plot (-g) with Ethernet as both the send and receive
          protocols (-p) using the IP method to map the IP addresses to a host (-l)
 - Only pie plots and horizontal bar plots are affected by the -l and -r options
 - All plots specified will be placed in one PNG file named:
     "[sanitized_IN_DIR_path]_[plot_names].png"

For more information, see the README.""".format(prog_name=PATH)

MISSING = BEG + "The \"%s\" %s is missing." + END
DOWNLOAD_DB = RED + "Please go to the README for instructions to download the databases.\n"\
              "    If the databases are already downloaded, please make sure they are\n"\
              "    in the correct directory." + END
NO_PERM = BEG + "The %s \"%s\" does not have %s permission." + END
PIE_STM = "***PiePlot currently does not function properly. Please choose a different plot.\n"\
          "   Currently available plots: BarHPlot, BarPlot, LinePlot, ScatterPlot, StackPlot"
RP_STM = "***RipeCountry currently does not function properly. Please choose a different IP"\
         " mapping method.\n   Currently available methods: Country, Host, IP, TSharkHost"

INVAL = BEG + "%s \"%s\" is not a %s." + END
WRONG_EXT = BEG + "%s must be a %s file. Received \"%s\"" + END

NO_IN_DIR = BEG + "Pcap input directory (-i) required." + END
NO_MAC = BEG + "Either the MAC address (-m) or device name (-d) must be specified." + END
INVAL_MAC = BEG + "Invalid MAC address \"%s\". Valid format xx:xx:xx:xx:xx:xx" + END
NO_DEV = BEG + "The device \"%s\" does not exist in the device list \"%s\"." + END
NON_POS = BEG + "The number of processes must be a positive integer. Received \"%s\"." + END

INVAL_PLT = BEG + "\"%s\" is not a valid plot type.\n    Must be either \"BarHPlot\","\
            " \"BarPlot\", \"LinePlot\", \"PiePlot\", \"ScatterPlot\", or \"StackPlot\"." + END
NO_PROT = BEG + "A protocol (-p) must be specified for \"%s\"." + END
INVAL_PROT = BEG + "Invalid set of protocols \"%s\" for \"%s\".\n"\
             "    Protocols should be in the form \"[send].[receive]\"." + END
INVAL_LOC = BEG + "Invalid IP locator method \"%s\" for \"%s\".\n    Must be"\
            " either \"Country\", \"Host\", \"IP\", \"TSharkHost\", or \"RipeCountry\"." + END
INVAL_ATTR = BEG + "Invalid IP Attribute \"%s\" for \"%s\".\n"\
             "    Must be either \"addrPcktNum\" or \"addrPcktSize\"." + END

NO_PCKT = BEG + "The file \"%s\" does not contain any packets." + END

