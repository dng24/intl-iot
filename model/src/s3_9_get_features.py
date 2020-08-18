import os
import sys
import time
import copy
from multiprocessing import Process
from multiprocessing import Manager

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels import robust

import Constants as c

cols_feat = ["start_time", "end_time", "meanBytes", "minBytes", "maxBytes", "medAbsDev",
             "skewLength", "kurtosisLength", "q10", "q20", "q30", "q40", "q50", "q60", "q70",
             "q80", "q90", "spanOfGroup", "meanTBP", "varTBP", "medianTBP", "kurtosisTBP",
             "skewTBP", "network_to", "network_from", "network_both", "network_to_external",
             "network_local", "anonymous_source_destination", "device", "state", "hosts"]

#In: in_decoded_dir out_features_dir [num_processes]
#Out: CSV w/ header: headings above in cols_feat containing device and state labels

random_ratio = 0.8
num_per_exp = 10

#is_error is either 0 or 1
def print_usage(is_error):
    print(c.GET_FEAT_USAGE, file=sys.stderr) if is_error else print(c.GET_FEAT_USAGE)
    exit(is_error)


def main():
    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % c.PATH)
    
    #error checking
    #check that there are 2 or 3 args
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(c.WRONG_NUM_ARGS % (2, (len(sys.argv) - 1)), file=sys.stderr)
        print_usage(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    str_num_proc = sys.argv[3] if len(sys.argv) == 4 else "1"

    #check in_dir
    errors = False
    if not os.path.isdir(in_dir):
        errors = True
        print(c.INVAL % ("Decoded pcap directory", in_dir, "directory"), file=sys.stderr)
    else:
        if not os.access(in_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("decoded pcap directory", in_dir, "read"), file=sys.stderr)
        if not os.access(in_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("decoded pcap directory", in_dir, "execute"), file=sys.stderr)

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

    print("Input files located in: %s\nOutput files placed in: %s\n" % (in_dir, out_dir))
    
    group_size = 50
    dict_dec = dict()
    dircache = os.path.join(out_dir, 'caches')
    if not os.path.exists(dircache):
        os.system('mkdir -pv %s' % dircache)
    #Parse input file names
    #in_dir/dev_dir/act_dir/dec_file
    for dev_dir in os.listdir(in_dir):
        if dev_dir.startswith("."):
            continue
        training_file = os.path.join(out_dir, dev_dir + '.csv') #Output file
        #Check if output file exists
        if os.path.exists(training_file):
            print('Features for %s prepared already in %s' % (dev_dir, training_file))
            continue
        full_dev_dir = os.path.join(in_dir, dev_dir)
        for act_dir in os.listdir(full_dev_dir):
            full_act_dir = os.path.join(full_dev_dir, act_dir)
            for dec_file in os.listdir(full_act_dir):
                full_dec_file = os.path.join(full_act_dir, dec_file)
                if not full_dec_file.endswith(".txt"):
                    print(c.WRONG_EXT % ("Decoded file", "text (.txt)", full_dec_file), file=sys.stderr)
                    continue
                if not os.path.isfile(full_dec_file):
                    print(c.INVAL % ("Decoded file", full_dec_file, "file"), file=sys.stderr)
                    continue
                if not os.access(full_dec_file, os.R_OK):
                    print(c.NO_PERM % ("decoded file", full_dec_file, "read"), file=sys.stderr)
                    continue

                if 'companion' in dec_file:
                    state = '%s_companion_%s' % (act_dir, dev_dir)
                    device = dec_file.split('.')[-2] # the word before pcap
                else:
                    state = act_dir
                    device = dev_dir
                feature_file = os.path.join(out_dir, 'caches', device + '_' + state
                               + '_' + dec_file[:-4] + '.csv') #Output cache files
                #the file, along with some data about it
                paras = (full_dec_file, feature_file, group_size, device, state)
                #Dict contains devices that do not have an output file
                if device not in dict_dec:
                    dict_dec[device] = []
                dict_dec[device].append(paras)

    devices = "None" if len(dict_dec) == 0 else ", ".join(dict_dec.keys())
    print("Feature files to be generated from the following devices:", devices)

    for device in dict_dec:
        training_file = os.path.join(out_dir, device + '.csv')
        list_paras = dict_dec[device]

        #create groups to run with processes
        params_arr = [ [] for _ in range(num_proc) ]

        #create results array
        results = Manager().list()

        #split pcaps into num_proc groups
        for i, paras in enumerate(list_paras):
            params_arr[i % num_proc].append(paras)

        procs = []
        for paras_list in params_arr:
            p = Process(target=run, args=(paras_list, results))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

        if len(results) > 0:
            pd_device = pd.concat(results, ignore_index=True) #Concat all cache files together
            pd_device.to_csv(training_file, index=False) #Put in CSV file
            print("Results concatenated to %s" % training_file)


def run(paras_list, results):
    for paras in paras_list:
        full_dec_file = paras[0]
        feature_file = paras[1]
        device = paras[3]
        state = paras[4]
        tmp_data = load_features_per_exp(full_dec_file, feature_file, device, state)
        if tmp_data is None or len(tmp_data) == 0:
            continue
        results.append(tmp_data)


def load_features_per_exp(dec_file, feature_file, device_name, state):
    #Load data from cache
    if os.path.exists(feature_file):
        print('    Load from %s' % feature_file)
        return pd.read_csv(feature_file)

    #Attempt to extract data from input files if not in previously-generated cache files
    feature_data = extract_features(dec_file, feature_file, device_name, state)
    if feature_data is None or len(feature_data) == 0: #Can't extract from input files
        print('No data or features from %s' % dec_file)
        return
    else: #Cache was generated; save to file
        feature_data.to_csv(feature_file, index=False)
    return feature_data


#Create CSV cache file
def extract_features(dec_file, feature_file, device_name, state):
    col_feat = cols_feat
    pd_obj_all = pd.read_csv(dec_file, sep="\t")
    pd_obj = pd_obj_all.loc[:, :]
    num_total = len(pd_obj_all)
    if pd_obj is None or num_total < 10:
        return
    print("In decoded: %s\n  Out features: %s" % (dec_file, feature_file))
    feature_data = pd.DataFrame()
    num_pkts = int(num_total * random_ratio)
    for di in range(0, num_per_exp):
        random_indices = list(np.random.choice(num_total, num_pkts))
        random_indices = sorted(random_indices)
        pd_obj = pd_obj_all.loc[random_indices, :]
        d = compute_tbp_features(pd_obj, device_name, state)
        feature_data = feature_data.append(pd.DataFrame(data=[d], columns=col_feat))
    return feature_data


#Use Pandas to perform stat analysis on raw data
def compute_tbp_features(pd_obj, device_name, state):
    start_time = pd_obj.ts.min()
    end_time = pd_obj.ts.max()
    group_len = end_time - start_time
    meanBytes = pd_obj.frame_len.mean()
    minBytes = pd_obj.frame_len.min()
    maxBytes = pd_obj.frame_len.max()
    medAbsDev = robust.mad(pd_obj.frame_len)
    skewL = skew(pd_obj.frame_len)
    kurtL = kurtosis(pd_obj.frame_len)
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentiles = np.percentile(pd_obj.frame_len, p)
    kurtT = kurtosis(pd_obj.ts_delta)
    skewT = skew(pd_obj.ts_delta)
    meanTBP = pd_obj.ts_delta.mean()
    varTBP = pd_obj.ts_delta.var()
    medTBP = pd_obj.ts_delta.median()
    network_to = 0 # Network going to 192.168.10.204, or home.
    network_from = 0 # Network going from 192.168.10.204, or home.
    network_both = 0 # Network going to/from 192.168.10.204, or home both present in source.
    network_local = 0
    network_to_external = 0 # Network not going to just 192.168.10.248.
    anonymous_source_destination = 0

    for i, j in zip(pd_obj.ip_src, pd_obj.ip_dst):
        if i == "192.168.10.204":
            network_from += 1
        elif j == "192.168.10.204":
            network_to += 1
        elif i == "192.168.10.248,192.168.10.204":
            network_both += 1
        elif j == "192.168.10.204,129.10.227.248":
            network_local += 1
        elif j != "192.168.10.204" and i != "192.168.10.204":
            network_to_external += 1
        else:
            anonymous_source_destination += 1

    #host is either from the host column, or the destination IP if host doesn't exist
    hosts = set([ str(pd_obj.ip_dst.iloc[i]) if host == "" else host for i, host in enumerate(pd_obj.host.fillna("")) ])
    
    d = [start_time, end_time, group_len, meanBytes, minBytes, maxBytes, medAbsDev, skewL, kurtL,
         percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4],
         percentiles[5], percentiles[6], percentiles[7], percentiles[8], meanTBP, varTBP, medTBP,
         kurtT, skewT, network_to, network_from, network_both, network_to_external, network_local,
         anonymous_source_destination, device_name, state, ";".join(hosts)]
    return d


if __name__ == '__main__':
    main()

