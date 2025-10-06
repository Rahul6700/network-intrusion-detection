import kagglehub
import numpy as np
import pandas as pd
import os

path = kagglehub.dataset_download("hassan06/nslkdd")
print("Path to dataset files:", path)
os.listdir(path)

#using the train+.txt and the test.txt
train_file = os.path.join(path, "KDDTrain+.txt")
test_file = os.path.join(path, "KDDTest+.txt")

#Manually label the dataset
col_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
            'root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
             'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
             'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
             'dst_host_rerror_rate','dst_host_srv_rerror_rate','class','difficulty']

#loading the data in pandas
test_df = pd.read_csv(test_file, header=None, names = col_names)
train_df = pd.read_csv(train_file, header=None, names = col_names)

#implement target variable grouping
# need to reduce the many different outputs to 5 major ones
# using a hashmap for O(1) lookups
# Each specific attack type is a key and the values are their corresponding high level type
attack_map = {
    # DoS mappings
    'back':'DoS', 'land':'DoS', 'neptune':'DoS', 'pod':'DoS', 'smurf':'DoS', 'teardrop':'DoS',
    'apache2':'DoS', 'udpstorm':'DoS', 'processtable':'DoS', 'worm':'DoS',
    # Probe
    'satan':'Probe', 'ipsweep':'Probe', 'nmap':'Probe', 'portsweep':'Probe', 'mscan':'Probe', 'saint':'Probe',
    # R2L
    'ftp_write':'R2L', 'guess_passwd':'R2L', 'imap':'R2L', 'phf':'R2L', 'multihop':'R2L', 'warezmaster':'R2L',
    'warezclient':'R2L', 'spy':'R2L', 'xlock':'R2L', 'xsnoop':'R2L', 'snmpguess':'R2L', 'snmpgetattack':'R2L',
    'httptunnel':'R2L', 'sendmail':'R2L', 'named':'R2L',
    # U2R
    'buffer_overflow':'U2R', 'loadmodule':'U2R', 'rootkit':'U2R', 'perl':'U2R', 'sqlattack':'U2R',
    'xterm':'U2R', 'ps':'U2R'
}

print(train_df.shape())
print(train_df.head())

