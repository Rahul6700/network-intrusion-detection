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

#todo-> target variable grouping
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

# now we do one hot encoding, since our model cannot take string inputs like "tcp" or "udp"
# we do one hot encoding and convert these to binary digits
train_df_encoded = pd.get_dummies(train_df, columns=['protocol_type', 'service', 'flag'])
test_df_encoded = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'])

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

#TODO: target variable grouping
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

# now we do one hot encoding, since our model cannot take string inputs like "tcp" or "udp"
# we do one hot encoding and convert these to binary digits
train_df_encoded = pd.get_dummies(train_df, columns=['protocol_type', 'service', 'flag'])
test_df_encoded = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'])

#print(list(test_df_encoded.columns) == list(train_df_encoded.columns)) # this is returning false, so there is a column mismatch now due to one hot encoding

# so we add all the column's in the training set that are not there in the test set, to the test set (fill the values with 0)
# this function does the same for us -> .reindex()
test_df_encoded = train_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)

#print(list(test_df_encoded.columns) == list(train_df_encoded.columns)) # now this returns true

# next we split our dataset into the input features (connection, server type, etc) and the output values (the type of attack) -> do this for both training and testing set
#print(list(train_df_encoded.columns))
# removing 'class' since its our final output and difficulty cuz its redundent
inp_train = train_df_encoded.drop(['class','difficulty'], axis = 1)
inp_test = test_df_encoded.drop(['class','difficulty'], axis = 1)

# output set has only class
out_train = train_df_encoded['class']
out_test = test_df_encoded['class']

#inp_train.head()
#out_train.head()

#print(list(test_df_encoded.columns) == list(train_df_encoded.columns)) # this is returning false, so there is a column mismatch now due to one hot encoding

# so we add all the column's in the training set that are not there in the test set, to the test set (fill the values with 0)
# this function does the same for us -> .reindex()
test_df_encoded = train_df_encoded.reindex(columns=train_df_encoded.columns, fill_value=0)

#print(list(test_df_encoded.columns) == list(train_df_encoded.columns)) # now this returns true

# next we split our dataset into the input features (connection, server type, etc) and the output values (the type of attack) -> do this for both training and testing set
#print(list(train_df_encoded.columns))
# removing 'class' since its our final output and difficulty cuz its redundent
inp_train = train_df_encoded.drop(['class','difficulty'], axis = 1)
inp_test = test_df_encoded.drop(['class','difficulty'], axis = 1)

# output set has only class
out_train = train_df_encoded['class']
out_test = test_df_encoded['class']

#inp_train.head()
#out_train.head()

