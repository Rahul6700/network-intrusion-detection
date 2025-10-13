import kagglehub
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

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

# mapping all the 40 something attacks to those 5 attacks using a lambda function
train_df['class'] = train_df['class'].map(lambda x: attack_map[x] if x in attack_map else 'normal')
test_df['class'] = test_df['class'].map(lambda x: attack_map[x] if x in attack_map else 'normal')

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

# now we scale the dataset
# say one col has values like 5000,6000,4300,7500 and another col has values like 0.2, 0.4, 0.009, 0.5
# the col with bigger values will overshadow the smaller val's -> example when using distance formula -> sqrt(5000^2 + 0.3^2), the 0.3 is negligible
# so the 2nd col will barely affect the models prediction
# so we scale all the numerical val col's, so everything has small values between 0 and 1 or something like that -> hence giving equal weightage to everything
# we'll import and use standardScalar for this

# all the col's which have numerical values
numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

scaler = StandardScaler() # create an scaler obj

# applying the scaler to all the numeric col's in both the testing and training input set
inp_train[numeric_cols] = scaler.fit_transform(inp_train[numeric_cols])
inp_test[numeric_cols] = scaler.fit_transform(inp_test[numeric_cols])

# print(inp_train.shape, inp_test.shape)
# inp_train.head()

# now to train the model's
# starting off with decision tree

# For faster iteration, let take a subset of 25,000 samples from the training set
inp_train_subset = inp_train.sample(n=25000, random_state=42)
out_train_subset = out_train.loc[inp_train_subset.index]

# decision tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(inp_train_subset, out_train_subset)

# make predictions on the test set
dt_predictions = dt_model.predict(inp_test)

print("decision tree accuracy: ", accuracy_score(out_test, dt_predictions))
print(classification_report(out_test, dt_predictions))

print("- - - - - - - - - - - - - - - - - - - - - - - - -- ")

# random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(inp_train_subset, out_train_subset)

# make predictions on the test set
rf_predictions = rf_model.predict(inp_test)

print("Random Forest Accuracy:", accuracy_score(out_test, rf_predictions))
print(classification_report(out_test, rf_predictions))

print("- - - - - - - - - - - - - - - - - - - - - - - - -- ")

# Gaussian Naive Bayes (only on raw numerical features, before scaling)
gnb_model = GaussianNB()

# Extract the numerical columns directly from the ORIGINAL train_df and test_df (not the encoded or scaled ones)
inp_train_gnb = train_df[numeric_cols]
inp_test_gnb = test_df[numeric_cols]

out_train_gnb = train_df['class']
out_test_gnb = test_df['class']

# Train the GNB model
gnb_model.fit(inp_train_gnb, out_train_gnb)

# Make predictions
gnb_predictions = gnb_model.predict(inp_test_gnb)

# Evaluate the model
print("Gaussian Naive Bayes Accuracy:", accuracy_score(out_test_gnb, gnb_predictions))
print(classification_report(out_test_gnb, gnb_predictions))

