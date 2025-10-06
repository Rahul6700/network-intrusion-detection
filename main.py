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

#loading the data in pandas
test_df = pd.read_csv(test_file, header=None)
train_df = pd.read_csv(train_file, header=None)

print(train_df.head());


