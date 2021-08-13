import numpy as np

#import datafile
data_file = open('Dataset/dataset.txt','r')
data_string = data_file.read()
data_rows = data_string.split('\n')
data_arr = np.zeros((len(data_rows),31))

for i in range(len(data_rows)):
    vals = data_rows[i].split(",")
    data_arr[i,:] = vals[:]

#split into data and labels
data_full = data_arr[:,0:30]
labels_full = data_arr[:,30]

data_full = np.array(data_full).reshape(11055,30)
labels_full =np.array(labels_full).reshape(11055,1)

#first we randomly shuffle the data
s = np.arange(len(data_full))
np.random.shuffle(s)
data_full = data_full[s]
labels_full = labels_full[s]
 
# now split into test vs training vs validation data
#there is 11055 data points
#we train using 9000 points
#we test using 1555 points
#we validate on 500 points
train_data = data_full[0:9000]
train_labels = labels_full[0:9000]

test_data = data_full[9000:10555]
test_labels = labels_full[9000:10555]

val_data = data_full[10555:]
val_labels = labels_full[10555:]

feat_file = open('Dataset/featureNames.txt','r')
feat_string = feat_file.read()
features = np.array(feat_string.split('\n'))