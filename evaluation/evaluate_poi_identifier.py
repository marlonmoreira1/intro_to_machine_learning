#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys_wd.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=.3,random_state=42)

###clf = DecisionTreeClassifier()
###clf.fit(features_train,labels_train)
###pred = clf.predict(features_test)
###print(accuracy_score(pred,labels_test))
###print(len(features_test))
###print(len([e for e in labels_test if e == 1.0]))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print(confusion_matrix(true_labels, predictions))
print(precision_score(true_labels,predictions))

print(recall_score(true_labels,predictions))