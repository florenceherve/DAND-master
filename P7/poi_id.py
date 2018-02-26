#!/usr/bin/python

import os
import sys
import nltk
import scipy
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from time import time
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier #decision tree
from sklearn.cross_validation import train_test_split #for splitting the data into training and test sets

# For evaluation metrics:
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import sys
import pickle
sys.path.append("../tools/")



from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features= ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features= ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']

features_list = poi_label + financial_features + email_features
print features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### Visualize outliers
features = ['salary', 'total_payments']
data = featureFormat(data_dict,features)

for point in data:
    salary = point[0]
    total_payments = point[1]
    matplotlib.pyplot.scatter(total_payments, salary)

matplotlib.pyplot.xlabel("total_payments")
matplotlib.pyplot.ylabel("salary")
matplotlib.pyplot.show()

### Delete "Total" and "The Travel Agency in the Park" lines
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

### Visualize new scatterplot with the those two outliers removed
data = featureFormat(data_dict,features)

for point in data:
    salary = point[0]
    total_payments = point[1]
    matplotlib.pyplot.scatter(total_payments, salary)

matplotlib.pyplot.xlabel("total_payments")
matplotlib.pyplot.ylabel("salary")
matplotlib.pyplot.show()

### Identify lines with no values:
for person in data_dict:
    if data_dict[person]["total_payments"] == "NaN":
        if data_dict[person]["total_stock_value"] == "NaN":
            print person, data_dict[person]["poi"], data_dict[person]['to_messages']

# Delete null lines
data_dict.pop("CHAN RONNIE",0)
data_dict.pop("POWERS WILLIAM",0)
data_dict.pop("LOCKHART EUGENE E",0)

### Visualize salary vs loan advances
features = ['salary', 'loan_advances', 'poi']
data = featureFormat(data_dict,features)

for point in data:
    salary = point[0]
    loan_advances = point[1]
    mark_poi = point[2]
    if mark_poi:
        color = 'red'
    else:
        color = 'blue'
    matplotlib.pyplot.scatter(loan_advances, salary, color = color)

matplotlib.pyplot.xlabel("loan_advances")
matplotlib.pyplot.ylabel("salary")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

import math
def computeFraction(ratio1, ratio2):
    if math.isnan(float(ratio1)) == True:
        fraction = 0.
    if math.isnan(float(ratio2)) == True:
        fraction = 0.
    if ratio2 == 0:
        fraction = 0.
    else:
        fraction = float(ratio1) / float(ratio2)
    return fraction

def addNumbers(add1, add2):
    if str(add1).lower() == 'nan' and str(add2).lower() == 'nan':
        add = 0.
    if str(add1).lower() == 'nan':
        add = float(add2)
    if str(add2).lower() == 'nan':
        add = float(add1)
    if math.isnan(float(add1)) == True and math.isnan(float(add2)) == True:
        add = 0.
    if math.isnan(float(add1)) == True:
        add = float(add2)
    if math.isnan(float(add2)) == True:
        add = float(add1)
    else:
        add = float(add1 + add2)
    return add

for person in my_dataset:
    mark_poi = my_dataset[person]["poi"]
    #all_messages = my_dataset[person]["from_messages"] + my_dataset[person]["to_messages"]
    #all_poi_messages = my_dataset[person]["from_this_person_to_poi"] + my_dataset[person]["from_poi_to_this_person"]
    all_messages1 = addNumbers(my_dataset[person]["from_messages"], my_dataset[person]["to_messages"])
    all_poi_messages1 = addNumbers(my_dataset[person]["from_this_person_to_poi"], my_dataset[person]["from_poi_to_this_person"])
    fraction_to_poi = computeFraction(my_dataset[person]["from_this_person_to_poi"], my_dataset[person]["from_messages"])
    fraction_from_poi = computeFraction(my_dataset[person]["from_poi_to_this_person"],my_dataset[person]["to_messages"])
    fraction_total_poi = computeFraction(all_poi_messages1, all_messages1)
    if mark_poi:
        color = 'red'
    else:
        color = 'blue'
    matplotlib.pyplot.scatter(fraction_from_poi, fraction_to_poi, color=color)
     
matplotlib.pyplot.xlabel("Fraction from POI")
matplotlib.pyplot.ylabel("Fraction to POI")
matplotlib.pyplot.show()

for name in my_dataset.iteritems():
    #all_messages = float(name[1]['to_messages']) + float(name[1]['from_messages'])
    #all_poi_messages = float(name[1]['from_this_person_to_poi']) + float(name[1]['from_poi_to_this_person'])
    all_messages2 = addNumbers(name[1]['to_messages'], name[1]['from_messages'])
    all_poi_messages2 = addNumbers(name[1]['from_this_person_to_poi'], name[1]['from_poi_to_this_person'])
    name[1]['fraction_to_poi'] = computeFraction(name[1]['from_this_person_to_poi'], name[1]['from_messages'])
    name[1]['fraction_from_poi'] = computeFraction(name[1]['from_poi_to_this_person'], name[1]['to_messages'])
    name[1]['fraction_total_poi'] = computeFraction(all_poi_messages2, all_messages2)

import math

for person in my_dataset:
    if math.isnan(my_dataset[person]['fraction_to_poi']) == True:
        my_dataset[person]['fraction_to_poi'] = 0.
    if math.isnan(my_dataset[person]['fraction_from_poi']) == True:
        my_dataset[person]['fraction_from_poi'] = 0.
    if math.isnan(my_dataset[person]['fraction_total_poi']) == True:
        my_dataset[person]['fraction_total_poi'] = 0.

# Univariate feature selection
## Extract features and labels from dataset for local testing
features_list = poi_label + financial_features + email_features + ['fraction_to_poi'] + ['fraction_from_poi'] + ['fraction_total_poi']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Univariate feature selection: SelectKBest selects the K features that are most powerful (where K is a parameter).
# Perform feature selection
selector = SelectKBest(f_classif, k='all')
selector = selector.fit(features, labels)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
for s in range(len(features_list) - 1):
    print features_list[s+1]
    print scores[s]

# Print best features
#features_selected = [features_list[i+1] for i in selector.get_support(indices=True)]
#features_selected=selector.get_support()
#best_features = []
#for bool, feature in zip(features_selected, features_list):
    #if bool:
        #best_features.append(feature)
#print best_features

# Plot the scores.
plt.bar(range(len(features_list)-1), scores)
plt.xticks(range(len(features_list)-1), features_list[1:], rotation='vertical')
plt.show()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Univariate feature selection: SelectKBest selects the K features that are most powerful (where K is a parameter).
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Perform feature selection
selector = SelectKBest(k=6) # change k value
selector = selector.fit(features, labels)

features_selected = [features_list[i+1] for i in selector.get_support(indices=True)]
#features_selected=selector.get_support()
best_features = []
for bool, feature in zip(features_selected, features_list):
    if bool:
        best_features.append(feature)
print best_features

data = featureFormat(my_dataset, best_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Split into a training and testing set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Naive Bayes
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
clf = GaussianNB() #random_state = 42 put the same random state for every clf, to compare their performance on the same data

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "---------"

print "Accuracy score: ", accuracy_score(pred, labels_test)

target_names = ["Not POI", "POI"]
classification = classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
print '\n Classification Report:'
print classification

# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 42)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "---------"

print "Accuracy score: ", accuracy_score(pred, labels_test)

target_names = ["Not POI", "POI"]
classification = classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
print '\n Classification Report:'
print classification

# Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=42)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "---------"

print "Accuracy score: ", accuracy_score(pred, labels_test)

target_names = ["Not POI", "POI"]
classification = classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
print '\n Classification Report:'
print classification

# Adaboost

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=42) #algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50

t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "---------"

print "Accuracy score: ", accuracy_score(pred, labels_test)

target_names = ["Not POI", "POI"]
classification = classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
print '\n Classification Report:'
print classification

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Apply GridSearchCV using all the features available to have it issue the recommended number of features
features_list = poi_label + financial_features + email_features + ['fraction_to_poi'] + ['fraction_from_poi'] + ['fraction_total_poi']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print features_list

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

folds = 10
sss = StratifiedShuffleSplit(folds, random_state = 42) #test_size = 0.2

#create a dictionary with all the parameters we want to search through
parameters={'criterion': ('gini', 'entropy'),'min_samples_split' : range(4,40,2),'max_depth': range(4,20,2), 
            'max_features': range(3,12,1)} #'random_state': range(2,100,2)
clf_tree = tree.DecisionTreeClassifier() #random_state = 42
gridsearch = GridSearchCV(clf_tree, param_grid = parameters, cv=sss, scoring="f1") 
gridsearch.fit(features_train, labels_train)

print "Best Score: {}".format(gridsearch.best_score_)
print "Best params: {}".format(gridsearch.best_params_)

# Apply recommended number of features
selector = SelectKBest(k=7) # select k as the recommended number of features
selector = selector.fit(features_train, labels_train)

features_selected = [features_list[i+1] for i in selector.get_support(indices=True)]
#features_selected=selector.get_support()
best_features = []
for bool, feature in zip(features_selected, features_list):
    if bool:
        best_features.append(feature)
print best_features

data = featureFormat(my_dataset, best_features, sort_keys = True)

### split into labels and features (this line assumes that the first feature in the array is the label, 
###which is why "poi" must always be first in features_list
labels, features = targetFeatureSplit(data)

print 'Decision Tree'
t0 = time()
clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=14, max_depth=12) #random_state=80
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "Predicting time:", round(time()-t1, 3), "s"

accuracy = clf.score(features_test, labels_test)
print "Accuracy:", accuracy

#from sklearn.metrics import precision_score
#precision = precision_score(labels_test, pred, average='binary')
#print "Precision Score:", precision

#from sklearn.metrics import recall_score
#recall = recall_score(labels_test, pred, average='binary')
#print "Recall Score:", recall

target_names = ["Not POI", "POI"]
classification = classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
print '\n Classification Report:'
print classification

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)