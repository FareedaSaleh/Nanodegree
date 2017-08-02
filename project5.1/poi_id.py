#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary', 
             'total_payments', 
               'loan_advances', 
                 'bonus', 
               'deferred_income', 
                'total_stock_value',
               'exercised_stock_options', 
                'from_this_person_to_poi', 
                 'shared_receipt_with_poi',
                 "fraction_from_poi",
                 "fraction_to_poi",
                 "fraction_shared_receipt",
                 "Salary_Bonus"
                 ] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
##Ask qustions about dataset

#Number of data points
Count_point=len(data_dict.keys())

print"The Number of people are:",Count_point
#Number of featrues
for key in data_dict.iterkeys():
    fet=len(data_dict[key])
    
print" Number of Featurse:",fet

#Number of POI

count=0
non_poi=0  
for key in data_dict.iterkeys():
    # print (enron_data[key])
    if data_dict[key]["poi"]==1:
        count=count+1
    else:
        non_poi=non_poi+1

print "Number of POI =",count
print "Number of Non POI=",non_poi


#missing featrues:

#https://www.accelebrate.com/blog/using-defaultdict-python/
from collections import defaultdict 
   
missing_feartures = defaultdict(int)		
for k,v in data_dict.iteritems():
	for i in v:
		if v[i] == 'NaN':
			missing_feartures[i]+=1
print 'The Number of missing features', missing_feartures 

### Task 2: Remove outliers

#Drow the poit of data to iedentfiae the outliers 

for point in data_dict.values():
    
    matplotlib.pyplot.scatter( point['salary'], point['bonus'] )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

#we notic after review the scatterplot and findlow pdf  the TOTAL is outliers!!

# remove the total as it's outliers

data_dict.pop( "TOTAL", 0 ) 

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)


### Task 3: Create new feature(s)

# fraction_from_poi


for k,v in data_dict.iteritems():
    if v["from_poi_to_this_person"]=='NaN' or v["to_messages"]=='NaN':
        
        v["fraction_from_poi"]='NaN'
    else:
        v["fraction_from_poi"]= float(v["from_poi_to_this_person"])/float(v["to_messages"])
    

#fraction_to_poi

for k,v in data_dict.iteritems():
    
   if v["from_this_person_to_poi"]=='NaN' or v["from_messages"]=='NaN':
        
        v["fraction_to_poi"]='NaN'
   else:
        v["fraction_to_poi"]= float(v["from_this_person_to_poi"])/float(v["from_messages"])


#fraction_shared_receipt


for k,v in data_dict.iteritems():
    
   if v["shared_receipt_with_poi"]=='NaN' or v["to_messages"]=='NaN':
        
        v["fraction_shared_receipt"]='NaN'
   else:
        v["fraction_shared_receipt"]= float(v["shared_receipt_with_poi"])/float(v["to_messages"])


#Salary_Bounace

for k,v in data_dict.iteritems():
    
   if v["salary"]=='NaN' or v["bonus"]=='NaN':
        
        v["Salary_Bonus"]='NaN'
   else:
        v["Salary_Bonus"]= float(v["bonus"])/float(v["salary"])
        


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:


### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


scaler = MinMaxScaler()
kbest = SelectKBest()

#classifier for algorth
dtclf = DecisionTreeClassifier()
svclf = SVC()
knclf = KNeighborsClassifier()
adclf = AdaBoostClassifier()
NBclf= GaussianNB()

# steps for pipeline

steps = [ ('min_max_scaler', scaler), 
         ('kbest', kbest),
         
       #("dtclf",dtclf)
#("SVC",svclf),
#('KN', knclf)
#("AD",adclf)
("NBclf", NBclf)
]
         
pipeline = Pipeline(steps) 


#parameters 


parameters = dict(
                kbest__k=[2, 3, 5,6], 
                
             #dtclf__criterion=['gini', 'entropy'],
                  
             #dtclf__splitter= ['best', 'random'],
                  
             #dtclf__max_depth=[None, 1, 2, 3, 4],
                  
             #dtclf__min_samples_split=[2, 3, 4, 25,30],
                 
              #dtclf__min_samples_leaf=[1, 2, 3, 4],
                  
                # (it =0 in kbest) dtclf__min_weight_fraction_leaf=[0, 0.25, 0.5],
                  
            #dtclf__class_weight= [None, 'balanced'],
                 
               #dtclf__random_state= [42]#,
               #SVC parameters
               
              #SVC__C=[ 0.1,1, 10, 100, 1000,10000],
             #SVC__kernel=['rbf'],
              #SVC__gamma=[0.001, 0.0001],
                  
                                 
                #KNeighbors
                
            #KN__n_neighbors=[1, 2, 3, 4, 5,8],
           #KN__leaf_size=[1, 10, 30, 60],
          #KN__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
                
            #AdaBoost parameters
            
             #AD__n_estimators= [10, 20, 30, 40, 50],
            #AD__learning_rate= [.5,.8, 1, 1.2, 1.5]
             )    
        

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
# Example starting point. Try investigating other evaluation techniques!


#from sklearn.cross_validation import train_test_split


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)




sss = StratifiedShuffleSplit(labels_train, n_iter = 20, test_size=0.5, random_state=0)

gs = GridSearchCV(pipeline,  param_grid=parameters, scoring="f1",cv=sss, error_score=0)
	           	               
	           
gs.fit(features_train, labels_train)
pred = gs.predict(features_test)

clf = gs.best_estimator_
print "fainally select the best!! ", gs.best_params_
###
'''
features_selected=[features_list[i+1] for i in clf.named_steps['kbest'].get_support(indices=True)]
print "The features selected:",features_selected
# Access the feature importances
##The code of importances from :https://discussions.udacity.com/t/feature-importances-/173319/10
# The step in the pipeline for the Decision Tree Classifier is called 'DTC'
# that step contains the feature importances

importances = clf.named_steps['dtclf'].feature_importances_

import numpy as np
indices = np.argsort(importances)[::-1]

# Use features_selected, the features selected by SelectKBest, and not features_list
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]])
'''
###

## http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
## target_names = ['class 0', 'class 1', 'class 2']
####


###

X_new = gs.best_estimator_.named_steps['kbest']
# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple

####
print "This is the validation matrix"

report = classification_report( labels_test, pred )
print report

#



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)