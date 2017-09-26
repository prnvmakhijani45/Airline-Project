import json
import csv
from urllib.request import urlopen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Getting the dataset
url = "http://ist.gmu.edu/~hpurohit/courses/ait582-proj-data-spring16.json"
data = urlopen(url).read()
data_parsed = json.loads(data.decode('utf-8'))

# open a file for writing
airline_data = open('Airline.csv', 'w')
# create the csv writer object
csvwriter = csv.writer(airline_data)
for row in data_parsed:
      csvwriter.writerow(row.values())
airline_data.close()

airline = pd.read_csv("Airline.csv")

#Metadata Extraction
#Populate Age & Title column from Description
airline["AGE"] = float(0)
airline["TITLE"] = ""
airline["GENDER"] = ""
nrow = len(airline.index)

for row in range(0,nrow):
    age = airline.loc[row]["DESCRIPTION"].split(';')[1]
    if(age != ""):
        airline.loc[row,"AGE"] = float(age)
    else:
        airline.loc[row,"AGE"] = 0
    
    airline.loc[row,"TITLE"] = airline.loc[row,"DESCRIPTION"].split(',')[1].split('.')[0]
                   

sns.distplot(airline['AGE'], bins=20)
newairline = airline[airline["AGE"] == 0]
newairline["TITLE"].unique()

#Median Age
airline["AGE"].median()
airline[airline["TITLE"] == " Mr"]["AGE"].median()
airline[airline["TITLE"] == " Mrs"]["AGE"].median()
airline[airline["TITLE"] == " Miss"]["AGE"].median()
airline[airline["TITLE"] == " Master"]["AGE"].median()
airline[airline["TITLE"] == " Dr"]["AGE"].median()


def ImputeMissingAge(Title):
    
    local_airline = airline[(airline["TITLE"] == Title) & (airline["AGE"] == 0)]
    medianAgeClass1 = airline[(airline["TITLE"] == Title) & (airline["SEATCLASS"] == 1)]["AGE"].median()
    medianAgeClass2 = airline[(airline["TITLE"] == Title) & (airline["SEATCLASS"] == 2)]["AGE"].median()
    medianAgeClass3 = airline[(airline["TITLE"] == Title) & (airline["SEATCLASS"] == 3)]["AGE"].median()
    
    for index, row in local_airline.iterrows():
        if(row["SEATCLASS"] == 1):
            airline.loc[index, "AGE"] = medianAgeClass1
        elif(row["SEATCLASS"] == 2):
            airline.loc[index, "AGE"] = medianAgeClass2
        elif(row["SEATCLASS"] == 3):
            airline.loc[index, "AGE"] = medianAgeClass3
        
ImputeMissingAge(" Mr")
ImputeMissingAge(" Mrs")
ImputeMissingAge(" Miss")
ImputeMissingAge(" Master")
ImputeMissingAge(" Dr")
    
sns.distplot(airline['AGE'], bins=20)


#Knowledge Engineering: 1. Better the seat class, more the fare. 2. More guests = more fare
missing_d = airline[airline["FARE"] == 0]
airline[airline["SEATCLASS"] == 1]["FARE"].mean()
airline[(airline["SEATCLASS"] == 1) & (airline["GUESTS"] == 0)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 1) & (airline["GUESTS"] == 1)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 1) & (airline["GUESTS"] == 2)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 1) & (airline["GUESTS"] == 3)]["FARE"].mean()

airline[airline["SEATCLASS"] == 2]["FARE"].mean()
airline[(airline["SEATCLASS"] == 2) & (airline["GUESTS"] == 0)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 2) & (airline["GUESTS"] == 1)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 2) & (airline["GUESTS"] == 2)]["FARE"].mean()

airline[airline["SEATCLASS"] == 3]["FARE"].mean()
airline[(airline["SEATCLASS"] == 3) & (airline["GUESTS"] == 0)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 3) & (airline["GUESTS"] == 1)]["FARE"].mean()
airline[(airline["SEATCLASS"] == 3) & (airline["GUESTS"] == 2)]["FARE"].mean()


def ImputeMissingFare():
    
    local_airline = airline[airline["FARE"] == 0]
    for index, row in local_airline.iterrows():
        airline.loc[index, "FARE"] = airline[(airline["SEATCLASS"] == row["SEATCLASS"]) & 
                   (airline["GUESTS"] == row["GUESTS"])]["FARE"].mean()

ImputeMissingFare()

#Metadata Extraction: Gender Column
uniqueTitles = airline["TITLE"].unique()

def ImputeGender():
    male = [' Mr', ' Master', ' Don', ' Rev', ' Dr', ' Major', ' Sir', ' Col', ' Capt']
    for index, row in airline.iterrows():
        if(row["TITLE"] in male):
            airline.loc[index, "GENDER"] = 'Male'
        else:
            airline.loc[index, "GENDER"] = 'Female'
            
ImputeGender()
sns.countplot

#Modelling - Artificial Neural Network
#Creating features and target variable
X = airline.iloc[:, [ 0, 2, 3, 6, 7, 8]].values
y = airline.iloc[:, 1].values

#Creating dummy variables for categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_gender = LabelEncoder()
X[:, 5] = labelencoder_gender.fit_transform(X[:, 5])

labelencoder_title = LabelEncoder()
X[:, 4] = labelencoder_title.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features=[4])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for title
X = X[:, 1:]

labelencoder_pclass = LabelEncoder()
X[:, 17] = labelencoder_pclass.fit_transform(X[:, 17])
onehotencoder = OneHotEncoder(categorical_features=[17])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for title
X = X[:, 1:]

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)


#Scaling all features to a similiar scale
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

#Initializing ANN
classifier = Sequential()
#Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 22))
#Adding second hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
#Adding third hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
#Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#Compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)
ann_pred = classifier.predict(X_test)
ann_pred = (ann_pred > 0.5)

        
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, ann_pred) 
print(classification_report(y_test, ann_pred))

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()



#Modelling: SVM
#Creating features and target variable
X = airline.iloc[:, [ 0, 2, 3, 6, 8]].values #seatclass1, seatclass 2, gender,guests,fare,age
y = airline.iloc[:, 1].values

#Creating dummy variables for categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_gender = LabelEncoder()
#X[:, 5] = labelencoder_gender.fit_transform(X[:, 5])

labelencoder_title = LabelEncoder()
X[:, 4] = labelencoder_title.fit_transform(X[:, 4]) #female, male, guests, seatclass, fare, age

onehotencoder = OneHotEncoder(categorical_features=[4])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for title
X = X[:, 1:]

labelencoder_pclass = LabelEncoder()
X[:, 2] = labelencoder_pclass.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[2])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for title
X = X[:, 1:]

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)


#Scaling all features to a similiar scale
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Support Vector Classification
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 101)
classifier.fit(X_train, y_train)
svm_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, svm_pred)

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()

#Cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Hyperparameter tuning for SVC
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']},
              {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.1, 0.01, 0.001, 0.0001]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy',
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

classifier = SVC(C=10, gamma=0.1, kernel = 'rbf', random_state = 101)
classifier.fit(X_train, y_train)
svm_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, svm_pred)
print(classification_report(y_test, svm_pred))
classifier.coef_
#ROC Curve
from sklearn.metrics import roc_curve, auc
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Area under curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print( pd.DataFrame(pca.components_,index = ['PC-1','PC-2']))

plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c = y_train, cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
 #seatclass1, seatclass 2, gender,guests,fare,age
sns.heatmap(pca.components_, cmap = 'plasma')
feat = ['seatclass1', 'seatclass 2', 'gender','guests','fare','age']

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
knn_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, knn_pred)
print(classification_report(y_test, knn_pred))

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()


#Modelling: Random Forest
#Creating features and target variable
X = airline.iloc[:, [ 0, 2, 3, 6, 8]].values #seatclass1, seatclass 2, gender,guests,fare,age
y = airline.iloc[:, 1].values

#Creating dummy variables for categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_gender = LabelEncoder()
#X[:, 5] = labelencoder_gender.fit_transform(X[:, 5])

labelencoder_title = LabelEncoder()
X[:, 4] = labelencoder_title.fit_transform(X[:, 4]) 

onehotencoder = OneHotEncoder(categorical_features=[4])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for title
#X = X[:, 1:]

labelencoder_pclass = LabelEncoder()
X[:, 3] = labelencoder_pclass.fit_transform(X[:, 3])#pclass1, pclass2, pclass3, female, male, guests, fare, age
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
#Dummy variable trap for title
#X = X[:, 1:]

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)


#Scaling all features to a similiar scale
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 101)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#clf = RandomForestClassifier()
#clf.fit(X_train, y_train.ravel())

features = ['pclass1', 'pclass2', 'pclass3', 'female', 'male', 'guests', 'fare', 'age']

importance = classifier.feature_importances_
importance = pd.DataFrame(importance, index=features, columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_
                            for tree in classifier.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

plt.bar(x, y, yerr=yerr, align="center")
plt.xlabel(features)
plt.show()
#77.65%



    
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()

'''
fare = airline.iloc[:, [6,3]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 101)
    kmeans.fit(fare)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 101)
y_kmeans = kmeans.fit_predict(fare)

plt.scatter(fare[y_kmeans == 0, 0], fare[y_kmeans == 0, 1], s=100, c='blue', label = 'Cluster 2')
plt.scatter(fare[y_kmeans == 1, 0], fare[y_kmeans == 1, 1], s=100, c='blue', label = 'Cluster 2')
plt.scatter(fare[y_kmeans == 2, 0], fare[y_kmeans == 2, 1], s=100, c='green', label = 'Cluster 3')
plt.scatter(fare[y_kmeans == 3, 0], fare[y_kmeans == 3, 1], s=100, c='cyan', label = 'Cluster 4')
plt.scatter(fare[y_kmeans == 4, 0], fare[y_kmeans == 4, 1], s=100, c='magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c= 'yellow', label = 'Centroids')
plt.title('Cluster of Fares')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.show()
'''

'''
#Model Ensemble - Stacking: Max vote
#Merging all classifiers into one dataframe
def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v

new_data = np.vectorize(boolstr_to_floatstr)(ann_pred).astype(int)
dfd = pd.DataFrame(new_data, columns=['ann'])
dfd['knn'] = pd.Series(knn_pred, index=dfd.index)
dfd['svm'] = pd.Series(svm_pred, index=dfd.index)

dfd['output'] = 0

def majority_vote():
    for i in range(0, 178):
        if(dfd.loc[i, 'ann'] == dfd.loc[i, 'knn']):
            dfd.loc[i, 'output'] = dfd.loc[i, 'ann']
        elif(dfd.loc[i, 'ann'] == dfd.loc[i, 'svm']):
            dfd.loc[i, 'output'] = dfd.loc[i, 'ann']
        elif(dfd.loc[i, 'knn'] == dfd.loc[i, 'svm']):
            dfd.loc[i, 'output'] = dfd.loc[i, 'knn']
        

majority_vote()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, dfd['output'])

'''




















































