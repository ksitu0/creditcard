import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score)

path = '/home/grace/ml_bootcamp/creditcard/' # path to local folder
seed = 12 # for random_state

# Load train/test data from CSVs to dataframes
X_train_path = path + 'train_features.csv'
y_train_path = path + 'train_labels.csv'
X_test_path = path + 'test_features.csv'
y_test_path = path + 'test_labels.csv'

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

# make decision tree classifier
tree = DecisionTreeClassifier(random_state = seed)
tree.fit(X_train, y_train)
y_trainer = tree.predict(X_train)
y_pred = tree.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Score tree
print("Accuracy on training set: {}".format(accuracy_score(y_train, y_trainer)))
print("Accuracy on test set: {}".format(accuracy_score(y_test, y_pred)))

print("The precision is {}".format(precision_score(y_test, y_pred)))
print("The recall/sensitivity is {}".format(recall_score(y_test, y_pred)))
print("The F1 score is {}".format(f1_score(y_test, y_pred)))