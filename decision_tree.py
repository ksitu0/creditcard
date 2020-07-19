import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score)
import pickle

path = '/home/grace/ml_bootcamp/creditcard/' # path to local folder
seed = 18 # for random_state

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
weights = {0:600, 1:1}

tree = DecisionTreeClassifier(max_depth=28, random_state=seed, class_weight=weights)
tree.fit(X_train, y_train)
y_trainer = tree.predict(X_train)
y_pred = tree.predict(X_test)

# Score tree
print(confusion_matrix(y_train,y_trainer))
print(confusion_matrix(y_test, y_pred))

acc_train = accuracy_score(y_train, y_trainer)
acc_test = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy on training set: {}".format(acc_train))
print("Accuracy on test set: {}".format(acc_test))
print("The precision is {}".format(precision))
print("The recall/sensitivity is {}".format(recall))
print("The F1 score is {}".format(f1))

# Save model and predictions
pickle.dump(tree, open("tree_model.pkl","wb"))

tree_results = pd.DataFrame(y_pred)
tree_results.to_csv(path + 'tree_results.csv', index=False)