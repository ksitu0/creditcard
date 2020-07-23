# Data 
import pandas as pd

path = ".../creditcard/"

# Load train/test data from CSVs to dataframes
X_train_path = path + 'train_features.csv'
y_train_path = path + 'train_labels.csv'
X_test_path = path + 'test_features.csv'
y_test_path = path + 'test_labels.csv'

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import numpy as np
import pickle

seed = 42

# Make random forest classifier
forest = RandomForestClassifier(n_estimators=800, random_state=seed)
print("fitting")
forest.fit(X_train, y_train)

print("predicting")
y_trainer = forest.predict(X_train)
y_pred = forest.predict(X_test)

# Results + Confusion matrix
print(confusion_matrix(y_train,y_trainer))
print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(tree, X_test, y_test, normalize='true', 
        include_values=False, cmap='Reds')
plt.savefig('tree_confusion.png', dpi=300, bbox_inches='tight')

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
pickle.dump(forest, open("forest_model.pkl","wb"))

forest_results = pd.DataFrame(y_pred)
forest_results.to_csv(path + 'forest_results.csv', index=False)
