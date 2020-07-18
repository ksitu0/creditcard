import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

path = '/home/grace/ml_bootcamp/creditcard/' # path to local folder
seed = 12 # for random_state when sampling

# Load data from CSV to dataframe
data_path = path + 'kaggle_data.csv'
data_df = pd.read_csv(data_path)

# Visualize class imbalance
class_counts = data_df['Class'].value_counts()

class_counts.plot.bar(color=['b','r'])
plt.title('Fraud Class Distribution')
plt.ylabel('Count')
plt.xticks([0,1], ['Non-Fraud','Fraud'], rotation=0)

# Separate features and labels
X = data_df.drop(columns = ['Class'])
y = data_df['Class']

print(y.value_counts())

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=seed)

print(y_test.value_counts())

# Balance data with SMOTE
sm = SMOTE(sampling_strategy=1.0, random_state=seed)
X_train_res, y_train_res = sm.fit_sample(X_train,y_train) 

print(y_train_res.value_counts())

# Save train/test from dataframe to CSV
X_train_res.to_csv(path + 'train_features.csv', index=False)
X_test.to_csv(path + 'test_features.csv', index=False)
y_train_res.to_csv(path + 'train_labels.csv', index=False)
y_test.to_csv(path + 'test_labels.csv', index=False)