import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

seed = 12 # for random_state when sampling

# Load data
data_path = '/home/grace/creditcard/kaggle_data.csv'
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

# Balance data with SMOTE
sm = SMOTE(sampling_strategy=1.0, random_state=seed)
X_res, y_res = sm.fit_sample(X,y) 

print(y_res.value_counts())

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=seed)

