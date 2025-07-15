import pandas as pd
import numpy as np

# Generate 100 data points, one per minute
data = {
    'timestamp': pd.date_range(start='2023-10-01', periods=100, freq='T'),
    'heart_rate': np.random.randint(60, 100, 100),  # normal range
    'blood_oxygen': np.random.randint(90, 100, 100),  # normal range
    'activity_level': np.random.choice(['low', 'moderate', 'high'], 100)
}

df = pd.DataFrame(data)
print(df.head())
# Simulate some missing values
df.loc[5:10, 'heart_rate'] = np.nan

# Fill missing values with mean
df['heart_rate'].fillna(df['heart_rate'].mean(), inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['activity_level_encoded'] = le.fit_transform(df['activity_level'])
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['heart_rate', 'blood_oxygen']] = scaler.fit_transform(df[['heart_rate', 'blood_oxygen']])
from sklearn.ensemble import IsolationForest

# Use numerical features only
features = df[['heart_rate', 'blood_oxygen']]

# Create and train the model
model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = model.fit_predict(features)

# Convert -1 to "Anomaly" and 1 to "Normal"
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# View results
print(df[['timestamp', 'heart_rate', 'blood_oxygen', 'anomaly']].head())
def recommend(row):
    if row['anomaly'] == 'Anomaly':
        if row['heart_rate'] < 60:
            return 'Heart rate low - try light exercise or consult doctor'
        elif row['blood_oxygen'] < 92:
            return 'Low SpO2 - breathe deeply, avoid stress'
        else:
            return 'Unusual activity detected - monitor closely'
    return 'All metrics normal'

df['recommendation'] = df.apply(recommend, axis=1)
print(df[['heart_rate', 'blood_oxygen', 'anomaly', 'recommendation']].head())

# 4 .MODEL TRAINING AND EVALUATION
# 1 = Anomaly, 0 = Normal
df['anomaly_label'] = df['anomaly'].apply(lambda x: 1 if x == 'Anomaly' else 0)
X = df[['heart_rate', 'blood_oxygen']]
y = df['anomaly_label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import IsolationForest

# Re-train model on training data
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict on test data
y_pred = model.predict(X_test)

# Convert predictions to 0 = normal, 1 = anomaly
y_pred = [1 if i == -1 else 0 for i in y_pred]
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt

plt.scatter(df['heart_rate'], df['blood_oxygen'], c=df['anomaly_label'], cmap='coolwarm')
plt.xlabel('Heart Rate')
plt.ylabel('Blood Oxygen')
plt.title('Anomaly Detection')
plt.show()
