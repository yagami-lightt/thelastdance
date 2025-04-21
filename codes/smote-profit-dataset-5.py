import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset (assuming the file is named 'data.csv')
df = pd.read_csv('EXP5ADS.csv')

# Create a binary target: 1 if profit > median, else 0
df['Profit_Class'] = (df['Profit'] > df['Profit'].median()).astype(int)

print(df['Profit_Class'])

# Features and target
X = df.drop(['Profit', 'Profit_Class'], axis=1)  # Features
y = df['Profit_Class']  # Target

# Convert categorical features (State) into numeric
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_res, y_res)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
