import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('2.csv')  # Replace with your actual file name

# 1. Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())  # Filling missing with median
df['Name'] = df['Name'].fillna('Unknown')  # Filling missing with 'Unknown'
df['Salary'] = df['Salary'].fillna(df['Salary'].median())  # Filling missing with median
df['Joining_Date'] = df['Joining_Date'].fillna('2000-01-01')  # Filling missing with default date

# 2. Remove Duplicates
df = df.drop_duplicates()  # Drop duplicates and assign back to df

# 3. Fix Inconsistencies
# Standardize gender values
df['Gender'] = df['Gender'].replace({'F': 'Female', 'M': 'Male', 'f': 'Female', 'm': 'Male'})
df['Gender'] = df['Gender'].str.capitalize()  # Ensure proper capitalization

# Standardize department
df['Department'] = df['Department'].replace({'Unknown': 'Other'})

# 4. Remove Outliers
# Remove negative ages
df = df[df['Age'] > 0]

# Use IQR method for outlier detection on Salary
# Calculate IQR
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Filter the DataFrame based on conditions
df = df[(df['Salary'] >= lower) & (df['Salary'] <= upper)]


# 5. Standardize Formats
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'], errors='coerce')  # Convert to datetime

# Preview cleaned data
print(df)