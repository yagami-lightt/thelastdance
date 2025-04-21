# !pip install seaborn --quiet

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('tips.csv')
# General summary
print(df.describe())
print(df.info())

# Distribution of Total Bill

sns.histplot(df['total_bill'], kde=True)
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill ($)')
plt.ylabel('Frequency')
plt.show()

# Distribution of Tip

sns.histplot(df['tip'], kde=True, color='green')
plt.title('Distribution of Tip Amount')
plt.xlabel('Tip ($)')
plt.ylabel('Frequency')
plt.show()

# Tip vs Total Bill

sns.scatterplot(data=df, x='total_bill', y='tip', hue='sex')
plt.title('Tip vs Total Bill by Sex')
plt.show()

# Boxplot of tip by gender

sns.boxplot(x='sex', y='tip', data=df)
plt.title('Tip Distribution by Gender')
plt.show()

# Barplot of average tip per day

sns.barplot(x='day', y='tip', data=df, estimator=sum)
plt.title('Total Tips by Day')
plt.show()

# Countplot of smoker vs non-smoker

sns.countplot(x='smoker', data=df)
plt.title('Smokers vs Non-Smokers')
plt.show()

# Pairplot
sns.pairplot(df, hue="sex")
plt.show()
