import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\prath\Downloads\CODING\spare\water_potability.csv')

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Create a countplot for 'Potability' class distribution
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Potability', palette='Set2')
plt.title("Distribution of Potability")
plt.xlabel("Potability")
plt.ylabel("Count")
plt.show()
