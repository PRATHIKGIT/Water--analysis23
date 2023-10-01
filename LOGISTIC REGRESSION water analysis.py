import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\prath\Downloads\CODING\spare\water_potability.csv')

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Select two features for visualization
X = df[['ph', 'Hardness']]
Y = df['Potability']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101, shuffle=True)

# Standardize features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Logistic Regression classifier
logistic_reg = LogisticRegression(random_state=101)

# Fit the Logistic Regression model on the training data
logistic_reg.fit(X_train_scaled, Y_train)

# Create a mesh grid of values for the entire feature space
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the class labels for the mesh grid
Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Calculate accuracy
Y_pred = logistic_reg.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)

# Create a contour plot to visualize the decision boundary
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
plt.title(f"Logistic Regression Decision Boundary\nAccuracy: {accuracy:.2f}")
plt.xlabel("ph")
plt.ylabel("Hardness")

# Scatter plot for the actual data points
sns.scatterplot(data=df, x='ph', y='Hardness', hue='Potability', palette='Set1')
plt.show()





