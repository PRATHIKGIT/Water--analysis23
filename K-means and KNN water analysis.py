import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\prath\Downloads\CODING\spare\water_potability.csv')

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Define the number of clusters for K-Means
k_clusters = 2  # Adjust as needed

# Split the data into features (X) and target (Y)
X = df[['ph', 'Hardness']]  # Use only 'ph' and 'Hardness'
Y = df['Potability']

# Perform K-Means clustering and add the cluster labels to the dataset
kmeans = KMeans(n_clusters=k_clusters, random_state=101)
df['Cluster'] = kmeans.fit_predict(X)  # Add the 'Cluster' column to the original dataframe

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101, shuffle=True)

# Standardize features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
k_neighbors = 3  # Adjust as needed
knn = KNeighborsClassifier(n_neighbors=k_neighbors)

# Fit the KNN model on the training data
knn.fit(X_train_scaled, Y_train)

# Make predictions with KNN
prediction_knn = knn.predict(X_test_scaled)

# Calculate accuracy for KNN
accuracy_knn = accuracy_score(Y_test, prediction_knn) * 100
print("KNN Accuracy:", accuracy_knn)

# Rest of the code for visualization, Random Forest, and Silhouette Score remains the same.
rf = RandomForestClassifier(random_state=101)

# Define the hyperparameters to search
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Perform grid search with cross-validation for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, Y_train)

# Get the best Random Forest model
best_rf = grid_search_rf.best_estimator_

# Make predictions with Random Forest
prediction_rf = best_rf.predict(X_test)

# Calculate accuracy for Random Forest
accuracy_rf = accuracy_score(Y_test, prediction_rf) * 100
print("Random Forest Accuracy:", accuracy_rf)

# Silhouette Score for K-Means
silhouette_avg = silhouette_score(X, X['Cluster'])
print("Silhouette Score for K-Means:", silhouette_avg)

# Create a mesh grid of values for the entire feature space
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the class labels for the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)



sns.pairplot(df,hue='Potability')


