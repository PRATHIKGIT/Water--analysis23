import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\prath\Downloads\CODING\spare\water_potability.csv')

df.fillna(df.mean(), inplace=True)


from sklearn.tree import plot_tree
X = df.drop('Potability',axis=1)
Y= df['Potability']

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101, shuffle=True)

# Define the hyperparameters to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3],
    'min_samples_split': [2, 5,8,10,12,15, 20]
}

# # Create a DecisionTreeClassifier
# dt = DecisionTreeClassifier()

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, Y_train)

# # Get the best estimator and parameters
# best_dt = grid_search.best_estimator_
# best_params = grid_search.best_params_

# # Fit the best model
# best_dt.fit(X_train, Y_train)

# # Make predictions
# prediction = best_dt.predict(X_test)

# # Calculate accuracy
# accuracy_dt = accuracy_score(Y_test, prediction) * 100
# print("Accuracy:", accuracy_dt)
# print("Best Hyperparameters:", best_params)


from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier
rf = RandomForestClassifier()

# Perform grid search with cross-validation for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search_rf.fit(X_train, Y_train)

# Get the best estimator and parameters for Random Forest
best_rf = grid_search_rf.best_estimator_
best_params_rf = grid_search_rf.best_params_

# Fit the best Random Forest model
best_rf.fit(X_train, Y_train)

# Make predictions with Random Forest
prediction_rf = best_rf.predict(X_test)

# Calculate accuracy for Random Forest
accuracy_rf = accuracy_score(Y_test, prediction_rf) * 100
print("Random Forest Accuracy:", accuracy_rf)
print("Best Hyperparameters for Random Forest:", best_params_rf)
