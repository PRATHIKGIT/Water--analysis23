import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\prath\Downloads\CODING\spare\water_potability.csv')

df.fillna(df.mean(), inplace=True)



X = df.drop('Potability',axis=1)
Y= df['Potability']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
dt=DecisionTreeClassifier(criterion= 'entropy', min_samples_split= 10, splitter= 'best')
dt.fit(X_train,Y_train)

# Plot the Decision Tree

plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=list(X.columns), class_names=['Not Potable', 'Potable'], filled=True, rounded=True, max_depth=2)
plt.show()

