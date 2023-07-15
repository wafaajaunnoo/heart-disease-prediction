import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('disease.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


plt.scatter(X_train, y_train, color = 'purple')
plt.plot(X_train, classifier.predict_proba(X_train)[:, 1], color = 'green')
plt.title('Cholesterol Level vs Heart Disease (Training set)')
plt.xlabel('Cholesterol Level')
plt.ylabel('Heart Disease')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, classifier.predict_proba(X_train)[:, 1], color = 'blue')
plt.title('Cholesterol Level vs Heart Disease (Test set)')
plt.xlabel('Cholesterol Level')
plt.ylabel('Heart Disease')
plt.show()


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
