from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the data
from wine_data2 import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# Define the classifier
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(1,50)
             }
clf = GridSearchCV(knn, param_grid=parameters, cv=5)

# Run the classifier
clf.fit(X_train_std, y_train)

# Identify training and test accuracy
y_pred_train = clf.predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('KNN: train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# Print diagnostics
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)
print('gridscores:')
print(clf.grid_scores_)

# Show learning curve
scores = [x[1] for x in clf.grid_scores_]
plt.plot(parameters['n_neighbors'], scores)
plt.xlabel('neighbors')
plt.ylabel('Mean score')
plt.show()