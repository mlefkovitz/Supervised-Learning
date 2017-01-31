from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np

# Load the data
from wine_data2 import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# Define the classifier
svm = SVC(random_state=1)
parameters = {'kernel':('linear', 'rbf')
             ,'C':[1, 10]
             ,'gamma':(0.1, 0.5)
             }
clf = GridSearchCV(svm, parameters)

# Run the classifier
clf.fit(X_train_std, y_train)

# Identify training and test accuracy
y_pred = clf.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
y_pred_train = clf.predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('SVM Kernel train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# Print diagnostics
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)
print('gridscores:')
print(clf.grid_scores_)

# Print diagnostics
scores = [x[1] for x in clf.grid_scores_]
print('scores:')
print(scores)
scores = np.array(scores).reshape(len(parameters['C']),len(parameters['kernel']) * len(parameters['gamma']))
scores = scores.transpose()
print('scores:')
print(scores)

# Show learning curve
test = [x[0] for x in clf.grid_scores_]
total = len(parameters['gamma'])*len(parameters['kernel'])
gammas = [x['gamma'] for x in test[0:total]]
kernels = [x['kernel'] for x in test[0:total]]
iterator = np.column_stack((gammas, kernels))
for ind, i in enumerate(iterator):
    # print('kernel: ' + str(i[1]) + '; gamma: ' + str(i[0]))
    # print('C:' + str(parameters['C']))
    # print('Score:' + str(scores[ind]))
    plt.plot(parameters['C'], scores[ind], label='kernel: ' + str(i[1]) + '; gamma: ' + str(i[0]))
plt.legend()
plt.xlabel('C')
plt.ylabel('Mean score')
plt.show()