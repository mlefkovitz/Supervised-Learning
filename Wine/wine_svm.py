from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from plot_learning_curve import plot_learning_curve as plc
from plot_learning_curve2 import data_size_response
from plot_learning_curve2 import plot_response
from plot_learning_curve3 import drawLearningCurve
from Print_Timer_Results import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from wine_data import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define the classifier
svm = SVC(random_state=1)
parameters = {'kernel':('linear', 'rbf')
             ,'C':[1, 10]
             ,'gamma':(0.1, 0.5)
             }
clf = GridSearchCV(svm, parameters)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training and test accuracy
y_pred = clf.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('SVM Kernel train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# cv = 3
# max_instances = int(X_train.shape[0]*(cv-1)/cv)
# # Create learning curve
# train_sizes = [int(max_instances/80), int(max_instances/40), int(max_instances/20), int(max_instances/10), int(max_instances/5), int(max_instances*2/5), int(max_instances*3/5), int(max_instances*4/5), max_instances-5]
# train_sizes, train_scores, valid_scores = learning_curve(clf.best_estimator_, X_train, y_train, train_sizes=train_sizes, cv=cv)
#
# # Print learning curve diagnostics
# print(train_sizes)
# print(train_scores.shape)
# print(train_scores)
# print(valid_scores.shape)
# print(valid_scores)
#
# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"

# # Plot learning curves
# plc(clf.best_estimator_, title, X_train, y_train, cv=3, n_jobs=1, train_sizes=train_sizes)

# # data size reponse
# response = data_size_response(clf,X_train,X_test,y_train,y_test,prob=False)
# plot_response(*response)

# Draw learning curve
drawLearningCurve(clf, X_train, X_test, y_train, y_test, min_size=1000, numpoints=50)

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

Stop_Timer(start_time)

# # Show learning curve
# test = [x[0] for x in clf.grid_scores_]
# total = len(parameters['gamma'])*len(parameters['kernel'])
# gammas = [x['gamma'] for x in test[0:total]]
# kernels = [x['kernel'] for x in test[0:total]]
# iterator = np.column_stack((gammas, kernels))
# for ind, i in enumerate(iterator):
#     # print('kernel: ' + str(i[1]) + '; gamma: ' + str(i[0]))
#     # print('C:' + str(parameters['C']))
#     # print('Score:' + str(scores[ind]))
#     plt.plot(parameters['C'], scores[ind], label='kernel: ' + str(i[1]) + '; gamma: ' + str(i[0]))
# plt.legend()
# plt.xlabel('C')
# plt.ylabel('Mean score')
plt.show()