from sklearn.metrics import accuracy_score
from PrunedTrees import dtclf_pruned
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

# Start timer
start_time = time.time()

# Load the data
from wine_data import X_train, X_test, y_train, y_test

# Define the classifier
#tree = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=0)
tree = dtclf_pruned(alpha=0.006)
ada = AdaBoostClassifier(base_estimator=tree, random_state=0)

parameters = {'n_estimators': [10, 20, 40, 70, 100, 150, 200]
             ,'learning_rate':[3, 1, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001]
             }

clf = GridSearchCV(ada, parameters)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training and test accuracy
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('Ada boost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

# Print diagnostics
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)
print('gridscores:')
print(clf.grid_scores_)
scores = [x[1] for x in clf.grid_scores_]
print('scores:')
print(scores)
#scores = np.array(scores).reshape(len(parameters['n_estimators']), len(parameters['learning_rate']))
scores = np.array(scores).reshape(len(parameters['learning_rate']), len(parameters['n_estimators']))
print('scores:')
print(scores)

print('This function took', time.time()-start_time, 'seconds.')

# Show learning curve
for ind, i in enumerate(parameters['learning_rate']):
    print('learning_rate: ' + str(i))
    print('n_estimators:' + str(parameters['n_estimators']))
    print('Score:' + str(scores[ind]))
    plt.plot(parameters['n_estimators'], scores[ind], label='learning_rate: ' + str(i))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Mean score')
plt.show()

# from sklearn.tree import export_graphviz
# export_graphviz(clf.best_estimator_, out_file = 'Wine2boosted.dot', feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])