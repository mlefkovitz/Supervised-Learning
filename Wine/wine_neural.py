import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

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
nn = MLPClassifier(solver='lbfgs', random_state=1)

grid_params = {'alpha': [0.05, 0.01, 0.005, 0.001]
              ,'hidden_layer_sizes': [3, 4, 7, 8, 12, 15, 20, 30, 50]
              }

# grid_params = {'alpha': [0.05, 0.01, 0.005, 0.001]
#               ,'hidden_layer_sizes': [4, 8, 12]
#               }

rand_params = {'alpha': stats.uniform(0.001, 0.05)
              ,'hidden_layer_sizes': stats.randint(3, 50)
              }

clf = GridSearchCV(nn, param_grid=grid_params, cv=5)
#clf = RandomizedSearchCV(nn, param_distributions=rand_params, n_iter=100, cv=5)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training accuracy
y_train_pred = clf.predict(X_train)
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

# Identify test set accuracy
y_test_pred = clf.predict(X_test)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))

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
scores = np.array(scores).reshape(len(grid_params['alpha']), len(grid_params['hidden_layer_sizes']))
print('scores:')
print(scores)

print('This function took', time.time()-start_time, 'seconds.')

# Show learning curve
for ind, i in enumerate(grid_params['hidden_layer_sizes']):
    # print('hidden_layer_sizes: ' + str(i))
    # print('alpha:' + str(grid_params['alpha']))
    # print('Score:' + str(scores[:,ind]))
    plt.plot(grid_params['alpha'], scores[:,ind], label='hidden_layer_sizes: ' + str(i))
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Mean score')
plt.show()