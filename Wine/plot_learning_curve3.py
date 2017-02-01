import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# seperating data sets for cross validation

# compute the rms error
def compute_error(x, y, model):
    yfit = model.predict(x)
    rms = np.sqrt(np.mean((y - yfit) ** 2))
    ascore = accuracy_score(y, yfit)
    #return rms
    return ascore


def drawLearningCurve(model, X_train, X_test, y_train, y_test, min_size=1, numpoints=50):
    sizes = np.linspace(min_size, X_train.shape[0], numpoints, endpoint=True).astype(int)
    train_error = np.zeros(sizes.shape)
    crossval_error = np.zeros(sizes.shape)

    for i, size in enumerate(sizes):
        # getting the predicted results of the GaussianNB
        model.fit(X_train[:size, :], y_train[:size])
        predicted = model.predict(X_train)

        # compute the validation error
        crossval_error[i] = compute_error(X_test, y_test, model)

        # compute the training error
        train_error[i] = compute_error(X_train[:size, :], y_train[:size], model)

    # draw the plot
    fig, ax = plt.subplots()
    ax.plot(sizes, crossval_error, lw=2, label='test error')
    ax.plot(sizes, train_error, lw=2, label='training error')
    ax.set_xlabel('training examples')
    ax.set_ylabel('rms error')

    ax.legend(loc=0)
    ax.set_xlim(0, X_train.shape[0]+1)
    ax.set_title('Learning Curve')