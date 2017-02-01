from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np

def data_size_response(model,trX,teX,trY,teY,score_func,prob=True,n_subsets=20):

    train_errs,test_errs = [],[]
    subset_sizes = np.exp(np.linspace(3,np.log(trX.shape[0]),n_subsets)).astype(int)

    for m in subset_sizes:
        model.fit(trX[:m],trY[:m])
        if prob:
            train_err = score_func(trY[:m],model.predict_proba(trX[:m]))
            test_err = score_func(teY,model.predict_proba(teX))
        else:
            train_err = score_func(trY[:m],model.predict(trX[:m]))
            test_err = score_func(teY,model.predict(teX))
        print("training error: %.3f test error: %.3f subset size: %.3f" % (train_err,test_err,m))
        train_errs.append(train_err)
        test_errs.append(test_err)

    return subset_sizes,train_errs,test_errs

def plot_response(subset_sizes,train_errs,test_errs):

    plt.plot(subset_sizes,train_errs,lw=2)
    plt.plot(subset_sizes,test_errs,lw=2)
    plt.legend(['Training Error','Test Error'])
    plt.xscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Error')
    plt.title('Model response to dataset size')
    plt.show()

# model = # put your model here
# score_func = # put your scoring function here
# response = data_size_response(model,trX,teX,trY,teY,score_func,prob=True)
# plot_response(*response)