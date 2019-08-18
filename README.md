# Supervised Learning

### Introduction

In this project I compared 5 popular machine learning algorithms against 2 different data sets (wine quality and adult income) with the goal being accurate prediction.

Algorithms:
- Decision trees (pruned)
- Neural network
- Boosted Decision Trees
- Support Vector Machines
- k-nearest neighbors

The full report is available here: [Report](/Analysis.pdf)

#### Wine Quality

The wine quality problem includes 11 independent variable attributes (acidity, sugar, alcohol, etc). The dependent variable the learner will try to predict is a score on the 1-10 scale (whole numbers).

**Model Results Comparison:**

![Wine Model Results Compared](./Wine/Final%20Graphs/Wine%20Model%20Results%20Compared.png)

While all of the models performed well (better than a simplistic model), two performed better than each of the others: SVM (support vector machines) and KNN (k-nearest neighbors). 

See the learning curve for SVM below:

![Wine SVM Learning Curve](./Wine/Final%20Graphs/SVM%20Learning%20Curve.png)

See the learning curve for KNN below:

![Wine KNN Learning Curve](./Wine/Final%20Graphs/KNN%20Learning%20Curve.png)

When evaluating a model’s performance, we should consider: the score against training data, the mean score against cross
validated data, the score against a held-out test set, we should review the learning curve (the aforementioned scores plotted
against the number of training examples) for bias/variance, and we should consider the model’s training time.

SVM Model:
- This model’s performance against training data is 99.1%
- This model’s performance against CV data is 58.6%
- This model’s performance against test data is 63.6%
- This model took approximately 9 minutes to run

KNN Model:
- This model’s performance against training data is 100%
- This model’s performance against CV data is 57.7%
- This model’s performance against test data is 63.7%
- This model took approximately 11 minutes to run

By reviewing the learning curves for both models above we can see that they exhibit high variance. We can tell, because the training set scores are very far from the test scores.

Knowing that the model shows high variance, we can tell that adding more training examples, or simplifying the model might improve it. Additionally, with training scores above 99% it’s clear that the model is overfitting the data. There’s likely room to remove some of the
attributes that add only noise, or increase regularization to reduce overfitting.

#### Adult Income

The adult income problem classifies adults into one of two income categories: ‘>50K’, or ‘<=50K’. The ‘>50K’ category identifies individuals that earned more than $50,000 in the given year, 1994. The ‘<=50K’ category identifies individuals that earned less than or equal to $50,000. $50,000 in 1994 is approximately $81,000 in today’s terms. The data has 13 attributes, 5 of which are real valued (age, hours worked per week, etc), and 8 of which are categorical (education, marital status, race, etc).

**Model Results Comparison:**

![Income Model Results Compared](./Income/Final%20Graphs/Income%20Model%20Results%20Compared.png)

All of the models tested performed well, but SVM (support vector machines) produced the highest test score in nearly the fastest time. See the learning curve for SVM below:

![Income SVM Learning Curve](./Income/Final%20Graphs/SVM%20Learning%20Curve.png)

When evaluating a model’s performance, we should consider: the score against training data, the mean score against cross
validated data, the score against a held-out test set, we should review the learning curve (the aforementioned scores plotted
against the number of training examples) for bias/variance, and we should consider the model’s training time.

- This model’s performance against training data is 85.5%
- This model’s performance against CV data is 83.8%
- This model’s performance against test data is 83.9%
- This model took approximately 5 minutes to run

By reviewing the learning curve above, we can see that the model exhibits high bias. We can tell, because it is consistent and fairly close to the test score.

Knowing that the model shows high bias, we can tell that adding more training examples, or simplifying the model won’t
improve it. This makes conceptual sense, since we determined the pruning factor ‘alpha’ that minimized CV error. The pruning
factor alpha is an attempt to regularize or reduce the impact of additional features. Since this model exhibits high bias, if we
wanted to improve it we would need to search for more features for the existing instances. For example, we could see if region
within the US was a strong indicator of income.