# Supervised Learning

### This directory contains two subfolders:
- /Wine/
- /Income/

Each subfolder has all of the code used in this assignment. 

#### Wine
For the Wine Quality data set, data is loaded in the wine_data.py file, which pulls it directly from the UCI site.

The learners are split into their own file, and call the other files in the folder as necessary.
Learner files are:

- Decision trees: wine_decision.py
- Neural network: wine_nerual.py
- Boosting: wine_adaboost.py
- SVM: wine_svm.py
- kNN: wine_knn.py

#### Income
For the Adult Income data set, data is loaded in the income_data.py file, which pulls it from the train.csv file in the directory. I hard-coded my directory in this file, so you may have to adjust this.

The learners are split into their own file, and call the other files in the folder as necessary. The structure mirrors the structure for the Wine directory.
Learner files are:

- Decision trees: income_decision.py
- Neural network: income_nerual.py
- Boosting: income_adaboost.py
- SVM: income_svm.py
- kNN: income_knn.py

