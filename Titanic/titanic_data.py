import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

titanic_df = pd.read_csv("C:/Users/Myles/Documents/OMSCS/CS7641 ML/Assignment 1/Supervised-Learning/Titanic/input/fulltrain.csv")

# Drop non-predictive factors:
#   PassengerID (an ID assigned by Kaggle, the data set curator)
#   Name
#   Ticket (an ID that is unique per record)
#   Cabin (filled in less than 25% of cases)
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

# Populate instances missing 'Embarked' factor with most popular element
# There are 2 instances that are missing 'Embarked' data.
# S is the most popular element, with around 70% of the other instances using "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# Convert "Embarked" factor into numbers
titanic_df_embarked = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
titanic_df = pd.concat([titanic_df,titanic_df_embarked], axis=1)
titanic_df = titanic_df.drop(['Embarked'], axis=1)

# Convert "Sex" factor into numbers
titanic_df['Sex'].replace('female', 1, inplace=True)
titanic_df['Sex'].replace('male', 0, inplace=True)

# Populate instances missing 'Age' factor with popular random ages
# There are 263 instances missing 'Age' data
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
titanic_df['Age'] = titanic_df['Age'].astype(int)

# Populate instances missing 'Fare' factor with popular random ages
# There is 1 instance missing 'Fare' data
average_fare_titanic = titanic_df["Fare"].mean()
titanic_df["Fare"][np.isnan(titanic_df["Fare"])] = average_fare_titanic

# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
# titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
# titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
# titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0
# titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

# we are classifying on all features
y = titanic_df['Survived'].values
X = titanic_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values

le = LabelEncoder()

# encode classifications as 1 and 0 rather than 2 and 3 (original state)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)