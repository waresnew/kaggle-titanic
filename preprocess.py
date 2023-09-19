import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess(data):
    data['Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    data['Family'] = data['SibSp'] + data['Parch']+1
    data['Alone'] = data['Family'].apply(lambda x: 1 if x == 1 else 0)
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    # replace uncommon titles with "Other"
    data['Title'] = data['Title'].replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder(sparse_output=False)) # need dense array to output as dataframe
    ])
    data_pipeline = ColumnTransformer([
        ('num', num_pipeline, ['Age', 'Fare']),
        ('cat', cat_pipeline, ['Embarked', 'Sex', 'Title']),

    ])
    data_pipeline.set_output(transform='pandas')
    extra = data_pipeline.fit_transform(data)
    data = data.drop(['Name', 'Title', 'SibSp', 'Parch', 'Ticket', 'Age', 'Embarked', 'Sex', 'Fare'], axis=1)
    data = pd.concat([data, extra], axis=1)
    data = data.astype(np.float32)
    return data


if __name__ == '__main__':
    preprocess(pd.read_csv('train.csv'))
