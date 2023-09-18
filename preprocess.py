import pandas as pd
import numpy as np
import math


def preprocess(data):
    hot_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    hot_sex = pd.get_dummies(data['Sex'], prefix='Sex')
    data = data.drop(['Name', 'Cabin', 'Ticket', 'Embarked', 'Sex'], axis=1)
    data = data.join(hot_embarked).join(hot_sex)
    mean_age = math.floor(data['Age'].mean())+0.5 # estimates end with 0.5 in existing data so be consistent
    data.fillna(mean_age, inplace=True)
    data = data.astype(np.float32)
    return data


if __name__ == '__main__':
    preprocess(pd.read_csv('train.csv'))
