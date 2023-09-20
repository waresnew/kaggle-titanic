import os.path

import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import preprocess
import optuna.integration.lightgbm as lgb

csv = preprocess(pd.read_csv('train.csv'))
ans = csv['Survived']
csv = csv.drop(['Survived'], axis=1)
print(csv.head())

x_train, x_test, y_train, y_test = train_test_split(csv, ans, test_size=0.2, random_state=42)
data = lgb.Dataset(x_train, label=y_train)
test = lgb.Dataset(x_test, label=y_test)
params = {
    'objective': 'binary', # binary classification
    'metric': 'auc', # auc is good for binary classification
    'verbosity': -1
}
model = lgb.train(params, data, num_boost_round=1000, valid_sets=[data, test], callbacks=[early_stopping(10, first_metric_only=True)])
print('Accuracy: ', accuracy_score(y_test, model.predict(x_test, num_iteration=model.best_iteration).round()))
print("Params: \n", model.params)
test = preprocess(pd.read_csv('test.csv'))
predictions = model.predict(test, num_iteration=model.best_iteration)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions.round()}, dtype=int)
output.to_csv('output.csv', index=False)
