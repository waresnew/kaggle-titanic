import optuna.logging
import pandas as pd
from lightgbm import early_stopping
from optuna.integration import LightGBMTunerCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import preprocess
import lightgbm as lgb

csv = preprocess(pd.read_csv('train.csv'))
ans = csv['Survived']
csv = csv.drop(['Survived'], axis=1)
print(csv.head())
x_train, x_test, y_train, y_test = train_test_split(csv, ans, test_size=0.4, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
data = lgb.Dataset(x_train, label=y_train) # train dataset
test = lgb.Dataset(x_test, label=y_test) # test dataset, to be used at the VERY END for diagnostics
val = lgb.Dataset(x_val, label=y_val) # validation dataset, to be used for early stopping
params = {
    'objective': 'binary',  # binary classification
    'metric': 'binary_logloss',  # auc overfits bc it only cares about ranking, not values
    'verbosity': -1
}
# only do cv on train data, not test data (do not touch test data until final diagnostic) (good practice)
# early stopping is a hyperparamter that can be tuned from validation data
tuner = LightGBMTunerCV(params, data, verbose_eval=False, num_boost_round=1000, early_stopping_rounds=10)
tuner.run()
print('Best score: ', tuner.best_score)
print('Best params: ', tuner.best_params)
model = lgb.train(tuner.best_params, data, num_boost_round=1000, callbacks=[early_stopping(10)], valid_sets=[data, val])
print('Accuracy: ', accuracy_score(y_test, model.predict(x_test, num_iteration=model.best_iteration).round()))

submit = preprocess(pd.read_csv('test.csv'))
predictions = model.predict(submit, num_iteration=model.best_iteration)
output = pd.DataFrame({'PassengerId': submit['PassengerId'], 'Survived': predictions.round()}, dtype=int)
output.to_csv('output.csv', index=False)
