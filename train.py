import pandas as pd
from preprocess import preprocess
import lightgbm as lgb

data = preprocess(pd.read_csv('train.csv'))
ans = data['Survived']
data = data.drop(['Survived'], axis=1)
print(data.head())
data = lgb.Dataset(data, label=ans)
params = {
    'objective': 'binary', # binary classification
    'num_leaves': 15, # num leaves should be less than 2^max_depth,
    'min_data_in_leaf': 100, # generally 100s to 1000s
    'metric': 'auc' # "best metric" for classification bc it's immune to unbalanced data
}
model = lgb.train(params, data, 10)
print(min(lgb.cv(params, data, 10, nfold=5)['valid auc-mean'])) # maximize this value

test = preprocess(pd.read_csv('test.csv'))
predictions = model.predict(test)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions.round()}, dtype=int)
output.to_csv('output.csv', index=False)
