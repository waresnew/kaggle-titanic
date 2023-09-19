
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from preprocess import preprocess

data = preprocess(pd.read_csv('train.csv'))
ans = data['Survived']
data = data.drop(['Survived'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(data, ans, test_size=0.2, random_state=69)
print(data.head())
model = GradientBoostingClassifier(n_estimators=700, learning_rate=0.1, max_depth=3, random_state=69)
model = model.fit(x_train, y_train)
test_scores = cross_val_score(model, x_test, y_test, cv=5)
train_scores = cross_val_score(model, x_train, y_train, cv=5)
print(train_scores)
print(test_scores)
test = preprocess(pd.read_csv('test.csv'))
predictions = model.predict(test)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions}, dtype=int)
output.to_csv('output.csv', index=False)
