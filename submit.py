import pandas as pd

from model import get_model
from preprocess import preprocess

data = preprocess(pd.read_csv('test.csv'))
print(data.head())
model = get_model()
predictions = model.predict(data).flatten()
ans = pd.DataFrame(columns=['PassengerId', 'Survived'])
for i in range(len(predictions)):
    ans.loc[len(ans)] = {'PassengerId': data.loc[i, 'PassengerId'], 'Survived': round(predictions[i])}
ans['PassengerId'] = ans['PassengerId'].astype(int)
ans['Survived'] = ans['Survived'].astype(int)
ans.to_csv('output.csv', index=False)
