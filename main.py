import pandas
import numpy as np
import xgboost as xgb
import ast
import csv


# Solution is kept trivial and highly inefficient on purpose as it's provided
# purely as an example which should be straightforward to beat by anyone
def convert_dataset(dataset):
    examples = []
    for blob in dataset['txData']:
        txData = ast.literal_eval(blob)
        examples.append([
            int(txData['from'], 0) % (2**30),
            (int(txData['to'], 0) if txData['to'] is not None else 0) % (2**30),
            int(txData['gas'], 0),
            int(txData['gasPrice'], 0),
            (int(txData['input'][:10], 0) if txData['input'] != '0x' else 0) % (2**30),
            int(txData['nonce'], 0),
        ])
    return np.array(examples)


train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')
testFeatures = convert_dataset(test)

binaryModel = xgb.XGBClassifier(n_estimators=50)
binaryModel.fit(convert_dataset(train), train['Label0'])
binaryPredictions = binaryModel.predict_proba(testFeatures)[:, 1]

regressionModel = xgb.XGBRegressor(n_estimators=50)

regressionModel.fit(
    convert_dataset(train[train['Label0'] == True]),
    train[train['Label0'] == True]['Label1']
)

regressionPredictions = regressionModel.predict(testFeatures)

submission = csv.writer(open('submission.csv', 'w', encoding='UTF8'))
for x, y in zip(binaryPredictions, regressionPredictions):
    submission.writerow([x, y])
