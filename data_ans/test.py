import numpy as np
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
import random
import os
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import auc, f1_score


# 固定随机数种子
def Seed_Everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


Seed_Everything(2021)

import sklearn.model_selection

data = pd.read_excel('test.xlsx')
# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit(data['y1'])
# data['y1_temp'] = label_encoder.transform(data['y1'])


x_train, x_test, y_train, y_test = train_test_split(data[['x1', 'x2']], data['y1'], shuffle=True, random_state=2021,
                                                    test_size=0.1)

lgb_model = LGBMRegressor(random_state=2021)

lgb_model.fit(x_train, y_train)

ans_test = lgb_model.predict(x_test)
x_test['y1'] = y_test
x_test['pred_y1'] = ans_test
# x_test['pred_y1'] = ans_test
# accuracy_score_test = accuracy_score(y_test,ans_test)
# #auc_test = auc(y_test,ans_test)
# f1_score_test = f1_score(y_test,ans_test)
print(mean_squared_error(y_test, ans_test))

x_train, x_test2, y_train, y_test2 = train_test_split(data[['x1', 'x2']], data['y2'], shuffle=True, random_state=2021,
                                                      test_size=0.2)
lgb_model2 = LGBMRegressor(random_state=2021)
lgb_model2.fit(x_train, y_train)

ans_test2 = lgb_model2.predict(x_test2)
x_test2['y2'] = y_test2
x_test2['pred_y2'] = ans_test2
# x_test['pred_y1'] = ans_test
# accuracy_score_test = accuracy_score(y_test,ans_test)
# #auc_test = auc(y_test,ans_test)
# f1_score_test = f1_score(y_test,ans_test)
print(mean_squared_error(y_test2, ans_test2))

x_train, x_test3, y_train, y_test3 = train_test_split(data[['x1', 'x2']], data['y3'], shuffle=True, random_state=2021,
                                                      test_size=0.2)
lgb_model3 = LGBMClassifier(random_state=2021)
lgb_model3.fit(x_train, y_train)

ans_test3 = lgb_model3.predict(x_test3)
x_test3['y3'] = y_test3
x_test3['pred_y3'] = ans_test3
print(len(x_test3[x_test3['y3'] == x_test3['pred_y3']]) / len(x_test3))
a = 2
