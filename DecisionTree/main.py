import mysql.connector
import pandas as pd
import numpy as np

from ConfusionMatrix import compute_tp_tn_fn_fp, compute_accuracy, compute_precision, \
    compute_recall, compute_f1_score
from InsertEvaluateData import insertEvaluateData

from train_test_split import train_test_split

from ConnectDatabase import connectDatabase
from DecisionTree import Node
df,db = connectDatabase()


# df = df.head(15000)
# print(df)




#Chia tập train test
df = df.drop(columns={12,9,11,13,10,3,4,15})
X_train, X_test, y_train, y_test = train_test_split(df,0.7,5)



Y = y_train
X = X_train
features = list(X.columns)
# print(features)
hp = {
    'max_depth': 5,
    'min_samples_split': 200
}
#Khởi tạo node
root = Node(Y, X, **hp)
#Split tốt nhất
root.grow_tree()
#Print thông tin cây
root.print_tree()
#Dự đoán
results = X.copy()
predict = root.predict(X_test)

# print(predict)



result = pd.DataFrame({'Actual:': y_test, 'Predict:': predict})
print(result)

y_test = y_test.tolist()
y_test = np.array(y_test)
predict = np.array(predict)
print(type(y_test), type(predict))
# predict = np.array()



tp_rf, tn_rf, fp_rf, fn_rf = compute_tp_tn_fn_fp(y_test, predict)
print('TP for Random Forest :', tp_rf)
print('TN for Random Forest :', tn_rf)
print('FP for Random Forest :', fp_rf)
print('FN for Random Forest :', fn_rf)

acc = compute_accuracy(tp_rf, tn_rf, fn_rf, fp_rf)
precision_ = compute_precision(tp_rf, fp_rf)
recall = compute_recall(tp_rf, fn_rf)
f1_score = compute_f1_score(y_test, predict)
print('Accuracy for Random Forest :', acc)
print('Precision for Random Forest :', precision_)
print('Recall for Random Forest :', recall)
print('F1 score for Random Forest :', f1_score)
arr = [ acc, precision_, recall, f1_score]


insertEvaluateData(arr,db)