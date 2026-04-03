import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


df = pd.read_csv('D:\\AI_ML\\lab1\\processed_ds.csv')

X = df.drop(['bmi', 'id'], axis=1)
y = df['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = DecisionTreeRegressor(max_depth=2, random_state=42)

reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print("Регрессия:   ")
print("RMSE: ", root_mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))

tree.plot_tree(reg_model, feature_names=X.columns, filled=True, fontsize=10)
plt.show()

X_cl = df.drop(['stroke', 'id'], axis=1)
y_cl = df['stroke']

X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X_cl, y_cl, test_size=0.2, random_state=42)

class_model = DecisionTreeClassifier(max_depth=2, random_state=42)
class_model.fit(X_cl_train, y_cl_train)

y_cl_pred = class_model.predict(X_cl_test)

y_proba = class_model.predict_proba(X_cl_test)

fpr, tpr, thresholds = roc_curve(y_cl_test, y_proba[:, 1])
uc_metric = auc(fpr, tpr)

plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')

print(f"ROC-AUC: {uc_metric}")
plt.show()