import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.linear_model import ElasticNetCV

df = pd.read_csv('D:\\AI_ML\\lab1\\processed_ds.csv')

X = df.drop(['bmi', 'id'], axis=1)
y = df['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lin = linear_model.predict(X_test)

print("Линейная регрессия")
print("RMSE:", root_mean_squared_error(y_test, y_pred_lin))
print("MAE:", mean_absolute_error(y_test, y_pred_lin))

n = 2
poly_features = PolynomialFeatures(n)

X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

print("\nПолиномиальная регрессия")
print("RMSE:", root_mean_squared_error(y_test, y_pred_poly))
print("MAE:", mean_absolute_error(y_test, y_pred_poly))

ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
model_cv = ElasticNetCV(l1_ratio=ratios, alphas=100, cv=5, max_iter=10000)
model_cv.fit(X_train_poly, y_train)

print(f"\nИдеальное alpha: {model_cv.alpha_}")
print(f"Идеальное l1_ratio: {model_cv.l1_ratio_}")

y_pred = model_cv.predict(X_test_poly)

print("\nПосле регуляризации")
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

X_cl = df.drop(['stroke', 'id'], axis=1)
y_cl = df['stroke']

X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X_cl, y_cl, test_size=0.4, random_state=42)
X_cl_test, X_cl_val, y_cl_test, y_cl_val = train_test_split(X_cl_test, y_cl_test, test_size=0.4, random_state=42)

logreg_model = LogisticRegression(class_weight='balanced')
logreg_model.fit(X_cl_train, y_cl_train)

y_pred_logreg = logreg_model.predict(X_cl_test)

cm = confusion_matrix(y_cl_test, y_pred_logreg)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

print("\nЛогистическая регрессия")

print(classification_report(y_cl_test, y_pred_logreg))
plt.show()

