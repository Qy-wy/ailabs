import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(df)

num_missing_cols = df.isnull().any().sum()
print("Количество столбцов с пропусками:", num_missing_cols)
df_rows = len(df)

cols_to_drop = []

print("\nСтолбцы, где пропусков > 5%")
for col in df:
    quantity_of_na = df[col].isnull().sum()
    if (quantity_of_na/df_rows) < 0.05:
       cols_to_drop.append(col)
    else:
        print(col, ": ", df[col].isnull().sum())
        print("Статистика:\n", df[col].describe())

for col in cols_to_drop:
    print("Удаление строк с пустыми значениями там, где пропусков в столбце < 5%")
    df.dropna(subset=[col], inplace=True)

df['bmi'] = df['bmi'].fillna(df['bmi'].median())

print("\nСтолбцы после заполнения пропусков\n")
for col in df:
    print(col)
    print(df[col].isnull().sum())

num_cols = ['age', 'avg_glucose_level', 'bmi']
df[num_cols] = scaler.fit_transform(df[num_cols])

df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
df.to_csv("processed_ds.csv", index=False)