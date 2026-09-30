import pandas as pd
#step1
df = pd.read_csv('titanic.csv')
print('step1')
print(df.shape); print(df.head()); df.info(); print(df.describe())

#step2
kids = df[df['age'] < 12]
print('step2')
print(len(kids), kids['survived'].mean())

#step3
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
if 'deck' in df.columns:
    df = df.drop(columns=['deck'])
else:
    print("Столбец 'deck' не найден")
print('step3')
print(df.isna().sum())

#step4
print('step4')
print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())
print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))

#step5
import matplotlib.pyplot as plt, seaborn as sns
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(df['age'], bins=30); ax[0].set_title('Возраст')
sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]); ax[1].set_title('Цена по классу')
sns.barplot(x='sex', y='survived', data=df, ax=ax[2]); ax[2].set_title('Выживаемость')
fig.tight_layout(); fig.savefig('titanic.png', dpi=120)
plt.show()

#step6
df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv('C:/Users/Егор/Desktop/task4 pandas/clean.csv', index=False)

# step7
df_check = pd.read_csv('C:/Users/Егор/Desktop/task4 pandas/clean.csv') 
print("Проверка - ")
print(df_check.isna().sum()) 
