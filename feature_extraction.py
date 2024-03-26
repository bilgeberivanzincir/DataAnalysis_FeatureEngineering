import pandas as pd

from datetime import date
from statsmodels.stats.proportion import proportions_ztest

from feature_engineering import load

# BINARY FEATURES
df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
df.head()

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

#kayda değer bir ilişki midir?
#oran testi
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#pvalue<0.05 ise anlamlı bir farklılık vardır fikri verebilir. modellemede(çok değişkenli etkide) daha net anlaşılır.

# LETTER COUNT

df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

# ÖZEL YAPILARI YAKALAMAK

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

# Regex ile değişken türetmek

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean",
                                                                "Age": ["count", "mean"]})

# Date Değişkenleri Üretmek

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d %H:%M:%S")

dff["year"] = dff['Timestamp'].dt.year
dff["month"] = dff['Timestamp'].dt.month
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

# month diff= iki tarih arasındaki ay farkı : yıl farkı + ay farkı
dff["month_diff"] = (date.today().year - dff['Timestamp'].dt.year) * 12 + (date.today().month - dff['Timestamp'].dt.month)

dff['day_name'] = dff['Timestamp'].dt.day_name()

# Feature Interactions

df = load()
df["NEW_AGE_PCLASS"] = df["Age"] + df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.groupby("NEW_SEX_CAT")["Survived"].mean()