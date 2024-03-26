import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from data_analysis import cat_summary
from feature_engineering import load, load_application_train, grab_col_names

# Label Encoding & Binary Encoding

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
                and df[col].nunique() == 2]

df.dtypes

for col in binary_cols:
    label_encoder(df, col)

df.head()

dfff = load_application_train()
binary_cols = [col for col in dfff.columns if dfff[col].dtype not in ["int64", "float64"]
                and dfff[col].nunique() == 2]

#label encoder na'lere de bir değer atar. bunun farkında ol.

for col in binary_cols:
    label_encoder(dfff, col)

dfff[binary_cols].head()


# One Hot Encoding
# Kategorik değişkenin sınıfları değişkenlere dönüşmektedir.
# get_dummies() ile binary encoding ve one hot encoding aynı anda yapılabilir.

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe



cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


one_hot_encoder(df, ohe_cols).head()


# RARE ENCODİNG
# bonus

df = load_application_train()
df.head()
df["NAME_EDUCATION_TYPE"].value_counts()

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.


cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in cat_cols:
    cat_summary(df, col)

# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}),end="\n\n")

rare_analyser(df, "TARGET", cat_cols)

# 3. Rare encoder yazacağız.


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)   # sınıf oranları
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

new_df["OCCUPATION_TYPE"].value_counts()


############
# Feature Scaling
##########

# StandardScaler: Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s


df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

# RobustScaler: Medyanı çıkar iqr'a böl. Aykırı değerlere karşı dayanıklı.

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

# MinMaxScaler
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

# Numeric to Categorical: Sayısal değişkenleri kategorik değişkenlere çevirme
# Binning

df["Age_qcut"] = pd.qcut(df['Age'], 5)
