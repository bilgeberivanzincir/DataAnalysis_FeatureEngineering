import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


df = load_application_train()
df.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

# outliers

sns.boxplot(x=df["Age"])
plt.show()

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index


# function

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


low, up = outlier_thresholds(df, "Age")


def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outliers(df, "Fare")

#grab_col_names

dff = load_application_train()
dff.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_col'un içerisindedir.

    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['int64', 'float64']]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ['category', 'object']]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerIf"]

for col in num_cols:
    print(col, check_outliers(df,col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outliers(dff, col))


# Aykırı değerlerin kendisine erişmek


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10 :
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up ))].index
        return outlier_index


age_index = grab_outliers(df, "Age", True)


# Aykırı Değer Problemini Çözme
# Silme Yöntemi


df.shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    new_df = remove_outlier(df, col)

# Baskılama Yöntemi
# Silme yönteminden ortaya çıkan veri kaybını yaşamamak için kullanılır.

def replace_with_thresholds(dataframe, col):
    low_limit, up_limit = outlier_thresholds(dataframe, col)
    dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
    dataframe.loc[(dataframe[col] > up_limit), col] = up_limit

df = load()
df.shape

for col in num_cols:
    print(col, check_outliers(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)


###################
# Çok değişkenli Aykırı Değer Analizi: Local Outlier Factor
# Belirli bir örneklemin yoğunluğunun komşularına göre yerel sapmasını ölçen' yöntem
###################

# tek başına aykırılık ifade etmeyip baika bir değerle aykırı olanlar

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include = ['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col,check_outliers(df,col))

low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

#çok değişken


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5] #en kötü beş gözlem, outlier scores

#eşik değer ne olmalı problemi
#en son sert değişikliğin olduğu noktayı alabiliriz.
#grafiğe göre 3.index marjinal değişikliğin olduğu score

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50], style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th] #bunlar neden aykırı? 3 tane

df.describe().T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# çok fazla gözlem olduğunda baskılama yapamayız, gürültü yaratır.
# gözlem sayısına göre az outlier olduğunda direkt silebiliriz.
# ağaç yöntemleri kullanılıyorsa outliers değerlerine dokunmamalıyız.
# opsiyon olarak quartile değerini değiştirebiliriz.

#doğrusal yöntemler kullanıyorsak tek değişkenli yaklaşıp baskılamak tercih edilebilir veyahut çok değişkenli ve azsa silebililiriz.


##########
# Missing values(Eksik değerler)
###########

# Silme

df = load()

df.isnull().values.any()
df.isnull().sum()

def missing_values_table(dataframe, na_name= False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

# Eksik değer problemini çözme

#hızlıca silmek
df.dropna().shape

# basit atama yöntemleri ile doldurmak

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)

dff = df.apply(lambda x : x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# Kategorik Değişken Kırılımında Değer Atama
# kadınlara ve erkeklere kendi yaş ortamalarını atama
df.groupby("Sex")["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female")]


df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

# Tahmine dayalı atama ile doldurma

df = load()

cat_cols, num_cols, cat_bur_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# encoder, one hot coding
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.dtypes
df.dtypes

# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
dff.dtypes


# knn'in uygulanması


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(),["Age", "age_imputed_knn"]]

dff.head()

#############
# Gelişmiş Analizler, eksik değerlerin rastsallığı
#############

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

# Eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesş


na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe,target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FlAG'] = np.where(temp_df[col].isnull(),1,0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "COUNT": temp_df.groupby(col)[target].count()}), end="\n\n")

missing_vs_target(df,"Survived", na_cols)

