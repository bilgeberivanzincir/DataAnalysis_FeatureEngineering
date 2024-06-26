import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df.isnull().values.any()
def check_df(dataFrame, head=5):
    print("####### Shape #######")
    print(dataFrame.shape)
    print("####### Type ########")
    print(dataFrame.dtypes)
    print("####### Head ########")
    print(dataFrame.head(head))
    print("####### Tail ########")
    print(dataFrame.tail(head))
    print("####### NA ########")
    print(dataFrame.isnull().sum())
    print("####### Quantiles ########")
    print(dataFrame.describe().T)

check_df(df)

df['sex'].unique()
df['sex'].nunique()
df["survived"].value_counts()

df.info()

# Analysis of Categorical Variables

# bool, categoric, object
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# numeric ama categoric olanlar, benzersiz sınıf sayısı 10'dan küçükse categoric diyebiliriz.
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ['int', 'float']]

# object veya categoric tipli olup  sınıf sayısı çok fazla (isim soyisim) ise kardinalitesi yüksek değişkendir.
# ölçüm değeri, açıklanabilirlik taşımazlar. categoric ama cardinaş
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ['category', 'object']]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100* dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

# df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)


################################
# Analysis of Numerical Variables

num_cols = [col for col in df.columns if df[col].dtypes in ['int','float']]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(df[numerical_col].describe(quantiles).T)

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

df = sns.load_dataset("titanic")
# Değişkenlerin Yakalanması
# docstring

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

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['int64', 'int32', 'float32', 'float64']]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ['category', 'object']]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int', 'float']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car =  grab_col_names(df)

df.head()
