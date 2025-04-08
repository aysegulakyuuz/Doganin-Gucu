import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



energy = pd.read_csv("energy_dataset.csv")
weather = pd.read_csv("weather_features.csv")
df = pd.concat([energy, weather], axis=1)

columns_to_drop = [
    "generation hydro pumped storage aggregated",
    "forecast wind offshore eday ahead",
    "generation fossil coal-derived gas",
    "generation fossil oil shale",
    "generation fossil peat",
    "generation geothermal",
    "generation marine",
    "generation wind offshore",
    "rain_3h"
]

df = df.drop(columns=columns_to_drop, axis=1)


df.head()
df.tail()


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
#    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

######################################
# TARİH TİP DÖNÜŞÜMÜ
######################################
df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.tz_convert(None)



##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=35):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)


######################################
# Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

def cat_summary(dataframe, col_name, plot=True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)




######################################
# Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)


######################################
# Hedef Değişkenin Belirlenmesi
######################################

generation_columns = [col for col in df.columns if 'generation' in col]
df['total_generation'] = df[generation_columns].sum(axis=1)

df.head(5)

######################################
# Hedef Değişken Analizi (Analysis of Target Variable)
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"total_generation",col)


# TRANSFORMATION
# Bağımlı değişkenin incelenmesi
df["total_generation"].hist(bins=100)
plt.show(block=True)

# Bağımlı değişkenin logaritmasının incelenmesi
np.log1p(df['total_generation']).hist(bins=50)
plt.show(block=True)


######################################
# Korelasyon Analizi (Analysis of Correlation)
######################################

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    dataframe = dataframe.select_dtypes(include=['float64', 'int64'])

    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()

    return drop_list

high_correlated_cols(df, True)



######################################
#  Feature Engineering
######################################

######################################
# Aykırı Değer Analizi
######################################

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "total_generation":
      print(col, check_outlier(df, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "total_generation":
        replace_with_thresholds(df,col)

for col in num_cols:
    if col != "total_generation":
      print(col, check_outlier(df, col))



######################################
# Eksik Değer Analizi
######################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# EKSİK DEĞERLERİN SİLİNMESİ (Yaklaşık 35000 satırdan Max 36 tanesi boştur)

df = df.dropna()

######################################
# Rare analizi yapınız ve rare encoder uygulayınız.
######################################

# Kategorik kolonların dağılımının incelenmesi
def rare_analyser_with_enhanced_plot(dataframe, target, cat_cols):
    for col in cat_cols:
        # Kategorik değişkenin değer dağılımını alıyoruz
        temp_df = pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        })

        # COUNT ve TARGET_MEAN grafiği
        plt.figure(figsize=(12, 8))

        # COUNT çubuk grafiği
        sns.barplot(x=temp_df.index, y="COUNT", data=temp_df, palette="viridis", edgecolor="black", label="COUNT")
        plt.xticks(rotation=45)
        plt.xlabel(col, fontsize=14)
        plt.ylabel("COUNT", fontsize=14)
        plt.title(f"Distribution and Target Mean of {col}", fontsize=16, fontweight="bold", color="darkblue")

        # İkincil y eksenine TARGET_MEAN çiziyoruz
        ax2 = plt.twinx()
        sns.lineplot(x=temp_df.index, y="TARGET_MEAN", data=temp_df, color="orange", marker="o", markersize=8,
                     linewidth=2.5, label="TARGET_MEAN", ax=ax2)
        ax2.set_ylabel("TARGET_MEAN", fontsize=14)

        # Grafiklerin birleşik bir şekilde gözükmesi için etiketleri ve başlıkları özelleştiriyoruz
        ax2.legend(loc="upper left")
        plt.legend(loc="upper right")

        # Grid ve görsel düzenlemeler
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Grafiği göster
        plt.show(block=True)


# Örneğin kullanım:
rare_analyser_with_enhanced_plot(df, "total_generation", cat_cols)
#Sunuma grafik ekleyelim buradan


# Nadir sınıfların tespit edilmesi
# Sunuma grafik


# Nadir kategorileri 'Rare' olarak etiketleyen fonksiyon
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


# Pasta grafikleriyle kategorik kolonların analizi
def rare_analyser_with_pie_plot(dataframe, target, cat_cols):
    for col in cat_cols:
        print(f"{col} - Unique Categories: {len(dataframe[col].value_counts())}")

        # summary_df oluşturuluyor
        summary_df = pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        })

        print(summary_df, end="\n\n")

        # Pasta grafiği oluşturma
        plt.figure(figsize=(8, 8))
        plt.pie(
            summary_df["COUNT"],
            labels=summary_df.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel")
        )
        plt.title(f'{col} - Distribution')
        plt.show(block=True)


# Örnek kullanım
df = rare_encoder(df, 0.01)  # Nadir kategoriler 'Rare' olarak etiketleniyor
rare_analyser_with_pie_plot(df, "total_generation", cat_cols)  # Pasta grafikleriyle dağılım analizi


########################################
# Yeni Değişkenlerin Oluşturulması
########################################
print(df.columns)

# --- 1. Zaman Bazlı Değişkenler ---
df['NEW_Hour'] = df['time'].dt.hour
df['NEW_Day'] = df['time'].dt.day
df['NEW_Weekday'] = df['time'].dt.weekday
df['NEW_IsWeekend'] = df['NEW_Weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
df['NEW_Month'] = df['time'].dt.month
df['NEW_Season'] = df['NEW_Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                          'Spring' if x in [3, 4, 5] else
                                          'Summer' if x in [6, 7, 8] else
                                          'Fall')

# Sabah (5-12), Öğlen (12-18), Akşam (18-22), Gece (22-5)
df['NEW_TimeOfDay'] = df['NEW_Hour'].apply(lambda x: 'Morning' if 5 <= x < 12 else
                                           'Afternoon' if 12 <= x < 18 else
                                           'Evening' if 18 <= x < 22 else
                                           'Night')

# --- 2. Enerji Kaynağıyla İlgili Değişkenler ---
renewable_sources = ['generation hydro run-of-river and poundage',
                     'generation hydro water reservoir',
                     'generation wind onshore',
                     'generation solar']
fossil_sources = ['generation fossil hard coal',
                  'generation fossil gas',
                  'generation fossil oil']

# Yenilenebilir enerji oranı
df['NEW_RenewableRatio'] = df[renewable_sources].sum(axis=1) / df['total_generation']

# Fosil enerji oranı
df['NEW_FossilRatio'] = df[fossil_sources].sum(axis=1) / df['total_generation']

# En yüksek enerji kaynağı
df['NEW_HighestEnergySource'] = df[generation_columns].idxmax(axis=1)

# --- 3. Hava Durumu Değişkenleri ---
# Hissedilen sıcaklık
df['NEW_TempFeeling'] = df['temp'] - (0.55 * (1 - (df['humidity'] / 100)) * (df['temp'] - 14.5))

# Fırtınalı hava
df['NEW_IsStormy'] = ((df['wind_speed'] > 20) & (df['humidity'] > 70)).astype(int)

# Günlük sıcaklık farkı
df['NEW_TempRange'] = df['temp_max'] - df['temp_min']

# Sıcaklık anormalliği (ortalama sapması)
daily_avg_temp = df['temp'].mean()
df['NEW_TempAnomaly'] = df['temp'] - daily_avg_temp

# --- 4. Oran ve Kombinasyon Değişkenleri ---
# Güneş ve rüzgar enerjisinin toplamı
df['NEW_SolarWindTotal'] = df['generation wind onshore'] + df['generation solar']

# Güneş ve rüzgar enerjisi oranı
df['NEW_SolarWindRatio'] = df['generation solar'] / (df['generation wind onshore'] + 1e-9)

# Nem ve rüzgar etkisi
df['NEW_HumidityWindImpact'] = df['humidity'] * df['wind_speed']

# --- 5. İstatistiksel Değişkenler ---
# Enerji üretiminde zaman içindeki farklar
df['NEW_GenerationChange'] = df['total_generation'].diff().fillna(0)

# Hareketli ortalama (ör. 3 saatlik)
df['NEW_MovingAvg_3h'] = df['total_generation'].rolling(window=3).mean().fillna(method='bfill')

# Enerji üretim trendi (zaman serisi eğilimi)
df['NEW_GenerationTrend'] = df['total_generation'].rolling(window=12).apply(lambda x: x[-1] - x[0], raw=True).fillna(0)

# --- 6. Kategorik Değişken Türetilmesi ---
# Ortalama nem seviyesine göre kategorik sınıflama
df['NEW_HumidityCategory'] = pd.cut(df['humidity'], bins=[0, 30, 60, 100], labels=['Low', 'Moderate', 'High'])

# --- 7. Özel İlişkiler ---
# Yenilenebilir enerji ve sıcaklık arasındaki etkileşim
df['NEW_TempRenewableImpact'] = df['temp'] * df['NEW_RenewableRatio']

# Hava sıcaklığı ve rüzgar hızının enerji üretimine etkisi
df['NEW_WeatherImpactOnEnergy'] = (df['temp'] + df['wind_speed']) * df['total_generation']

########################################
# Yeni Değişkenlerin Kontrolü
########################################
print("Yeni değişkenler oluşturuldu.")
df.head()


######!!!!!!!!!!!1drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]
# drop_list'teki değişkenlerin düşürülmesi
########!!!!!!!!!!!!!!!!!!!!!!!!!!!1df.drop(drop_list, axis=1, inplace=True)


#ENCODING

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape

#Hedef Değişkenin Dönüşümü

print("Skewness:", df['total_generation'].skew())
print("Kurtosis:", df['total_generation'].kurt())
Skewness: 0.24140772741494837
Kurtosis: -0.4522752011770632
#Çarpıklık ve basıklık değerleri normal kabul edilebilecek sınırlar içinde olduğu için dönüşüm yapmamıza gerek yok
#Bu haliyle modellemeye geçebiliriz

# Histogram
df["total_generation"].hist(bins=50)
plt.title("Hedef Değişken Dağılımı")
plt.show(block=True)

# Boxplot
sns.boxplot(x=df["total_generation"])
plt.title("Hedef Değişken Boxplot")
plt.show(block=True)

##################################
# MODELLEME
##################################

y = df["total_generation"]
X = df.drop("total_generation", axis=1)

#veri türü kontrol etme
print(X.dtypes)
#verileri modele hazırlama

# Tarihi bileşenlerine ayırma
X['year'] = X['time'].dt.year
X['month'] = X['time'].dt.month
X['day'] = X['time'].dt.day
X['hour'] = X['time'].dt.hour

print(X.columns)  # Tüm sütun adlarını yazdır

# Orijinal time sütununu kaldır
X.drop('time', axis=1, inplace=True)

# Tüm bool sütunlarını 0 ve 1'e dönüştür
X = X.astype({'NEW_Weekday_5': 'int',
              'NEW_Weekday_6': 'int',
              'NEW_IsWeekend_1': 'int',
              'NEW_HumidityCategory_Moderate': 'int',
              'NEW_HumidityCategory_High': 'int'})



print(X.dtypes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim öncesi RMSE (ortalama tahmini)
baseline_prediction = np.mean(y_train)
baseline_rmse = np.sqrt(mean_squared_error(y_test, [baseline_prediction] * len(y_test)))
print(f"Eğitim öncesi RMSE: {baseline_rmse:.4f}")


#tüm model türlerinin performansını karşılaştırma
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
RMSE: 167.2790244208568

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse

print(pd.DataFrame(results.items(), columns=["Model", "RMSE"]))



"""
              Model    RMSE
0  LinearRegression  33.814
1             Ridge  33.480
2             Lasso  32.605
3      RandomForest 170.408
4           XGBoost 136.740
5          LightGBM 118.964
6          CatBoost  89.956


Model sonuçlarını incelediğinizde, 
Lasso modelinin en düşük RMSE'ye sahip olduğunu görebiliyoruz. Bu, Lasso'nun test verisi üzerinde en iyi performansı gösterdiği anlamına gelir.

İşte her modelin RMSE değerlerinin kısa bir özeti:

Lasso: 32.605 (En iyi performans)
Ridge: 33.480
LinearRegression: 33.814
CatBoost: 89.956
LightGBM: 118.964
XGBoost: 136.740
RandomForest: 170.408 (En yüksek RMSE, yani en kötü performans)




"""

# MAE
from sklearn.metrics import mean_absolute_error
# Modeli eğitme, tahmin yapma ve MAE'yi hesaplama
for name, regressor in models.items():
    model = regressor.fit(X_train, y_train)  # Modeli eğitim verisiyle eğitiyoruz
    y_pred = model.predict(X_test)  # Test verisi ile tahmin yapıyoruz
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)  # MAE'yi hesaplıyoruz
    print(f"MAE: {mae} ({name})")  # Sonucu yazdırıyoruz

    """
MAE: 20.421830001855245 (LinearRegression)
MAE: 20.011101467356617 (Ridge)
MAE: 18.737169584223498 (Lasso)
MAE: 101.20557824100513 (RandomForest)
MAE: 98.05249901841805 (XGBoost)
MAE: 83.56496815162923 (LightGBM)
MAE: 65.12167259082241 (CatBoost)

 
"""


# Ortalama ve standart sapma hesaplama
mean_value = df['total_generation'].mean()
std_value = df['total_generation'].std()

# Sonuçları yazdırma
print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_value}")

# Histogram oluşturma
plt.figure(figsize=(10, 6))  # Grafik boyutunu ayarlama
df["total_generation"].hist(bins=100, color='skyblue', edgecolor='black', alpha=0.7)

# Eksen başlıkları ve başlık ekleme
plt.xlabel('Total Generation', fontsize=12, fontweight='bold')  # X ekseni başlığı
plt.ylabel('Frequency', fontsize=12, fontweight='bold')         # Y ekseni başlığı
plt.title('Histogram of Total Generation', fontsize=14, fontweight='bold')  # Grafik başlığı

# Izgara ekleme
plt.grid(True, linestyle='--', alpha=0.7)  # Şeffaf ve kesikli ızgara

# Eksenleri daha belirgin hale getirme
plt.tick_params(axis='both', which='major', labelsize=10)

# Grafiği gösterme
plt.show(block=True)







#CATBOOST ÜZERİNE HİPERPARAMETRE


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


catboost_model = CatBoostRegressor(loss_function='RMSE', verbose=0)

param_grid = {
    'learning_rate': [0.01, 0.05],  # Daha düşük öğrenme oranı
    'max_depth': [3, 5],            # Ağaç derinliğini sınırlama
    'n_estimators': [300, 600],     # İterasyon sayısı
    'reg_lambda': [1, 5],           # L2 düzenleme parametresi
}


random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_grid,
    n_iter=5,                             # 5 farklı parametre kombinasyonu dene
    cv=5,                                 # 5 katlamalı çapraz doğrulama
    scoring='neg_mean_squared_error',     # MSE üzerinden değerlendirme yap
    verbose=2,
    random_state=42                      # Sonuçların tekrarlanabilir olması için sabit random_state
)


fit_params = {
    "eval_set": [(X_val, y_val)],
    "early_stopping_rounds": 100,
    "verbose": 10
}
random_search.fit(X_train, y_train, **fit_params)

# En iyi parametreleri yazdır
best_params_random = random_search.best_params_
print(f"En iyi parametreler (Randomized Search): {best_params_random}")

# En iyi modeli al ve test et
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)

# RMSE ve MAE hesapla
rmse_random = np.sqrt(mean_squared_error(y_test, y_pred_random))
print(f"En iyi modelin RMSE: {rmse_random}")

mae_random = mean_absolute_error(y_test, y_pred_random)
print(f"En iyi modelin MAE: {mae_random}")


"""
En iyi parametreler (Randomized Search): {'reg_lambda': 1, 'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.05}
En iyi modelin RMSE: 118.364084075655
En iyi modelin MAE: 87.14236149985852
"""

# Test verileri ve tahminler
y_pred = best_model_random.predict(X_test)  # Modelin tahmin ettiği değerler
residuals = y_test - y_pred          # Hataları hesapla (Gerçek - Tahmin)

# Hata Dağılım Histogramı
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color="green", edgecolor="black", alpha=0.7)
plt.title("Hata Dağılımı", fontsize=16)
plt.xlabel("Hata (Gerçek - Tahmin)", fontsize=12)
plt.ylabel("Frekans", fontsize=12)
plt.grid(True)
plt.show()



# Tahminler ve Gerçek Değerler
y_pred = best_model_random.predict(X_test)

# Gerçek ve Tahmin Edilen Değerleri Karşılaştırma
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45 derece doğru
plt.title("Gerçek vs Tahmin Edilen Değerler", fontsize=16)
plt.xlabel("Gerçek Değerler", fontsize=12)
plt.ylabel("Tahmin Edilen Değerler", fontsize=12)
plt.grid(True)
plt.show()




# Tahmin ve Gerçek Değerlerin Görselleştirilmesi

# Gerçek ve tahmini değerler
y_true = y_test  # Gerçek değerler
y_pred = best_model_random.predict(X_test)  # Tahmini değerler

plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolor='k')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, color='red')
plt.title('Gerçek vs Tahmini Değerler', fontsize=16)
plt.xlabel('Gerçek Değerler', fontsize=14)
plt.ylabel('Tahmini Değerler', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# Hata Dağılımı
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=50, kde=True, color="red")
plt.title("Tahmin Hatalarının Dağılımı (Gerçek - Tahmin) Catboost")
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.show(block=True)


# 2. Gerçek ve Tahmin Değerlerinin Karşılaştırılması
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Gerçek Değerler", color="blue", alpha=0.8)
plt.plot(y_pred, label="Tahmin Değerleri", color="orange", linestyle="--", alpha=0.8)
plt.title("Gerçek ve Tahmin Değerlerinin Karşılaştırılması Catboost")
plt.xlabel("Örnekler (Sıralı)")
plt.ylabel("Değerler")
plt.legend()
plt.show(block=True)

# 3. Hata Dağılımı
errors = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=50, kde=True, color="red", alpha=0.6)
plt.title("Hata Dağılımı (Gerçek - Tahmin) Catboost")
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.show(block=True)

# 4. Gerçek vs Tahmin (Dağılım Grafiği)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.title("Gerçek Değerler vs Tahmin Değerleri Catboost")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerleri")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.show(block=True)





# feature importance fonksiyonu
def plot_importance(model, features, num=len(X), save=False):
    # Modelin özellik önem derecelerini alıyoruz
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})

    # Grafik ayarları
    plt.figure(figsize=(12, 8))  # Grafik boyutunu ayarlıyoruz
    sns.set(font_scale=1.2)  # Yazı boyutunu büyütüyoruz
    sns.set_style("whitegrid")  # Beyaz arka plan ve grid hatları

    # Önem derecelerine göre sıralama
    sorted_feature_imp = feature_imp.sort_values(by="Value", ascending=False).head(num)

    # Barplot çizimi
    ax = sns.barplot(x="Value", y="Feature", data=sorted_feature_imp, palette="viridis")

    # Barların üzerine değerleri ekleyelim
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=12, color='black')

    # Başlık ve etiketler
    plt.title("Feature Importance", fontsize=16)
    plt.xlabel("Importance Value", fontsize=14)
    plt.ylabel("Features", fontsize=14)

    plt.tight_layout()

    plt.show(block=True)


# Eğitilen CatBoost modelini kullanarak görselleştirme
plot_importance(model, X)  # Tüm özellikler için önem grafiğini çiz
plot_importance(model, X, num=30)  # En önemli 30 özelliği çiz





#LGBM HİPERPARAMETRE

# Veri setini böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM modelini tanımla
lgbm_model = LGBMRegressor(objective='regression', random_state=42, verbose=-1)

# Parametre aralıklarını belirle
param_grid = {
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5],
    'n_estimators': [300, 600],
    'reg_lambda': [1, 5]
}

# RandomizedSearchCV ile parametre araması yap
random_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_grid,
    n_iter=5,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42
)

# Modeli eğitirken eval_set ve early_stopping callback'ini kullan
random_search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],               # Doğrulama veri seti
    callbacks=[early_stopping(stopping_rounds=100)]  # Erken durdurma için callback
)

# En iyi parametreleri yazdır
best_params_random = random_search.best_params_
print(f"En iyi parametreler (Randomized Search): {best_params_random}")

# En iyi modeli al ve test et
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)

# Performans metriklerini hesapla
rmse_random = np.sqrt(mean_squared_error(y_test, y_pred_random))
mae_random = mean_absolute_error(y_test, y_pred_random)
r2_random = r2_score(y_test, y_pred_random)

print(f"En iyi modelin RMSE: {rmse_random}")
print(f"En iyi modelin MAE: {mae_random}")
print(f"En iyi modelin R-squared (R2): {r2_random}")


"""
En iyi parametreler (Randomized Search): {'reg_lambda': 1, 'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.05}
En iyi modelin RMSE: 104.112347153653
En iyi modelin MAE: 71.25197876517399
En iyi modelin R-squared (R2): 0.9993780421213722
"""


# ---- Görselleştirmeler ---- #
from lightgbm import plot_importance
import matplotlib.pyplot as plt

# Özellik önemini çiz
plt.figure(figsize=(10, 8))
plot_importance(best_model_random, max_num_features=30, importance_type='split', title="Feature Importance")
plt.show()

# Rastgele örnekleme
sample_indices = np.random.choice(len(y_test), size=500, replace=False)  # 500 örnek al
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]

# Scatter Plot: Tahmin vs. Gerçek
plt.figure(figsize=(8, 6))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.6, color="orange")
plt.plot([min(y_test_sample), max(y_test_sample)], [min(y_test_sample), max(y_test_sample)], color="blue", linewidth=2)
plt.title("Gerçek ve Tahmin Değerlerinin Scatter Grafiği LGBM")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerleri")
plt.show(block=True)

# Hata Dağılımı
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=50, kde=True, color="red")
plt.title("Tahmin Hatalarının Dağılımı (Gerçek - Tahmin) LGBM")
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.show(block=True)


# 2. Gerçek ve Tahmin Değerlerinin Karşılaştırılması
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Gerçek Değerler", color="blue", alpha=0.8)
plt.plot(y_pred, label="Tahmin Değerleri", color="orange", linestyle="--", alpha=0.8)
plt.title("Gerçek ve Tahmin Değerlerinin Karşılaştırılması LGBM")
plt.xlabel("Örnekler (Sıralı)")
plt.ylabel("Değerler")
plt.legend()
plt.show(block=True)

# 3. Hata Dağılımı
errors = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=50, kde=True, color="red", alpha=0.6)
plt.title("Hata Dağılımı (Gerçek - Tahmin) LGBM")
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.show(block=True)

# 4. Gerçek vs Tahmin (Dağılım Grafiği)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.title("Gerçek Değerler vs Tahmin Değerleri LGBM")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerleri")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.show(block=True)




# RandomForest Hiperparametre

# RandomForest modelini tanımla
rf_model = RandomForestRegressor(random_state=42)

# Hiperparametre aralıklarını belirle
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# RandomizedSearchCV ile hiperparametre araması
random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Modeli eğit
random_search_rf.fit(X_train, y_train)

# En iyi parametreleri yazdır
best_params_rf = random_search_rf.best_params_
print(f"En iyi hiperparametreler (RandomForest): {best_params_rf}")

# En iyi modeli al ve test et
best_rf_model = random_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Performans metrikleri
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"RandomForest RMSE: {rmse_rf}")
print(f"RandomForest MAE: {mae_rf}")
"""
En iyi hiperparametreler (RandomForest): {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
RandomForest RMSE: 330.4748612001989
RandomForest MAE: 246.23040823909113
"""
# Özellik önem sırası
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# En önemli 30 faktörü yazdır
top_30_factors = feature_importances.head(30)
print("En önemli 30 faktör:")
print(top_30_factors)

# En önemli 30 özelliği çubuk grafikte görselleştir
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_30_factors,
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title('En Önemli 30 Özellik', fontsize=16)
plt.xlabel('Önem Skoru', fontsize=12)
plt.ylabel('Özellikler', fontsize=12)
plt.tight_layout()
plt.show()

# Hata dağılım grafiği
errors = y_test - y_pred_rf

plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='red', alpha=0.7)
plt.title('Hata Dağılım Grafiği', fontsize=16)
plt.xlabel('Hata (Gerçek - Tahmin)', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Gerçek vs. Tahmin Analizi
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Gerçek vs. Tahmin Analizi (RandomForest)', fontsize=16)
plt.xlabel('Gerçek Değerler', fontsize=12)
plt.ylabel('Tahmin Değerleri', fontsize=12)
plt.tight_layout()
plt.show()






# Korelasyon analizi
correlation = df.corr()
print(correlation)
# Enerji üretimi ile hava durumu değişkenleri arasındaki korelasyon
correlation_with_energy = correlation["total_generation"].sort_values(ascending=False)
print(correlation_with_energy)





 # Tarih sütununu datetime formatına dönüştürme
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Mevsim sütunu oluşturma
df['season'] = df['time'].dt.month % 12 // 3 + 1  # 1: Kış, 2: İlkbahar, 3: Yaz, 4: Sonbahar

# Mevsimlere göre enerji üretim ortalamaları
seasonal_effect = df.groupby("season")["total_generation"].mean()
print(seasonal_effect)

# Günün saatlerine göre enerji üretimi
df['hour'] = df['time'].dt.hour
hourly_effect = df.groupby("hour")["total_generation"].mean()
print(hourly_effect)


# Yenilenebilir ve fosil enerji oranlarının ortalamalarını hesaplama
renewable_avg = df['NEW_RenewableRatio'].mean()
fossil_avg = df['NEW_FossilRatio'].mean()

# Pasta grafiği
labels = ['Yenilenebilir Enerji', 'Fosil Enerji']
sizes = [renewable_avg, fossil_avg]
colors = ['#4CAF50', '#FF5722']  # Renk paleti
explode = (0.1, 0)  # Yenilenebilir enerjiyi vurgulamak için
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
plt.title('Enerji Üretiminde Yenilenebilir ve Fosil Kaynakların Oranı')
plt.show(block=True)

# Sütun grafiği
avg_data = pd.DataFrame({
    'Kaynak Türü': ['Yenilenebilir Enerji', 'Fosil Enerji'],
    'Oran': [renewable_avg, fossil_avg]
})

plt.figure(figsize=(8, 6))
sns.barplot(data=avg_data, x='Kaynak Türü', y='Oran', palette=colors)
plt.title('Yenilenebilir ve Fosil Enerji Oranları', fontsize=14)
plt.xlabel('Kaynak Türü')
plt.ylabel('Ortalama Oran')
plt.ylim(0, 1)  # Oranlar yüzde olduğu için 0 ile 1 arasında
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show(block=True)

# Korelasyon matrisi
correlation_matrix = df.corr()

# Enerji üretimi ile değişkenler arasındaki korelasyonları sıralama
correlation_with_energy = correlation_matrix["total_generation"].abs().sort_values(ascending=False)

# İlk 30 değişkeni seçme (total_generation dahil)
top_10_features = correlation_with_energy.head(10).index

# İlk 30 değişkenin korelasyon matrisini oluşturma
top_10_correlation_matrix = df[top_10_features].corr()

# Korelasyon tablosunu görselleştirme
plt.figure(figsize=(15, 10))
sns.heatmap(
    top_10_correlation_matrix,
    annot=True,               # Hücrelere korelasyon değerlerini yazdır
    fmt=".2f",                # Ondalık hassasiyet
    cmap="RdBu",              # Renk paleti
    vmin=-1, vmax=1,          # Korelasyon değer aralığı
    linewidths=0.5            # Hücre çizgi kalınlığı
)
plt.title("İlk 10 Değişkenin Korelasyon Tablosu", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show(block=True)

 # Tarih sütununu datetime formatına dönüştürme
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Mevsim sütunu oluşturma
df['season'] = df['time'].dt.month % 12 // 3 + 1  # 1: Kış, 2: İlkbahar, 3: Yaz, 4: Sonbahar

# Mevsimlere göre enerji üretim ortalamaları
seasonal_effect = df.groupby("season")["total_generation"].mean()
print(seasonal_effect)

# Günün saatlerine göre enerji üretimi
df['hour'] = df['time'].dt.hour
hourly_effect = df.groupby("hour")["total_generation"].mean()
print(hourly_effect)

# Aylara göre enerji üretim ortalamaları
monthly_effect = df.groupby(df['time'].dt.month)["total_generation"].mean()
print(monthly_effect)

# Aylara göre enerji üretimi görselleştirme
plt.figure(figsize=(12, 6))
colors = sns.color_palette("Spectral", n_colors=12)
sns.barplot(x=monthly_effect.index, y=monthly_effect.values, palette=colors)
plt.title("Aylara Göre Ortalama Enerji Üretimi", fontsize=18, fontweight="bold")
plt.xlabel("Ay", fontsize=14, labelpad=10)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=14, labelpad=10)
plt.xticks(ticks=range(12), labels=["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                                     "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
           fontsize=12, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine()
plt.show(block=True)


# Yıllara göre enerji üretim ortalamaları
yearly_effect = df.groupby(df['time'].dt.year)["total_generation"].mean()
print(yearly_effect)

# Yıllara göre enerji üretimi görselleştirme
plt.figure(figsize=(12, 6))
colors = sns.color_palette("Spectral", n_colors=len(yearly_effect))
sns.barplot(x=yearly_effect.index, y=yearly_effect.values, palette=colors)
plt.title("Yıllara Göre Ortalama Enerji Üretimi", fontsize=18, fontweight="bold")
plt.xlabel("Yıl", fontsize=14, labelpad=10)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=14, labelpad=10)
plt.xticks(ticks=range(len(yearly_effect)), labels=yearly_effect.index, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine()
plt.show(block=True)



# Mevsimlere göre enerji üretimi
plt.figure(figsize=(10, 6))
sns.barplot(x=seasonal_effect.index, y=seasonal_effect.values, palette="coolwarm")
plt.title("Mevsimlere Göre Ortalama Enerji Üretimi", fontsize=16)
plt.xlabel("Mevsim", fontsize=12)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3], labels=["Kış", "İlkbahar", "Yaz", "Sonbahar"])
plt.show(block=True)

# Günün saatlerine göre enerji üretimi
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_effect.index, y=hourly_effect.values, marker="o", color="blue")
plt.title("Günün Saatlerine Göre Ortalama Enerji Üretimi", fontsize=16)
plt.xlabel("Saat", fontsize=12)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
plt.grid(True)
plt.show(block=True)


# Seçilen değişkenler ve hedef değişken
selected_columns = ['temp', 'wind_speed', 'NEW_WeatherImpactOnEnergy', "NEW_Hour"]
target_column = 'total_generation'

# Seçilen değişkenler ve hedef değişkenin korelasyonlarını hesaplama
selected_correlation_matrix = df[selected_columns + [target_column]].corr()

# Korelasyon tablosunu görselleştirme (heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(
    selected_correlation_matrix,
    annot=True,                # Hücrelere korelasyon değerlerini yazdır
    fmt=".2f",                 # Ondalık hassasiyet
    cmap="RdBu",               # Renk paleti
    vmin=-1, vmax=1,           # Korelasyon değer aralığı
    linewidths=0.5,            # Hücre çizgi kalınlığı
    cbar_kws={"shrink": 0.8}   # Renk çubuğu boyutunu ayarlama
)
plt.title("Seçilen Değişkenlerin Korelasyon Grafiği", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Bulutluluk ile enerji türleri arasındaki korelasyonu hesaplama
energy_columns = [
    "generation solar",
    "generation wind onshore",
    "generation hydro water reservoir",
    "generation biomass",
"generation fossil gas",
    "generation fossil oil",
    'generation nuclear',

]
clouds_column = "clouds_all"

correlation_with_clouds = df[energy_columns + [clouds_column]].corr()[clouds_column].drop(clouds_column)
print("Bulutluluk ile Enerji Türleri Arasındaki Korelasyon:")
print(correlation_with_clouds)

# Görselleştirme: Korelasyon grafiği
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.barplot(
    x=correlation_with_clouds.values,
    y=correlation_with_clouds.index,
    palette="coolwarm"
)
plt.title("Bulutluluğun Enerji Türleri Üzerindeki Etkisi (Korelasyon)", fontsize=14)
plt.xlabel("Korelasyon")
plt.ylabel("Enerji Türleri")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Bulutluluk seviyesini gruplama (bulutlu ve bulutsuz)
df["is_cloudy"] = (df[clouds_column] > 50).astype(int)

# Bulutlu ve bulutsuz günlerdeki enerji üretim ortalamalarını karşılaştırma
cloudy_effect = df.groupby("is_cloudy")[energy_columns].mean().T
cloudy_effect.columns = ["Bulutsuz Günler", "Bulutlu Günler"]

# Sonuçları yazdırma
print("Bulutlu ve Bulutsuz Günlerde Enerji Üretimi Ortalamaları:")
print(cloudy_effect)

# Görselleştirme
cloudy_effect.plot(kind="bar", figsize=(12, 6), colormap="viridis")
plt.title("Bulutlu ve Bulutsuz Günlerde Enerji Üretimi", fontsize=14)
plt.ylabel("Ortalama Enerji Üretimi")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Dağılım Grafikleri (Scatterplot)
for energy in energy_columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[clouds_column], y=df[energy], alpha=0.6, color='blue')
    plt.title(f"Bulutluluk ({clouds_column}) ile {energy} Arasındaki İlişki", fontsize=14)
    plt.xlabel("Bulutluluk (%)")
    plt.ylabel(energy)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Kutu Grafikleri (Boxplot): Bulutlu ve bulutsuz günlerin etkisi
df["is_cloudy"] = (df[clouds_column] > 50).astype(int)

for energy in energy_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="is_cloudy", y=energy, data=df, palette="coolwarm")
    plt.title(f"Bulutlu ve Bulutsuz Günlerde {energy} Üretimi", fontsize=14)
    plt.xlabel("Bulutlu (1: Bulutlu, 0: Bulutsuz)")
    plt.ylabel(energy)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

temp_effect = df.groupby("temp")[target_column].mean()

# Sıcaklığa göre enerji üretimi grafiği
plt.figure(figsize=(12, 6))
sns.lineplot(x=temp_effect.index, y=temp_effect.values, marker="o", color="green")
plt.title("Sıcaklığa Göre Ortalama Enerji Üretimi", fontsize=16)
plt.xlabel("Sıcaklık", fontsize=12)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
plt.grid(True)
plt.show(block=True)

wind_effect = df.groupby("wind_speed")[target_column].mean()

# Rüzgar hızına göre enerji üretimi grafiği
plt.figure(figsize=(12, 6))
sns.lineplot(x=wind_effect.index, y=wind_effect.values, marker="o", color="purple")
plt.title("Rüzgar Hızına Göre Ortalama Enerji Üretimi", fontsize=16)
plt.xlabel("Rüzgar Hızı", fontsize=12)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
plt.grid(True)
plt.show(block=True)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sayısal veriyi 50 gruba ayırmak için pd.cut kullanma (min ve max'a göre)
num_bins = 50  # 50 gruba ayırmak için
bins = pd.cut(df["NEW_WeatherImpactOnEnergy"], bins=num_bins)

# Her grup için ortalama enerji üretimi
weather_effect = df.groupby(bins)[target_column].mean()

# Hava durumunun enerji üretimine etkisi
plt.figure(figsize=(12, 6))
sns.lineplot(x=weather_effect.index.astype(str), y=weather_effect.values, marker="o", color="blue")
plt.title("Hava Durumuna Göre Ortalama Enerji Üretimi (50 Gruplama)", fontsize=16)
plt.xlabel("Hava Durumu Etkisi (Gruplar)", fontsize=12)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
plt.xticks(rotation=90)  # X eksenindeki etiketleri döndürme, çünkü 50 grup var
plt.tight_layout()
plt.show(block=True)

print(df.columns)


sources = ["generation fossil gas", "generation fossil oil", "generation solar", "generation biomass", 'generation hydro water reservoir', 'generation nuclear']

for source in sources:
    source_effect = df.groupby(source)[target_column].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=source_effect.index, y=source_effect.values, marker="o")
    plt.title(f"{source} Üretimine Göre Ortalama Enerji Üretimi", fontsize=16)
    plt.xlabel(source, fontsize=12)
    plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
    plt.grid(True)
    plt.show(block=True)

solar_wind_ratio_effect = df.groupby("NEW_SolarWindRatio")[target_column].mean()

# Güneş ve rüzgar üretim oranına göre enerji üretimi
plt.figure(figsize=(12, 6))
sns.lineplot(x=solar_wind_ratio_effect.index, y=solar_wind_ratio_effect.values, marker="o", color="orange")
plt.title("Güneş ve Rüzgar Oranına Göre Ortalama Enerji Üretimi", fontsize=16)
plt.xlabel("Güneş/Rüzgar Oranı", fontsize=12)
plt.ylabel("Ortalama Enerji Üretimi", fontsize=12)
plt.grid(True)
plt.show(block=True)


df["is_raining"] = ((df['rain_1h_0.3'] > 0) | (df['rain_1h_0.9'] > 0) | (df['rain_1h_3.0'] > 0)).astype(int)



#YAĞMUR İÇİN GRAFİKLER
correlation_with_rain = df[energy_columns + ['is_raining']].corr()['is_raining'].drop('is_raining')
print("Yağmur ile Enerji Türleri Arasındaki Korelasyon:")
print(correlation_with_rain)

# Korelasyon Grafiği
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_rain.values, y=correlation_with_rain.index, palette="coolwarm")
plt.title("Yağmurun Enerji Türleri Üzerindeki Korelasyonu", fontsize=14)
plt.xlabel("Korelasyon")
plt.ylabel("Enerji Türleri")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Yağmur Durumuna Göre Enerji Üretimi
rain_effect = df.groupby("is_raining")[energy_columns].mean().T
rain_effect.columns = ["Yağmursuz Günler", "Yağmurlu Günler"]
print("Yağmurlu ve Yağmursuz Günlerde Enerji Üretimi Ortalamaları:")
print(rain_effect)

rain_effect.plot(kind="bar", figsize=(12, 6), colormap="viridis")
plt.title("Yağmurlu ve Yağmursuz Günlerde Enerji Üretimi", fontsize=14)
plt.ylabel("Ortalama Enerji Üretimi")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Rüzgar Hızı İçin Enerji Türleri Arasındaki Korelasyon
correlation_with_wind = df[energy_columns + ['wind_speed']].corr()['wind_speed'].drop('wind_speed')
print("Rüzgar Hızı ile Enerji Türleri Arasındaki Korelasyon:")
print(correlation_with_wind)

# Korelasyon Grafiği
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_wind.values, y=correlation_with_wind.index, palette="coolwarm")
plt.title("Rüzgar Hızının Enerji Türleri Üzerindeki Korelasyonu", fontsize=14)
plt.xlabel("Korelasyon")
plt.ylabel("Enerji Türleri")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Rüzgar Hızı Durumuna Göre Enerji Üretimi
df["is_windy"] = (df["wind_speed"] > df["wind_speed"].median()).astype(int)
wind_effect = df.groupby("is_windy")[energy_columns].mean().T
wind_effect.columns = ["Düşük Rüzgar", "Yüksek Rüzgar"]
print("Düşük ve Yüksek Rüzgar Durumlarında Enerji Üretimi Ortalamaları:")
print(wind_effect)

wind_effect.plot(kind="bar", figsize=(12, 6), colormap="viridis")
plt.title("Düşük ve Yüksek Rüzgar Durumlarında Enerji Üretimi", fontsize=14)
plt.ylabel("Ortalama Enerji Üretimi")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Sıcaklık İçin Enerji Türleri Arasındaki Korelasyon
correlation_with_temp = df[energy_columns + ['temp']].corr()['temp'].drop('temp')
print("Sıcaklık ile Enerji Türleri Arasındaki Korelasyon:")
print(correlation_with_temp)

# Korelasyon Grafiği
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_temp.values, y=correlation_with_temp.index, palette="coolwarm")
plt.title("Sıcaklığın Enerji Türleri Üzerindeki Korelasyonu", fontsize=14)
plt.xlabel("Korelasyon")
plt.ylabel("Enerji Türleri")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. Sıcaklık Durumuna Göre Enerji Üretimi
df["is_hot"] = (df["temp"] > df["temp"].median()).astype(int)
temp_effect = df.groupby("is_hot")[energy_columns].mean().T
temp_effect.columns = ["Soğuk", "Sıcak"]
print("Soğuk ve Sıcak Hava Durumlarında Enerji Üretimi Ortalamaları:")
print(temp_effect)

temp_effect.plot(kind="bar", figsize=(12, 6), colormap="viridis")
plt.title("Soğuk ve Sıcak Hava Durumlarında Enerji Üretimi", fontsize=14)
plt.ylabel("Ortalama Enerji Üretimi")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 7. Bulutluluk İçin Enerji Türleri Arasındaki Korelasyon
correlation_with_clouds = df[energy_columns + ['clouds_all']].corr()['clouds_all'].drop('clouds_all')
print("Bulutluluğun Enerji Türleri Arasındaki Korelasyon:")
print(correlation_with_clouds)

# Korelasyon Grafiği
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_clouds.values, y=correlation_with_clouds.index, palette="coolwarm")
plt.title("Bulutluluğun Enerji Türleri Üzerindeki Korelasyonu", fontsize=14)
plt.xlabel("Korelasyon")
plt.ylabel("Enerji Türleri")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 8. Bulutluluk Durumuna Göre Enerji Üretimi
df["is_cloudy"] = (df["clouds_all"] > 50).astype(int)
cloud_effect = df.groupby("is_cloudy")[energy_columns].mean().T
cloud_effect.columns = ["Bulutsuz", "Bulutlu"]
print("Bulutlu ve Bulutsuz Günlerde Enerji Üretimi Ortalamaları:")
print(cloud_effect)

cloud_effect.plot(kind="bar", figsize=(12, 6), colormap="viridis")
plt.title("Bulutlu ve Bulutsuz Günlerde Enerji Üretimi", fontsize=14)
plt.ylabel("Ortalama Enerji Üretimi")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 9. Dağılım Grafikleri (Scatter Plot) - Tüm Hava Durumları
for weather in ['is_raining', 'is_windy', 'is_hot', 'is_cloudy']:
    for energy in energy_columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[weather], y=df[energy], alpha=0.6, color='green')
        plt.title(f"{weather} ile {energy} Arasındaki İlişki", fontsize=14)
        plt.xlabel(weather)
        plt.ylabel(energy)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# 10. Kutu Grafikleri (Box Plot) - Tüm Hava Durumları
for weather in ['is_raining', 'is_windy', 'is_hot', 'is_cloudy']:
    for energy in energy_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=weather, y=energy, data=df, palette="coolwarm")
        plt.title(f"{weather} Durumunda {energy} Üretimi", fontsize=14)
        plt.xlabel(f"{weather} Durumu")
        plt.ylabel(energy)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def correlation_with_total_generation(df, weather_cols, target_columns):
    # Hedef değişken (toplam enerji üretimi ve diğer hedef sütunları) ile hava durumu değişkenlerini seçiyoruz
    cols = target_columns + weather_cols

    # Korelasyon matrisini hesaplıyoruz
    corr_matrix = df[cols].corr()

    # Korelasyon matrisini görselleştiriyoruz
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='black', fmt='.2f')
    plt.title(f'Correlation between Total Generation and Weather Conditions', fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


weather_columns = ['is_raining', 'is_windy', 'is_hot', 'is_cloudy']  # Hava durumu kolonları
target_columns = ["total_generation", "generation solar", "generation wind onshore",
                  "generation hydro water reservoir", "generation biomass",
                  "generation fossil gas", "generation fossil oil", "generation nuclear"]

correlation_with_total_generation(df, weather_cols=weather_columns, target_columns=target_columns)
