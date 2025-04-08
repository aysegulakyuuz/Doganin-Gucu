🌿 Doğanın Gücü – Proje Özeti
🎯 Proje Amacı
Bu proje, Madrid bölgesinde 2015-2018 yılları arasında kaydedilen hava durumu verileri ile enerji üretimi verilerini birleştirerek, hava koşullarının enerji üretimine olan etkisini analiz etmeyi amaçlamaktadır. Hangi hava durumu faktörlerinin enerji üretiminde ne ölçüde belirleyici olduğu araştırılmıştır.

🔍 Veri Seti
Kaynaklar: Yenilenebilir (güneş, rüzgar, hidro) ve fosil enerji üretimi verileri

Hava durumu değişkenleri: Sıcaklık, rüzgar hızı, nem, bulutluluk, yağmur gibi faktörler

Gözlem sayısı: ~35.000

Zaman aralığı: 2015–2018

🛠️ Yapılan Çalışmalar
Keşifsel Veri Analizi (EDA)

Zaman serisi grafikleri ve eksik/veri aykırı kontrolleri yapıldı.

Özellik Mühendisliği

20’den fazla yeni değişken oluşturuldu (ör. NEW_TempRenewableImpact, NEW_WeatherImpactOnEnergy).

Modelleme

Hedef değişken: total_generation (toplam enerji üretimi)

Kullanılan modeller:

Lasso (RMSE: 32.6 – En iyi sonuç)

Ridge, LinearRegression

CatBoost, LightGBM, RandomForest, XGBoost

Hiperparametre optimizasyonları yapıldı (RandomizedSearchCV ile)

Model Performansı

Lasso: RMSE 32.6, MAE 18.7

CatBoost: RMSE 89.9, MAE 65.1

LightGBM: RMSE 104.1, MAE 71.2

RandomForest: RMSE 330.4, MAE 246.2

Korelasyon Analizleri ve Görselleştirmeler

Hava koşullarının enerji kaynakları üzerindeki etkisi detaylı analizlerle görselleştirildi (boxplot, scatter, heatmap, barplot).

💡 Ana Bulgular
Rüzgar hızı ve sıcaklık, yenilenebilir enerji üretiminde oldukça etkili.

Bulutluluk, güneş enerjisini negatif etkiliyor; rüzgar enerjisini ise pozitif etkileyebilir.

Yağmur, güneş üretimini düşürürken bazı durumlarda rüzgar üretimini arttırabiliyor.

En iyi tahmin performansı Lasso Regresyon modelinden elde edildi.

📌 İş Önerileri
Enerji yönetimi, hava durumu tahminleriyle desteklenmeli.

Bulutlu ve yağmurlu günlerde güneş yerine rüzgar veya hidro gibi kaynaklara ağırlık verilmeli.

Zaman bazlı (mevsimsel, saatlik) üretim tahminleriyle optimizasyon sistemleri geliştirilebilir.
