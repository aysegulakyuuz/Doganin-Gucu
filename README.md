🌿 The Power of Nature - Project Summary
🎯 Project Objective
This project aims to analyze the impact of weather conditions on energy production by combining weather data and energy production data recorded in the Madrid region between 2015 and 2018. The extent to which weather factors are determinants of energy production is investigated.

🔍 Dataset
Sources: Renewable (solar, wind, hydro) and fossil energy production data

Weather variables: Factors such as temperature, wind speed, humidity, cloudiness, rain

Number of observations: ~35.000

Time span: 2015-2018

🛠️ Studies
Exploratory Data Analysis (EDA)

Time series graphs and missing/data outlier checks were performed.

Feature Engineering

More than 20 new variables created (ör. NEW_TempRenewableImpact, NEW_WeatherImpactOnEnergy).

Modeling

Target variable: total_generation (total energy production)

Models used:

Lasso (RMSE: 32.6 – Best result)

Ridge, LinearRegression

CatBoost, LightGBM, RandomForest, XGBoost

Hyperparameter optimizations were made (RandomizedSearchCV ile)

Model Performance

Lasso: RMSE 32.6, MAE 18.7

CatBoost: RMSE 89.9, MAE 65.1

LightGBM: RMSE 104.1, MAE 71.2

RandomForest: RMSE 330.4, MAE 246.2

Correlation Analysis and Visualizations

The impact of weather conditions on energy resources visualized with detailed analysis (boxplot, scatter, heatmap, barplot).

💡 Main Findings
Wind speed and temperature are highly influential in renewable energy production.

Cloudiness negatively affects solar energy, but can positively affect wind energy.

Rain decreases solar production, while in some cases it can increase wind production.

The best prediction performance was obtained from the Lasso Regression model.

📌 Business Recommendations
Energy management should be supported by weather forecasts.

On cloudy and rainy days, wind or hydro resources should be used instead of solar.

Optimization systems can be developed with time-based (seasonal, hourly) production forecasts.
