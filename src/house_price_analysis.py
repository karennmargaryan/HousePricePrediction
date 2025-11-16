import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'

conn = sqlite3.connect(DATA_DIR / 'project.db')

command = "SELECT * FROM Houses"
df = pd.read_sql_query(command, conn)
conn.close()

# Converting to Datetime
df['date'] = pd.to_datetime(df['date'])
df['sale_year'] = df['date'].dt.year
df['sale_month'] = df['date'].dt.month

# Adding the age of the house for more precise predictions
df['house_age'] = df['sale_year'] - df['yr_built']

# Filtering df for better visualisation
upper_limit = df['price'].quantile(0.99)
df_filtered = df[df['price'] < upper_limit]

# House Prices Distribution
sns.set_theme(style="whitegrid")

g = sns.displot(df['price'], kde = True, bins = 100)
g.set(title='Distribution of House Prices',
      xlabel='Price',
      ylabel='Frequency (Count)')
plt.show()

g = sns.displot(df_filtered['price'], kde=True, bins=100)
g.set(title='Distribution of House Prices (Filtered)',
      xlabel='Price',
      ylabel='Frequency (Count)')

plt.show()
# ----------------------------------------------

# Price VS sqft
plt.figure(figsize=(10, 6))
g = sns.scatterplot(x='sqft_living', y='price', data=df)
g.set(
    title = 'House Price vs. Square Feet (Living)',
    xlabel = 'Square Feet (Living)',
    ylabel = 'Price'
)

plt.show()

g = sns.scatterplot(x='sqft_living', y='price', data=df_filtered)
g.set(
    title = 'House Price vs. Square Feet (Living) (Filtered)',
    xlabel = 'Square Feet (Living)',
    ylabel = 'Price'
)

plt.show()
# ----------------------------------------------

# Price VS Condition
plt.figure(figsize=(12, 7))

g = sns.boxplot(x='condition', y='price', data=df)
g.set(
    title = 'House Price by Condition',
    xlabel = 'Condition (1=Poor, 5=Excellent',
    ylabel = 'Price',
)
plt.show()

g = sns.boxplot(x='condition', y='price', data=df_filtered)
g.set(
    title = 'House Price by Condition (Filtered)',
    xlabel = 'Condition (1=Poor, 5=Excellent',
    ylabel = 'Price',
)
plt.show()
# ----------------------------------------------

y = df_filtered['price']
y_log = np.log(y)
X = df_filtered.drop(columns=['date', 'street', 'statezip', 'country', 'price', 'yr_built', 'sale_year'])
X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(X, y, y_log, test_size=0.2, random_state=42)

text_columns = ['city']
numeric_columns = X_train.columns.drop(text_columns)

scaler = StandardScaler()
scaler.fit(X_train[numeric_columns])
X_train_num_scaled = scaler.transform(X_train[numeric_columns])
X_test_num_scaled = scaler.transform(X_test[numeric_columns])

X_train_cat_processed = pd.get_dummies(X_train[text_columns], drop_first = True)
X_test_cat_processed = pd.get_dummies(X_test[text_columns], drop_first = True)

X_test_cat_processed = X_test_cat_processed.reindex(columns = X_train_cat_processed.columns, fill_value=0)

X_train_num_scaled_df = pd.DataFrame(
    X_train_num_scaled,
    columns=numeric_columns,
    index=X_train.index
)
X_test_num_scaled_df = pd.DataFrame(
    X_test_num_scaled,
    columns=numeric_columns,
    index=X_test.index
)
X_train_processed = pd.concat([X_train_num_scaled_df, X_train_cat_processed], axis=1)
X_test_processed = pd.concat([X_test_num_scaled_df, X_test_cat_processed], axis=1)

linear_model = LinearRegression()
linear_model.fit(X_train_processed, y_train)
linear_predictions = linear_model.predict(X_test_processed)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mape = mean_absolute_percentage_error(y_test, linear_predictions)

random_model = RandomForestRegressor(random_state=42)
random_model.fit(X_train_processed, y_log_train)
random_predictions = np.exp(random_model.predict(X_test_processed))

random_mae = mean_absolute_error(y_test, random_predictions)
random_mape = mean_absolute_percentage_error(y_test, random_predictions)

print(f"Linear Regression MAE: {linear_mae:.2f}")
print(f"Linear Regression MAPE: {linear_mape:.2%}")
print(f"Random Regression MAE: {random_mae:.2f}")
print(f"Random Regression MAPE: {random_mape:.2%}")

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train_processed, y_log_train)
gb_predictions = np.exp(gb_model.predict(X_test_processed))
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_mape = mean_absolute_percentage_error(y_test, gb_predictions)
print(f"Gradient Boosting MAE: {gb_mae:.2f}")
print(f"Gradient Boosting MAPE: {gb_mape:.2%}")


import joblib
import json

print("\n--- Saving model and processing artifacts ---")

ARTIFACTS_DIR = ROOT_DIR / 'artifacts'

joblib.dump(random_model, ARTIFACTS_DIR / 'random_forest_model.joblib')
joblib.dump(linear_model, ARTIFACTS_DIR / 'linear_model.joblib')
joblib.dump(gb_model, ARTIFACTS_DIR / 'gradient_model.joblib')
joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.joblib')

model_columns = list(X_train_processed.columns)
with open(ARTIFACTS_DIR / 'model_columns.json', 'w') as f:
    json.dump(model_columns, f)

print("Artifacts saved successfully.")