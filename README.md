House Price Prediction Project

This project is a complete, end-to-end machine learning workflow to predict house sale prices based on a real-world dataset. The pipeline includes data loading from a SQL database, exploratory data analysis (EDA), data cleaning, feature engineering, and model comparison.

The final model is deployed as a working REST API using Flask.

Project Workflow & Key Decisions

This project follows the standard machine learning workflow.

1. Data Sourcing & Cleaning

Source: Data was loaded from a Kaggle CSV (housedata.csv).

Migration: The data was migrated into a SQLite database (project.db) using Pandas' .to_sql() method for efficient and robust data handling.

Cleaning:

Target Variable: Investigated the price column and found 49 entries with a value of $0. As this is impossible data, these rows were removed.

Outliers: Visual analysis (histograms) showed the price was heavily right-skewed by a few multi-million dollar properties. To build a model for the majority of houses, I filtered out the top 1% of prices, which dramatically improved model performance.

2. Feature Engineering & Preprocessing

To prepare the data for the model, I created several new features:

Date Handling: Converted the date column into datetime objects.

New Features Created:

sale_year: Extracted from the date.

sale_month: Extracted from the date to check for seasonal trends.

house_age: Calculated by sale_year - yr_built to create a more informative feature than yr_built alone.

Dropped Redundant Features: yr_built and sale_year were dropped to avoid multicollinearity.

3. The "Split First" Pipeline (Preventing Data Leakage)

To prevent the test set from influencing the training process, I strictly followed the "Split First, Process Second" rule.

Split: The cleaned data was split into X_train and X_test before any processing.

Process Separately:

Categorical Features: The city column was one-hot encoded using pd.get_dummies(..., drop_first=True).

Numerical Features: All numerical columns were scaled using StandardScaler.

Align Columns: X_test_processed was aligned to X_train_processed using .reindex() to ensure the columns matched perfectly, preventing errors from categories that might only appear in the training or test set.

4. Target Variable Transformation

The price (our target y) was also heavily skewed. To help the models learn better, I applied a log transform (np.log(y)) before training. Predictions were then converted back to dollars using np.exp().

Model Evaluation & Results

I trained and compared three different regression models to find the best performer. The models were evaluated on their Mean Absolute Error (MAE) (average dollar error) and Mean Absolute Percentage Error (MAPE) (average percentage error).

Final Model Results (Trained on 99% of data, outliers removed):

Model

MAE (in Dollars)

MAPE

Linear Regression

**$101,564.53**

**23.70%**

Random Forest

**$104,010.70**

**21.04%**

Gradient Boosting

**$102,878.31**

**21.54%**

Conclusion

The LinearRegression model achieved the lowest absolute dollar error, but the RandomForestRegressor achieved the lowest percentage error.

For a general-purpose model, MAPE is a better metric as it is not as skewed by high-value homes. A 21% average error is a strong result. The RandomForest model is the best all-around performer for this dataset.

How to Run This Project

1. Run the Analysis Pipeline

The file house_price_analysis.py contains the complete workflow.

# First, install all required packages
pip install -r requirements.txt

# Run the script to train models and see the results
python house_price_analysis.py


2. Run the API

The file api.py loads the saved model (.joblib) and artifacts to serve live predictions.

# Run the server
python api.py

# In a new terminal, send a POST request with test data
curl -Uri [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) -Method Post -ContentType "application/json" -InFile test_house.json