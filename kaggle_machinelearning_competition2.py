# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:04:54 2022

@author: Yunus
"""

# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
# iowa_file_path = '../input/train.csv'
# home_data = pd.read_csv(iowa_file_path)
home_data = pd.read_csv("train.csv")
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'MiscVal', 'PoolArea', 'ScreenPorch', 'Fireplaces', 'WoodDeckSF',
'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'MSSubClass', 'OverallQual', 'OverallCond']

# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

rf_val_predictions2 = rf_model_on_full_data.predict(val_X)
rf_val_mae2 = mean_absolute_error(rf_val_predictions2, val_y)

print("Validation MAE for Random Forest Model on Full Data: {:,.0f}".format(rf_val_mae2))



# Then in last code cell
# test_data_path = '../input/test.csv'
test_data = pd.read_csv("test.csv")
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)


# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission2.csv', index=False)



