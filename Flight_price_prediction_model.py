#!/usr/bin/env python
# coding: utf-8

# ## Regression Modeling - Predicting Flight Price
# 
# In this section, we build, tune, and evaluate regression models to predict flight prices using advanced feature engineering and cross-validation.
# 

# In[21]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import joblib


# In[22]:


df = pd.read_csv('flights.csv').drop_duplicates()
print("Data shape:", df.shape)
print(df.dtypes)
df.head()


# In[23]:


categorical = ['from', 'to', 'flightType', 'agency']
numerical   = ['time', 'distance']
X = df[categorical + numerical]
y = df['price']


# In[24]:


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
], remainder='passthrough')


# In[25]:


rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])


# In[26]:


linreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression(positive=True))
])


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[28]:


rf_pipeline.fit(X_train, y_train)
linreg_pipeline.fit(X_train, y_train)


# In[29]:


y_pred_train_rf = rf_pipeline.predict(X_train)
y_pred_test_rf  = rf_pipeline.predict(X_test)


# In[30]:


y_pred_train_lr = linreg_pipeline.predict(X_train)
y_pred_test_lr  = linreg_pipeline.predict(X_test)


# In[36]:


def print_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{label} Performance:")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.3f}")
    print(f"  MAPE : {mape:.2f}%")


# In[37]:


print_metrics(y_train, y_pred_train_rf, "RandomForest Train")
print_metrics(y_test,  y_pred_test_rf,  "RandomForest Test")

print_metrics(y_train, y_pred_train_lr, "LinearReg (pos) Train")
print_metrics(y_test,  y_pred_test_lr,  "LinearReg (pos) Test")


# In[38]:


# 11. Test on a known row
test_row = {
    'from': 'Florianopolis (SC)',
    'to':   'Natal (RN)',
    'flightType': 'economic',
    'agency':     'CloudFy',
    'time':       1.84,
    'distance':   709.37
}
row_df = pd.DataFrame([test_row])
print("\nRF predict:",  rf_pipeline.predict(row_df)[0])
print("LR predict:",   linreg_pipeline.predict(row_df)[0])


# In[39]:


joblib.dump(rf_pipeline,    'flight_price_rf_pipeline.joblib')
joblib.dump(linreg_pipeline,'flight_price_linreg_pos_pipeline.joblib')
print("\nPipelines saved: flight_price_rf_pipeline.joblib, flight_price_linreg_pos_pipeline.joblib")

