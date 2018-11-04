from __future__ import print_function
import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
from subprocess import call
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Set directory to where data is
#os.chdir(r'Stock-Modeling')

# Execute script to get new data for today (after reddit web scraping)
#scall('python process_reddit.py')
import pandas as pd

# Set Console formatting for panda prints
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

# **********************************************************************************************************************
# Modeling / Prepare Data

data = pd.read_csv('DJ_NEWS_SENTIMENT_DATA.csv')
data_tomorrow = pd.read_csv('DJ_NEWS_SENTIMENT_DATA.csv')

# Move certain columns up by one row for data_tomorrow
data_tomorrow.Anger = data_tomorrow.Anger.shift(+1)
data_tomorrow.Anticipation = data_tomorrow.Anticipation.shift(+1)
data_tomorrow.Disgust = data_tomorrow.Disgust.shift(+1)
data_tomorrow.Fear = data_tomorrow.Fear.shift(+1)
data_tomorrow.Joy = data_tomorrow.Joy.shift(+1)
data_tomorrow.Sadness = data_tomorrow.Sadness.shift(+1)
data_tomorrow.Surprise = data_tomorrow.Surprise.shift(+1)
data_tomorrow.Trust = data_tomorrow.Trust.shift(+1)
data_tomorrow.Negative = data_tomorrow.Negative.shift(+1)
data_tomorrow.Positive = data_tomorrow.Positive.shift(+1)
data_tomorrow.Max_Sentiment = data_tomorrow.Max_Sentiment.shift(+1)
data_tomorrow.Sentiment_Proportion = data_tomorrow.Sentiment_Proportion.shift(+1)

# Delete the first row of data_tomorrow
data_tomorrow.drop(data_tomorrow.head(1).index, inplace=True)

train_data = data[:-1]  # train data
today_record = data.tail(1)  # test data (validate current day and predict from following day)
train_data_tomorrow = data_tomorrow[:-1]  # train data
tomorrow_record = data_tomorrow.tail(1)  # test data (validate current day and predict from following day)


########################################################################################################################
# TODAY: Local method to identify most significant feature in dataset compared to y
def identify_sig_feature_4_today(y_variable, graph_data):
    # Split Data Into X, which are ALL the features
    x = data.iloc[:, 9:18].values

    # Split Data Into y, which are the associated targets/classifications; looking at Volume
    y = data[np.unicode(y_variable)].values

    # Get the Column Names, Ignore Index
    feat_labels = data.columns[9:19]

    # Randomly choose 20% of the data for testing; want a large train set (set random_state as 0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Declare the StandardScaler
    std_scaler = StandardScaler()

    # Standardize the features in the training data
    X_train = std_scaler.fit_transform(X_train)

    # Standardize the features in testing data
    X_test = std_scaler.transform(X_test)

    # Start The Random Forest Regressor
    treereg = RandomForestRegressor(n_estimators=100, max_depth=11, random_state=0)

    # Execute The Data With The Random Forest Regressor
    treereg.fit(X_train, y_train)

    print('The accuracy of the random forest for today sentiment is: ' + str(treereg.score(X_test, y_test)))

    # Get The Important Features From The Regressor
    importances = treereg.feature_importances_

    # Sort The Features By The Most Important
    indices = np.argsort(importances)[::-1]

    # Return data
    df_cols = ['Sentiment', 'Importance']
    master_df = pd.DataFrame(columns=df_cols)

    for f in range(x.shape[1]):
        sentiment = feat_labels[f]
        importance = importances[indices[f]]
        temp_data = {'Sentiment': sentiment,
                     'Importance': importance}
        master_df = master_df.append(temp_data, ignore_index=True)

    highest_sentiment = master_df['Sentiment'].iloc[0]
    highest_importance = master_df['Importance'].iloc[0]

    if graph_data == "TRUE":
        # Output Data As A Plot for Overall Data set
        plt.title('Today Feature Importances ' + np.unicode(y_variable))
        plt.bar(range(x.shape[1]), importances[indices], color='lightblue', align='center')
        plt.xticks(range(x.shape[1]), feat_labels, rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.tight_layout()
        plt.show()

    return highest_sentiment, highest_importance


########################################################################################################################
# TOMORROW: Local method to identify most significant feature in dataset compared to y
def identify_sig_feature_4_tomorrow(y_variable, graph_data):
    # Split Data Into X, which are ALL the features
    x = data_tomorrow.iloc[:, 9:18].values

    # Split Data Into y, which are the associated targets/classifications; looking at Volume
    y = data_tomorrow[np.unicode(y_variable)].values

    # Get the Column Names, Ignore Index
    feat_labels = data_tomorrow.columns[9:19]

    # Randomly choose 20% of the data for testing; want a large train set (set random_state as 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Declare the StandardScaler
    std_scaler = StandardScaler()

    # Standardize the features in the training data
    X_train = std_scaler.fit_transform(X_train)

    # Standardize the features in testing data
    X_test = std_scaler.transform(X_test)

    # Start The Random Forest Regressor
    treereg = RandomForestRegressor(n_estimators=100, max_depth=11, random_state=1)

    # Execute The Data With The Random Forest Regressor
    treereg.fit(X_train, y_train)

    print('The accuracy of the random forest for tomorrow sentiment is: ' + str(treereg.score(X_test, y_test)))

    # Get The Important Features From The Regressor
    importances = treereg.feature_importances_

    # Sort The Features By The Most Important
    indices = np.argsort(importances)[::-1]

    # Return data
    df_cols = ['Sentiment', 'Importance']
    master_df = pd.DataFrame(columns=df_cols)

    for f in range(x.shape[1]):
        sentiment = feat_labels[f]
        importance = importances[indices[f]]
        temp_data = {'Sentiment': sentiment,
                     'Importance': importance}
        master_df = master_df.append(temp_data, ignore_index=True)

    highest_sentiment = master_df['Sentiment'].iloc[0]
    highest_importance = master_df['Importance'].iloc[0]

    if graph_data == "TRUE":
        # Output Data As A Plot for Overall Data set
        plt.title('Tomorrow Feature Importances ' + np.unicode(y_variable))
        plt.bar(range(x.shape[1]), importances[indices], color='lightblue', align='center')
        plt.xticks(range(x.shape[1]), feat_labels, rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.tight_layout()
        plt.show()

    return highest_sentiment, highest_importance


########################################################################################################################
# Local method to correctly retrieve appropriate paramters for Regularized Fit Regression based on Ridge Regression
def get_fit_regression_params(significant_sentiment, target_variable, sentiment_value):
    # Define the data needed for this section, and as defined by highest_sentiment
    x = data[significant_sentiment].values.reshape(-1, 1)

    y = data[np.unicode(target_variable)].values  # used to be just data.High

    # Standardize features
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    # Create ridge regression with alpha values from .1 to 10.0, in increments of 0.1
    regr_cv = RidgeCV(alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                              1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                              2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                              3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.0,
                              4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
                              5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
                              6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
                              7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0,
                              8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0,
                              9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0])

    # Place x and y variables in the proper format for model_cv.
    y = np.array(y)
    x_std = x_std.reshape((len(y), 1))
    y = y.reshape((len(y), 1))

    # Determine the best alpha value to use.
    model_cv = regr_cv.fit(x_std, y)
    alpha_val_today = model_cv.alpha_

    # Set the L1 value based on significant_sentiment_value
    if sentiment_value >= 0.7:
        weight_value = 0.4
    elif sentiment_value >= 0.4:
        weight_value = 0.5
    else:
        weight_value = 0.6

    return alpha_val_today, weight_value


########################################################################################################################
# MODELING 4 TODAY #####################################################################################################
# Prepare formula to predict closing of stock data for today

highest_sentiment1_today, significant_value1_today = identify_sig_feature_4_today("Close", "False")
formula = ('Close ~ Open + High + Low + ' + np.unicode(highest_sentiment1_today))
dta = train_data[['Close', 'Open', 'High', 'Low', 'Anger', 'Anticipation',
                  'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                  'Trust', 'Negative', 'Positive', 'Sentiment_Proportion']].copy()

# set seed
np.random.seed(1)

alpha_val, weight_val = get_fit_regression_params(highest_sentiment1_today, "Close", significant_value1_today)

# Create a Ordinary Least Squares regression model
lm1_today = smf.ols(formula=formula, data=dta).fit_regularized(alpha=alpha_val, L1_wt=weight_val)
fig1 = plt.figure(figsize=(12, 8))
fig1 = sm.graphics.plot_partregress_grid(lm1_today, fig=fig1)
fig1.savefig('Today_Close_Regression.png')  # Show Partial regression plot of model

# Predicts closing value based on train data and model above
today_close_prediction = lm1_today.predict(today_record)

# Prepare formula to predict High of stock data for today
highest_sentiment2_today, significant_value2_today = identify_sig_feature_4_today("High", "False")
formula = ('High ~ Open + Close + Low + ' + np.unicode(highest_sentiment2_today))
dta = train_data[['High', 'Open', 'Close', 'Low', 'Anger', 'Anticipation',
                  'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                  'Trust', 'Negative', 'Positive', 'Sentiment_Proportion']].copy()

alpha_val, weight_val = get_fit_regression_params(highest_sentiment2_today, "High", significant_value2_today)

# Create a Ordinary Least Squares regression model
lm2_today = smf.ols(formula=formula, data=dta).fit_regularized(alpha=alpha_val, L1_wt=weight_val)
fig2 = plt.figure(figsize=(12, 8))
fig2 = sm.graphics.plot_partregress_grid(lm2_today, fig=fig2)
fig2.savefig('Today_High_Regression.png')  # Show Partial regression plot of model

# Predicts high value based on train data and model above
today_high_prediction = lm2_today.predict(today_record)

# Prepare formula to predict Low of stock data for today
highest_sentiment3_today, significant_value3_today = identify_sig_feature_4_today("Low", "False")
formula = ('Low ~ Open + Close + High + ' + np.unicode(highest_sentiment3_today))
dta = train_data[['Low', 'Open', 'Close', 'High', 'Anger', 'Anticipation',
                  'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                  'Trust', 'Negative', 'Positive', 'Sentiment_Proportion']].copy()

alpha_val, weight_val = get_fit_regression_params(highest_sentiment3_today, "Low", significant_value3_today)

# Create a Ordinary Least Squares regression model
lm3_today = smf.ols(formula=formula, data=dta).fit_regularized(alpha=alpha_val, L1_wt=weight_val)
fig3 = plt.figure(figsize=(12, 8))
fig3 = sm.graphics.plot_partregress_grid(lm3_today, fig=fig3)
fig3.savefig('Today_Low_Regression.png')  # Show Partial regression plot of model

# Predicts Low value based on train data and model above
today_low_prediction = lm3_today.predict(today_record)

print("The Close value for today's stock is estimated to be: " + str(today_close_prediction.iloc[0]))
print("The High value for today's stock is estimated to be: " + str(today_high_prediction.iloc[0]))
print("The Low value for today's stock is estimated to be: " + str(today_low_prediction.iloc[0]))
print("")

########################################################################################################################
# MODELING 4 NEXT DAY###################################################################################################

highest_sentiment1_tom, significant_value1_tom = identify_sig_feature_4_tomorrow("Close", "False")
formula = ('Close ~ Open + High + Low + ' + np.unicode(highest_sentiment1_tom))
dta = train_data_tomorrow[['Close', 'Open', 'High', 'Low', 'Anger', 'Anticipation',
                           'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                           'Trust', 'Negative', 'Positive', 'Sentiment_Proportion']].copy()
# Set seed again
np.random.seed(2)

alpha_val, weight_val = get_fit_regression_params(highest_sentiment1_tom, "Close", significant_value1_tom)

# Create a Ordinary Least Squares regression model
lm1_tom = smf.ols(formula=formula, data=dta).fit_regularized(alpha=alpha_val, L1_wt=weight_val)
fig4 = plt.figure(figsize=(12, 8))
fig4 = sm.graphics.plot_partregress_grid(lm1_tom, fig=fig4)
fig4.savefig('Tomorrow_Close_Regression.png')  # Show Partial regression plot of model

# Predicts closing value based on train data and model above
close_prediction_tom = lm1_tom.predict(tomorrow_record)

# Prepare formula to predict High of stock data for today
highest_sentiment2_tom, significant_value2_tom = identify_sig_feature_4_tomorrow("High", "False")
formula = ('High ~ Open + Close + Low + ' + np.unicode(highest_sentiment2_tom))
dta = train_data_tomorrow[['High', 'Open', 'Close', 'Low', 'Anger', 'Anticipation',
                           'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                           'Trust', 'Negative', 'Positive', 'Sentiment_Proportion']].copy()

alpha_val, weight_val = get_fit_regression_params(highest_sentiment2_tom, "High", significant_value2_tom)

# Create a Ordinary Least Squares regression model
lm2_tom = smf.ols(formula=formula, data=dta).fit_regularized(alpha=alpha_val, L1_wt=weight_val)
fig5 = plt.figure(figsize=(12, 8))
fig5 = sm.graphics.plot_partregress_grid(lm2_tom, fig=fig5)
fig5.savefig('Tomorrow_High_Regression.png')  # Show Partial regression plot of model

# Predicts high value based on train data and model above
high_prediction_tom = lm2_tom.predict(tomorrow_record)

# Prepare formula to predict Low of stock data for today
highest_sentiment3_tom, significant_value3_tom = identify_sig_feature_4_tomorrow("Low", "False")
formula = ('Low ~ Open + Close + High + ' + np.unicode(highest_sentiment3_tom))
dta = train_data_tomorrow[['Low', 'Open', 'Close', 'High', 'Anger', 'Anticipation',
                           'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                           'Trust', 'Negative', 'Positive', 'Sentiment_Proportion']].copy()

alpha_val, weight_val = get_fit_regression_params(highest_sentiment3_tom, "Low", significant_value3_tom)

# Create a Ordinary Least Squares regression model
lm3_tom = smf.ols(formula=formula, data=dta).fit_regularized(alpha=alpha_val, L1_wt=weight_val)
fig6 = plt.figure(figsize=(12, 8))
fig6 = sm.graphics.plot_partregress_grid(lm3_tom, fig=fig6)
fig6.savefig('Tomorrow_Low_Regression.png')  # Show Partial regression plot of model

# Predicts Low value based on train data and model above
low_prediction_tom = lm3_tom.predict(tomorrow_record)

print("The Close value for tomorrow's stock is estimated to be: " + str(close_prediction_tom.iloc[0]))
print("The High value for tomorrow's stock is estimated to be: " + str(high_prediction_tom.iloc[0]))
print("The Low value for tomorrow's stock is estimated to be: " + str(low_prediction_tom.iloc[0]))
print("")

if float(today_close_prediction.iloc[0]) < float(close_prediction_tom.iloc[0]):
    print("Based on our algorithm, the Closing value for the stock tomorrow will: Increase")
else:
    print("Based on our algorithm, the Closing value for the stock tomorrow will: Decrease")

if float(today_high_prediction.iloc[0]) < float(high_prediction_tom.iloc[0]):
    print("Based on our algorithm, the High value for the stock tomorrow will: Increase")
else:
    print("Based on our algorithm, the High value for the stock tomorrow will: Decrease")

if float(today_low_prediction.iloc[0]) < float(low_prediction_tom.iloc[0]):
    print("Based on our algorithm, the Low value for the stock tomorrow will: Increase")
else:
    print("Based on our algorithm, the Low value for the stock tomorrow will: Decrease")

# Record data today data and predictions to analyze accuracy:
with open('predictions_djia.csv', 'a') as csvfile:
    now = datetime.datetime.now()
    date = now.strftime("%m/%d/%Y")
    names = ['Date', 'Actual Close Today', 'Actual High Today', 'Actual Low Today', 'Predicted Close Today',
             'Predicted High Today', 'Predicted Low Today', 'Predicted Close Tomorrow', 'Predicted High Tomorrow',
             'Predicted Low Tomorrow']
    w = csv.DictWriter(csvfile, fieldnames=names, lineterminator='\n')
    w.writerow({'Date': str(date),
                'Actual Close Today': today_record['Close'].iloc[0],
                'Actual High Today': today_record['High'].iloc[0],
                'Actual Low Today': today_record['Low'].iloc[0],
                'Predicted Close Today': today_close_prediction.iloc[0],
                'Predicted High Today': today_high_prediction.iloc[0],
                'Predicted Low Today': today_low_prediction.iloc[0],
                'Predicted Close Tomorrow': close_prediction_tom.iloc[0],
                'Predicted High Tomorrow': high_prediction_tom.iloc[0],
                'Predicted Low Tomorrow': low_prediction_tom.iloc[0]})

    csvfile.close()

# **********************************************************************************************************************
########################################################################################################################
