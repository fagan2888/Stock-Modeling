"""
This script will process reddit news data downloaded from the Reddit Data Extractor into a csv file.
Perform Sentiment Analysis on it using a separate R script
Collect Stock data from yahoo finance
Join the sentiment analysis and yahoo finance data together
To modify data collection edit lines 33 and 41 with different ticker, line 55 with path to news data, 58 with path to 
where this script is saved, and line 79, 80 to path where Rscript.exe and sentiment_extraction.R are saved
This script can be run separately w/o Data_Mining_Analysis.py just for the data collection
"""
import os
from subprocess import call
from datetime import datetime
import time
import shutil


########################################################################################################################
# execute yahoo-finance to get the DJIA data first
# Parameters that can be changed: Stock Ticker and Data Timeframe for Extraction
# Currently set to DJIA and Data for Today
def get_yahoo_data():
    from pandas_datareader import data as pdr
    from datetime import datetime, timedelta
    import fix_yahoo_finance as yf
    yf.pdr_override()

    now = datetime.now()
    end_day = now + timedelta(days=1)

    # download dataframe from yahoo (^DJI or BTCUSD=X)
    data = pdr.get_data_yahoo("^DJI", start=now.strftime("%Y-%m-%d"), end=end_day.strftime("%Y-%m-%d"))
    try:
        data['change'] = (data['Close'].values.astype(int) - data['Open'].values.astype(int)) / data['Open'].values.astype(int)
        data['change'] = data['change'] * 100

    except KeyError:
        import fix_yahoo_finance as yf
        yf.pdr_override()
        data = pdr.get_data_yahoo("^DJI", start=end_day.strftime("%Y-%m-%d"), end=now.strftime("%Y-%m-%d"))
        data['change'] = ((data['Close'].values).astype(int) - (data['Open'].values).astype(int)) / (
            data['Open'].values).astype(int)
        data['change'] = data['change'] * 100

    return data


new_djia_data = get_yahoo_data()
new_djia_data = new_djia_data.tail(1)
########################################################################################################################
# re-import pandas to reset the override that is needed for yahoo-finance
# set the path to where the reddit news data is
# set the paths to Rscript.exe from your R build and path to the sentiment_extraction.r file
import pandas as pd

# set directory where scripts are
os.chdir(r'E:\Programming\Data Science\')

dir_name = os.getcwd() + r'\reddit_data\worldnews'  # change to where the reddit files are
data_path = os.listdir(dir_name)
date = time.strftime("%Y-%m-%d")
df_cols = ['Date', 'News']
master_df = pd.DataFrame(columns=df_cols)

# This chunk process the reddit news data into one master csv file for sentiment analysis ##############################
for item in data_path:
    string = str(item)
    if string[-4:] == ".txt" and len(string) > 10:
        news = string.replace('.txt', '')
        temp_data = {'Date': date,
                     'News': news}
        temp_df = pd.DataFrame(temp_data, columns=df_cols, index=[0])
        master_df = master_df.append(temp_data, ignore_index=True)

master_df.to_csv('Today_News.csv', encoding='utf-8', index=False, sep=',')
# The above file will be made where this python script is
# The absolute path to this file must be used in the sentiment_extraction.R file (please revise the R file)

shutil.rmtree(dir_name)  # removes all the files collected from the reddit web scraper
########################################################################################################################
# run R script and return new_sentiment_data.csv
# NOTE: param1 = path to Rscript.exe from the R build, param2 is the path to the R file to execute
call(["C:/Users/Matt Wilchek/Documents/R/R-3.4.3/bin/Rscript", "E:/Programming/Data Science/sentiment_extraction.r"])
# E:/Programming/Data Mining/GitHub Data Mining Project/GWU-Data-Mining-Proposal-1/sentiment_extraction.r
sentiment = pd.read_csv("new_sentiment_data.csv")
sentiment.drop('Unnamed: 0', axis=1, inplace=True)
########################################################################################################################
# Format DJIA dataframe to original dataset format
new_djia_data = new_djia_data.reset_index(drop=True)
now = datetime.now()
new_djia_data['Date'] = now.strftime("%m/%d/%Y")  # 8/8/2008
cols = new_djia_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
new_djia_data = new_djia_data[cols]

# Append Sentiment Dataframe columns to DJIA Dataframe columns
new_djia_data['Anger'] = sentiment['Anger']
new_djia_data['Anticipation'] = sentiment['Anticipation']
new_djia_data['Disgust'] = sentiment['Disgust']
new_djia_data['Fear'] = sentiment['Fear']
new_djia_data['Joy'] = sentiment['Joy']
new_djia_data['Sadness'] = sentiment['Sadness']
new_djia_data['Surprise'] = sentiment['Surprise']
new_djia_data['Trust'] = sentiment['Trust']
new_djia_data['Negative'] = sentiment['Negative']
new_djia_data['Positive'] = sentiment['Positive']

# Calculate Sentiment Proportions of Original Data
feat_labels = new_djia_data.columns[8:18]
sentiment_data = new_djia_data.iloc[:, 8:18].values
temp_df = pd.DataFrame(sentiment_data, columns=feat_labels, index=None)
temp_list = list()
for l, n in temp_df.iterrows():
    max_sentiment = n.idxmax()
    max_sentiment_value = ((n.max(axis=0)).astype(int)).max()
    total_values = sum(n)
    temp_list.append([str(max_sentiment), max_sentiment_value / total_values])

temp_df2 = pd.DataFrame(temp_list, columns=['Max_Sentiment', 'Sentiment_Proportion'], index=None)
new_djia_data = pd.concat([new_djia_data, temp_df2], axis=1)
new_djia_data.columns.values[5] = "Adj.Close"
new_djia_data = new_djia_data.reset_index(drop=True)
new_order = [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
new_djia_data = new_djia_data[new_djia_data.columns[new_order]]

new_djia_data.to_csv('new_record.csv', encoding='utf-8', index=False, sep=',')
os.remove('new_sentiment_data.csv')  # removes the combined DJIA and sentiment data file

########################################################################################################################
# Add New Data if it exists to master dataset
new_record = os.path.exists('new_record.csv')

if new_record:
    df_sentiment_dj = pd.read_csv('DJ_NEWS_SENTIMENT_DATA.csv')
    new_record_row = pd.read_csv('new_record.csv')
    df_sentiment_dj = df_sentiment_dj.append(new_record_row, ignore_index=True)
    df_sentiment_dj = df_sentiment_dj[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj.Close', 'change', 'Anger',
                                       'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust',
                                       'Negative', 'Positive', 'Max_Sentiment', 'Sentiment_Proportion']]
    os.remove('new_record.csv')

    # Update old dataset
    os.remove('DJ_NEWS_SENTIMENT_DATA.csv')
    df_sentiment_dj.to_csv('DJ_NEWS_SENTIMENT_DATA.csv', encoding='utf-8', index=False, sep=',')

__author__ = 'Matt Wilchek'
