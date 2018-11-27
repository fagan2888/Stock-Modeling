# Predicting the Stock Market

This is a repo for Predicting stock market data using machine learning with sentiment data on related news.
Below are the steps in order to run models:

## Step 1: Web Scrape News Data with Reddit Data Extractor
The reddit data extractor can be found here: https://github.com/NSchrading/redditDataExtractor

This will save text files into a directory of your choosing. The text files will be named with the headlines of the data collected.

## Step 2: Update File references in the following scripts

process_reddit.py
- A. Line 31/40 - change the stock ticker in the first param of 'get_data_yahoo()' to what you need (Note: if the dataframe comes back empty, it means Yahoo has not updated the data yet for the API to collect; try again later)
- B. Line 57 - change to the directory the reddit news data was saved to
- C. Line 83 & 84 - change the file paths to where Rscript.exe is from your R build, and the path to the sentiment_extraction.r script saved in this repo
- D. Line 135, 144, and 145 - change file name to name of stock data collected

sentiment_extraction.R
- A. Line 4 - change to the file path to where the file 'process_reddit.py' is saved to and append '.../Today_News.csv' to the end
- B. Line 15 - change path to file made from line 86 in 'process_reddit.py'
- C. Line 16 - Make sure path to file matches line 4

## Step 3: Execute Stock_Modeling.ipynb
- This notebook will process the existing stock data collected and joined sentiment data for modeling (e.g. DJ_NEWS_SENTIMENT_DATA.csv)
- Then it will perform modeling on the High, Low, and Close to create a prediction for today's final stock values and the following day's values.
- Note: Ensure data read-in for 'data' or 'data_tomorrow' are set to correct locations

### Note:
The following Python Build was used in the development: Anaconda 3.6

The following are also necessary packages for this to work:
- R: syuzhet
- Python: numpy, pandas, statsmodels, pandas_datareader, matplotlib, subprocess, sklearn, shutil, fix_yahoo_finance, datetime, csv, os
