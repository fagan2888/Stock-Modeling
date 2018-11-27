#! /usr/bin/Rscript
require(syuzhet)

today_news <- read.csv("E:/Programming/Data Science/Today_News.csv") # CHANGE PATH TO DATA FILE

#Sentiments words table:
records <- as.character(today_news$News)
sentiment <- get_nrc_sentiment(records)

sent <- cbind(today_news$Date, sentiment)
colnames(sent)<-c("Date","Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust", "Negative", "Positive")
sentiment_data <- as.data.frame(colSums(sent[,-1]))
sentiment_df <- do.call(rbind, sentiment_data)
colnames(sentiment_df)<-c("Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust", "Negative", "Positive")
write.csv(sentiment_df, file = "E:/Programming/Data Science/new_sentiment_data.csv")
file.remove("E:/Programming/Data Science/Today_News.csv")
