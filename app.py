import streamlit as st
import joblib
import datetime
import snscrape.modules.twitter as sntwitter
import pandas as pd

with open('model_lstm.tf', 'rb') as file_1:
  nlp = joblib.load(file_1)

def run():
    # Creating list to append tweet data to
    attributes_container = []

    stock = st.selectbox('Pick a stock:', ('BBNI', 'BBRI', 'BBTN', 'BMRI'))
    
    today = datetime.datetime.now()
    today = today.strftime('%Y-%m-%d')
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(stock.lower() + ' since:' + yesterday + ' until:' + today).get_items()):
        attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])

    # Creating a dataframe to load the list
    tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
    tweets = tweets_df[tweets_df['Tweet'].str.contains(stock)]
    tweets_daily = pd.DataFrame(pd.to_datetime(tweets['Date Created']).dt.tz_localize(None))
    tweets['Date Created'] = tweets_daily
    tweets['Date Created'] = pd.to_datetime(tweets['Date Created']).dt.date
    tweets

    if st.button('Predict sentiment'):
        df = pd.DataFrame()

        for i in range(0, len(tweets)):
            row = nlp(tweets['Tweet'].iloc[i])
            df = df.append(row, )
        df
        df['label'].value_counts()    

if __name__ == '__main__':
    run()