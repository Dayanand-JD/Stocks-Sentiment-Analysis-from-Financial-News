from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
# Add a selectbox to the sidebar:

finviz_url = "https://finviz.com/quote.ashx?t="


st.title('Stock Sentiment Analysis application')
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'which stock would like to be analayzed ?',
    ('GME', 'TSLA', 'NVDA', 'ATVI', 'MTCH'), 
)
st.write(add_selectbox)

tickers = []
tickers.append(add_selectbox)
news_tables = {}

for ticker in tickers:
    # form url for chosen ticker
    url = finviz_url + ticker
    # submit request to finwiz website
    req = Request(url=url, headers={"user-agent": "senti-app"})
    # parse html from request object
    
    html = BeautifulSoup(urlopen(req), "html")
    # extract table which holds news titles
    news_tables[ticker] = html.find(id="news-table")

# extract article info from table rows
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll("tr"):
        # specify elements
        title = row.a.text
        date_data = row.td.text.split()

        # if only time is given
        if len(date_data) == 1:
            time = date_data[0]
        # if both time and date are given
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

# apply sentiment analysis to headlines using nltk vader package
df = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])
vader = SentimentIntensityAnalyzer()
vander = SentimentIntensityAnalyzer()
df["Postive_Sentiment"] = df.title.apply(lambda t: vader.polarity_scores(t)["pos"])
df["Netural_Sentiment"] = df.title.apply(lambda t: vader.polarity_scores(t)["neu"])
df["Neagtive_Sentiment"] = df.title.apply(lambda t: vader.polarity_scores(t)["neg"])
data=[df["Postive_Sentiment"].mean(),df["Netural_Sentiment"].mean(),df["Neagtive_Sentiment"].mean()]
sentiments=['Postive','Netural','Neagtive']
plt.pie(data, labels = sentiments, explode=[0.09, 0.2, 0.09],autopct='%1.1f%%', shadow = True, colors =['#aee1e1', '#ece2e1', '#fcd1d1'])
plt.savefig('saved_fiagure.png')
st.write(df)
st.image(Image.open('saved_fiagure.png'))