# stocks-SentimentAnalysis
# Introduction  
The Reddit Stock Sentiment Analysis project is designed to scrape Reddit comments related to various stocks, perform sentiment analysis using a machine learning program, and then visualize the sentiment results. This project aims to provide insights into the overall sentiment of Reddit users towards specific stocks, helping investors and traders gauge public sentiment.
# How it works
1. The scraper scrapes reddit comments from every post for the past one month from the subreddit (https://www.reddit.com/r/wallstreetbets/)) r/wallstreetbets.
2. It then filters the data based on various keywords related to the stocks being tracked.
3. We can then perform preprocessing on the data to help the model in sentiment analysis. Preprocessing steps include remiving grammar and emojis, removing stop words, stemming and lemmatizing, and finally tokenization.
4. We can then pass the data into the model. The model is a tensorflow model that is built on a Bidirectional LSTM architecture in order to capture the largest amount of data from each sentence.
5. In order to obtain sentiment scores for each stock. The sentiment score is calculated as +1 for a positive comment, and a +0 for a negative comment. This is done in order to prevent sentiment scores from going into the negative.
6. The data for the sentiment score for each stock from each day from the past month is then graphed using Chart.JS

# Demo
I am working on a live demo that streams real-time sentiment values for the stocks. The real-time streaming isn't active yet but will be very soon. However, plots for historic values of the sentiment of the stocks are available in the live demo. Please note that it may take upto 2 minutes to render the webpage due to renders restrictions. (https://stockssentimentanalysis.onrender.com/)
# <img width="1297" alt="Screenshot 2023-07-27 at 1 09 20 AM" src="https://github.com/Aaddy-1/stocks-SentimentAnalysis/assets/83650351/43104302-0db4-4d68-ad76-97e0235ea149">

