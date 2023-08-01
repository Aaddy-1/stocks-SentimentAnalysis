# SentiStocks
# Introduction  
This stock sentiment analysis project is designed to scrape Reddit comments related to various stocks in order to gauge market sentiment towards these brands. This project aims to provide insights into the overall sentiment of online users towards specific stocks, which helps investors and traders gauge public sentiment on any company before investing. This tool can be invaluable for companies who are wanting to gauge public perception about their brand in the context of the financial markets.
# How it works
1. The scraper scrapes reddit comments from every post for the past one month from the subreddit [r/wallstreetbets](https://www.reddit.com/r/wallstreetbets/).
2. It then filters the data based on various keywords related to the stocks being tracked.
3. The data is preprocessed to help the model in sentiment analysis. Preprocessing steps include removing grammar and emojis, removing stop words, stemming and lemmatizing, and finally tokenization.
4. The balance between positive and negative sentiments in the training data is heavily skewed in the favor of positive sentiments. To offset this, we perform data augmentation on the training data in order to attain a balance between positive and negative values.
5. We then vectorize the texts using the built in text vectorizer of keras, I opted out of using TF-IDF vectorization because I found that it didnt provide a big benefit to the overall accuracy and F-1 score of the model.
6. We can then pass the data into the model. The model is a tensorflow model that is built on a Bidirectional LSTM architecture along with a dense layer in order to capture the largest amount of data from each sentence. I add a Dense layer with 2 nodes and a softmax activation functio as the last layer to obtain probabilities for both of the possible outcomes.
7. In order to obtain sentiment scores for each stock. The sentiment score is calculated as +1 for a positive comment, and a +0 for a negative comment. This is done in order to prevent sentiment scores from going into the negative.
8. The data for the sentiment score for each stock from each day from the past month is then graphed using Chart.JS

# Demo
I am working on a live demo that streams real-time sentiment values for the stocks. The real-time streaming isn't active yet but will be very soon. However, plots for historic values of the sentiment of the stocks are available in the live demo. Please note that it may take upto 2 minutes to render the webpage due to renders restrictions. (https://stockssentimentanalysis.onrender.com/)
# <img width="1297" alt="Screenshot 2023-07-27 at 1 09 20 AM" src="https://github.com/Aaddy-1/stocks-SentimentAnalysis/assets/83650351/43104302-0db4-4d68-ad76-97e0235ea149">

