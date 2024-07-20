import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "web scraping"

	
limit = 100
tweets = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
   if len(tweets) == limit:
       break
   else:
       tweets.append([tweet.date, tweet.user.username, tweet.content])
   print(vars(tweet))
   break

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])#print(df)
df.to_csv('scraped-tweets.csv', index=False, encoding='utf-8')