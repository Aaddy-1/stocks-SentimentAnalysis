import praw
import os
import pandas as pd
from dotenv import load_dotenv

# loading env file
load_dotenv('environment.env')


reddit = praw.Reddit(client_id = os.getenv('CLIENT_ID'),
                    client_secret = os.getenv('CLIENT_SECRET'),
                    user_agent = os.getenv('USER_AGENT'))

subreddit = reddit.subreddit("wallstreetbets")

# # Display the name of the Subreddit
# print("Display Name:", subreddit.display_name)
 
# # Display the title of the Subreddit
# print("Title:", subreddit.title)
 
# # Display the description of the Subreddit
# print("Description:", subreddit.description)

posts_url = []

for post in subreddit.hot(limit = 5):
    print(post.title)
    print(post.url)
    print(post.id)
    if (post.selftext):
        print("contains text")
    print()
    posts_url.append(post.url)

print(posts_url)

from praw.models import MoreComments
# comments = []
# for url in posts_url:
#     print(url)
#     submission = reddit.submission(url=url)
#     for comment in submission.comments[1:]:
#         if isinstance(comment, MoreComments):
#             continue
#         comments.append(comment.body)
# print(comments)

# url = "https://v.redd.it/408dou8scy9b1"
# id = '14q1z3k'
# submission = reddit.submission(id = id)

# posts = []
# for top_level_comment in submission.comments[1:]:
#     if isinstance(top_level_comment, MoreComments):
#         continue
#     posts.append(top_level_comment.body)

# print(posts)

posts_data = {"ID" : [], "url" : [], "Title" : [], "Total Comments" : [], "Score" : []}

# Collecting data from the past week
for post in subreddit.top(time_filter = "week"):
    print(post.title)
    posts_data["ID"].append(post.id)
    posts_data["url"].append(post.url)
    posts_data["Title"].append(post.title)
    posts_data["Total Comments"].append(post.num_comments)
    posts_data["Score"].append(post.score)

print(posts_data)

# Saving the past weeks posts data into a dataframe

top_posts_df = pd.DataFrame(posts_data)
top_posts_df.to_csv("Top_This_Week.csv", index=False)