import praw
import os
import pandas as pd
from dotenv import load_dotenv
from praw.models import MoreComments
import datetime

# loading env file
load_dotenv('environment.env')

reddit = praw.Reddit(client_id = os.getenv('CLIENT_ID'),
                    client_secret = os.getenv('CLIENT_SECRET'),
                    user_agent = os.getenv('USER_AGENT'))

subreddit = reddit.subreddit("wallstreetbets")
for comment in subreddit.stream.comments(skip_existing=True):
    print(comment)