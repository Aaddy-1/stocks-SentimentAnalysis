import praw
import os
from dotenv import load_dotenv

# loading env file
load_dotenv('environment.env')


reddit = praw.Reddit(client_id = os.getenv('CLIENT_ID'),
                    client_secret = os.getenv('CLIENT_SECRET'),
                    user_agent = os.getenv('USER_AGENT'))

