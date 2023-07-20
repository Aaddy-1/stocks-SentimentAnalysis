import praw
import os
import pandas as pd
from dotenv import load_dotenv
from praw.models import MoreComments
import datetime
import time

# loading env file
load_dotenv('environment.env')

reddit = praw.Reddit(client_id = os.getenv('CLIENT_ID'),
                    client_secret = os.getenv('CLIENT_SECRET'),
                    user_agent = os.getenv('USER_AGENT'))

subreddit = reddit.subreddit("wallstreetbets")

def get_subreddit_info(subreddit):
    # Display the name of the Subreddit
    print("Display Name:", subreddit.display_name)
 
    # Display the title of the Subreddit
    print("Title:", subreddit.title)
 
    # Display the description of the Subreddit
    print("Description:", subreddit.description)

def getPosts(subreddit, time, limit):
    print("Getting posts...")
    startTime = time.time()
    posts_data = {"ID" : [], "url" : [], "Title" : [], "Total Comments" : [], "Score" : []}
    # Collecting data from the past week
    for post in subreddit.top(time_filter = time, limit = limit):
        posts_data["ID"].append(post.id)
        posts_data["url"].append(post.url)
        posts_data["Title"].append(post.title)
        posts_data["Total Comments"].append(post.num_comments)
        posts_data["Score"].append(post.score)
    # print(posts_data)
    endTime = time.time()
    elapsedTime = endTime - startTime
    return posts_data, elapsedTime

# Saving the past weeks posts data into a dataframe
# top_posts_df = pd.DataFrame(posts_data)
# top_posts_df.to_csv("Top_This_Week.csv", index=False)

def getComments(posts):
    print("Getting comments...")
    startTime = time.time()
    comments = {"Post ID" : [], "Title" : [], "Date" : [], "Comment" : [], "Length" : []}
    postIDs = posts["ID"]
    for i in range(len(postIDs)):
        submission = reddit.submission(id = postIDs[i])
        for commentInstance in submission.comments[1:]:
            if isinstance(commentInstance, MoreComments):
                continue
            id =  postIDs[i]
            date = datetime.datetime.utcfromtimestamp(submission.created_utc)
            title = submission.title
            comments["Post ID"].append(id)
            comments["Title"].append(title)
            comments["Date"].append(date)
            comments["Comment"].append(commentInstance.body)
            comments["Length"].append(len(commentInstance.body))
    endTime = time.time()
    timeElapsed = endTime - startTime


    return comments, timeElapsed

pastWeekPosts = getPosts(subreddit, "month", 1000)
posts_df, timeTaken = pd.DataFrame(pastWeekPosts)
print("Done")
print("Time taken: ", timeTaken)
print("Number of Posts collected: ", len(posts_df.index))

pastWeekComments = getComments(pastWeekPosts)
comments_df, timeTaken = pd.DataFrame(pastWeekComments)
print("Done")
print("Time taken: ", timeTaken)
print("Number of Comments collected: ", len(comments_df.index))

# Saving all of our scraping results into csv files
# We have a file which contains information about the top posts from the past weeks
current_datetime = datetime.now().strftime("%Y-%m-%d")
str_current_datetime = str(current_datetime)

posts_df.to_csv("redditData/Posts/Posts[" + str_current_datetime + "].csv", index = False)

# We have our primary file which contains information about all of the comments from the past week
comments_df.to_csv("redditData/Comments/Comments[" + str_current_datetime + "].csv", index = False)
subreddit = reddit.subreddit("wallstreetbets")

def get_subreddit_info(subreddit):
    # Display the name of the Subreddit
    print("Display Name:", subreddit.display_name)
 
    # Display the title of the Subreddit
    print("Title:", subreddit.title)
 
    # Display the description of the Subreddit
    print("Description:", subreddit.description)

def getPosts(subreddit, time, limit):
    print("Getting posts...")
    startTime = time.time()
    posts_data = {"ID" : [], "url" : [], "Title" : [], "Total Comments" : [], "Score" : []}
    # Collecting data from the past week
    for post in subreddit.top(time_filter = time, limit = limit):
        posts_data["ID"].append(post.id)
        posts_data["url"].append(post.url)
        posts_data["Title"].append(post.title)
        posts_data["Total Comments"].append(post.num_comments)
        posts_data["Score"].append(post.score)
    # print(posts_data)
    endTime = time.time()
    elapsedTime = endTime - startTime
    return posts_data, elapsedTime

# Saving the past weeks posts data into a dataframe
# top_posts_df = pd.DataFrame(posts_data)
# top_posts_df.to_csv("Top_This_Week.csv", index=False)

def getComments(posts):
    print("Getting comments...")
    startTime = time.time()
    comments = {"Post ID" : [], "Title" : [], "Date" : [], "Comment" : [], "Length" : []}
    postIDs = posts["ID"]
    for i in range(len(postIDs)):
        submission = reddit.submission(id = postIDs[i])
        for commentInstance in submission.comments[1:]:
            if isinstance(commentInstance, MoreComments):
                continue
            id =  postIDs[i]
            date = datetime.datetime.utcfromtimestamp(submission.created_utc)
            title = submission.title
            comments["Post ID"].append(id)
            comments["Title"].append(title)
            comments["Date"].append(date)
            comments["Comment"].append(commentInstance.body)
            comments["Length"].append(len(commentInstance.body))
    endTime = time.time()
    timeElapsed = endTime - startTime


    return comments, timeElapsed

pastWeekPosts = getPosts(subreddit, "month", 1000)
posts_df, timeTaken = pd.DataFrame(pastWeekPosts)
print("Done")
print("Time taken: ", timeTaken)
print("Number of Posts collected: ", len(posts_df.index))

pastWeekComments = getComments(pastWeekPosts)
comments_df, timeTaken = pd.DataFrame(pastWeekComments)
print("Done")
print("Time taken: ", timeTaken)
print("Number of Comments collected: ", len(comments_df.index))

# Saving all of our scraping results into csv files

current_datetime = datetime.now().strftime("%Y-%m-%d")
str_current_datetime = str(current_datetime)

posts_df.to_csv("redditData/Posts/Posts[" + str_current_datetime + "].csv", index = False)

# We have our primary file which contains information about all of the comments from the past week
comments_df.to_csv("redditData/Comments/Comments[" + str_current_datetime + "].csv", index = False)
