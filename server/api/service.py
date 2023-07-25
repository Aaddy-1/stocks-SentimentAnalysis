import sys
from pathlib import Path
SERVER_FOLDER = Path(__file__).parent.parent.resolve()
data_folder = SERVER_FOLDER / "Data/redditData/Posts/processed_df.csv"


# sys.path.insert(0, '/Users/aadeesh/redditSentiment/server/model')
# from modelClasses import textTransformer, customModel
import pandas as pd
import numpy as np

def dataframeProcessor(df, classifier):

    keywords = {"Tesla" : ["$tsla", "tsla", "tesla", "elon musk", "musk"],
            "Apple" : ["$aapl", "aapl", "apple", "mac"], 
            "Nvidia" : ["$nvda", "nvda", "nvidia"], 
            "Google" : ["$googl", "googl", "google", "alphabet", "bard"],
            "Amazon" : ["$amzn", "amzn", "amazon", "aws"],
            "Microsoft" : ["$msft", "msft", "microsoft", "windows", "azure"],
            "Meta" : ["$meta", "meta", "instagram", "facebook"]
        }
    keywords2 = ["$tsla", "tsla", "tesla", "elon musk", "musk", 
             "$aapl", "aapl", "apple", "mac",
             "$nvda", "nvda", "nvidia",
             "$googl", "googl", "google", "alphabet", "bard",
             "$amzn", "amzn", "amazon", "aws",
             "$msft", "msft", "microsoft", "windows", "azure",
             "$meta", "meta", "instagram", "facebook"
        ]

    filtered_df = df[df['Comment'].str.contains('|'.join(keywords2), case = False)]

    # Add an extra column to the filtered dataframe that indicates which keyword was present in that comment
    def keyWordBuilder(comment):
        returnString = ""
        for keyword in keywords2:
            if keyword in comment.lower():
                for key in keywords:
                    if keyword in keywords[key]:
                        if key not in returnString:
                            returnString += key + ' '
        if returnString == "":
            return "None"
        return returnString

    keyWordList = filtered_df['Comment'].apply(keyWordBuilder)

    filtered_df = filtered_df.assign(Keyword = keyWordList)

    newDates = pd.to_datetime(filtered_df['Date'])
    newDates = newDates.dt.date
    filtered_df = filtered_df.assign(Date = newDates)
    filtered_df = filtered_df.sort_values(by='Date', ascending=True)

    comments = filtered_df.Comment
    preds = classifier.predict(comments)

    sentiments = np.argmax(preds, axis = 1)
    # preds

    filtered_df = filtered_df.assign(Sentiment = sentiments)

    return filtered_df

def jsonBuilder(filtered_df):
    # filtered_rows = filtered_df[filtered_df['Keyword'].str.contains('tesla', case=False)]
    # filtered_rows['Date'] = pd.to_datetime(filtered_rows['Date'])

    # # Extract only the date part from the 'Date' column
    # filtered_rows['Date'] = filtered_rows['Date'].dt.date
    # print(filtered_rows.head())
    tesla_df, apple_df, nvda_df, google_df, amzn_df, msft_df, meta_df = {}, {}, {}, {}, {}, {}, {}

    done = []
    for i in (filtered_df.Date):
        # date_string = i.strftime('%m-%d')
        date_string = i
        if date_string not in done:
            tesla_df[date_string] = 0
            apple_df[date_string] = 0
            nvda_df[date_string] = 0
            google_df[date_string] = 0
            amzn_df[date_string] = 0
            msft_df[date_string] = 0
            meta_df[date_string] = 0
        done.append(date_string)

    for i, j, k in zip(filtered_df.Date, filtered_df.Keyword, filtered_df.Sentiment):
        # date_string = i.strftime('%m-%d')
        date_string = i
        val = 1
        if k == 0:
            val = 0
        for keyword in j.split():
            if keyword == "Tesla":
                tesla_df[date_string] += val
            if keyword == "Apple":
                apple_df[date_string] += val
            if keyword == "Nvidia":
                nvda_df[date_string] += val
            if keyword == "Google":
                google_df[date_string] += val
            if keyword == "Amazon":
                amzn_df[date_string] += val
            if keyword == "Microsoft":
                msft_df[date_string] += val
            if keyword == "Meta":
                meta_df[date_string] += val
    return [tesla_df, apple_df, nvda_df, google_df, amzn_df, msft_df, meta_df]

def get_data():
    # df = pd.read_csv("/Users/aadeesh/redditSentiment/server/Data/redditData/Posts/post.csv")
    
    # filtered_df = dataframeProcessor(df, classifier)
    filtered_df = pd.read_csv(data_folder)
    return_list = jsonBuilder(filtered_df)

    # tesla_df, apple_df, nvda_df, google_df, amzn_df, msft_df, meta_df
    labels = list(return_list[0].keys())
    values = [list(return_list[i].values()) for i in range(len(return_list))]
    # values = list(return_list[0].values())
    return labels, values