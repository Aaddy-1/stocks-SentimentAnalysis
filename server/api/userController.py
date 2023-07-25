import sys
import service
import os
from flask import Flask, render_template
from dotenv import load_dotenv
from pathlib import Path
SERVER_FOLDER = Path(__file__).parent.parent.resolve()
template_folder = SERVER_FOLDER / "template"
model_folder = SERVER_FOLDER / "model"

# sys.path.insert(0, model_folder)
# import model
# from modelClasses import textTransformer, customModel






# load_dotenv('/Users/aadeesh/redditSentiment/environment.env')


userController = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder=template_folder)
# classifier = model.init_model()

# tesla_df, apple_df, nvda_df, google_df, amzn_df, msft_df, meta_df
labels, values = service.get_data()

@app.route('/', methods=['GET'])
def get():
    
    # return data
    # return render_template("home.html")
    print("HELLO HELLO HELLO ")
    # print(labels)
    # print(values)
    return render_template(
        template_name_or_list='sentiment_charts.html',
        tesla_vals = values[0],
        apple_vals = values[1],
        nvda_vals = values[2],
        google_vals = values[3],
        amzn_vals = values[4],
        msft_vals = values[5],
        meta_vals = values[6],
        labels=labels
    )


if __name__ == '__main__':
    userController.run(debug=True, port = 8080)