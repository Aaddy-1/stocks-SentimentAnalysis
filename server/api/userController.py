import sys
import service
import os
from flask import Flask, render_template
from dotenv import load_dotenv
sys.path.insert(0, '/Users/aadeesh/redditSentiment/server/model')
import model
from modelClasses import textTransformer, customModel
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()
my_file = THIS_FOLDER / "myfile.txt"



load_dotenv('/Users/aadeesh/redditSentiment/environment.env')


app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='/Users/aadeesh/redditSentiment/server/template')
classifier = model.init_model()

# tesla_df, apple_df, nvda_df, google_df, amzn_df, msft_df, meta_df
labels, values = service.get_data(classifier)

@app.route('/get', methods=['GET'])
def get():
    
    # return data
    # return render_template("home.html")
    print(labels)
    print(values)
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
    app.run(debug=True, port = os.getenv('PORT'))