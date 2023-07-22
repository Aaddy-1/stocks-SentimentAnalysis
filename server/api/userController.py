import json
import sys
import service
import os
from flask import Flask, request
from dotenv import load_dotenv
sys.path.insert(0, '/Users/aadeesh/redditSentiment/server/model')
import model
from modelClasses import textTransformer, customModel

load_dotenv('/Users/aadeesh/redditSentiment/environment.env')

app = Flask(__name__)
classifier = model.init_model()

@app.route('/get', methods=['GET'])
def get():
    return json.dumps({'name': 'Alice'})

@app.route('/post', methods=['POST'])
def post():
    name = request.get_json()['name']
    return json.dumps({'name': name})
if __name__ == '__main__':
    app.run(debug=True, port = os.getenv('PORT'))