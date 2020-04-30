import os
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/send")
def hello():
    return "MESSAGE"

@app.route("/toggle")
def mode():
    mode = request.args.get('mode')
    return mode

if __name__ == '__main__':
    app.run(use_reloader=True, debug=True, host='0.0.0.0', threaded=True)