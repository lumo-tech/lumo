from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'




if __name__ == '__main__':
    app.run()
app.run(host='0.0.0.0')
