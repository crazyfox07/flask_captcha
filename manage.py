from flask import Flask
from flask_restful import Resource, Api

from crack_captcha.dhb import DaiHouBang

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


api = Api(app)
api.add_resource(DaiHouBang, '/')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11112, debug=True)
