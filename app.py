from flask import Flask, request
from main import MLModel

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Home route</p>"


"""
/compute route is takes the data passed through the get request from client
and calls for MLModel method with this data to compute.
Returns the result of MLModel as reponse
"""

@app.route("/compute", methods=["GET"])
def compute():
    responseObj={}
    data = request.get_json(force=True)
    if data["text"] != []:
        responseObj = MLModel(data["text"])
    return responseObj
