import json
from flask import Flask, request
from main import MLModel, get_reviews

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Home route</p>"


"""
/compute route is takes the data passed through the get request from client
and calls for MLModel method with this data to compute.
Returns the result of MLModel as reponse
"""

# @app.route("/compute", methods=["GET"])
# def compute():
#     responseObj={}
#     data = request.get_json(force=True)
#     if data["text"] != []:
#         responseObj = MLModel(data["text"])
#     return responseObj

@app.route("/compute", methods=["GET"])
def compute():
    responseObj = {}
    URL = "https://www.amazon.in/iQOO-Storage-Snapdragon-Processor-44WFlashCharge/dp/B07WHS7MZ4/ref=cm_cr_arp_d_product_top?ie=UTF8&th=1"
    total_reviews_needed = 20
    try:
        reviews = get_reviews(URL, total_reviews_needed)
        formatted_reviews = json.dumps(reviews)
        if reviews:
            responseObj = MLModel(reviews)
    except Exception as e:
        responseObj = {"error": str(e)}
    return responseObj