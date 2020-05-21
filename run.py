import flask
from flask import request, render_template
from demo import *

# Initialize the app

app = flask.Flask(__name__)


@app.route("/", methods=['GET'])
def main():
    return render_template('predictor.html')


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    # print(request.args)
    print(request.args)
    print(request.form)
    html, predictions, sentiment = predict_model.inference(request.form['chat_in'])
    # print(x_input)
    print(sentiment)

    return render_template('predictor.html',
                           prediction=predictions,
                           sentiment=sentiment,
                           html=html)


# Start the server, continuously listen to requests.

if __name__ == "__main__":
    # For local development:
    predict_model = InferenceModel()
    app.run(debug=True, port=5000)
    # For public web serving:
    # app.run(host='0.0.0.0')
    app.run()
