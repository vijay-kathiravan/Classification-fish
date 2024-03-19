from flask import Flask, render_template, request
import Fish_Classifier as mdl  # Import your trained machine learning model
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs
    user_input1 = float(request.form["Weight"])  # Convert string inputs to float
    user_input2 = float(request.form["Length1"])
    user_input3 = float(request.form["Length2"])
    user_input4 = float(request.form["Length3"])
    user_input5 = float(request.form["Height"])
    user_input6 = float(request.form["Width"])

    # Create a DataFrame with the correct structure
    data = {
        'Weight': [user_input1],
        'Length1': [user_input2],
        'Length2': [user_input3],
        'Length3': [user_input4],
        'Height': [user_input5],
        'Width': [user_input6]
    }
    Data = pd.DataFrame(data)

    # Call model prediction function
    prediction = mdl.classifier.predict(Data)

    # Convert numpy array to list for easy handling in template
    prediction_list = prediction.tolist()
    #output = prediction[0]

    # Return result
    #return render_template('index.html', prediction_text=f'Species Predicted: {prediction_list}')

    return render_template("result.html", prediction=prediction_list)

if __name__ == "__main__":
    app.run(debug=True)  # Remove debug=True before deployment


