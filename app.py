# deploy the model to the cloud

# import the necessary libraries
import joblib
from flask import Flask, request, jsonify

# create a flask app
app = Flask(__name__)

# load the model
model = joblib.load('model.pkl')

# create a function to predict the survival of a passenger
def predict_survival():
    # get the data from the POST request
    data = request.get_json(force=True)
    
    # convert the data into a dataframe
    data_df = pd.DataFrame.from_dict(data)
    
    # make predictions using the model
    result = model.predict(data_df)
    
    # send back the results
    output = {'results': int(result[0])}
    
    return jsonify(results=output)

# create a route for the app
@app.route('/predict', methods=['POST'])
def predict():
    return predict_survival()

# run the app
if __name__ == '__main__':
    app.run(port=5000, debug=True)

