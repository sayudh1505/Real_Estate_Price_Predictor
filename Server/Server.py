from flask import Flask, request, jsonify, send_from_directory
import util

# Create a Flask app and set the static folder to serve HTML, CSS, JS
app = Flask(__name__, static_folder='../Client', static_url_path='')


@app.route('/')
def index():
    # Serve the main HTML page
    return send_from_directory('../Client', 'app.html')


@app.route('/api/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({'locations': util.get_location_names()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        estimated_price = util.get_estimated_price(
            location, total_sqft, bhk, bath)

        response = jsonify({'estimated_price': estimated_price})
    except Exception as e:
        response = jsonify({'error': str(e)})

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run(debug=True)
