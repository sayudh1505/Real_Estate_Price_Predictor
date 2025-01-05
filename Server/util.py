import pickle
import json
import numpy as np

# Global variables
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        # Locate index for the provided location
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    # Create a zeroed array of the same length as the number of columns
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    # If the location is found in columns.json, set the corresponding index to 1
    if loc_index >= 0:
        x[loc_index] = 1

    # Return the predicted price
    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    
    global __data_columns
    global __locations

    # Load column names from columns.json
    with open(r"C:\Users\sayud\OneDrive\Documents\Somaiya\3rd Year\5th Sem\Honours-DA\IA-2\Server\artifacts\columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # First 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        # Load the trained model from the pickle file
        with open(r"C:\Users\sayud\OneDrive\Documents\Somaiya\3rd Year\5th Sem\Honours-DA\IA-2\Server\artifacts\Mumbai_home_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)
    
    print("Loading saved artifacts...done")

def get_location_names():
    # Return all location names
    return __locations

def get_data_columns():
    # Return all data columns
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    
    # Example predictions, ensure the location names match those in your dataset
    print(get_location_names())  # List of locations
    print(get_estimated_price('andheri', 1000, 3, 3))  # Example for Andheri
    print(get_estimated_price('andheri', 1000, 2, 2))
    print(get_estimated_price('dadar', 1000, 2, 2))  # Example for Dadar
    print(get_estimated_price('worli', 1000, 2, 2))  # Example for Worli
