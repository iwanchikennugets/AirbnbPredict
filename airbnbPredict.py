import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Configuration & Setup ---
# Configure Streamlit page settings for title and layout
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load the pre-trained machine learning model
try:
    model = joblib.load('model_final_airbnb.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'model_final_airbnb.pkl' not found. Ensure it's in the correct directory.")
    st.stop()

# Define the central geographical reference point (Times Square/Midtown Manhattan)
REFERENCE_LATITUDE = 40.7580
REFERENCE_LONGITUDE = -73.9855

# Define the min/max for latitude and longitude for display
LAT_MIN = 40.5
LAT_MAX = 40.9
LON_MIN = -74.2
LON_MAX = -73.7

# --- Streamlit App UI ---
st.title("🏡 NYC Airbnb Price Predictor")
st.markdown("Enter details about the Airbnb listing to get a price estimation.")

# Create two columns for side-by-side layout (map on left, inputs on right)
col_map, col_inputs = st.columns([1, 1])

with col_map:
    st.header("Location Preview")
    st.write("See where your selected listing location is on the map:")

    # Prepare data for the interactive map, using session state for live updates
    map_data = pd.DataFrame({
        'lat': [st.session_state.get('latitude', 40.7)],
        'lon': [st.session_state.get('longitude', -74.0)]
    })
    st.map(map_data, zoom=11)

with col_inputs:
    st.header("Listing Details")

    # Use st.number_input for Latitude and Longitude with min/max values in the label
    latitude = st.number_input(
        f"Enter Latitude (Min: {LAT_MIN}, Max: {LAT_MAX})", # Included min/max in label
        min_value=LAT_MIN, max_value=LAT_MAX, value=st.session_state.get('latitude', 40.7),
        step=0.0001, format="%.4f", help="Latitude coordinate of the listing location. Values typically range from 40.5 to 40.9 in NYC." , key='latitude'
    )
    longitude = st.number_input(
        f"Enter Longitude (Min: {LON_MIN}, Max: {LON_MAX})", # Included min/max in label
        min_value=LON_MIN, max_value=LON_MAX, value=st.session_state.get('longitude', -74.0),
        step=0.0001, format="%.4f", help="Longitude coordinate of the listing location. Values typically range from -74.2 to -73.7 in NYC.", key='longitude'
    )

    # Input widgets for room type, minimum nights, and availability
    room_type_selected = st.selectbox(
        "Select Room Type",
        ['Private room', 'Shared room', 'Entire home/apt'], help="Type of room or entire property available for booking."
    )
    minimum_nights_input = st.number_input(
        "Enter Minimum Nights Required",
        min_value=1, max_value=365, value=1, help="The minimum number of nights required for a booking."
    )
    availability_365 = st.slider(
        "Select Availability in 365 Days",
        min_value=0, max_value=365, value=180, help="Number of days the listing is available for booking in the next year."
    )

    # --- Prediction Logic (Moved inside col_inputs) ---
    # Button to trigger the price prediction
    if st.button("Calculate Estimated Airbnb Price"):
        with st.spinner("Calculating price..."):
            # Collect raw user inputs into a dictionary
            raw_input_data = {
                'latitude': latitude, 'longitude': longitude,
                'minimum_nights': minimum_nights_input,
                'availability_365': availability_365,
                'room_type': room_type_selected
            }

            # Convert raw input to a Pandas DataFrame for preprocessing
            df_for_prediction = pd.DataFrame([raw_input_data])

            # Calculate 'dist_manhattan' feature, mirroring training preprocessing
            df_for_prediction['dist_manhattan'] = np.sqrt(
                (df_for_prediction['latitude'] - REFERENCE_LATITUDE)**2 +
                (df_for_prediction['longitude'] - REFERENCE_LONGITUDE)**2
            )

            # Bin 'minimum_nights' into categorical 'min_nights', mirroring training preprocessing
            min_nights_bins = [0, 3, 7, 30, 1000]
            min_nights_labels = ['1-3', '4-7', '8-30', '31+']
            df_for_prediction['min_nights'] = pd.cut(
                df_for_prediction['minimum_nights'],
                bins=min_nights_bins, labels=min_nights_labels, right=True, include_lowest=True
            )
            # Drop the original numerical 'minimum_nights' column
            df_for_prediction = df_for_prediction.drop('minimum_nights', axis=1)

            # Apply one-hot encoding to categorical features ('room_type', 'min_nights')
            df_for_prediction = pd.get_dummies(df_for_prediction, columns=['room_type'], prefix='room_type')
            df_for_prediction = pd.get_dummies(df_for_prediction, columns=['min_nights'], prefix='min_nights')

            # Define the exact list of feature names and their order expected by the trained model
            model_feature_names = [
                'latitude', 'longitude', 'availability_365', 'dist_manhattan',
                'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room',
                'min_nights_1-3', 'min_nights_4-7', 'min_nights_8-30', 'min_nights_31+'
            ]

            # Add any one-hot encoded columns that might be missing for the current input (fill with 0)
            for col in model_feature_names:
                if col not in df_for_prediction.columns:
                    df_for_prediction[col] = 0

            # Reorder columns to match the model's expected input order
            df_final_input = df_for_prediction[model_feature_names]

            # Attempt to make a prediction and display the result
            try:
                y_unseen_pred = model.predict(df_final_input)[0]
                st.success(f"**Predicted Airbnb Price: ${y_unseen_pred:,.2f}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write("Please check if the input features match the model's expectations.")
                st.dataframe(df_final_input) # Display DataFrame for debugging

# --- Styling ---
# Apply custom CSS for background, text, buttons, and component styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/842948/pexels-photo-842948.jpeg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white; /* Default text color */
    }
    .stApp > div:first-child {
        padding: 20px;
        border-radius: 10px;
    }

    /* Custom colors and shadows for headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FF69B4; /* Pink heading color */
        text-shadow: 1px 1px 2px black;
    }

    /* Ensure various Streamlit component labels remain white for visibility */
    div[data-testid="stForm"] label,
    div[data-testid="stVerticalBlock"] label,
    .st-emotion-cache-1wv7stf,
    .st-emotion-cache-1kvjp2x,
    .st-emotion-cache-r421ms,
    .st-emotion-cache-10qj79j {
        color: white;
    }

    /* Custom styling for the prediction button */
    .stButton > button {
        background-color: red; /* Red button */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF1493; /* Darker pink on hover */
        color: white;
    }

    /* Styling for the success message box */
    div[role="alert"] {
        background-color: #5d76a2 !important;
        color: white !important;
        padding: 15px 20px !important;
        border-radius: 1rem !important;
        opacity: 1 !important; /* Ensure full opacity */
    }

    /* Styling for the map container */
    .st-emotion-cache-1r6y40p {
        border-radius: 10px;
        overflow: hidden;
        height: 500px;
        /* Removed border and box-shadow */
    }

    /* Styling for number input fields */
    .st-emotion-cache-1c7y2qn input[type="number"] {
        border-radius: 5px;
        padding: 8px;
        color: white; /* Text color inside input */
        /* Removed border and background-color */
    }

    /* Styling for selectbox */
    .st-emotion-cache-1c7y2qn .stSelectbox > div[data-baseweb="select"] {
        border-radius: 5px;
        /* Removed border and background-color */
    }

    /* Styling for slider */
    .st-emotion-cache-1c7y2qn .stSlider > div[data-baseweb="slider"] {
        border-radius: 5px;
        padding: 8px;
        /* Removed border and background-color */
    }

    </style>
    """,
    unsafe_allow_html=True
)
