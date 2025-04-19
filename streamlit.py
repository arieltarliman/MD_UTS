import streamlit as st
import pickle
import pandas as pd
from preprocessor import HotelPreprocessor

# === 1. Load model and selected features ===
with open('hotel_booking_rf.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
selected_features = data['features']

# === 2. Define test cases ===
test_cases = {
    "Test Case 1": {
        'lead_time': 1,
        'avg_price_per_room': 200.0,
        'no_of_special_requests': 0,
        'repeated_guest': 0,
        'required_car_parking_space': 0,
        'type_of_meal_plan': 'Meal Plan 1',
        'room_type_reserved': 'Room_Type 2',
        'market_segment_type': 'Online',
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 0,
        'no_of_week_nights': 3,
        'arrival_year': 2025,
        'arrival_month': 4,
        'arrival_date': 20,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 1
    },
    "Test Case 2": {
        'lead_time': 1,
        'avg_price_per_room': 2000.0,
        'no_of_special_requests': 0,
        'repeated_guest': 0,
        'required_car_parking_space': 0,
        'type_of_meal_plan': 'Meal Plan 1',
        'room_type_reserved': 'Room_Type 1',
        'market_segment_type': 'Online',
        'no_of_adults': 10,
        'no_of_children': 3,
        'no_of_weekend_nights': 7,
        'no_of_week_nights': 10,
        'arrival_year': 2020,
        'arrival_month': 5,
        'arrival_date': 3,
        'no_of_previous_cancellations': 1000,
        'no_of_previous_bookings_not_canceled': 0
    }
}

# === Streamlit App ===
st.title("Hotel Booking Cancellation Predictor")
st.write("Enter booking details or choose a test case to get a prediction.")

mode = st.radio("Select Input Mode:", ["Use Test Case", "Manual Input"])

if mode == "Use Test Case":
    case_name = st.selectbox("Choose a test case:", list(test_cases.keys()))
    selected_case = test_cases[case_name]
    st.subheader("Selected Test Case Details:")
    st.json(selected_case)  # or st.write(selected_case) for a table view
    incoming = pd.DataFrame([selected_case])
else:
    # Manual input fields
    incoming = pd.DataFrame([{  
    'lead_time': st.slider('Lead Time (days)', min_value=0, max_value=600, value=0),
    'avg_price_per_room': st.number_input('Average Price per Room', min_value=0.0, max_value=1000000.0, value=100.0, step=0.01),
    'no_of_special_requests': st.selectbox('Special Requests', [0, 1, 2, 3, 4, 5]),
    'repeated_guest': st.selectbox('Repeated Guest', [0, 1]),
    'required_car_parking_space': st.selectbox('Parking Space Required', [0, 1]),
    'type_of_meal_plan': st.selectbox('Meal Plan', ['Meal Plan 1','Meal Plan 2','Meal Plan 3','Not Selected']),
    'room_type_reserved': st.selectbox('Room Type Reserved', ['Room_Type 1','Room_Type 2','Room_Type 3','Room_Type 4','Room_Type 5','Room_Type 6','Room_Type 7']),
    'market_segment_type': st.selectbox('Market Segment', ['Aviation','Complementary','Corporate','Offline','Online']),
    'no_of_adults': st.slider('Number of Adults', min_value=0, max_value=10, value=0, step=1),
    'no_of_children': st.slider('Number of Children', min_value=0, max_value=10, value=0, step=1),
    'no_of_weekend_nights': st.slider('Weekend Nights', min_value=0, max_value=50, value=0, step=1),
    'no_of_week_nights': st.slider('Week Nights', min_value=0, max_value=50, value=0, step=1),
    'arrival_year': st.slider('Arrival Year', min_value=2017, max_value=2025, value=2017, step=1),
    'arrival_month': st.slider('Arrival Month', min_value=1, max_value=12, value=1, step=1),
    'arrival_date': st.slider('Arrival Date', min_value=1, max_value=31, value=1, step=1),
    'no_of_previous_cancellations': st.number_input('Previous Cancellations', min_value=0, max_value=100, value=0, step=1),
    'no_of_previous_bookings_not_canceled': st.number_input('Previous Bookings (Not Canceled)', min_value=0, max_value=100, value=0, step=1)
}])


# Button to trigger prediction
def predict(input_df):
    # Preprocess incoming data
    processor = HotelPreprocessor(input_df)
    processed = processor.preprocess(is_training=False)
    # Select only what the model expects
    X = processed[selected_features]
    # Predict
    pred = model.predict(X)[0]
    return "Canceled" if pred == 1 else "Not Canceled"

if st.button("Predict Booking Status"):
    result = predict(incoming)
    st.success(f"Prediction Result: {result}")
