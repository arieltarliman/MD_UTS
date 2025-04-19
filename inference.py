import pickle
import pandas as pd
from preprocessor import HotelPreprocessor

# Load model
with open('hotel_booking_rf.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
selected_features = data['features']

# Simulated input
incoming_data = pd.DataFrame([{
    'lead_time': 100,
    'avg_price_per_room': 150.5,
    'no_of_special_requests': 1,
    'total_guests': 3,
    'total_nights': 2,
    'type_of_meal_plan': 1,
    'room_type_reserved': 2,
    'market_segment_type': 4,
    'repeated_guest': 0,
    'required_car_parking_space': 1,
    'Booking_ID': 'ABCS', 
    'no_of_adults' : 2, 
    'no_of_children': 0, 
    'no_of_weekend_nights': 2,
    'no_of_week_nights': 5,
    'type_of_meal_plan': 'Meal Plan 1', 
    'lead_time': 30, 
    'arrival_year':2018, 
    'arrival_month': 4,
    'arrival_date': 14, 
    'no_of_previous_cancellations': 0, 
    'no_of_previous_bookings_not_canceled':0,
}])
# Preprocess
processor = HotelPreprocessor(incoming_data)
processed_data = processor.preprocess(is_training=False)

# Select features
input_df = processed_data[selected_features]

# Predict
prediction = model.predict(input_df)
print("Prediction:", "Canceled" if prediction[0] == 1 else "Not Canceled")
