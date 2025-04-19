import pandas as pd

class HotelPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def preprocess(self, is_training=True):
        df = self.df

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Fill missing values
        df['avg_price_per_room'] = df.groupby('room_type_reserved')['avg_price_per_room'].transform(
            lambda x: x.fillna(x.mean())
        )
        df['required_car_parking_space'] = df['required_car_parking_space'].fillna(
            df['required_car_parking_space'].mode()[0]
        )
        df['type_of_meal_plan'] = df['type_of_meal_plan'].fillna(
            df['type_of_meal_plan'].mode()[0]
        )
        df.dropna(inplace=True)

        # Type conversions
        df['type_of_meal_plan'] = df['type_of_meal_plan'].astype('category')
        df['room_type_reserved'] = df['room_type_reserved'].astype('category')
        df['market_segment_type'] = df['market_segment_type'].astype('category')
        df['avg_price_per_room'] = df['avg_price_per_room'].astype('float32')

        if is_training and 'booking_status' in df.columns:
            df['booking_status'] = df['booking_status'].astype('category')

        # Feature engineering
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

        # Feature encoding
        meal_plan_mapping = {'Meal Plan 1': 0, 'Meal Plan 2': 1, 'Meal Plan 3': 2, 'Not Selected': 3}
        room_type_mapping = {
            'Room_Type 1': 0, 'Room_Type 2': 1, 'Room_Type 3': 2,
            'Room_Type 4': 3, 'Room_Type 5': 4, 'Room_Type 6': 5, 'Room_Type 7': 6
        }
        market_segment_mapping = {
            'Aviation': 0, 'Complementary': 1, 'Corporate': 2,
            'Offline': 3, 'Online': 4
        }

        df['type_of_meal_plan'] = df['type_of_meal_plan'].map(meal_plan_mapping)
        df['room_type_reserved'] = df['room_type_reserved'].map(room_type_mapping)
        df['market_segment_type'] = df['market_segment_type'].map(market_segment_mapping)

        col_to_drop = ['no_of_weekend_nights','no_of_week_nights','no_of_adults','no_of_children']
        
        df = df.drop(col_to_drop, axis=1, errors='ignore')

        if is_training and 'booking_status' in df.columns:
            df = df.drop('Booking_ID', axis = 1)
            df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

        return df
