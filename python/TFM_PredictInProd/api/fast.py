import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


from datetime import datetime
import pytz

# create a datetime object from the user provided datetime
pickup_datetime = "2021-05-30 10:12:00"
pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

# localize the user datetime with NYC timezone
eastern = pytz.timezone("US/Eastern")
localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

# localize the datetime to UTC
utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

@app.get("/") # predict
def predict(pickup_datetime, pickup_longitude, 
            pickup_latitude, dropoff_longitude, 
            dropoff_latitude, passenger_count):
    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    
    data = {"key": str("2013-07-06 17:18:00.000000119"),
            "pickup_datetime": [utc_pickup_datetime],
            "pickup_longitude": [float(pickup_longitude)],
            "pickup_latitude": [float(pickup_latitude)],
            "dropoff_longitude": [float(dropoff_longitude)],
            "dropoff_latitude": [float(dropoff_latitude)],
            "passenger_count": [int(passenger_count)]}
    model = joblib.load('model.joblib')
    df = pd.DataFrame(data)
    
    return {'prediction': model.predict(df)[0]}

