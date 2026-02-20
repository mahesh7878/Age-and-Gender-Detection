from pymongo import MongoClient
import datetime

client = MongoClient('mongodb://localhost:27017/')
db = client['age_gender_db']
detections_collection = db['detections']
users_collection = db['users']

def insert_detection(gender, age):
    detection = {
        'gender': gender,
        'age': age,
        'timestamp': str(datetime.datetime.now())
    }
    detections_collection.insert_one(detection)
    print(f"Inserted detection: {detection}")

def register_user(username, password, email):
    if users_collection.find_one({'username': username}):
        return False
    users_collection.insert_one({'username': username, 'password': password, 'email': email})
    return True

def check_user(username, password):
    user = users_collection.find_one({'username': username, 'password': password})
    return user is not None

def get_all_detections():
    return list(detections_collection.find({}, {'_id': 0}))