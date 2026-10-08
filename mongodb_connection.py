from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["MultiDiseasePrediction"]

feedback_collection = db["feedback"]


print("MongoDB database connected successfully")