from pymongo import MongoClient

MONGO_URL = "mongodb://127.0.0.1:27017/"

client = MongoClient(MONGO_URL)

db = client["medical_prediction_feedback"]

feedback_collection = db['feedback']


def test_connection():
  try:
    client.admin.command("ping")
    return True
  except Exception as e:
    print("MongoDB connnection error.",e)
    return False