from mongodb_connection import feedback_collection
feedback = {

  "name" : "Test User",
  "disease":"Diabetes",
  "prediction":"Positive",
  "feedback":"This is a test feedback",
  "status":"pending"
}

result = feedback_collection.insert_one(feedback)

print("Feedback inserted successfully")
print("Feedback ID",result.inserted_id)