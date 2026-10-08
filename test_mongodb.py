from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")


try:
  client.admin.command("ping")
  print("MongoDB connected Successfully")

except Exception as e:
  print("MongDB connected Failed")
  print(e)
