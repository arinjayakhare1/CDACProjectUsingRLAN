import pymongo


class mongoDBUtils:
	def __init__(self):
		self.client = pymongo.MongoClient("mongodb://localhost:27017/")
		self.db = self.client["CDAC"]
		self.tweetIdTextCollection = self.db["tweetIdText"]
		self.tweetIdStatusCollection = self.db["tweetIdStatus"]
		self.tweetIdScoreCollection = self.db["tweetIdScore"]
		self.twitterKeysCollection = self.db["twitterKeys"]
		self.classes = ["violentExtremism", "nonViolentExtremism", "radicalViolence", "nonRadicalViolence", "notRelevant"]
		
	def getDB(self, dbName):
		return self.db

	def getClient(self):
		return self.client

	def getCollection(self, db, collName):
		return db[collName]	

	def find(self, collection, parameters):
		return collection.find(parameters)	

	def insert(self, collection, parameters):
		collection.insert_one(parameters)	

	def insertMany(self, collection, parameters):
		collection.insert_many(parameters)	

	def update(self, collection, initParameter, updatedParameter):
		collection.update_one(initParameter, updatedParameter)	

	def disconnect(self):
		self.client.close()	