

class counter:
	def __init__(self):
		self.tweetsAdded = 0
		self.classifiedTweets = {"violentExtremism" : 0, "nonViolentExtremism" : 0, "radicalViolence" : 0, "nonRadicalViolence" : 0, "notRelevant" : 0}
		self.trainedTweets = {"violentExtremism" : 0, "nonViolentExtremism" : 0, "radicalViolence" : 0, "nonRadicalViolence" : 0, "notRelevant" : 0}