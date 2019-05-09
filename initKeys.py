from mongoDBUtils import mongoDBUtils
import time
import datetime

consumer_keys = []
consumer_secrets = []
access_tokens = []
access_token_secrets = []

with open("access_keys_total.txt") as f:
	lines = f.readlines()

p = 0

for y in lines:
	x = y[:-1]
	if(p == 0):
		consumer_keys.append(x)
	elif (p == 1):
		consumer_secrets.append(x)
	elif(p == 2):
		access_tokens.append(x)
	elif(p == 3):
		access_token_secrets.append(x)
	p = (p + 1) % 4

print(consumer_keys)
print(consumer_secrets)
print(access_tokens)
print(access_token_secrets)

client = mongoDBUtils()
db = client.db
twitterKeysCollection = client.twitterKeysCollection

endTime2 = time.time() + 86400
endTime2 = datetime.datetime.fromtimestamp(endTime2, None)

print(len(consumer_keys))
for x in range(0, len(consumer_keys)):
	paramDict = {}
	paramDict["consumer_key"] = consumer_keys[x]
	paramDict["consumer_secret"] = consumer_secrets[x]
	paramDict["access_token"] = access_tokens[x]
	paramDict["access_token_secret"] = access_token_secrets[x]
	paramDict["TTL"] = endTime2

	lastUsedTime = time.time()
	lastUsedTime = datetime.datetime.fromtimestamp(lastUsedTime, None)

	paramDict["lastUsedTime"] = lastUsedTime

	client.insert(twitterKeysCollection, paramDict)

client.disconnect()