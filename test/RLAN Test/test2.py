s = ""
with open("keywords.txt", 'r') as f:
	for line in f:
		s += line[:-1] + ", " 

print(s)