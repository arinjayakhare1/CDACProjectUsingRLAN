import csv

x = []
y = []
x0 = []
x1 = []
x2 = []
label = []
typeLabel = [0, 0, 0, 0, 0]
with open("newData.csv", 'r', encoding="utf8") as csvFile:
	reader = csv.reader(csvFile)
	p = 0
	for row in reader:
		if(p == 0):
			p = p + 1
			continue
		if(len(row) >= 2):
			s = row[0].split(",")[1]
			# print(s)
			x.append(s)
			sc = 4
			for i in range(1, 5):
				if(row[i] == "1.0" or row[i] == "2.0"):
					sc = i - 1
					break
			label.append(sc)
			typeLabel[sc] += 1
		p = p + 1
csvFile.close

print("text : ", len(x), "\t scores = ", typeLabel[0] + typeLabel[1] + typeLabel[2] + typeLabel[3] + typeLabel[4])
print("Types : \n", typeLabel)