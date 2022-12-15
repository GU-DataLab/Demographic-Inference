
import random

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
genders = ["male", "female"]

w_gt = open("imdb_gt.csv", 'w')

for i in range(32):
	idx = random.randint(4, 8)
	name = ""
	for j in range(idx):
		name += random.choice(letters)
	gender = random.choice(genders)
	w_emd = open("imdb_embeddings/"+name+".csv", 'w')
	w_gt.write(",".join([name, gender])+"\n")
	for k in range(32):
		emd = ["2020-02-01 19:55:27"]
		for i in range(512):
			emd.append(str(round(random.random(), 3)))
		w_emd.write(",".join(emd)+"\n")
	w_emd.close()
w_gt.close()