from scipy.io.arff import loadarff
import sys
import math
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.interactive(False)

weights = []
bias = 0.1
table = []
classes = []
expected = []

def partition(folds):
	p = [[] for i in range(0,folds)]
	pos = []
	neg = []
	mapping = {}

	for i in table:
		if i[-1] == classes[0]:
			neg.append(i)
		elif i[-1] == classes[1]:
			pos.append(i) 
	
	pos_per = len(pos)/folds
	neg_per = len(neg)/folds
	#print(pos_per)
	#print(neg_per)
	
	for i in range(0,folds):
		pos_sam = random.sample(pos,pos_per)
		for j in pos_sam:
			mapping[table.index(j)] = i+1
			pos.remove(j)
			p[i].append(j)	

	count = 0
	while len(pos) != 0:
		p[count].append(pos[0])
		mapping[table.index(pos[0])] = count + 1
		pos = pos[1:]
		count = (count+1)%folds
	
	for i in range(0,folds):
		neg_sam = random.sample(neg,neg_per)
		for j in neg_sam:
			mapping[table.index(j)] = i+1
			neg.remove(j)
			p[i].append(j)	

	count = 0
	while len(neg) != 0:
		p[count].append(neg[0])
		mapping[table.index(neg[0])] = count + 1
		neg = neg[1:]
		count = (count+1)%folds
	
	return p, mapping 	

def train_classify(train_data,learning_rate,num_epochs,test_data,bias, weights):
	train_table = []
	for i in train_data:
		for j in i:
			train_table.append(j)
	#Train		
	for e in range(0,num_epochs):
		for t in train_table:
			inp = t[:-1]
			net = bias
			for k in range(0,len(weights)):
				net = net + weights[k]*inp[k]
			if t[-1] == classes[0]:
				exp = expected[0]
			else:
				exp = expected[1]		
			act = 1.0/(1.0 + math.exp(-net))
			delta = act * (1 - act) * (exp - act)
			dw = learning_rate*delta
			bias = bias + dw*1
			for k in range(0,len(weights)):
				weights[k] = weights[k]+dw*inp[k]
	#Test
	test_set = []			  	
	for t in test_data:
		tv = []
		inp = t[:-1]
		net = bias
		for k in range(0,len(weights)):
			net = net + weights[k]*inp[k]
		tv.append(table.index(t))
		act = 1.0/(1.0 + math.exp(-net))
		if act >= 0.5:
			tv.append(classes[1])
		else:
			tv.append(classes[0])		
		tv.append(t[-1])
		tv.append(act)
		# tv contains [instance_id,prediction,actual class,activation/confidence]
		test_set.append(tv)
	return test_set				
	
if __name__ == "__main__":
	train_file = sys.argv[1]
	num_folds = int(sys.argv[2])
	learning_rate = float(sys.argv[3])
	num_epochs = int(sys.argv[4])

	#Parse file to get features and  attributes
	attributes = []
	train_data, train_meta = loadarff(train_file)
	table = [list(i) for i in train_data]
	att_names = train_meta.names()
	classes = list(train_meta.__getitem__(att_names[-1])[1])
	expected = [0.0,1.0]

	#Initialize weights and bias
	#for i in range(0,len(att_names)-1):
	#	weights.append(0.1)

	# Partition
	partitions , mapping = partition(num_folds)

	# Perform cross validation: Send data to train(learning rate , epochs and test data)
	classify = []
	#print(len(partitions))
	#print(len(partitions[9]))
	#print(len(mapping))
	train_data = partitions
	for i in range(0,num_folds):
		bias = 0.1
		#print(i)
		#print(len(train_data))
		for j in range(0,len(att_names)-1):
			weights.append(0.1)
		test_data = train_data[i]
		train_data.remove(test_data)
		res = train_classify(train_data,learning_rate,num_epochs,test_data,bias,weights)
		train_data.insert(i,test_data)
		for en in res:
			classify.append(en)
		weights = []
		 
	#print(att_names)
	#print(classes)
	#print(mapping)
	#print(len(partitions))
	#print(len(partitions[0]))
	#print(partitions[0])
	#print(len(mapping))
	#print(mapping)
	#print(expected)
	#print(table)
	
	
	# Sort classify by id and generate trace for printing
	output = sorted(classify,key =  lambda a: a[0])
	roc_table = sorted(classify, key = lambda a: -a[3])
	classifications = len(classify)
	right = 0.0
	for i in output:
		if i[1]==i[2]:
			right=right+1
	printtrace = []
	for p in output:
		inst = []
		inst.append(mapping[p[0]])
		inst.append(p[1])
		inst.append(p[2])
		inst.append(p[3])
		printtrace.append(inst)
		
	# Generate points for TPR and FPR	
	pos = 0
	neg = 0
	for i in roc_table:
		if i[2]==classes[0]:
			neg=neg+1
		else:
			pos=pos+1	
	roc_all = []
	nr_pos = 0.0
	nr_neg = 0.0
	for i in roc_table:
		ins = []
		if i[2]==classes[0]:
			nr_neg=nr_neg+1
		else:
			nr_pos=nr_pos+1
		ins.append(nr_pos/pos)
		ins.append(nr_neg/neg)
		roc_all.append(ins)		
	
	# Print Trace		
	#print(100*right/classifications)	
	#print(len(classify))
	#print(len(printtrace))
	#print(classify)
	#print(printtrace)
	for tr in printtrace:
		for t in tr:
			print t,
		print("")	
	
	#Code for RoC Plot
	#print(roc_all)
	y = [i[0] for i in roc_all]
	x = [i[1] for i in roc_all]
	#plt.plot(x,y,'ro')
	#plt.ylabel('TPR')
	#plt.xlabel('FPR')
	#plt.title("ROC Curve")
	#plt.show()
	#plt.savefig("Roc.png")
	#print("Generated graph")
