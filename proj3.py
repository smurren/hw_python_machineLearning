# Sean Murren
# CMSC 471 Project 3
# May 2015
import math


class Node(object):
		
	def __init__(self, splitIndex):
		
		self.splitIndex = splitIndex
		self.leftChild = None
		self.rightChild = None
		self.classified = False
		self.classification = None

	
	

class DecisionTree(object):
	
	
	def __init__(self):
		self.tree = [] 
		self.tree = []
		self.rootEntropy = 0.0
		self.totalExamples = 0
		self.classCount = {}
	
	
	
	def train(self, examples):
		
		if len(examples) > 1:
		
			# calculate root entropy.  I was unable to use entropy function for this
			#---------------------------------------------------------------------
			classCount1 = 0
			classCount0 = 0
			for example in examples:
				if example[0] == 1:
					classCount1 += 1.0
				else:	
					classCount0 += 1.0
				self.totalExamples += 1.0
			
			if classCount1 > 0.0:
				self.rootEntropy = -1.0 * (classCount1 / self.totalExamples * 
								   math.log(classCount1 / self.totalExamples, 2) +
								   classCount0 / self.totalExamples * 
								   math.log(classCount0 / self.totalExamples, 2))
			#print(self.rootEntropy)
			#---------------------------------------------------------------------
			
			
			# calculate information gains and splitting point
			#---------------------------------------------------------------------
			maxGain = 0.0
			maxIndex = 0		
			# find attribute index that has max info gain from split
			maxGain, maxIndex = self.gain(self.rootEntropy, examples)
			#---------------------------------------------------------------------
			
			
			# build tree
			#---------------------------------------------------------------------
			# set root node split information
			self.tree.append(Node(maxIndex))
			
			#if entropy is not 0.0
			if self.rootEntropy != 0.0:
				
				# add the split to the tree:
				leftboundExamples = []
				rightboundExamples = []
				for example in examples:
					if example[maxIndex] == 1:
						rightboundExamples.append(example)
					else: 
						leftboundExamples.append(example)
				
				# create left subtrees
				if len(leftboundExamples) >= 1:
					leftBranch = DecisionTree()
					leftSubTree = leftBranch.train(leftboundExamples)
					self.tree[0].leftChild = leftSubTree[0]
				
				# create right subtrees
				if len(rightboundExamples) >= 1:
					rightBranch = DecisionTree()
					rightSubTree = rightBranch.train(rightboundExamples)
					self.tree[0].rightChild = rightSubTree[0]
				
				
			else:  # if 0.0 entropy at root, all examples are of same class
				self.tree[0].classified = True
				self.tree[0].classification = examples[0][0]
			#---------------------------------------------------------------------
			
			
		# if node contains only 1 example, classify as that example
		#---------------------------------------------------------------------
		else:  # len(examples) == 1: 
			self.tree.append(Node(None))
			self.tree[0].classified = True
			self.tree[0].classification = examples[0][0]
		#---------------------------------------------------------------------
	
		
		# return new tree
		return self.tree
		
	
	
	def gain(self, baseEntropy, testExamples):
		
		gain = -1.0
		gainIndex = 0
		
		# find attribute index with max information gain
		for attrIndex in range(1,len(testExamples[0])):
			
			# information gain for attribute at attrIndex
			newGain = baseEntropy - self.entropy(testExamples, attrIndex)
			
			if newGain > gain:
				gain = newGain
				gainIndex = attrIndex

		return gain, gainIndex
		
		
		
	def entropy(self, examples, attrIndex):
		
		e = 0.0		# final expected new entropy (return value), combines e1, e2
		e1 = 0.0	# entropy of value 1
		e2 = 0.0	# entropy of value 0
		classCount1 = 0.0	# count of examples classified 1
		attr0Count = 0.0
		attr1Count = 0.0
		totalExamples = 0.0
		
		# calculate e1
		#---------------------------------------------------------------------
		for example in examples:
			if example[attrIndex] == 1:
				if example[0] == 1:
					classCount1 += 1.0
				attr1Count += 1.0
				totalExamples += 1.0
		
		
		if classCount1 > 0.0 and classCount1 < attr1Count:
			e1 = -1.0 * (classCount1 / attr1Count * 
					  math.log(classCount1 / attr1Count, 2) +
					  (totalExamples-classCount1) / attr1Count * 
					  math.log((totalExamples-classCount1) / attr1Count, 2))
		#---------------------------------------------------------------------
		
		classCount1 = 0.0
		
		# calculate e2
		#---------------------------------------------------------------------
		for example in examples:
			if example[attrIndex] == 0:
				if example[0] == 1:
					classCount1 += 1.0
				attr0Count += 1.0
				totalExamples += 1.0
		
		
		if classCount1 > 0.0 and classCount1 < attr0Count:
			e2 = -1.0 * (classCount1 / attr0Count * 
					  math.log(classCount1 / attr0Count, 2) +
					  (totalExamples-classCount1) / attr0Count * 
					  math.log((totalExamples-classCount1) / attr0Count, 2))
		#---------------------------------------------------------------------
		
		# compute expected new entropy
		e = (attr1Count / totalExamples * e1) + (attr0Count / totalExamples * e2)
			
		return e
		

	def classify(self, newItem):
		
		print(newItem)
		
		# check the splitIndex/value of root node and compare to newItem, 
		# move down tree until 100% same class
		node = self.tree[0]
		
		while node.classified == False:
			
			if newItem[node.splitIndex-1] == 1:
				node = node.rightChild
			else:
				node = node.leftChild
		
		##print("Classification: " + str(node.classification))
		return node.classification
	
		
	
	
	
class NaiveBayes(object):
    ############################
	#  P(b|a) = P(a|b) * P(b)  #
	#  ----------------------  #
	#         P(a)             #
	############################
	
	
	def __init__(self):

		self.count0 = []
		self.count1 = []
		self.countB0 = 0.0
		self.countB1 = 0.0
	
	
	def train(self, examples):
		
		for a in range(len(examples[0])-1):
			self.count0.append(0)
			self.count1.append(0)
		
		for example in examples:
			exI = 0
			if example[exI] == 0: 
				self.countB0 += 1
				
				exI += 1
				for index in range(len(examples[0])-1):
					self.count0[index] += example[exI]
					exI += 1
				exI = 0
				
			if example[exI] == 1:
				self.countB1 += 1
				
				exI += 1
				for index in range(len(examples[0])-1):
					self.count1[index] += example[exI]
					exI += 1
				exI = 1
					
		for index0 in range(len(self.count0)):
			self.count0[index0] /= self.countB0 
		for index1 in range(len(self.count1)):
			self.count1[index1] /= self.countB1 
		
			
	
	def classify(self, newItem):
		p0, p1 = self.bayes(newItem)
		
		print(newItem)

		if p0 < p1:
			#print("Classified:  1")
			return "classified 1"
		elif p0 > p1:
			#print("Classified:  0")
			return "classified 0"
		else:
			#print("Could be classified 0 or 1")
			return "classified 0 or 1"
		
	
	def bayes(self, vector):
		
		p0 = 1.0
		p1 = 1.0
		
		for index in range(len(vector)):
			if vector[index] == 1:
				p0 += self.count0[index]
				p1 += self.count1[index]
			else:
				p0 += (1.0 - self.count0[index])
				p1 += (1.0 - self.count1[index])
			
			
		p0 *= (self.countB0 / (self.countB0 + self.countB1))
		p1 *= (self.countB1 / (self.countB0 + self.countB1))
		
		return p0, p1
		
		
	
	
class NearestNeighbor(object):

	def __init__(self):
		trainSet = []
	
	def train(self, vectors):
		self.trainSet = vectors
	
	def classify(self, vector):
		k = 1
		print(str(vector) + ",  K = 1")
		class0 = 0
		class1 = 0
		neighbors = self.getNeighbors(self.trainSet, vector, k)
		
		for neighbor in neighbors:
			if neighbor[0] == 1:
				class1 += 1
			else:
				class0 += 1
		
		if class1 > class0:
			#print("Majority k-Nearest classified 1")
			return "classified 1"
		elif class1 < class0:
			#print("Majority k-Nearest classified 0")
			return "classified 0"
		else:
			#print("k-Nearest classified 0 or 1")
			return "classified 0 or 1"
		
		
	def getNeighbors(self, trainSet, testVector, k):
		distances = []
		length = len(testVector)
		for x in range(len(trainSet)):
			dist = self.euclideanDistance(testVector, trainSet[x], length)
			distances.append((dist, trainSet[x]))
		distances = sorted(distances)
		
		#print(distances)
		
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][1])
		return neighbors
	
	def euclideanDistance(self, vector1, vector2, length):
		distance = 0
		for x in range(1,length+1):
			distance += pow((vector1[x-1] - vector2[x]), 2)
		return math.sqrt(distance)
	
	
def main():

	#---------------------------------------------------------------------
	print("\n\nDecision Tree:\n")
	dtObj = DecisionTree()
	#dtObj.train([[1, 0, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1]])
	#dtObj.classify([1,1,1,0,1])
	dtObj.train([[1,1,1,1],[1,1,1,0],[0,1,0,1],[0,0,0,1],[0,0,1,1],[0,1,0,1],[0,0,0,0]])
	print("Classification: " + str(dtObj.classify([1,1,1])))
	print("Classification: " + str(dtObj.classify([1,1,0])))
	print("Classification: " + str(dtObj.classify([1,0,1])))
	print("Classification: " + str(dtObj.classify([0,0,1])))
	print("Classification: " + str(dtObj.classify([0,1,1])))
	print("Classification: " + str(dtObj.classify([1,0,1])))
	print("Classification: " + str(dtObj.classify([0,0,0])))
	#---------------------------------------------------------------------
	
	
	#---------------------------------------------------------------------
	print("\n\nNaive Bayes:\n")
	nbObj = NaiveBayes()
	nbObj.train([[1, 0, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1]])
	print("Classification: " + str(nbObj.classify([0,1,1,1,1])))
	#nbObj.train([[1,1,1,1],[1,1,1,0],[0,1,0,1],[0,0,0,1],[0,0,1,1],[0,1,0,1],[0,0,0,0]])
	#print("Classification: " + str(nbObj.classify([1,1,1])))
	#print("Classification: " + str(nbObj.classify([1,1,0])))
	#print("Classification: " + str(nbObj.classify([1,0,1])))
	#print("Classification: " + str(nbObj.classify([0,0,1])))
	#print("Classification: " + str(nbObj.classify([0,1,1])))
	#print("Classification: " + str(nbObj.classify([1,0,1])))
	#print("Classification: " + str(nbObj.classify([0,0,0])))
	#---------------------------------------------------------------------
	
	
	#---------------------------------------------------------------------
	print("\n\nNearestNeighbor:\n")
	knObj = NearestNeighbor()
	#knObj.train([[1, 0, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1]])
	#print("Classification: " + str(knObj.classify([0,0,1,0,1])))
	knObj.train([[1,1,1,1],[1,1,1,0],[0,1,0,1],[0,0,0,1],[0,0,1,1],[0,1,0,1],[0,0,0,0]])
	print("Classification: " + str(knObj.classify([1,1,1])))
	print("Classification: " + str(knObj.classify([1,1,0])))
	print("Classification: " + str(knObj.classify([1,0,1])))
	print("Classification: " + str(knObj.classify([0,0,1])))
	print("Classification: " + str(knObj.classify([0,1,1])))
	print("Classification: " + str(knObj.classify([1,0,1])))
	print("Classification: " + str(knObj.classify([0,0,0])))
	#---------------------------------------------------------------------
	
	
main()