import glob
import cv2
import numpy as np
import time

# store = {} #for memoization of the recurssion
# def get_intensity(img,i,j):
# 	if i < 0 or j < 0:
# 		key = (i,j)
# 		if key not in store:
# 			store[key] = 0
# 		return store[key]
# 	else:
# 		key = (i,j)
# 		if key not in store: # use memoization
# 			store[key] = get_intensity(img,i-1,j) + get_intensity(img,i,j-1) - get_intensity(img,i - 1,j - 1) + img[i][j]
# 		return store[key]

def get_integral_img(img):
	temp = [[0]*len(img[0]) for _ in range(len(img))]
	for i in range(1,len(img)):
		for j in range(1,len(img[0])):
			a = temp[i - 1][j] if i - 1 in range(len(img)) else 0
			b = temp[i][j-1] if j - 1 in range(len(img[0])) else 0
			c = temp[i - 1][j - 1] if (i - 1 in range(len(img)) and j - 1 in range(len(img[0]))) else 0
			# temp[i][j] = temp[i - 1][j] + temp[i][j-1] - temp[i - 1][j - 1] + img[i][j]
			temp[i][j] = a + b - c +img[i][j]
	return temp

##########################
#test = [[1,3,7,5],[12,4,8,2],[0,14,16,9],[5,11,6,10]] # try the example in the slide
##########################

def get_rectangle_sum(integral_img, left_top,right_top,left_bottom, right_bottom):#IMPORTANT: these coordinates are all inclusive
	n = len(integral_img)
	a = integral_img[left_top[0] - 1][left_top[1] - 1] if (left_top[0] - 1 in range(n) and left_top[1] - 1 in range(n)) else 0
	b = integral_img[left_bottom[0]][left_bottom[1] - 1] if (left_bottom[0] in range(n) and left_bottom[1] - 1 in range(n)) else 0
	c = integral_img[right_bottom[0]][right_bottom[1]] if (right_bottom[0] in range(n) and right_bottom[1] in range(n)) else 0
	d = integral_img[right_top[0] - 1][right_top[1]] if (right_top[0] - 1 in range(n) and right_top[1] in range(n)) else 0
	return c + a - (b + d)

def get_regtangle_sum_by_depth_width(integral_img, left_top, depth, width):
	return get_rectangle_sum(integral_img, left_top, (left_top[0],left_top[1]+width-1), (left_top[0] + depth - 1, left_top[1]), (left_top[0] + depth - 1,left_top[1]+width-1) )

def featureA(integral_img,left_top,depth,width):
	left_block = get_regtangle_sum_by_depth_width(integral_img,left_top, depth, width//2)
	right_block = get_regtangle_sum_by_depth_width(integral_img,(left_top[0], left_top[1] + width//2), depth, width//2)
	return right_block - left_block
def featureB(integral_img,left_top,depth,width):
	upper_block = get_regtangle_sum_by_depth_width(integral_img,left_top, depth//2, width)
	lower_block = get_regtangle_sum_by_depth_width(integral_img,(left_top[0] + depth//2, left_top[1]), depth//2, width)
	return upper_block - lower_block
def featureC(integral_img,left_top,depth,width):
	width1 = width//3
	width2 = width - 2 * (width1)
	width3 = width1
	block_1 = get_regtangle_sum_by_depth_width(integral_img,left_top, depth, width1)
	block_2 = get_regtangle_sum_by_depth_width(integral_img,(left_top[0], left_top[1] + width1), depth, width2)
	block_3 = get_regtangle_sum_by_depth_width(integral_img,(left_top[0], left_top[1] + width1 + width2), depth, width3)
	return block_2 - (block_1 + block_3)
def featureD(integral_img,left_top,depth,width):
	# we make use of feature A to compute minus upper half plus that of lower half
	return - featureA(integral_img, left_top,depth//2,width) + featureA(integral_img, (left_top[0]+depth//2,left_top[1]),depth//2,width)

def createFeatureVector(featureFunc,integral_img, initialDepth,initialWidth,depthIncre,widthIncre, get_particular_feature = -1):
	result = []
	test_d = len(integral_img)
	test_w = len(integral_img[0])
	countFeature = 0
	for i in range(test_d):
		for j in range(test_w):
			wid = initialWidth
			while j + wid <= test_w:
				dep = initialDepth
				while i + dep <= test_d:
					if get_particular_feature == -1:
						result.append(featureFunc(integral_img,(i,j),dep,wid))
					else:# this is to get a particular feature for testing use
						if (countFeature == get_particular_feature):
							return featureFunc(integral_img,(i,j),dep,wid)
					countFeature += 1
					dep += depthIncre
				wid += widthIncre
	return result

def joint_feature(integral_img):
	return createFeatureVector(featureA,integral_img,1,2,1,2) + createFeatureVector(featureB,integral_img,2,1,2,1) + createFeatureVector(featureC,integral_img,1,3,1,3) + createFeatureVector(featureD,integral_img,2,2,2,2)
# print(createFeatureVector(featureA,test,1,2,1,2)) # width of feature A needs to be bisect
# print(createFeatureVector(featureB,test,2,1,2,1)) # depth of feature B needs to be bisect
# print(createFeatureVector(featureC,test,1,3,1,3)) # width of feature C needs to be trisect
# print(createFeatureVector(featureD,test,2,2,2,2)) # both width and depth of feature D need to be bisect


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
abc = AdaBoostClassifier(n_estimators=10, algorithm = 'SAMME')
rfc = RandomForestClassifier(n_estimators=10, max_depth = 1) # require rfc to have depth 1 to be better compared with abc




from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree


train_face = [cv2.imread(file,0) for file in glob.glob("train/face/*.pgm")]
train_nonface = [cv2.imread(file,0) for file in glob.glob("train/non-face/*.pgm")]
test_face = [cv2.imread(file, 0) for file in glob.glob("test/face/*.pgm")]
test_nonface = [cv2.imread(file, 0) for file in glob.glob("test/non-face/*.pgm")]
systemKey = [1] * len(test_face) + [-1] * len(test_nonface)

#label face as 1, non-face as -1
X = train_face + train_nonface
y = [1] * len(train_face) + [-1] * len(train_nonface)




X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

#given that we know how long a subfeature vector is
(lA,lB,lC,lD) = (17100,17100,10830,8100)


def get_feature(integral_img,idx): #assuming feature index start from 1
	if 1 <= idx <= lA:
		return createFeatureVector(featureA,integral_img,1,2,1,2,idx - 1)
	elif idx <= lA+lB:
		return createFeatureVector(featureB,integral_img,2,1,2,1,(idx - 1) % lA )
	elif idx <= lA+lB+lC:
		return createFeatureVector(featureC,integral_img,1,3,1,3,(idx - 1) % (lA + lB) )
	else:
		return createFeatureVector(featureD,integral_img,2,2,2,2, (idx - 1) % (lA + lB + lC))


###########################################
###########################################
# Here there are two manipulatable index that can be modified to shorten the training time at the cose of less training data
# train_shorten_idx indicates how many traiining images we are going to use, for which we need to create a long feature vector for
# validation_shorten_idx is how many validation images we are going to use

# train_shorten_idx = 2000 # this is to short the training data a bit to save our time
# validation_shorten_idx = 50

train_shorten_idx = len(X_train) # use full train data
validation_shorten_idx = len(X_validation) # use full validation data
###########################################
###########################################



X_train_feature_vector = []
X_validation_feature_vector = []


start_time = time.time()
print("Start creating feature vector for train data")
for idx,img in enumerate(X_train[:train_shorten_idx]):
	integral_img = get_integral_img(img)
	X_train_feature_vector.append(joint_feature(integral_img))
train_time = time.time() - start_time
print("Time for creating feature vector of training data of size ", train_shorten_idx, "--- %s seconds ---" % (train_time))



start_time = time.time()
print("Start creating feature vector for validation data")
for idx,img in enumerate(X_validation[:validation_shorten_idx]):
	integral_img = get_integral_img(img)
	X_validation_feature_vector.append(joint_feature(integral_img))
validation_time = time.time() - start_time
print("Time for creating feature vector of training data of size ", validation_shorten_idx, "--- %s seconds ---" % (validation_time))


print()
print("---------------------AdaBoostClassifier---------------------")
print()



start_time = time.time()
abc.fit(X_train_feature_vector, y_train[:train_shorten_idx])
print("Time training AdaBoostClassifier: --- %s seconds ---" % (time.time() - start_time))
print("accuracy on the validation data: ", abc.score(X_validation_feature_vector,y_validation[:validation_shorten_idx]))







rows, cols = (len(abc.estimators_), 4)
all_classifiers = [[0 for i in range(cols)] for j in range(rows)]

for i in range(len(abc.estimators_)):
    print(f"Tree {i}: ")
    print(f"Weight: {abc.estimator_weights_[i]}")
    print(tree.export_text(abc.estimators_[i]))
    classifier_info_array = tree.export_text(abc.estimators_[i]).split()

    if classifier_info_array[2] != "<=":
    	print("Error!")
    all_classifiers[i][0] = int(classifier_info_array[1][8:])
    all_classifiers[i][1] = float(classifier_info_array[3])
    all_classifiers[i][2] = int(classifier_info_array[7])
    all_classifiers[i][3] = float(abc.estimator_weights_[i])


#all_classifier is of the form [feature number, boundary, class, weight]
# which means that feature number <= boundary then class with weight.

# print(len(test_face)) # = 472
# print(len(test_nonface)) # = 23573

def testOut(integral_img,all_classifiers):
	res = 0
	for item in all_classifiers:
		if get_feature(integral_img,item[0]+1) <= item[1]:
			res += item[2] * item[3]
		else:
			res += (-item[2] * item[3])
	return 1 if res > 0 else -1


print("Start to classify test data")
start_time = time.time()
answerKey = []
for idx,img in enumerate(test_face + test_nonface):
	integral_img = get_integral_img(img)
	res = testOut(integral_img,all_classifiers)
	answerKey.append(res)
	if idx%1000 == 0:
		print(idx," pictures have been classfied.")
test_time = (time.time() - start_time)
print("time to classify test: --- %s seconds ---" % test_time)
print("ABC: average classify time per picture: ", test_time / len(systemKey))



print("ABC: the accuracy score is:", accuracy_score(answerKey, systemKey))


from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cm = confusion_matrix(systemKey, answerKey, labels=[-1, 1])
print(cm)

df_cm = pd.DataFrame(cm, index = [-1, 1],
                  columns = [-1, 1])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


plt.savefig('abc_confusion_matrix.png')


print()
print("---------------------RandomForestClassifier---------------------")
print()




########### RFC ###########
start_time = time.time()
rfc.fit(X_train_feature_vector, y_train[:train_shorten_idx])
print("Time training RandomForestClassifier: --- %s seconds ---" % (time.time() - start_time))

print("accuracy on the validation data: ", rfc.score(X_validation_feature_vector,y_validation[:validation_shorten_idx]))


rows, cols = (len(rfc.estimators_), 3)
all_classifiers = [[0 for i in range(cols)] for j in range(rows)]

for i in range(len(rfc.estimators_)):
    print(f"Tree {i}: ")
    print(tree.export_text(rfc.estimators_[i]))
    classifier_info_array = tree.export_text(rfc.estimators_[i]).split()

    if classifier_info_array[2] != "<=": # meaning there is not classification at all
    	all_classifiers[i][0] = 1 #take any feature
    	all_classifiers[i][1] = float('inf')
    	all_classifiers[i][2] = float(classifier_info_array[2])
    else:
	    all_classifiers[i][0] = int(classifier_info_array[1][8:])
	    all_classifiers[i][1] = float(classifier_info_array[3])
	    all_classifiers[i][2] = float(classifier_info_array[7])
print(all_classifiers)

def testOut2(integral_img,all_classifiers):
	n = len(all_classifiers)
	res = 0
	for item in all_classifiers:
		if get_feature(integral_img,item[0]+1) <= item[1]:
			res += item[2]
		else:
			res += (item[2] + 1) % 2 # in the case of RFC, the classes are 0,1
	return 1 if (res/n > 0.5) else -1

print("Start to classify test data")
start_time = time.time()
answerKey = []
for idx,img in enumerate(test_face + test_nonface):
	integral_img = get_integral_img(img)
	res = testOut2(integral_img,all_classifiers)
	answerKey.append(res)
	if idx%1000 == 0:
		print(idx," pictures have been classfied.")
test_time = (time.time() - start_time)
print("RFC: time to classify test: --- %s seconds ---" % test_time)
print("RFC: average classify time per picture: ", test_time / len(systemKey))
print("RFC: the accuracy score is:", accuracy_score(answerKey, systemKey))



from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cm = confusion_matrix(systemKey, answerKey, labels=[-1, 1])
print(cm)

df_cm = pd.DataFrame(cm, index = [-1, 1],
                  columns = [-1, 1])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


plt.savefig('rfc_confusion_matrix.png')