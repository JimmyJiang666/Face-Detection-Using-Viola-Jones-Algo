The "p3.py" needs to run in a directory where the "train" and "test" folders containing the training and testing images are present. Python libraries including cv2, numpy, glob, sklearn, seaborn, panda,matplotlib,etc., need to be installed.

We first used a AdaBoostClassifier for the image classification then used RandomForestClassifier with trees with depth at most 1. Both classifiers are allowed to have 10 maximum sub-estimators. The log of the execution can be found below.

Analysis:
The training time of the RandomForestClassifier is around 60s which is 6 times shorter than that of the AdaBoostClassifier. This is probably because of the ability of RFC to do the parallel computing and training of the trees due to their independency. The testing time for each classifier does not differ too much, with AdaBoostClassifier having 375s and RandomForestClassifier has 307s. This is due to the fact that both classifiers in our case have 10 trees which means there are 10 features of the feature vectors that are actually computed and used.

In terms of the testing performance, AdaBoostClassifier achieves an accuracy score of 0.89 and RandomForestClassifier achieves 0.87. From the confusion matrices, we can see that AdaBoostClassifier has a better performance on the face-like image than the RandomForestClassifier.


[LOG]:

Start creating feature vector for train data
Time for creating feature vector of training data of size  5581 --- 3591.7532000541687 seconds ---
Start creating feature vector for validation data
Time for creating feature vector of training data of size  1396 --- 905.1195330619812 seconds ---

---------------------AdaBoostClassifier---------------------

Time training AdaBoostClassifier: --- 360.3146197795868 seconds ---
accuracy on the validation data:  0.9505730659025788
Tree 0: 
Weight: 1.9257895637711815
|--- feature_22186 <= -1222.00
|   |--- class: 1
|--- feature_22186 >  -1222.00
|   |--- class: -1

Tree 1: 
Weight: 1.5780061953255071
|--- feature_7206 <= 100.50
|   |--- class: -1
|--- feature_7206 >  100.50
|   |--- class: 1

Tree 2: 
Weight: 1.1896438275478236
|--- feature_52925 <= -101.50
|   |--- class: 1
|--- feature_52925 >  -101.50
|   |--- class: -1

Tree 3: 
Weight: 1.0224427204420774
|--- feature_27873 <= 16.50
|   |--- class: -1
|--- feature_27873 >  16.50
|   |--- class: 1

Tree 4: 
Weight: 0.8608782457578306
|--- feature_12941 <= 14.50
|   |--- class: -1
|--- feature_12941 >  14.50
|   |--- class: -1

Tree 5: 
Weight: 1.127312336824253
|--- feature_12941 <= 13.50
|   |--- class: -1
|--- feature_12941 >  13.50
|   |--- class: 1

Tree 6: 
Weight: 0.7633271265178505
|--- feature_20115 <= 3.50
|   |--- class: -1
|--- feature_20115 >  3.50
|   |--- class: 1

Tree 7: 
Weight: 1.0003866563111061
|--- feature_7279 <= -69.50
|   |--- class: 1
|--- feature_7279 >  -69.50
|   |--- class: -1

Tree 8: 
Weight: 0.6692565609902248
|--- feature_9176 <= -17.50
|   |--- class: 1
|--- feature_9176 >  -17.50
|   |--- class: -1

Tree 9: 
Weight: 0.7387068955942196
|--- feature_28227 <= 107.00
|   |--- class: -1
|--- feature_28227 >  107.00
|   |--- class: -1

Start to classify test data
0  pictures have been classfied.
1000  pictures have been classfied.
2000  pictures have been classfied.
3000  pictures have been classfied.
4000  pictures have been classfied.
5000  pictures have been classfied.
6000  pictures have been classfied.
7000  pictures have been classfied.
8000  pictures have been classfied.
9000  pictures have been classfied.
10000  pictures have been classfied.
11000  pictures have been classfied.
12000  pictures have been classfied.
13000  pictures have been classfied.
14000  pictures have been classfied.
15000  pictures have been classfied.
16000  pictures have been classfied.
17000  pictures have been classfied.
18000  pictures have been classfied.
19000  pictures have been classfied.
20000  pictures have been classfied.
21000  pictures have been classfied.
22000  pictures have been classfied.
23000  pictures have been classfied.
24000  pictures have been classfied.
time to classify test: --- 375.6193959712982 seconds ---
ABC: average classify time per picture:  0.015621517819559085
ABC: the accuracy score is: 0.8933250155957579
[[21278  2295]
 [  270   202]]

---------------------RandomForestClassifier---------------------

Time training RandomForestClassifier: --- 63.69787907600403 seconds ---
accuracy on the validation data:  0.9111747851002865
Tree 0: 
|--- feature_5985 <= -65.50
|   |--- class: 1.0
|--- feature_5985 >  -65.50
|   |--- class: 0.0

Tree 1: 
|--- feature_25940 <= -62.50
|   |--- class: 1.0
|--- feature_25940 >  -62.50
|   |--- class: 0.0

Tree 2: 
|--- feature_24815 <= -402.50
|   |--- class: 1.0
|--- feature_24815 >  -402.50
|   |--- class: 0.0

Tree 3: 
|--- feature_7203 <= 57.50
|   |--- class: 0.0
|--- feature_7203 >  57.50
|   |--- class: 1.0

Tree 4: 
|--- feature_24150 <= -785.50
|   |--- class: 1.0
|--- feature_24150 >  -785.50
|   |--- class: 0.0

Tree 5: 
|--- feature_7203 <= 48.50
|   |--- class: 0.0
|--- feature_7203 >  48.50
|   |--- class: 1.0

Tree 6: 
|--- feature_25338 <= -92.50
|   |--- class: 1.0
|--- feature_25338 >  -92.50
|   |--- class: 0.0

Tree 7: 
|--- feature_22706 <= -957.50
|   |--- class: 1.0
|--- feature_22706 >  -957.50
|   |--- class: 0.0

Tree 8: 
|--- feature_7204 <= 67.50
|   |--- class: 0.0
|--- feature_7204 >  67.50
|   |--- class: 1.0

Tree 9: 
|--- feature_22595 <= -1283.00
|   |--- class: 1.0
|--- feature_22595 >  -1283.00
|   |--- class: 0.0

[[5985, -65.5, 1.0], [25940, -62.5, 1.0], [24815, -402.5, 1.0], [7203, 57.5, 0.0], [24150, -785.5, 1.0], [7203, 48.5, 0.0], [25338, -92.5, 1.0], [22706, -957.5, 1.0], [7204, 67.5, 0.0], [22595, -1283.0, 1.0]]
Start to classify test data
0  pictures have been classfied.
1000  pictures have been classfied.
2000  pictures have been classfied.
3000  pictures have been classfied.
4000  pictures have been classfied.
5000  pictures have been classfied.
6000  pictures have been classfied.
7000  pictures have been classfied.
8000  pictures have been classfied.
9000  pictures have been classfied.
10000  pictures have been classfied.
11000  pictures have been classfied.
12000  pictures have been classfied.
13000  pictures have been classfied.
14000  pictures have been classfied.
15000  pictures have been classfied.
16000  pictures have been classfied.
17000  pictures have been classfied.
18000  pictures have been classfied.
19000  pictures have been classfied.
20000  pictures have been classfied.
21000  pictures have been classfied.
22000  pictures have been classfied.
23000  pictures have been classfied.
24000  pictures have been classfied.
RFC: time to classify test: --- 307.94624304771423 seconds ---
RFC: average classify time per picture:  0.012807080184974598
RFC: the accuracy score is: 0.8735703888542317
[[20869  2704]
 [  336   136]]