# Face-Detection-Using-Viola-Jones-Algo

In this exercise, we will implement the Viola-Jones algorithm for detecting faces. For training and testing, we will use the CBCL dataset which you can download from http://cbcl. mit.edu/software-datasets/FaceData2.html. If you unzip the file you will find two files faces.train.tar.gz and faces.test.tar.gz which contain pictures of faces and non-faces for training and testing respectively. The pictures are in Portable GrayMap (.pgm) format and can be opened using the cv2.imread function from the library cv2. Split the pictures in faces.train.tar.gz further into training and validation sets.

• For all pictures in the training and validation sets, compute the four types of Haar-like features shown in Figure 1 of the paper of Viola and Jones (https://www.cs.cmu.edu/ ~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf). For efficiency you should use the integral image data structure.

• Train an AdaBoost classifier (sklearn.ensemble.AdaBoostClassifier) with decision stumps as the base estimator (this the default) and the algorithm parameter set to ‘SAMME’1, on the training data with just enough estimators so that you obtain over 90% accuracy on the validation set (about 10 estimators should suffice).
  
Using the above information, to classify all pictures in the test set. For pictures in the test set, you should only compute the necessary features. Report the average time required per image including the time require for computing the necessary features.
• Report the confusion matrix (see https://en.wikipedia.org/wiki/Confusion_matrix).
• Replace the AdaBoost classifier by a Random Forest Classifier2 and do a quantitative comparison of the the two classifiers with respect to the time required for training them and their speeds at test time (including the time for computing necessary features).
