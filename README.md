# Machine-Learning
The skin dataset consist of (B, G, R) values where B represents blue , G represents green and R represents red. Each of the specified values takes on an integer value between 0 and 255 (RGB colour format 8 bit representation). The dataset  was collected by randomly sampling B,G,R values from face images of various age groups (young, middle, and old), race groups(white, black, and asian), and genders obtained from FERET database and PAL database. 
The main aim of collecting the dataset was to try and classify a given (B, G, R) value from the data as human skin or not human skin. All data points were pre-labelled, where a label of 1 represents the class human skin and a label of 2 represents the class not human skin. The dataset consists of 245057 points, out of which 50859 is the human skin sample and 194198 is the not human skin sample. The attributes of the dataset is the B, G, R values and the class labels.       Some sample of the data points:    
B  G  R  Class  
74  85  123  1  
72  83  121  1  
80  86  129  1  
172  172  126  2  

The dataset was split into training data, validation data and testing data. The training data is the randomly picked 60% of the original data (which is 147035 training data), validation data is the randomly picked 20% of the original data (which is 49011 of validation data) and the testing data is the randomly picked 20% of the original data 
(which is 49011 of testing data).

In trying to classify the data a naïve bayes classifier and a logistic regression classifier was used.     

Naïve Bayes 
The naive bayes classifier is often considered when we have labelled data in which we trying to predict discrete classes with probabilistic outputs. The dataset we have satisfies the above criteria. The naive bayes classifier is easy to program, fast to train and it can deal with uncertainty. Although the naive bayes classifier introduces the zero frequency problem, this can be tackled by using smoothing or assigning a probability density function such as a normal distribution or any other distribution suitable for the dataset.  
 
Logistic Regression
Logistic regression is considered a good classifier when we want to predict an outcome variable that is categorical form, given predictor variables that are continuous and categorical.    
