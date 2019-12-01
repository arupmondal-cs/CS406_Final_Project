# CS406_Final_Project (SPAM FILTER)
**Unstructured Information Processing Final Project:**

In recent times, unwanted commercial bulk emails called spam has become a huge problem on theinternet.   Spam  prevents  the  user  from  making  full  and  good  use  of  time,  storage  capacity  andnetwork bandwidth. 

Machine learning methods of recent are being used to successfully detect and filter spam emails. Majority of the email spam filtering methods uses text categorization approaches. Consequently, spam filters perform poorly and cannot efficiently prevent spam mails from getting to the inbox of the users. In this project mainly we focused the Random Forests (RF) algorithm to extract important features from emails, and classify the emails into either spam or non-spam. In this review we focus some of the most popular machine learning methods that have been applied to spam detection.

 1. K-nearest neighbours (kNN) classifier
 2. Naive Bayes (NB) classifier 
   * Naive Bayes
   * Multinomial Naive Bayes
   * Bernoulli Naive Bayes
3. Support Vector Machine (SVM) classifier
4. Decision Tree (DT) classifier
5. Random Forests (RF) classifier
6. Neural Network (NN) - Multi-Layer Perceptron (MLP) classifier


## **File Description:**

* **SpamFilter.py :** contains the actual python code of the _spam filter_.

* **spambase.data :** contains the _traing_ and _test_ data (in this work we consider 80% data is traing and 20% data is test data of the all data, and we choose it randomly).

* **input.txt :** is the _user given_ test input data for testing our model.

* **fet.txt :** contains the text features we considered in this proect to detect the spam or non-spam email. 


## **Results:**

In this project we use [Spambase Data Set](https://archive.ics.uci.edu/ml/datasets/Spambase) which is very old data set for train our model, and it's also used very limited number features. But if we incease the features set then it will also provide good result for the recent data set.

In this project we used 6 different technique to train and test our model, out of all the technique **Random Forest** classifier gives highest accuracy, i.e., 95.87%. You can find [results](https://github.com/arupmondal-cs/CS406_Final_Project/tree/master/Image) for the all classifier.

So, the example result for the Random Forest:

![Confusion Matrix for Random Forest](https://github.com/arupmondal-cs/CS406_Final_Project/blob/master/Image/RF.png)
