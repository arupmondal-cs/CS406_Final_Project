import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
from sklearn.naive_bayes import GaussianNB # Import Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB # Import Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB # Import Bernoulli Naive Bayes Classifier
from sklearn.neighbors import KNeighborsClassifier # Import KNN Classifier
from sklearn.svm import SVC # Import SVM Classifier
from sklearn.neural_network import MLPClassifier # Import MLP Classifier


'''
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
'''


#-----------------------------------------------------------------------------#
                ##########  DATA READ  ##########     
#-----------------------------------------------------------------------------#
    
def DataRead():
    # load dataset
    col_num=['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6', 'FT7', 'FT8', 'FT9', 'FT10', 'FT11', 'FT12', 'FT13', 'FT14', 'FT15', 'FT16', 'FT17', 'FT18', 'FT19', 'FT20', 'FT21', 'FT22', 'FT23', 'FT24', 'FT25', 'FT26', 'FT27', 'FT28', 'FT29', 'FT30', 'FT31', 'FT32', 'FT33', 'FT34', 'FT35', 'FT36', 'FT37', 'FT38', 'FT39', 'FT40', 'FT41', 'FT42', 'FT43', 'FT44', 'FT45', 'FT46', 'FT47', 'FT48', 'FT49', 'FT50', 'FT51', 'FT52', 'FT53', 'FT54', 'FT55', 'FT56', 'FT57', 'label']
    data = pd.read_csv("spambase.data", header=None, names=col_num)

    data.head()
    
    #split dataset in features and target variable
    feature_cols = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6', 'FT7', 'FT8', 'FT9', 'FT10', 'FT11', 'FT12', 'FT13', 'FT14', 'FT15', 'FT16', 'FT17', 'FT18', 'FT19', 'FT20', 'FT21', 'FT22', 'FT23', 'FT24', 'FT25', 'FT26', 'FT27', 'FT28', 'FT29', 'FT30', 'FT31', 'FT32', 'FT33', 'FT34', 'FT35', 'FT36', 'FT37', 'FT38', 'FT39', 'FT40', 'FT41', 'FT42', 'FT43', 'FT44', 'FT45', 'FT46', 'FT47', 'FT48', 'FT49', 'FT50', 'FT51', 'FT52', 'FT53', 'FT54', 'FT55', 'FT56', 'FT57']
    X = data[feature_cols] # Features
    y = data.label # Target variable
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50) # 80% training and 20% test
    return feature_cols, X_train, X_test, y_train, y_test

def Test_DataRead(fn):
    d=open("input.txt", "r")
    data=[re.sub(r'[\n]', '', i) for i in d]  

    email=""
    for i in data: email+=" "+i

    spdata=re.split(' |\n', email.strip())


    f=open("fet.txt", "r")
    fet=[re.sub(r'[\n]', '', i) for i in f]  

    FetCo=[0 for i in range(57)]

    for i in range(len(fet)): FetCo[i]+=FetCo[i]+email.count(fet[i])

    count=0
    length=0
    templen=0
    for s in spdata:
        length+=len(''.join([c for c in s if c.isupper()]))
        count+=1
    
        if len(''.join([c for c in s if c.isupper()]))>templen:
            templen=len(''.join([c for c in s if c.isupper()]))

    if length>0:
        FetCo[54]=count/length
    else:
        FetCo[54]=0
        
    FetCo[55]=templen
    FetCo[56]=length
    
    return FetCo

#-----------------------------------------------------------------------------#
                ##########  CONFUSION MATRIX COMPUTING  ##########     
#-----------------------------------------------------------------------------#

# Confusion Matrix to inspect the performance of the classification model 
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',  cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 16)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


#-----------------------------------------------------------------------------#
                   ##########  NAIVE BAYES  ##########     
#-----------------------------------------------------------------------------#

def NaiveBaise_Train(X_train, y_train):

    # Create Decision Tree classifer object
    clf1 = GaussianNB()

    # Train Decision Tree Classifer
    clf1 = clf1.fit(X_train,y_train)

    
    ##########--------Multinomial Naive Bayes Classifier--------##########
    
    clf2 = MultinomialNB()
    # Train Decision Tree Classifer
    clf2 = clf2.fit(X_train,y_train)

    
    ##########--------Multinomial Naive Bayes Classifier--------##########
    
    clf3 = BernoulliNB()
    # Train Decision Tree Classifer
    clf3 = clf3.fit(X_train,y_train)

    return clf1, clf2, clf3


def NaiveBaise_Test(clf1, clf2, clf3, X_test, y_test):
    #Predict the response for test dataset
    y_pred = clf1.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Naive Bayes Classifier:",metrics.accuracy_score(y_test, y_pred))

    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of Naive Bayes Classifier')
    
    
    ##########--------Multinomial Naive Bayes Classifier--------##########
    
    #Predict the response for test dataset
    y_pred = clf2.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Multinomial Naive Bayes Classifier:",metrics.accuracy_score(y_test, y_pred))

    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of Multinomial Naive Bayes Classifier')
    
    
    ##########--------Multinomial Naive Bayes Classifier--------##########
    
    #Predict the response for test dataset
    y_pred = clf3.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Bernoulli Naive Bayes Classifier:",metrics.accuracy_score(y_test, y_pred))

    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of Bernoulli Naive Bayes Classifier')
    
    
#-----------------------------------------------------------------------------#
                   ##########  DECISION TREE  ##########     
#-----------------------------------------------------------------------------#
                
def DecisionTree_Train(feature_cols, X_train, y_train):
    
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    return clf

    
def DecisionTree_Test(clf, feature_cols, X_test, y_test):
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Decision Tree:",metrics.accuracy_score(y_test, y_pred))
    
    
    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of Decision Tree')
    
    

#-----------------------------------------------------------------------------#
                    ##########  RANDOM FOREST  ##########     
#-----------------------------------------------------------------------------#
    
def RandomForest_Train(feature_cols, X_train, y_train):
    
    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100, random_state=50, max_features = 'sqrt')
    model = model.fit(X_train,y_train)
    
    return model

def RandomForest_Test(model, feature_cols, X_test, y_test):
    y_pred = model.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Random Forest:",metrics.accuracy_score(y_test, y_pred))

    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of Random Forest')
    
    
#-----------------------------------------------------------------------------#
           ##########  KNN Classifier  ##########     
#-----------------------------------------------------------------------------#
                
def KNN_Train(X_train, y_train):
    
    # Create Decision Tree classifer object
    clf = KNeighborsClassifier(n_neighbors=7)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    
    return clf
    
def KNN_Test(clf, X_test, y_test):

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of KNN Classifier:",metrics.accuracy_score(y_test, y_pred))
    
    
    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of KNN Classifier')
    
 
#-----------------------------------------------------------------------------#
           ##########  SVM Classifier  ##########     
#-----------------------------------------------------------------------------#
                
def SVM_Train(X_train, y_train):
    
    # Create Decision Tree classifer object
    clf = SVC(C=62, gamma=0.00057)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    
    return clf

def SVM_Test(clf, X_test, y_test):
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of SVM Classifier:",metrics.accuracy_score(y_test, y_pred))
    
    
    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of SVM Classifier')

 
#-----------------------------------------------------------------------------#
           ##########  MLP Classifier  ##########     
#-----------------------------------------------------------------------------#
           
           
def MLP_Train(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(13, 22), random_state=10, max_iter=24, warm_start=True)
    
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    
    return clf
    
def MLP_Test(clf, X_test, y_test):
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of MLP Classifier:",metrics.accuracy_score(y_test, y_pred))
    
    
    # Confusion Matrix for Decision Tree
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes = ['Spam', 'Not-Spam'], title = 'Spam Filter Confusion Matrix of MLP Classifier')
    

#-----------------------------------------------------------------------------#
           ##########  SPAM FILTER  ##########     
#-----------------------------------------------------------------------------#    

def SpamFilter(opt, cls, feature_cols, X_train, X_test, y_train, y_test, GX):
    if opt==1:
        if cls==1:
            clf=DecisionTree_Train(feature_cols, X_train, y_train)
            DecisionTree_Test(clf, feature_cols, X_test, y_test)
            
        elif cls==2:
            model=RandomForest_Train(feature_cols, X_train, y_train)
            RandomForest_Test(model, feature_cols, X_test, y_test)
            
        elif cls==3:
            clf1, clf2, clf3=NaiveBaise_Train(X_train, y_train)
            NaiveBaise_Test(clf1, clf2, clf3, X_test, y_test)            
            
        elif cls==4:
            clf=SVM_Train(X_train, y_train)
            SVM_Test(clf, X_test, y_test)  
            
        elif cls==5:
            clf=KNN_Train(X_train, y_train)
            KNN_Test(clf, X_test, y_test)    
            
        elif cls==6:
            clf=MLP_Train(X_train, y_train)
            MLP_Test(clf, X_test, y_test)
            
    elif opt==2:
        if cls==1:
            clf=DecisionTree_Train(feature_cols, X_train, y_train)
            SHOW(clf.predict(GX))
            
        elif cls==2:
            model=RandomForest_Train(feature_cols, X_train, y_train)
            SHOW(model.predict(GX))
            
        elif cls==3:
            clf1, clf2, clf3=NaiveBaise_Train(X_train, y_train)
            SHOW(clf1.predict(GX))            
            
        elif cls==4:
            clf=SVM_Train(X_train, y_train)
            SHOW(clf.predict(GX))
            
        elif cls==5:
            clf=KNN_Train(X_train, y_train)
            SHOW(clf.predict(GX)) 
            
        elif cls==6:
            clf=MLP_Train(X_train, y_train)
            SHOW(clf.predict(GX)) 

def SHOW(clas):
    if clas==1:
        print("\nThe given data is Spam!")
    else:
        print("\nThe given data is Not Spam!")
    
    
def Choice_sel():
    print("\n1) Spam Filter for the given Data")
    print("2) Spam Filter for the User given Test Data")
    print("\n# Enter '0' for ending execution!")
    opt=int(input("Enter your choice: "))
    if opt==0:
        print("Successfully exit!")
        
    return opt

def Cls_sel():
    print("\n1) Decision Tree")
    print("2) Random Forest")
    print("3) Naive Bayes")
    print("4) Support Vector Machine")
    print("5) K-Nearest Neighbour")
    print("6) Multi-Layer Perceptron")
    cls=int(input("Enter your choice: "))

    return cls


#-----------------------------------------------------------------------------#
                ##########  MAIN FUNCTION  ##########     
#-----------------------------------------------------------------------------#
    
if __name__ == "__main__":
    print("\n------- Please wait, training data is processing -------\n")
    feature_cols, X_train, X_test, y_train, y_test = DataRead()
    GX=Test_DataRead("input.txt")
    GX=pd.DataFrame(np.array(GX).reshape(1,57), columns = feature_cols)
    
    opt=Choice_sel()
    while(opt):
        if opt not in [0,1,2]:
            print("Please enter right option!")
            opt=Choice_sel()
            continue        
        
        cls=Cls_sel()
        while(cls not in [1,2,3,4,5,6]):
            if cls not in [1,2,3,4,5,6]:
                print("Please enter right option!")
                cls=Cls_sel()
                continue
        
        SpamFilter(opt, cls, feature_cols, X_train, X_test, y_train, y_test, GX)
        print("\n ------------------ ************* ------------------\n")
        opt=Choice_sel()
