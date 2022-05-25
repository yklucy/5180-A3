import pandas as pd
from nltk.metrics import jaccard_distance
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# read the train data, dev data, test data
df_train_01 = pd.read_csv('./data_preprocessing/train_01.csv')
df_dev_01 = pd.read_csv('./data_preprocessing/dev_01.csv')
df_test_01 = pd.read_csv('./data_preprocessing/test_01.csv')

# ############################calculate the jaccard distance for train data##########################
jaccard_dis = []
#print(len(df_train_01)-1)

for i in range(0,(len(df_train_01)-1)):
    s1 = str(df_train_01['Sent_1'][i])
    s2 = str(df_train_01['Sent_2'][i])

    # transfer to TF Matrix
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # disjuction
    numerator = np.sum(np.min(vectors, axis=0))
    # conjuction
    denominator = np.sum(np.max(vectors, axis=0))
    # jaccard distance
    jaccard_dis.append(1.0 * numerator / denominator)
    #print(jaccard_dis)

# train data for model training
#axis=1, add jaccard_dis and Label 
train_data = pd.concat([df_train_01['Topic_Id'],pd.DataFrame(jaccard_dis),df_train_01['Label']],axis=1)
train_data.columns = ['Topic_Id','distance','Label']

if(train_data.isna().sum().sum()) > 0:
    #drop NaN values
    train_data = train_data.dropna()

# write the train data to csv file
train_data.to_csv("./data_preprocessing/train_data_alg2.csv",index=0)


# ############################calculate the jaccard distance for dev data##########################
jaccard_dis_dev = []
#print(len(df_dev_01)-1)

for i in range(0,(len(df_dev_01)-1)):
    s1 = str(df_dev_01['Sent_1'][i])
    s2 = str(df_dev_01['Sent_2'][i])
    # transfer to TF Matrix
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # disjuction
    numerator = np.sum(np.min(vectors, axis=0))
    # conjuction
    denominator = np.sum(np.max(vectors, axis=0))
    # jaccard distance
    jaccard_dis_dev.append(1.0 * numerator / denominator)
    #print(jaccard_dis_dev)

# dev data for model training
#axis=1, add jaccard_dis_dev and Label 
dev_data = pd.concat([df_dev_01['Topic_Id'],pd.DataFrame(jaccard_dis_dev),df_dev_01['Label']],axis=1)
dev_data.columns = ['Topic_Id','distance','Label']

if(dev_data.isna().sum().sum()) > 0:
    #drop NaN values
    dev_data = dev_data.dropna()

# write the dev data to csv file
dev_data.to_csv("./data_preprocessing/dev_data_alg2.csv",index=0)



# ############################calculate the jaccard distance for test data##########################
jaccard_dis_test = []
#print(len(df_test_01)-1)

for i in range(0,(len(df_test_01)-1)):
    s1 = str(df_test_01['Sent_1'][i])
    s2 = str(df_test_01['Sent_2'][i])
    # transfer to TF Matrix
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # disjuction
    numerator = np.sum(np.min(vectors, axis=0))
    # conjuction
    denominator = np.sum(np.max(vectors, axis=0))
    # jaccard distance
    jaccard_dis_test.append(1.0 * numerator / denominator)
    #print(jaccard_dis_test)

# test data for model training
#axis=1, add jaccard_dis_test and Label 
test_data = pd.concat([df_test_01['Topic_Id'],pd.DataFrame(jaccard_dis_test),df_test_01['Label']],axis=1)
test_data.columns = ['Topic_Id','distance','Label']

if(test_data.isna().sum().sum()) > 0:
    #drop NaN values
    test_data = test_data.dropna()

# write the test data to csv file
test_data.to_csv("./data_preprocessing/test_data_alg2.csv",index=0)



# ############################Algorithm 2: train the model and test ##########################
from sklearn import tree
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# train data_X, train_data_y
train_data_y = train_data['Label']
train_data_X = train_data.copy()
train_data_X.drop(['Label'],axis=1,inplace=True)


# dev_data_X, dev_data_y
dev_data_y = dev_data['Label']
dev_data_X = dev_data.copy()
dev_data_X.drop(['Label'],axis=1,inplace=True)


# test_data_X, test_data_y
test_data_y = test_data['Label']
test_data_X = test_data.copy()
test_data_X.drop(['Label'], axis=1, inplace=True)

# train the model with DecisionTreeClassifier
clf_tree = tree.DecisionTreeClassifier(criterion="entropy")
clf_tree = clf_tree.fit(train_data_X,train_data_y)
score_tree = clf_tree.score(dev_data_X, dev_data_y)
print("\n score_tree:",score_tree)
print("\n")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

dev_y_pred = clf_tree.predict(dev_data_X)
dev_y_true = dev_data_y
f1 = f1_score(dev_y_true,dev_y_pred)
print("f1_score:",f1)
print("\n")

print(classification_report(dev_y_true,dev_y_pred))

# test
test_y_pred = clf_tree.predict(test_data_X)
test_y_true = test_data_y
f1_test = f1_score(test_y_true,test_y_pred)
print("\n f1_score_test:",f1_test)
print("\n test:")
print(classification_report(test_y_true, test_y_pred))

'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(train_data_X, train_data_y)
clf.score(dev_data_X, dev_data_y)


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
clf_rf = clf_rf.fit(train_data_X,train_data_y)
score_rf = clf_rf.score(dev_data_X, dev_data_y)

'''

