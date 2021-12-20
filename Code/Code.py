import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import graphviz
from sklearn.tree import export_graphviz
import matplotlib. pyplot as plt
from sklearn import tree



warnings.filterwarnings('ignore')

## Data Fetch
od.download(r'https://www.kaggle.com/benfattori/league-of-legends-diamond-games-first-15-minutes')


{"username":"caishuting","key":"1b669a73659c98b47cba906aa8d8fc5f"}

# read in data
df = pd.read_csv("MatchTimelinesFirst15.csv")

#distribution
plt.figure(figsize=(20,18))
for i, col in enumerate(df):
    plt.subplot(5,4,i+1); sns.distplot(df[col])
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

#
df.drop(labels = ["Unnamed: 0", "matchId", "blueDragonKills", "redDragonKills"]
        , axis = 1, inplace=True)
pd.set_option("display.max_columns", len(df.columns))
#print(df.head(5))
#print(df.columns)
#print(df.dtypes)
df.info()   # no missing value
#print(df.describe())

# data and target
df_data = df.drop(labels="blue_win", axis=1)
target = df['blue_win']

# distribution of the target
plt.hist(target)
plt.title('Distribution of the target')
plt.show()


 # outliners
plt.figure(figsize=(15,18))
for i, col in enumerate(df_data):
    plt.subplot(5,3,i+1); sns.boxplot(df_data[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

redMinionsKilled_min = df.loc[df['redMinionsKilled']==14]
df1 = df.drop(labels=26207,axis=0,inplace=False)
df1.info()



plt.figure(figsize=(15,18))
for i, col in enumerate(df_data):
    plt.subplot(5,3,i+1); sns.boxplot(df_data[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()





# normolization (no difference on model)
scaler = MinMaxScaler()
scaler = scaler.fit(df_data)
df_data = scaler.transform(df_data)
#x = pd.DataFrame(x)
# x.columns = ['blue_win', 'blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled',
#        'blueAvgLevel', 'redGold', 'redMinionsKilled', 'redJungleMinionsKilled',
#        'redAvgLevel', 'blueChampKills', 'blueHeraldKills',
#        'blueTowersDestroyed', 'redChampKills', 'redHeraldKills',
#        'redTowersDestroyed']



##pca
# pca_f = PCA(n_components=0.95)
# pca_f = pca_f.fit(df_data)
# df_f = pca_f.transform(df_data)
# print(df_f.shape)
# print(df_data.shape)

xtrain,xtest,ytrain,ytest = train_test_split(df_data,target,test_size=0.3)


### Decision Tree
clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, min_samples_split=40)
clf.fit(xtrain, ytrain)
print("score on test: " + str(clf.score(xtest, ytest)))
print("score on train: " + str(clf.score(xtrain, ytrain)))


test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(criterion='entropy'
                                  ,random_state=20
                                  ,splitter='random'
                                  ,max_depth=i+1
                               #  ,min_samples_leaf=10
                                )
    clf = clf.fit(xtrain,ytrain)
    score = clf.score(xtest,ytest)
    test.append(score)
plt.plot(range(1,11),test,color = "red",label = "max_depth")
plt.title("Parameter Curve for Max_depth")
plt.legend()
plt.show()

# decision tree
from pydotplus import graph_from_dot_data
dot_data = tree.export_graphviz(clf
                                ,feature_names=['blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled',
    'blueAvgLevel', 'redGold', 'redMinionsKilled', 'redJungleMinionsKilled',
      'redAvgLevel', 'blueChampKills', 'blueHeraldKills',
   'blueTowersDestroyed', 'redChampKills', 'redHeraldKills',
        'redTowersDestroyed']
                                ,class_names=['blue win','red win']
                                ,filled = True
                                ,rounded = True
                               )
graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy.pdf")
#webbrowser.open_new(r'decision_tree_entropy.pdf')





###Logistic regression
Xtrain,Xtest,Ytrain,Ytest = train_test_split(df_data,target,test_size=0.3)
lr = LR(penalty='l2',solver='liblinear', C=0.9, max_iter=5)
lr=lr.fit(Xtrain,Ytrain)
print(accuracy_score(lr.predict(Xtrain),Ytrain))
print(accuracy_score(lr.predict(Xtest),Ytest))

X_embedded = SelectFromModel(lr, norm_order=2).fit_transform(df_data,target)
print(cross_val_score(lr,X_embedded,target,cv=10).mean())


y_pred_score = clf.predict_proba(xtest)
y_pred = clf.predict(xtest)
print(y_pred_score)
print(y_pred)



l1=[]
l2=[]
l1test = []
l2test = []
for i in np.linspace(0.05,2,19):
    lrl1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=500)
    lrl2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=500)

    lrl1 = lrl1.fit(xtrain, ytrain)
    l1.append(accuracy_score(lrl1.predict(xtrain),ytrain))
    l1test.append(accuracy_score(lrl1.predict(xtest),ytest))

    lrl2 = lrl2.fit(xtrain, ytrain)
    l2.append(accuracy_score(lrl2.predict(xtrain), ytrain))
    l2test.append(accuracy_score(lrl2.predict(xtest), ytest))

graph = [l1,l2,l1test,l2test]
color = ['green','black','lightgreen','gray']
label = ['L1','L2','L1test','L2test']

plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,1,19),graph[i],color[i],label=label[i])

plt.legend(loc=4 ) 
plt.title('Parameter of LR')
plt.show()