# 0种机器学习算法（附Python代码）

# 转载roger_royer 最后发布于2018-01-15 11:20:34 阅读数 11432  收藏
1. sklearn python API

#%% LinearRegression
from sklearn.linear_model import LinearRegression         # 线性回归 #
module = LinearRegression()
module.fit(x, y)
module.score(x, y)
module.predict(test)


#%% LogisticRegression
from sklearn.linear_model import LogisticRegression         # 逻辑回归 #
module = LogisticRegression()
module.fit(x, y)
module.score(x, y)
module.predict(test)





#%% KNN
from sklearn.neighbors import KNeighborsClassifier     #K近邻#
from sklearn.neighbors import KNeighborsRegressor
module = KNeighborsClassifier(n_neighbors=6)
module.fit(x, y)
predicted = module.predict(test)
predicted = module.predict_proba(test)



#%% SVM
from sklearn import svm                                #支持向量机#
module = svm.SVC()
module.fit(x, y)
module.score(x, y)
module.predict(test)
module.predict_proba(test)


#%% naive_bayes
from sklearn.naive_bayes import GaussianNB            #朴素贝叶斯分类器#
module = GaussianNB()
module.fit(x, y)
predicted = module.predict(test)




#%% DecisionTree
from sklearn import tree                              #决策树分类器#
module = tree.DecisionTreeClassifier(criterion='gini')
module.fit(x, y)
module.score(x, y)
module.predict(test)





#%% K-Means
from sklearn.cluster import KMeans                    #kmeans聚类#
module = KMeans(n_clusters=3, random_state=0)
module.fit(x, y)
module.predict(test)




#%% RandomForest
from sklearn.ensemble import RandomForestClassifier  #随机森林#
from sklearn.ensemble import RandomForestRegressor
module = RandomForestClassifier()
module.fit(x, y)
module.predict(test)





#%% GBDT
from sklearn.ensemble import GradientBoostingClassifier      #Gradient Boosting 和 AdaBoost算法#
from sklearn.ensemble import GradientBoostingRegressor
module = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
module.fit(x, y)
module.predict(test)





#%% PCA
from sklearn.decomposition import PCA              #PCA特征降维#
train_reduced = PCA.fit_transform(train)
test_reduced = PCA.transform(test)
