import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import os
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD


os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# sl:satisfaction_level---False:MinMaxScaler;True:StandardScaler
# le:last_evaluation---False:MinMaxScaler;True:StandardScaler
# npr:number_project---False:MinMaxScaler;True:StandardScaler
# amh:average_monthly_hours---False:MinMaxScaler;True:StandardScaler
# wa:work_accident---False:MinMaxScaler;True:StandardScaler
# tsc:time_spend_company---False:MinMaxScaler;True:StandardScaler
# pl5:promotion_last_5years---False:MinMaxScaler;True:StandardScaler
# dp:department---False:LabelEncoder;True:OneHotEncoder
# slr:salary---False:LabelEncoder;True:OneHotEncoder
def hr_preprocessing(sl=False,le=False,npr=False,amh=False,wa=False,tsc=False,pl5=False,dp=False,slr=False,low_d=False,ld_n=1):
    df = pd.read_csv('./HR.csv')
    #1 清洗数据
    df = df.dropna(subset=['satisfaction_level','last_evaluation'])
    df = df[df['satisfaction_level'] <= 1][df['salary'] != 'nme']
    # 2 得到标注
    label = df['left']
    df = df.drop('left', axis=1)
    #3 特征选择
    #4 特征处理
    scaler_list = [sl,le,npr,amh,tsc,wa,pl5]
    column_list = ['satisfaction_level','last_evaluation','number_project','average_monthly_hours',\
                   'time_spend_company','Work_accident','promotion_last_5years']
    for i in range(len(scaler_list)):
        if not scaler_list[i]:
            df[column_list[i]] = \
            MinMaxScaler().fit_transform(df[column_list[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df[column_list[i]] = \
            StandardScaler().fit_transform(df[column_list[i]].values.reshape(-1,1)).reshape(1,-1)[0]
    scaler_list = [dp,slr]
    column_list = ['department','salary']
    for i in range(len(column_list)):
        if not scaler_list[i]:
            if column_list[i] == 'salary':
                df[column_list[i]] = [map_salary(s) for s in df[column_list[i]].values]
            else:
                df[column_list[i]] = LabelEncoder().fit_transform(df[column_list[i]])
            df[column_list[i]] = \
                MinMaxScaler().fit_transform(df[column_list[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df = pd.get_dummies(df,columns=[column_list[i]])

    if low_d:
        return PCA(n_components=ld_n).fit_transform(df.values),label

    return df,label

d = dict([('low',0),('medium',1),('high',2)])
def map_salary(s):
    return d.get(s,0)

def hr_modeling(features,label):
    f_v = features.values
    f_names = features.columns.values
    #print(f_names)
    l_v = label.values
    X_tt,X_validation,Y_tt,Y_validation = train_test_split(f_v,l_v,test_size=0.2)
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size=0.25)
    print(len(X_train),len(X_validation),len(X_test))

    mdl = Sequential()
    mdl.add(Dense(50,input_dim=len(f_v[0])))
    mdl.add(Activation('sigmoid'))
    mdl.add(Dense(2))
    mdl.add(Activation('softmax'))
    sgd = SGD(lr=0.1)
    mdl.compile(loss='mean_squared_error',optimizer='adam')
    mdl.fit(X_train,np.array([[0,1] if i==1 else [1,0] for i in Y_train]),nb_epoch=1000,batch_size=8999)
    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
    for i in range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        Y_pred = mdl.predict_classes(X_part)
        print(i)
        print('NN', '-ACC', accuracy_score(Y_part, Y_pred))
        print('NN', '-REC', recall_score(Y_part, Y_pred))
        print('NN', '-F1', f1_score(Y_part, Y_pred))

    models = []
    models.append(('KNN',KNeighborsClassifier(n_neighbors=3)))
    models.append(('GaussianNB',GaussianNB()))
    models.append(('BernoulliNB',BernoulliNB()))
    models.append(('DecisionTreeGini',DecisionTreeClassifier()))
    models.append(('DecisionTreeEntropy',DecisionTreeClassifier(criterion='entropy')))
    models.append(("SVM Classifier", SVC(C=1000)))
    models.append(('RandomForest',RandomForestClassifier(max_features=None)))
    models.append(('AdaBoostClassifier',AdaBoostClassifier()))
    models.append(('LogisticRegression',LogisticRegression()))
    for clf_name,clf in models:
        clf.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name,'-ACC',accuracy_score(Y_part,Y_pred))
            print(clf_name, '-REC', recall_score(Y_part, Y_pred))
            print(clf_name, '-F1', f1_score(Y_part, Y_pred))
            # dot_data = export_graphviz(clf,out_file=None,
            #                            feature_names=f_names,
            #                            class_names=["NL","L"],
            #                            filled=True,
            #                            rounded=True,
            #                            special_characters=True)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf('dt_tree.pdf')

    # joblib.dump(knn_clf,'knn_clf')
    # joblib.load('knn_clf')
    # Y_pred = knn_clf.predict(X_test)
    # print('test2')
    # print('ACC:', accuracy_score(Y_test, Y_pred))
    # print('REC', recall_score(Y_test, Y_pred))
    # print('F-score', f1_score(Y_test, Y_pred))

def regr_test(features,labels):
    print('X',features)
    print('Y',labels)
    # regr = Ridge(alpha=0.1)
    regr = LinearRegression()
    # regr = Lasso(alpha=0.0001)
    regr.fit(features.values, labels.values)
    Y_pred = regr.predict(features.values)
    print('coef', regr.coef_)
    from sklearn.metrics import mean_squared_error
    print('mse', mean_squared_error(Y_pred, labels.values))

def main():
    features, label = hr_preprocessing()
    regr_test(features[['number_project','average_monthly_hours']],features[['last_evaluation']])
    #hr_modeling(features,label)

if __name__ == '__main__':
    pd.set_option('display.max_columns',1000)
    main()

 