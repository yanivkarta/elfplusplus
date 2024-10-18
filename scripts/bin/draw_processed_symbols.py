''' draw processed_symbols.csv '''


import pandas as pd
import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
#itertools
import itertools
#for smote
from imblearn.over_sampling import SMOTE
#for adasyn 
from imblearn.over_sampling import ADASYN

from imblearn.over_sampling import RandomOverSampler
#classifiers :
#gaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier    
from sklearn.model_selection import train_test_split

#classification metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
#roc/auc
from sklearn.metrics import roc_curve, auc
#for cycle
from itertools import cycle

#mpi parallelization support:
from joblib import Parallel, delayed

classifiers =[
    ExtraTreesClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    SVC(),
    KNeighborsClassifier(),
    MLPClassifier() ]


def plot_class_regions_for_classifier_subplot(classifier, X_train, y_train, X_test, y_test, title, subplot, target_names=None, plot_decision_regions=True, ax=None):
    print ("[+]plot_class_regions_for_classifier_subplot\n")
    if ax is None:
        if subplot == 1:
            plt.subplot(2, 2, 1)
        else:
            plt.subplot(2, 2, 2)

        ax = plt.gca()

    if plot_decision_regions:
        x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdBu')
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)
    markers = ('s', 'x', 'o', '^', 'v')
    if target_names is not None:
        for i, target_name in enumerate(target_names):
            #plot class samples
            #avoid invalid index error 
            scatterX = X_train[y_train == i] if i < len(target_names) else X_train[y_train == 0]  # 
            if scatterX.shape[0] > 0: 
                ax.scatter(scatterX[:, 0], scatterX[:, 1], alpha=0.8, c=colors[i%len(colors)], marker=markers[i%len(markers)], label=target_name) 
            else:
                continue


                


    if (X_test is not None) and (y_test is not None):
        for i, target_name in enumerate(target_names):
            scatterX = X_test[X_test[y_test == i]] if i < len(target_names) else X_test[X_test[y_test == 0]] 
            if scatterX.shape[0] > 0:
                ax.scatter(scatterX[:, 0], scatterX[:, 1], alpha=0.8, c=colors[i%len(colors)], marker=markers[i%len(markers)], label=target_name)    
            else:
                continue


    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.axis('equal')
    ax.axhline(y=0, color='k')
    ax.axhline(y=1, color='k')
    ax.axvline(x=0, color='k')
    ax.axvline(x=1, color='k')

    return ax
def plot_confusion_matrix(cm, classes, ax, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    print ("[+]plot_confusion_matrix\n")
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):      
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="green" if cm[i, j] > thresh else "red") 

    ax.set_ylabel('True label')
    
def plot_area_under_curve(fpr, tpr, ax, title=None):
    print ("[+]plot_area_under_curve\n")
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=1, label='area under curve (AUC = %0.2f)' % (roc_auc))
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    sns.despine(ax=ax, offset=10)


def plot_decision_function(data, target, classifier, ax, title=None):
    print ("[+]plot_decision_function\n")
    step_size = 0.02
    x_min, x_max = data.min(axis=0), data.max(axis=0)
    x_min, x_max = x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1 
    #set y_min, y_max according to the target
    y_min, y_max = target.min( axis=0), target.max(axis=0)
    y_min, y_max = y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1
    ax.set_xlim(x_min.min(), x_max.max())
    ax.set_ylim(y_min, y_max) 

    #create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size)) 
 
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(data[:, 0], data[:, 1], alpha=0.8, c=target, edgecolor="k")
    if title:
        ax.set_title(title)



def plot_feature_importances(clf,ax,title=None):
    print ("[+]plot_feature_importances\n")
    feature_importance = 100.0 * (clf.feature_importances_ / clf.feature_importances_.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    ax.barh(pos, feature_importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(df.columns)[sorted_idx])
    ax.set_xlabel('Relative Importance')
    ax.set_title(title) 

def plot_feature_importances_by_name(feature_names, feature_importances, ax, title=None):
    print ("[+]plot_feature_importances\n")
    feature_names = np.array(feature_names)
    feature_importances = np.array(feature_importances)
    sorted_idx = np.argsort(feature_importances)
    ax.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)

def plot_auc_roc_curve(X_train, y_train, X_test, y_test, classifier, y_pred, target_column, ax, title=None):
    print ("[+]plot_auc_roc_curve\n")
    try:
        ax2 = ax.twinx()
        n_classes = len(np.unique(y_test))
        print ("[+]n_classes ",n_classes,"\n")
        # Compute ROC curve and ROC area for each class 
        #plotting roc_curve
        colors = cycle(['blue', 'red', 'green', 'yellow', 'orange'])
        for i, color in zip(range(n_classes), colors):
            
            
            
            fpr, tpr, thresholds =roc_curve(y_train, y_pred, pos_label=i)
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label='ROC curve of training class {0} (area = {1:0.2f})' 
                    ''.format(i, auc_score), color=color,linestyle='dashed')             
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=i)
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label='ROC curve of testing class {0} (area = {1:0.2f})' 
                    ''.format(i, auc_score), color=color) 
            
            
            

            ax2.plot(fpr, tpr, color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, auc(fpr, tpr)))
        #metrics.plot_roc_curve(clf, df_test, df_test[target_column], ax=ax2, name=classifier.__name__ ,)
#                                ax2.plot(fpr, tpr, color=color, lw=2,
#                                        label='ROC curve of class {0} (area = {1:0.2f})'
#                                        ''.format(i, metrics.auc(fpr, tpr)))
                            #metrics.plot_roc_curve(clf, df_test, df_test[target_column], ax=ax2, name=classifier.__name__ ,)
                            

            ax2.show_legend = False
            #ax2.plot([0, 1], [0, 1], 'k--', lw=2) 

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend(loc="lower right")
        ax2.set_title(title)

    except Exception as e:
        print ("[+]Exception in plot_auc_roc_curve ",e) 


def plot_resampling(X, y, sampler, ax, title=None):
    print ("[+]plot_resampling\n")
    X_res, y_res = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)


#read the csv file
df = pd.read_csv('processed_symbols.csv')
 

print ('rows:',df.shape[0],'columns:',df.shape[1],'\n')

#plot the dataframe


#set type as the label
labels = df[df.columns[-1]]


#set the features
features = df[df.columns[:-1]] 


print ('[+]splitting data into train and test, 50/50')

#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=0) 


print ('[+]applying random oversampling')
#apply random oversampling
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X_train, y_train)


#should iterate through the list of classifiers here:
# for classifier in classifiers:
for classifier in classifiers:
    print ('[+]applying classifier :', classifier.__name__, '\n')
    print ('[+]fitting classifier', classifier.__name__,  'rows:',X_ros.shape[0],'columns:',X_ros.shape[1] ,'labels:',np.unique(y_ros) ,'\n')
    clf = classifier.fit(X_train, y_train)
    print ('[+]predicting classifier', classifier.__name__, '\n')
    y_predict = clf.predict(X_test)
    #plot the dataframe, X_ros, y_ros
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 11))
    print(classification_report(y_test, y_predict))
    cm = confusion_matrix(y_test, y_predict, labels=clf.classes_)
    plot_confusion_matrix(cm, clf.classes_, ax=ax1, title="Random Oversampling Confusion Matrix") 
    #plot_auc_roc_curve(clf, X_test, y_test, ax=ax2, title="Random Oversampling Area Under Curve")
    plot_auc_roc_curve(X_train, y_train, X_test, y_test, clf, y_predict, target_column=0, ax=ax2, title=None)
    sns.set_style("darkgrid")
    ax1.set(xlabel='Predicted', ylabel='True', title=classifier.__name__ + " Confusion Matrix") 
    print ('accuracy:',accuracy_score(y_test, y_predict)) 
    classifier_name = str(clf.__class__.__name__)
    plt.savefig(classifier_name + '.png', dpi=300)
    #clear the figure
    plt.clf()
    plt.close('all')
    #save results to results.txt
    with open(classifier.__name__ +'results.txt', 'a') as f:
        f.write(classifier.__name__ + ' ' + str(accuracy_score(y_test, y_predict)) + '\n') 
        #write confusion matrix
        f.write('\n' + str(cm) + '\n')
        f.write('\n' + str(classification_report(y_test, y_predict)) + '\n')
        f.write('\n' + str(accuracy_score(y_test, y_predict)) + '\n')
        


#plot_area_under_curve(fpr, tpr, ax=ax2, title="Random Oversampling Area Under Curve") 

#plot extratree feature importance
#plot_feature_importances(clf, ax=ax2,title="ExtraTreeRandom Oversampling Feature Importance") 

#save classification report, confusion matrix,and accuracy to results.txt

 
