# elfplusplus
discriminative and generative elf utils framework

![dist](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/dist.png?raw=true)


extract features from elf files,creates datasets from local linux distribution, monitor elf files , packages ,distributions,etc...


scripts - contains non optimal python scripts for building datasets from elf features . 

examples - elfpp examples : monitor,elf walk , dump, feature extractor...




![packages](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/packages.png?raw=true)



Scores of symbols by classes/labels/packages :

![sym_scores](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/sym_scores.png?raw=true)




sample results : 


![adaboost](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/AdaBoostClassifier.png?raw=true)


AdaBoostClassifier 0.9156735898308932

[[   636    135      0      0      0]
 [   345 368626      0    318      0]
 [     0    176   2770    553      0]
 [   267  44971      0 136252      0]
 [     0     44      0      0      0]]

              precision    recall  f1-score   support

           0       0.51      0.82      0.63       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.79      0.88      3499
           3       0.99      0.75      0.86    181490
           4       0.00      0.00      0.00        44

    accuracy                           0.92    555093
   macro avg       0.68      0.67      0.66    555093
weighted avg       0.92      0.92      0.91    555093



![decision](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/DecisionTreeClassifier.png?raw=true)

0.9156735898308932
DecisionTreeClassifier 0.9197431781701445

[[   639    131      0      1      0]
 [     2 368981      0    306      0]
 [     0    176   2818    505      0]
 [     0  43391      0 138099      0]
 [     0     38      0      0      6]]

              precision    recall  f1-score   support

           0       1.00      0.83      0.91       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.76      0.86    181490
           4       1.00      0.14      0.24        44

    accuracy                           0.92    555093
   macro avg       0.98      0.71      0.77    555093
weighted avg       0.93      0.92      0.92    555093



![extratree](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/ExtraTreesClassifier.png?raw=true)

0.9197431781701445
ExtraTreesClassifier 0.9197467811700022

[[   641    129      0      1      0]
 [     2 368981      0    306      0]
 [     0    176   2818    505      0]
 [     0  43391      0 138099      0]
 [     0     38      0      0      6]]

              precision    recall  f1-score   support

           0       1.00      0.83      0.91       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.76      0.86    181490
           4       1.00      0.14      0.24        44

    accuracy                           0.92    555093
   macro avg       0.98      0.71      0.77    555093
weighted avg       0.93      0.92      0.92    555093



![gaussianNB](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/GaussianNB.png?raw=true)

0.9197467811700022
GaussianNB 0.08223306725179384

[[   766      5      0      0      0]
 [103114  41465      0      9 224701]
 [   176      0   2818      0    505]
 [  4529     90      0    576 176295]
 [    22      0      0      0     22]]

              precision    recall  f1-score   support

           0       0.01      0.99      0.01       771
           1       1.00      0.11      0.20    369289
           2       1.00      0.81      0.89      3499
           3       0.98      0.00      0.01    181490
           4       0.00      0.50      0.00        44

    accuracy                           0.08    555093
   macro avg       0.60      0.48      0.22    555093
weighted avg       0.99      0.08      0.14    555093


0.08223306725179384





![gradientboost](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/GradientBoostingClassifier.png?raw=true)
GradientBoostingClassifier 0.9197341706705002

[[   634    136      0      1      0]
 [     1 368982      0    306      0]
 [     0    176   2818    505      0]
 [     0  43392      0 138098      0]
 [     0     38      0      0      6]]

              precision    recall  f1-score   support

           0       1.00      0.82      0.90       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.76      0.86    181490
           4       1.00      0.14      0.24        44

    accuracy                           0.92    555093
   macro avg       0.98      0.70      0.77    555093
weighted avg       0.93      0.92      0.92    555093


0.9197341706705002



![knn](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/KNeighborsClassifier.png?raw=true)

KNeighborsClassifier 0.9521665738894204

[[   626    142      0      3      0]
 [     4 364178      0   5107      0]
 [     0    176   2818    505      0]
 [     0  20577      0 160913      0]
 [     0     38      0      0      6]]

              precision    recall  f1-score   support

           0       0.99      0.81      0.89       771
           1       0.95      0.99      0.97    369289
           2       1.00      0.81      0.89      3499
           3       0.97      0.89      0.92    181490
           4       1.00      0.14      0.24        44

    accuracy                           0.95    555093
   macro avg       0.98      0.73      0.78    555093
weighted avg       0.95      0.95      0.95    555093


0.9521665738894204



![logistic](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/LogisticRegression.png?raw=true)

LogisticRegression 0.9153313048444135

[[     0    771      0      0      0]
 [     0 368969      0    320      0]
 [     0    176   2818    505      0]
 [     0  45183      0 136307      0]
 [     0     44      0      0      0]]

              precision    recall  f1-score   support

           0       0.00      0.00      0.00       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.75      0.86    181490
           4       0.00      0.00      0.00        44

    accuracy                           0.92    555093
   macro avg       0.58      0.51      0.54    555093
weighted avg       0.92      0.92      0.91    555093


0.9153313048444135


![mlpc](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/MLPClassifier.png?raw=true)


MLPClassifier 0.9153313048444135

[[     0    771      0      0      0]
 [     0 368969      0    320      0]
 [     0    176   2818    505      0]
 [     0  45183      0 136307      0]
 [     0     44      0      0      0]]

              precision    recall  f1-score   support

           0       0.00      0.00      0.00       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.75      0.86    181490
           4       0.00      0.00      0.00        44

    accuracy                           0.92    555093
   macro avg       0.58      0.51      0.54    555093
weighted avg       0.92      0.92      0.91    555093


0.9153313048444135



![randomforest](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/RandomForestClassifier.png?raw=true)
RandomForestClassifier 0.919748582669931

[[   641    129      0      1      0]
 [     2 368981      0    306      0]
 [     0    176   2818    505      0]
 [     0  43390      0 138100      0]
 [     0     38      0      0      6]]

              precision    recall  f1-score   support

           0       1.00      0.83      0.91       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.76      0.86    181490
           4       1.00      0.14      0.24        44

    accuracy                           0.92    555093
   macro avg       0.98      0.71      0.77    555093
weighted avg       0.93      0.92      0.92    555093


0.919748582669931



![svcc](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/SVC.png?raw=true)


SVC 0.9153421138439864

[[     0    771      0      0      0]
 [     0 368969      0    320      0]
 [     0    176   2818    505      0]
 [     0  45183      0 136307      0]
 [     0     38      0      0      6]]

              precision    recall  f1-score   support

           0       0.00      0.00      0.00       771
           1       0.89      1.00      0.94    369289
           2       1.00      0.81      0.89      3499
           3       0.99      0.75      0.86    181490
           4       1.00      0.14      0.24        44

    accuracy                           0.92    555093
   macro avg       0.78      0.54      0.59    555093
weighted avg       0.92      0.92      0.91    555093


0.9153421138439864



Examples : 

elfdump - dump sections, mark injectable/non-injectable objects/offsets  :

```
./examples/elfdump  ./examples/elfmonitor

File: ./examples/elfmonitor
Mode: 33277
Size: 416024
Magic: ELF
Class: 
Data: 
Version: 
Type: 3
Machine: 62
Version: 1
Entry: 25120
Flags: 0
Header size: 64
Program header offset: 64
Section header offset: 413912
Flags: 0
Header size: 64
Program header entry size: 56
Program header entry count: 13
Section header entry size: 64
Section header entry count: 33
Section header string table index: 32
...
...
[injectable section] name: GLIBCXX_3.4.22
[injectable section] type: 1
[injectable section] flags: 6
[injectable] addr: 25120
[injectable] offset: 25120
[injectable] size: 107217
[injectable] link: 0
[injectable] info: 0
[injectable] addralign: 16
[injectable] entsize: 0


[injectable section] name: deregister_tm_clones
[injectable section] type: 1
[injectable section] flags: 6
[injectable] addr: 132340
[injectable] offset: 132340
[injectable] size: 13
[injectable] link: 0
[injectable] info: 0
[injectable] addralign: 4
[injectable] entsize: 0

....



[non-injectable section] name: _ZL6sigint
[non-injectable section] type: 1
[non-injectable section] flags: 2
[non-injectable] addr: 138928
[non-injectable] offset: 138928
[non-injectable] size: 10188
[non-injectable] link: 0
[non-injectable] info: 0
[non-injectable] addralign: 4
[non-injectable] entsize: 0





```

feature extractor: 

```
./examples/feature_extractor /usr/local/lib /usr/bin

[+]link to string table: 0
[+]link name: __gmon_start__
[!] unhandled section type 8
[!]invalid section name
[+]link to string table: 0
[+]link name: __gmon_start__
[!]multiple symbol tables
[!] unhandled section type 7
[!] unhandled section type 7
[!]multiple symbol tables
[!] unhandled section type 1879048191
[!] unhandled section type 4
[!] unhandled section type 4
[!]multiple dynamic segments
[!] unhandled section type 8
[!]multiple symbol tables

...

file id: 10912200 cols:cols: 11319 rows: 11319
...



[+] creating feature matrix
Features Matrix
===================================================
||entropy: 1.4254e+07
||mean: 0.160631
||min: 0.01
||max: 1
||sum: 2.058e+07

[+]/usr/local/lib/python3.10/dist-packages/sqlalchemy/cyextension/util.cpython-310-x86_64-linux-gnu.so::PyUnicode_FromFormat


[+] global training matrix size: 13837,11319
 
...
[+] rowsum[479]4.83001
[+] rowsum[480]4.83031
[+] rowsum[481]483
[+] rowsum[482]4.83016
...
...
[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::PyNumber_Remainder


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::PyNumber_Index


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::PyCode_NewWithPosOnlyArgs


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::PyBytes_Type


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::PyBool_Type


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::PySet_Pop


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::.dynsym


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::.init


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::.eh_frame


[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/strptime.cpython-310-x86_64-linux-gnu.so::_init



[+]/usr/local/lib/python3.10/dist-packages/pandas/_libs/tslibs/conversion.cpython-310-x86_64-linux-gnu.so::PyErr_Occurred


Features Matrix
===================================================
||entropy: 21831.1
||mean: 0.175604
||min: 0.01
||max: 1
||sum: 33228.7
||divergence size: 0
===================================================

....

```
the feature extractor will attempt to extract features from the elf files in the lib folder and classify / test on the /bin folder

The results are then printed out and csv files of the model are saved.
 

This demonstrates unsupervised method to fit for various datasets infered from the data such as security,performance, connectivity,etc... 

