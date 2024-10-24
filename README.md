# elfplusplus
discriminative and generative elf utils framework

![dist](https://github.com/yanivkarta/elfplusplus/blob/main/scripts/dist.png?raw=true)


extract features from elf files,creates datasets from local linux distribution, monitor elf files , packages ,distributions,etc...


scripts - contains non optimal python scripts for building datasets from elf features . 

examples - elfpp 

examples : monitor,elf walk , dump, feature extractor...




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

elfwalk:

```   
./examples/elfwalk /usr/local/lib
```
sample output:

```
File: /usr/local/bin/grpc_php_plugin
Magic: ELF
Class: 
Data: 
Version: 
OS/ABI: 
ABI Version: 
Type: 3
Machine: 3e
Version: 1
Entry point address: 0x7f480
Start of program headers: 0x40
Start of section headers: 0x2549a0
[+]Program headers: 14
[+]Section headers: 35
[+]Section header string table index: 34
[+][+]Program Header: 0
[+][+]Type: 6
[+][+]Offset: 40
[+][+]Virtual address: 40
[+][+]Physical address: 40
[+][+]File offset: 64
[+][+]Memory size: 784
[+][+]Flags: 4
[+][+]Alignment: 8
[+][+]Program Header: 1
[+][+]Type: 3
[+][+]Offset: 350
[+][+]Virtual address: 350
[+][+]Physical address: 350
[+][+]File offset: 848
[+][+]Memory size: 28
[+][+]Flags: 4
[+][+]Alignment: 1
[+][+]Program Header: 2
[+][+]Type: 1
[+][+]Offset: 0
[+][+]Virtual address: 0
[+][+]Physical address: 0
[+][+]File offset: 0
[+][+]Memory size: 476248
[+][+]Flags: 4
[+][+]Alignment: 4096
[+][+]Program Header: 3
[+][+]Type: 1
[+][+]Offset: 75000
[+][+]Virtual address: 75000
[+][+]Physical address: 75000
[+][+]File offset: 479232
[+][+]Memory size: 1098441
[+][+]Flags: 5
[+][+]Alignment: 4096
[+][+]Program Header: 4
[+][+]Type: 1
[+][+]Offset: 182000
[+][+]Virtual address: 182000
[+][+]Physical address: 182000
[+][+]File offset: 1581056
[+][+]Memory size: 305406
[+][+]Flags: 4
[+][+]Alignment: 4096
[+][+]Program Header: 5
[+][+]Type: 1
[+][+]Offset: 1cd8a0
[+][+]Virtual address: 1ce8a0
[+][+]Physical address: 1ce8a0
[+][+]File offset: 1890464
[+][+]Memory size: 22704
[+][+]Flags: 6
[+][+]Alignment: 4096
[+][+]Program Header: 6
[+][+]Type: 2
[+][+]Offset: 1d1798
[+][+]Virtual address: 1d2798
[+][+]Physical address: 1d2798
[+][+]File offset: 1906584
[+][+]Memory size: 576
[+][+]Flags: 6
[+][+]Alignment: 8
[+][+]Program Header: 7
[+][+]Type: 4
[+][+]Offset: 370
[+][+]Virtual address: 370
[+][+]Physical address: 370
[+][+]File offset: 880
[+][+]Memory size: 48
[+][+]Flags: 4
[+][+]Alignment: 8
[+][+]Program Header: 8
[+][+]Type: 4
[+][+]Offset: 3a0
[+][+]Virtual address: 3a0
[+][+]Physical address: 3a0
[+][+]File offset: 928
[+][+]Memory size: 68
[+][+]Flags: 4
[+][+]Alignment: 4
[+][+]Program Header: 9
[+][+]Type: 7
[+][+]Offset: 1cd8a0
[+][+]Virtual address: 1ce8a0
[+][+]Physical address: 1ce8a0
[+][+]File offset: 1890464
[+][+]Memory size: 32
[+][+]Flags: 4
[+][+]Alignment: 16
[+][+]Program Header: 10
[+][+]Type: 6474e553
[+][+]Offset: 370
[+][+]Virtual address: 370
[+][+]Physical address: 370
[+][+]File offset: 880
[+][+]Memory size: 48
[+][+]Flags: 4
[+][+]Alignment: 8
[+][+]Program Header: 11
[+][+]Type: 6474e550
[+][+]Offset: 1908e4
[+][+]Virtual address: 1908e4
[+][+]Physical address: 1908e4
[+][+]File offset: 1640676
[+][+]Memory size: 31236
[+][+]Flags: 4
[+][+]Alignment: 4
[+][+]Program Header: 12
[+][+]Type: 6474e551
[+][+]Offset: 0
[+][+]Virtual address: 0
[+][+]Physical address: 0
[+][+]File offset: 0
[+][+]Memory size: 0
[+][+]Flags: 6
[+][+]Alignment: 16
[+][+]Program Header: 13
[+][+]Type: 6474e552
[+][+]Offset: 1cd8a0
[+][+]Virtual address: 1ce8a0
[+][+]Physical address: 1ce8a0
[+][+]File offset: 1890464
[+][+]Memory size: 18272
[+][+]Flags: 4
[+][+]Alignment: 1
[+]Section headers: 35
[+][+]Section Header: 0
[+][+]Type: 0
[+][+]Flags: 0
[+][+]Address: 0
[+][+]Offset: 0
[+][+]Size: 0
[+][+]Link: 0
[+][+]Info: 0
[+][+]Alignment: 0
[+][+]Entry size: 0
[!]Empty section
[+][+]Section Header: 1
[+][+]Type: 1
[+][+]Flags: 2
[+][+]Address: 350
[+][+]Offset: 848
[+][+]Size: 28
[+][+]Link: 0
[+][+]Info: 0
[+][+]Alignment: 1
[+][+]Entry size: 0
[!]SHT_PROGBITS
[!]Invalid section: /usr/local/bin/grpc_php_plugin
[+][+]Section Header: 2
[+][+]Type: 7
[+][+]Flags: 2
[+][+]Address: 370
[+][+]Offset: 880
[+][+]Size: 48
[+][+]Link: 0
[+][+]Info: 0
[+][+]Alignment: 8
[+][+]Entry size: 0
[+][+]Section Header: 3
[+][+]Type: 7
[+][+]Flags: 2
[+][+]Address: 3a0
[+][+]Offset: 928
[+][+]Size: 36
[+][+]Link: 0
[+][+]Info: 0
[+][+]Alignment: 4
[+][+]Entry size: 0
[+][+]Section Header: 4
[+][+]Type: 7
[+][+]Flags: 2
[+][+]Address: 3c4
[+][+]Offset: 964
[+][+]Size: 32
[+][+]Link: 0
[+][+]Info: 0
[+][+]Alignment: 4
[+][+]Entry size: 0
[+][+]Section Header: 5
[+][+]Type: 6ffffff6
[+][+]Flags: 2
[+][+]Address: 3e8
[+][+]Offset: 1000
[+][+]Size: 27632
[+][+]Link: 6
[+][+]Info: 0
[+][+]Alignment: 8
[+][+]Entry size: 0
[+][+]Section Header: 6
[+][+]Type: b
[+][+]Flags: 2
[+][+]Address: 6fd8
[+][+]Offset: 28632
[+][+]Size: 95040
[+][+]Link: 7
[+][+]Info: 1
[+][+]Alignment: 8
[+][+]Entry size: 24
[!]SHT_DYNSYM
[!]Tag: 255865753264588419 343597384334
[!]Tag: -1484564035636664 54389558871065073
[!]Tag: 102738[+][+]Flags: 0
[+][+]Address: 0
[+][+]Offset: 2443340
[+][+]Size: 335
[+][+]Link: 0
[+][+]Info: 0
[+][+]Alignment: 1
[+][+]Entry size: 0
[!]SHT_STRTAB
Time taken: 13883094 nanoseconds
[+]strings extracted: 9141
=====================

```
