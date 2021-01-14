---

---

# Chicago_Car_Crash_Prediction 



## Overview



•The project was to build a classifier to **predict the** **COST OF DAMAGE** whenever a crash is recorded**. •**

The damages were divided into two parts above $1500 & below $1500

•Data base analysed is of the crash from 2015-2020

•The data that has been analysed is from the website https://data.cityofchicago.org/ 



#### Exploratory Data Analysis:

The feature crash_type & most severe injury surprisingly revealed that the damage claims were higher even though the there were no injury or its indication

The feature year revealed that the cost of damages have been steadily increasing by each passing year

The feature hour suggested that most crash happen between 3am to 5am 

The feature weather condition showed crashes and damages when the weather was clear , it also suggested as the severity of weather increased people became more careful and drove with precaution

The first crash feature revealed rear end crash were the highest

Similarly straight road, dry condition , road with no defects amounted for the highest damages

#### Exploratory Data Analysis & Pandas Profile report Overview:



Dataset info

|           Number of variables | 33      |
| ----------------------------: | ------- |
|        Number of observations | 9622    |
|             Total Missing (%) | 0.0%    |
|          Total size in memory | 2.4 MiB |
| Average record size in memory | 264.0 B |

Variables types

|       Numeric | 27   |
| ------------: | ---- |
|   Categorical | 0    |
|       Boolean | 3    |
|          Date | 0    |
| Text (Unique) | 0    |
|      Rejected | 3    |
|   Unsupported | 0    |

Warnings

- [`posted_speed_limit`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_posted_speed_limit) has 143 / 1.5% zeros **Zeros**
- [`injuries_no_indication`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_injuries_no_indication) has 171 / 1.8% zeros **Zeros**
- [`crash_hour`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_crash_hour) has 165 / 1.7% zeros **Zeros**
- [`year_notified`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_year_notified) is highly correlated with [`crash_year`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_crash_year) (ρ = 0.99941) **Rejected**
- [`month_notified`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_month_notified) is highly correlated with [`crash_month`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_crash_month) (ρ = 0.98822) **Rejected**
- [`day_notified`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_day_notified) is highly correlated with [`crash_day`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_crash_day) (ρ = 0.91956) **Rejected**
- [`hour_notified`](http://localhost:8888/notebooks/Chicago_Car_Crash_Prediction/Modelling_2.ipynb#pp_var_hour_notified) has 153 / 1.6% zeros **Zeros**





## Modelling Preparation steps carried out

1-Model preparation & Train Test Split

2-Before applying algorithm checking whether the data is equally splitted or not,to avoid data imbalacing problem

3-Scaling with Standard Scaler



## **ML models**

Different machine learning algorithm to be used and try to find algorithm which predict accurately.

1. Logistic Regression (Baseline)
2. Naive Bayes
3. Random Forest Classifier
4. Extreme Gradient Boost
5. K-Nearest Neighbour
6. Decision Tree
7. Support Vector Machine

## All the models were analysed based on 

**Confusion matrix**

**Accuracy**

 **precision**    

**recall**  

**f1-score**   

### Baseline model Logistic Regression

```
confussion matrix
[[380 479]
 [301 765]]


Accuracy of Logistic Regression: 59.480519480519476 

              precision    recall  f1-score   support

           0       0.56      0.44      0.49       859
           1       0.61      0.72      0.66      1066

    accuracy                           0.59      1925
   macro avg       0.59      0.58      0.58      1925
weighted avg       0.59      0.59      0.59      1925
```

### Model 2 Naive Bayes

```
confussion matrix
[[627 232]
 [588 478]]


Accuracy of Naive Bayes model: 57.4025974025974 

              precision    recall  f1-score   support

           0       0.52      0.73      0.60       859
           1       0.67      0.45      0.54      1066

    accuracy                           0.57      1925
   macro avg       0.59      0.59      0.57      1925
weighted avg       0.60      0.57      0.57      1925
```

### Model 3 Random Forest Classfier

```
confussion matrix
[[246 613]
 [187 879]]


Accuracy of Random Forest: 58.44155844155844 

              precision    recall  f1-score   support

           0       0.57      0.29      0.38       859
           1       0.59      0.82      0.69      1066

    accuracy                           0.58      1925
   macro avg       0.58      0.56      0.53      1925
weighted avg       0.58      0.58      0.55      1925
```

### Model 4 Extreme Gradient Boost

```
confussion matrix
[[380 479]
 [272 794]]


Accuracy of Extreme Gradient Boost: 60.98701298701299 

              precision    recall  f1-score   support

           0       0.58      0.44      0.50       859
           1       0.62      0.74      0.68      1066

    accuracy                           0.61      1925
   macro avg       0.60      0.59      0.59      1925
weighted avg       0.61      0.61      0.60      1925
```

### Model 5 K-NeighborsClassifier

```
confussion matrix
[[519 340]
 [488 578]]


Accuracy of K-NeighborsClassifier: 56.98701298701299 

              precision    recall  f1-score   support

           0       0.52      0.60      0.56       859
           1       0.63      0.54      0.58      1066

    accuracy                           0.57      1925
   macro avg       0.57      0.57      0.57      1925
weighted avg       0.58      0.57      0.57      1925
```

### Model 6 DecisionTreeClassifier

```
confussion matrix
[[575 284]
 [478 588]]


Accuracy of DecisionTreeClassifier: 60.41558441558441 

              precision    recall  f1-score   support

           0       0.55      0.67      0.60       859
           1       0.67      0.55      0.61      1066

    accuracy                           0.60      1925
   macro avg       0.61      0.61      0.60      1925
weighted avg       0.62      0.60      0.60      1925
```

### Model 7 Support Vector Classifier

```
confussion matrix
[[439 420]
 [327 739]]


Accuracy of Support Vector Classifier: 61.19480519480519 

              precision    recall  f1-score   support

           0       0.57      0.51      0.54       859
           1       0.64      0.69      0.66      1066

    accuracy                           0.61      1925
   macro avg       0.61      0.60      0.60      1925
weighted avg       0.61      0.61      0.61      1925
```



# **Model Evaluation**

| Model | Accuracy               |           |
| :---- | :--------------------- | --------- |
| 0     | Logistic Regression    | 59.480519 |
| 1     | Naive Bayes            | 57.402597 |
| 2     | Random Forest          | 58.441558 |
| 3     | Extreme Gradient Boost | 60.155844 |
| 4     | K-Nearest Neighbour    | 56.987013 |
| 5     | Decision Tree          | 60.415584 |
| 6     | Support Vector Machine | 61.194805 |

### Stacking technique_Ensembling to improve accuracy

```
confussion matrix
[[440 419]
 [327 739]]


Accuracy of StackingCVClassifier: 61.24675324675325 

              precision    recall  f1-score   support

           0       0.57      0.51      0.54       859
           1       0.64      0.69      0.66      1066

    accuracy                           0.61      1925
   macro avg       0.61      0.60      0.60      1925
weighted avg       0.61      0.61      0.61      1925
```

Observation:

1) Support Vector Machine gives the best Accuracy compared to other models.

2) Ensembling technique increased the accuracy of the model by a small margin.



### GridSearchCV for improving Decision tree & Random forest model

```
Decision tree grid search:  0.6192851205320034
Random forest grid search:  0.6296758104738155
```

### Confusion Matrix & Classification Report

- For Random forest & Random Forest GridSearch



**Random forest **

```
confussion matrix
[[ 386  686]
 [ 260 1074]]


Accuracy of Random Forest: 60.68162926018288 

              precision    recall  f1-score   support

           0       0.60      0.36      0.45      1072
           1       0.61      0.81      0.69      1334

    accuracy                           0.61      2406
   macro avg       0.60      0.58      0.57      2406
weighted avg       0.60      0.61      0.59      2406
```



**Random forest GridSearch**

```
confussion matrix
[[542 530]
 [361 973]]


Accuracy of Random Forest GridSearch: 62.96758104738155 

              precision    recall  f1-score   support

           0       0.60      0.51      0.55      1072
           1       0.65      0.73      0.69      1334

    accuracy                           0.63      2406
   macro avg       0.62      0.62      0.62      2406
weighted avg       0.63      0.63      0.62      2406
```



### Overall Analysis

- Support Vector Machine  had the best accuracy score during the first modelling 61.194805, followed by Decision Tree  60.415584 & Extreme Gradient Boost  60.155844

- Stacking technique_Ensembling showed minor improved Accuracy of StackingCVClassifier: 61.24675324675325 

  **GridSearchCV improved**  

- Decision tree model from 60.415584 to Decision tree grid search:  61.92851205320034

- Random forest model from  58.441558 & 60.68162926018288  Random forest grid search:  62.96758104738155

- 



Thank You 