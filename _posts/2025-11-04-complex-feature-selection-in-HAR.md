---
layout: post
title: Feature Selection For Classification Accuracy
image: "/posts/classification-title-img.png"
tags: [Human Activity Recognition, Machine Learning, Classification, Python]
---

Our client, a HAR team, wants to utilise Machine Learning to predict a movement activity type!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Vector Calculations](#veccalc-title)
- [04. Linear SVC + RFECV Feature Selection](#linSVCRFECV-title)
- [05. CFS Feature Selection](#cfs-title)
- [06. KNN](#knn-title)
- [07. Modelling Summary](#modelling-summary)
- [08. Application](#modelling-application)
- [09. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, an Human Activity Recognition (HAR) study team, ran a study using HAR sensors to provide real-time feedback of weightlifting technique.  They want to classify a subject's movement on a bicep curl as using 'correct technique' or one of four common technique errors. 

For this, they collected data on six subjects using sensors strapped to their belt, forearm, arm, and a held dumbbell. The subjects performed a biceps curl with correct and common incorrect technique. 

For the next subject they would like to classify the movement they are doing, and provide this as feedback.

Based upon the data they've collected, we will look to understand the *prediction accuracy* of a model built to classify the lift, and test it on new lifters. This would allow the HAR team to provide a new lifter with technique cues to improve their movement on the exercise.

Let's use Machine Learning to take on this task!
<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from each of the 6 subjects, gathering key sensor data that may help predict *class* of the movement.

Within our dataset from the 6 subjects, we found that 28.4% of sensor data across all collection windows indicated a correct lift (class A), and 71.6% came from incorrect lifts. 
Of the incorrect lifts, it was split by error type B-E: 19.3% : 18.4% : 17.4% : 16.4%. This tells us that while the data isn't perfectly balanced at 20:20:20:20:20 across lift class, it isn't *too* imbalanced either.

As we are predicting a class in a HAR context, we tested Random Forest modelling approaches.

For each model, we will import the data in the same way but will need to pre-process the data based upon different feature sets to include in our Random Forest algorithm.  We will train & test each model, refining our approach to feature setting, to provide optimal performance, and then measure this predictive performance based on accuracy score and performance on a new set of 20 observations from 20 new lifters.
<br>
<br>

### Results <a name="overview-results"></a>

The goal for the project was to build a model that would accurately predict the customers that would sign up for the *delivery club*.  This would allow for a much more targeted approach when running the next iteration of the campaign.  A secondary goal was to understand what the drivers for this are, so the client can get closer to the customers that need or want this service, and enhance their messaging.

Based upon these, the chosen the model is the Random Forest as it was a) the most consistently performant on the test set across classification accuracy, precision, recall, and f1-score, and b) the feature importance and permutation importance allows the client an understanding of the key drivers behind *delivery club* signups.

<br>
**Metric 1: Classification Accuracy**

* KNN = 0.936
* Random Forest = 0.935
* Decision Tree = 0.929
* Logistic Regression = 0.866

<br>
**Metric 2: Precision**

* KNN = 1.00
* Random Forest = 0.887
* Decision Tree = 0.885
* Logistic Regression = 0.784

<br>
**Metric 3: Recall**

* Random Forest = 0.904
* Decision Tree = 0.885
* KNN = 0.762
* Logistic Regression = 0.69

<br>
**Metric 4: F1 Score**

* Random Forest = 0.895
* Decision Tree = 0.885
* KNN = 0.865
* Logistic Regression = 0.734
<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

Since predictive accuracy was very high - our feature selection and modelling approach could be tested on new subjects doing different types of lifts, to see if this accuracy translates to different movements.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken for more complex movements, to similar to the complete data in the original study.
<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be predicting the categorical *classe* metric from the subset *training* sample adapted from the HAR study database.

The key variables hypothesised to predict this will come from the sensor data contained in this set.

We calculate vector statistics from the repeat sensor collections in windows of different length across (0.5 to 2.5 seconds) during the lift. For this training data, a lifter had repeated 10 repetitions of a bicep curl exactly according to instructions specified by a weight lfiting as correct technique (Class A), or by their instruction to perform one of the following errors:
Error 1: Throw the elbows to the front (Class B)
Error 2: Lift the dumbbell only halfway (Class C)
Error 3: Lower the dumbbell only halfway (Class D)
Error 4: Throw the hips to the front (Class E)

After some data pre-processing in Python, we have a dataset for modelling that contains the following fields...

<br>
<br>

| **Variable Names** | **Variable Type** | **Description** |
|---|---|---|
| classe | Dependent | A categorical variable showing the lift class |
| num_window | Independent | The time window in which the sensor data was collected - not used for modelling  |
| roll/pitch/yaw_belt/arm/forearm/dumbbell | Independent | The roll,pitch or yaw reading for a sensor collection in the time window at either belt/arm/forearm/dumbbell sensors |
| total_accel_belt | Independent | The total accelerometer data in 3 dimensions x,y,z at the belt sensor |
| gyros/accel/magnet_belt_x/y/z | Independent | The gyrometer, accelerometer and magnetometer belt sensor readings in x/y/z dimensions: forwards, sideways, upwards  |
| gyros/accel/magnet_arm_x/y/z | Independent | The gyrometer, accelerometer and magnetometer armband sensor readings in x/y/z dimensions: forwards, sideways, upwards |
| gyros/accel/magnet_dumbbell_x/y/z | Independent | The gyrometer, accelerometer and magnetometer dumbbell sensor readings in x/y/z dimensions: forwards, sideways, upwards |
| gyros/accel/magnet_forearm_x/y/z | Independent | The gyrometer, accelerometer and magnetometer forearm sensor readings in x/y/z dimensions: forwards, sideways, upwards |

<br>
# Modelling Overview  <a name="modelling-overview"></a>

We will build a model that looks to accurately classify *classe*, based upon the sensor measures listed above.

If that can be achieved, we can use this model to predict movement type for future movements (future weighted bicep curl movements).  This information can be used to provide feedback to a lifter, guiding correct movement technique.

As we are predicting a categorical output using granular data from many inputs in a HAR environment, we use a two step approach: 1) clacluate vector statistics from the raw sensor data and 2) pass these onto one of two feature selection approaches:

* LinearSVC + RFECV - an industry standard approach to HAR using sensor data.
* Correlation-Based Feature Selection (CFS) + RF - an elegant selector that trims highly correlated features.

<br>
# Vector Calculations <a name="veccalc-title"></a>

We utilise the numpy and pandas libraries within Python to compute vector magnitudes for all sensors. The code sections below are broken up into 2 key sections:

* Data Import
* Data Preprocessing - vector calculations

<br>
### Data Import <a name="veccalc-import"></a>

We import the raw training measures data.  We ensure we remove id and unnecessary columns, and keep only sensor and class data.

We also investigate the class balance of our dependent variable - which is important when assessing classification accuracy.

```python

import pandas as pd
import numpy as np

df = pd.read_csv("data/pml_training.csv")

# Drop non-sensor metadata
df = df.drop([
    "Unnamed: 0",
    "user_name",
    "raw_timestamp_part_1",
    "raw_timestamp_part_2",
    "new_window"
], axis=1)

# Remove columns with too many missing values (>=20): this removes all variables in raw data that were wiped - leaving only the complete sensor and time window data.
missing_counts = df.isna().sum()
df = df.loc[:, missing_counts < 20]

# Keep only numeric columns: keeps all sensor data
df = df.select_dtypes(include=[np.number])

# Target: sets outcome variable
y = pd.read_csv("data/pml_training.csv")["classe"]

# Class
y.value_counts(normalize = True)

```
<br>
From the last step in the above code, we see that **28% of movements were class A and 72% were an error class B-E: 19.3%, 18.4%, 17.4%, 16.4%, respectively**.  This tells us that while the data isn't perfectly balanced at 20:20:20:20:20, it isn't *too* imbalanced either.

<br>
### Data Preprocessing <a name="veccalc-preprocessing"></a>

For 3D Sensor data, we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* Summarising variables on axes to vectors

<br>
##### Missing Values

There were no missing values in the raw sensor data, as this subset of the study data had been pre-cleaned. So we will just move on to the next step.

We had removed any columns containing missing data on data import, and will create new calculated fields to replace them.

<br>

<br>
##### Summarise variables on axes to vectors

In the next code block we do four things, we firstly group our data into vector groups by sensor, calculate their vector magnitudes, define the vector statistics that we wish to calculate to use as features in the random forest model along with the grouping we will use to aggregate to these, and finally run our sensor data through this to create these new variables as vector summaries.

Once we have done this, we can perform feature selection using these summaries as features.

Vector magnitudes are calculated from 3 dimensions (front, sideways, upwards) as:

$\mathrm{magnitude}=\sqrt{x^2+y^2+z^2}$

<br>
```python

# Group to create vector magnitudes for all sensors
sensor_groups = {
    "belt_accel": ["accel_belt_x", "accel_belt_y", "accel_belt_z"],
    "belt_gyro":  ["gyros_belt_x", "gyros_belt_y", "gyros_belt_z"],
    "belt_mag":   ["magnet_belt_x", "magnet_belt_y", "magnet_belt_z"],

    "arm_accel":  ["accel_arm_x", "accel_arm_y", "accel_arm_z"],
    "arm_mag":    ["magnet_arm_x", "magnet_arm_y", "magnet_arm_z"],

    "dumb_accel": ["accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z"],
    "dumb_gyro":  ["gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z"],
    "dumb_mag":   ["magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z"],

    "fore_gyro":  ["gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z"],
    "fore_mag":   ["magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z"],
}

# Calculate vector magnitudes for all sensors
for name, cols in sensor_groups.items():
    df[f"{name}_mag"] = np.sqrt((df[cols]**2).sum(axis=1))

# Window grouping variable
num_window = pd.read_csv("data/pml_training.csv")["num_window"]

df["num_window"] = num_window

# Functions to compute
stats = {
    "mean": "mean",
    "var": "var",
    "max": "max",
    "min": "min",
    "range": lambda s: s.max() - s.min(),
    "sum": "sum"
}

# Apply to all magnitude features + roll/pitch/yaw
candidate_cols = [
    col for col in df.columns
    if any(key in col for key in ["roll", "pitch", "yaw", "_mag"])
]

for col in candidate_cols:
    for stat_name, func in stats.items():
        df[f"{col}_{stat_name}"] = df.groupby("num_window")[col].transform(func)

# drop the grouping variable as it is no longer needed and should not be a feature
df = df.drop(columns=["num_window"])

```

<br>
##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task.  It can be a very important addition or at least, consideration, in certain scenarios.  The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train, and faster to make predictions
* **Explainability** - understanding & explaining outputs for stakeholder & customers becomes much easier

There are many, many ways to apply Feature Selection.  These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

For our task in an HAR context, feature selection is extremely important. 

We attempt an industry standard variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* using a *Linear Support Vector Classifier* where we split the data into many "chunks" and iteratively train & validate models on each "chunk" separately.  This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was.  LinearSVC learns a linear decision boundary that distinguishes the five exercise‑quality classes using the accelerometer, gyroscope, and magnetometer features, combined with our calculated vector statistics features.
RFECV then uses the model’s coefficients to identify which of all of these features truly contribute to class separation and recursively removes the weakest ones.
One downside of this approach is the long run-times that LinearSVC can take, as we'll see!

We compare this to an elegant mathematical approach to feature selection called Correlation-Based feature Selection (CFS), which is also used in HAR.
CFS chooses the smallest set of features that are highly correlated with the class while being minimally correlated with each other. The core of this approach uses the merit function:

$\mathrm{Merit_{\mathnormal{S}}}=\frac{k\cdot \bar {r}_{cf}}{\sqrt{k+k(k-1)\bar {r}_{ff}}}$

Multicollinearity occurs when two or more input variables are *highly* correlated with each other, it is a scenario we attempt to avoid as in short, while it won't necessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics around how well the model is performing, and how much each input variable is truly having. CFS does a good job of reducing multicollinearity as it assigns more merit to features that are not correlated.

We'll code this approach in Python and see how it works out!

Let's first attempt the industry standard approach to feature selection in HAR tasks.

<br>
# LinearSVC + RFECV <a name="linSVCRFECV-title"></a>

![alt text](/img/posts/log-reg-feature-selection-plot.png "Logistic Regression Feature Selection Plot")

We will again utilise the scikit-learn library within Python to select features for our model. The code section below continues from the previous sections:

<br>
### Feature Selection <a name="linSVCRFECV-select"></a>

Continuing with our df object from adding our vector features, we import the packages we'll need: LinearSVC, RFECV and StratifiedKFold.

```python

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

estimator = LinearSVC(
    C=1.0,
    dual=False,
    max_iter=5000
)

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    n_jobs=-1
)

rfecv.fit(X, y)

selected_features = X.columns[rfecv.support_]

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X,y)

optimal_feature_count = feature_selector.n_features_
print(f"the optimal number of features is {optimal_feature_count}")

X_new = X.loc[:,feature_selector.get_support()]

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.xticks(range(1,len(fit.cv_results_['mean_test_score'])+1,1))
plt.title(f"Feature Selection using RFE w CV \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

```
Well, this is still running after an hour...
While this will eventually complete, and deliver a nice set of features to use in modelling our prediction, it can take minutes to over an hour! 

We'd need something faster if we were investigating a range of new exercises.
Let's see how we get on using a simpler approach.

# CFS Feature Selection <a name="cfs-title"></a>

```python

############################################################################
# Step 1: Implement a CFS selector
############################################################################

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CFSSelector(BaseEstimator, TransformerMixin):
    def __init__(self, max_no_improve=5):
        self.max_no_improve = max_no_improve

    def _subset_merit(self, idx, r_cf, r_ff):
        k = len(idx)
        if k == 0:
            return 0
        r_cf_bar = np.mean(np.abs(r_cf[idx]))
        if k == 1:
            r_ff_bar = 0
        else:
            sub = r_ff[np.ix_(idx, idx)]
            mask = ~np.eye(k, dtype=bool)
            r_ff_bar = np.mean(np.abs(sub[mask]))
        return (k * r_cf_bar) / np.sqrt(k + k*(k-1)*r_ff_bar + 1e-12)

    def fit(self, X, y):
        X = np.asarray(X, float)
        _, y_enc = np.unique(y, return_inverse=True)

        n = X.shape[1]

        # feature–class correlations
        r_cf = np.array([np.corrcoef(X[:, j], y_enc)[0, 1] for j in range(n)])

        # feature–feature correlations
        r_ff = np.corrcoef(X, rowvar=False)

        selected = []
        best_merit = 0
        no_improve = 0
        remaining = list(range(n))

        while remaining and no_improve < self.max_no_improve:
            best_candidate = None
            best_candidate_merit = best_merit

            for f in remaining:
                cand = selected + [f]
                merit = self._subset_merit(cand, r_cf, r_ff)
                if merit > best_candidate_merit:
                    best_candidate_merit = merit
                    best_candidate = f

            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                if best_candidate_merit > best_merit:
                    best_merit = best_candidate_merit
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        self.selected_indices_ = np.array(selected)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

###################################################################################
# Step 2 : Run CFS and inspect the selected features: using standardised scaling
###################################################################################

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

selector = CFSSelector().fit(X_scaled, y)

selected_features = df.columns[selector.selected_indices_]

print("Number of selected features:", len(selected_features))
print("\nSelected features:")
for f in selected_features:
    print("-", f)

###########################################################################################   
# when we run CFS on this dataset, 
# - CFS output is correct for this dataset
# - The 6 features here are the best subset in the feature space.
###########################################################################################

########################################################################
# Step 3: Build a dataframe containing only 'classe' + selected features
########################################################################

# Convert selected feature names to a list
selected_feature_list = list(selected_features)

# Build the final dataframe
df_selected = pd.DataFrame({
    "classe": y
})

# Add each selected feature column
for feat in selected_feature_list:
    df_selected[feat] = df[feat].values

print("\nFinal dataframe shape:", df_selected.shape)
print(df_selected.head())

```



















At a high level, there are two common ways to tackle this.  The first, often just called **Feature Importance** is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the gini impurity score (for a Classification problem) was before the split was made, and compare this to the gini impurity score after the split was made.  We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called **Permutation Importance** cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through.  Once all of these observations have been run through the Decision Tree, we obtain a classification accuracy score for these predictions.

In order to understand the *importance*, we *randomise* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score.  The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let's put them both in place, and plot the results...

<br>
![alt text](/img/posts/rf-classification-feature-importance.png "Random Forest Feature Importance Plot")
<br>
<br>
![alt text](/img/posts/rf-classification-permutation-importance.png "Random Forest Permutation Importance Plot")

<br>
The overall story from both approaches is very similar, in that by far, the most important or impactful input variables are *distance_from_store* and *transaction_count*

Surprisingly, *average_basket_size* was not as important as hypothesised.

There are slight differences in the order or "importance" for the remaining variables but overall they have provided similar findings.

___
<br>


We utilise the scikit-learn library within Python to model our data using KNN. The code sections below are broken up into 5 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment
* Optimal Value For K

<br>
### Data Import <a name="knn-import"></a>

Again, since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

As with the other approaches, we also investigate the class balance of our dependent variable - which is important when assessing classification accuracy.

```python

# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV

# import modelling data
data_for_model = pickle.load(open("data/delivery_club_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

# assess class balance of dependent variable
data_for_model["signup_flag"].value_counts(normalize = True)

```
<br>
From the last step in the above code, we see that **69% of customers did not sign up and 31% did**.  This tells us that while the data isn't perfectly balanced at 50:50, it isn't *too* imbalanced either.  Because of this, and as you will see, we make sure to not rely on classification accuracy alone when assessing results - also analysing Precision, Recall, and F1-Score.

<br>
### Data Preprocessing <a name="knn-preprocessing"></a>

For KNN, as it is a distance based algorithm, we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Feature Scaling
* Feature Selection

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

```python

# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Outliers

As KNN is a distance based algorithm, you could argue that if a data point is a long way away, then it will simply never be selected as one of the neighbours - and this is true - but outliers can still cause us problems here.  The main issue we face is when we come to scale our input variables, a very important step for a distance based algorithm.

We don't want any variables to be "bunched up" due to a single outlier value, as this will make it hard to compare their values to the other input variables.  We should always investigate outliers rigorously - in this case we will simply remove them.

In this code section, just like we saw when applying Logistic Regression, we use **.describe()** from Pandas to investigate the spread of values for each of our predictors.  The results of this can be seen in the table below.

<br>

| **metric** | **distance_from_store** | **credit_score** | **total_sales** | **total_items** | **transaction_count** | **product_area_count** | **average_basket_value** |
|---|---|---|---|---|---|---|---|
| mean | 2.61 | 0.60 | 968.17 | 143.88 | 22.21 | 4.18 | 38.03  |
| std | 14.40 | 0.10 | 1073.65 | 125.34 | 11.72 | 0.92 | 24.24  |
| min | 0.00 | 0.26 | 2.09 | 1.00 | 1.00 | 1.00 | 2.09  |
| 25% | 0.73 | 0.53 | 383.94 | 77.00 | 16.00 | 4.00 | 21.73  |
| 50% | 1.64 | 0.59 | 691.64 | 123.00 | 23.00 | 4.00 | 31.07  |
| 75% | 2.92 | 0.67 | 1121.53 | 170.50 | 28.00 | 5.00 | 46.43  |
| max | 400.97 | 0.88 | 7372.06 | 910.00 | 75.00 | 5.00 | 141.05  |

<br>
Again, based on this investigation, we see some *max* column values for several variables to be much higher than the *median* value.

This is for columns *distance_from_store*, *total_sales*, and *total_items*

For example, the median *distance_to_store* is 1.64 miles, but the maximum is over 400 miles!

Because of this, we apply some outlier removal in order to facilitate generalisation across the full dataset.

We do this using the "boxplot approach" where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

<br>
```python

outlier_investigation = data_for_model.describe()
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)

```

<br>
##### Split Out Data For Modelling

In exactly the same way we've done for the other three models, in the next code block we do two things, we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation. Again, we make sure to add in the stratify parameter to ensure that both our training and test sets have the same proportion of customers who did, and did not, sign up for the delivery club - meaning we can be more confident in our assessment of predictive performance.

<br>
```python

# split data into X and y objects for modelling
X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

```

<br>
##### Categorical Predictor Variables

As we saw when applying the other algorithms, in our dataset, we have one categorical variable *gender* which has values of "M" for Male, "F" for Female, and "U" for Unknown.

The KNN algorithm can't deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of *new* columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation.  These new columns would go into our model as input variables, and the original column is discarded.

We also drop one of the new columns using the parameter *drop = "first"*.  We do this to avoid the *dummy variable trap* where our newly created encoded columns perfectly predict each other - and we run the risk of breaking the assumption that there is no multicollinearity, a requirement or at least an important consideration for some models, Linear Regression being one of them! 

___
<br>
# Modelling Summary  <a name="modelling-summary"></a>

The goal for the project was to build a model that would accurately predict the customers that would sign up for the *delivery club*.  This would allow for a much more targeted approach when running the next iteration of the campaign.  A secondary goal was to understand what the drivers for this are, so the client can get closer to the customers that need or want this service, and enhance their messaging.

Based upon these, the chosen the model is the Random Forest as it was a) the most consistently performant on the test set across classification accuracy, precision, recall, and f1-score, and b) the feature importance and permutation importance allows the client an understanding of the key drivers behind *delivery club* signups.

<br>
**Metric 1: Classification Accuracy**

* KNN = 0.936
* Random Forest = 0.935
* Decision Tree = 0.929
* Logistic Regression = 0.866

<br>
**Metric 2: Precision**

* KNN = 1.00
* Random Forest = 0.887
* Decision Tree = 0.885
* Logistic Regression = 0.784

<br>
**Metric 3: Recall**

* Random Forest = 0.904
* Decision Tree = 0.885
* KNN = 0.762
* Logistic Regression = 0.69

<br>
**Metric 4: F1 Score**

* Random Forest = 0.895
* Decision Tree = 0.885
* KNN = 0.865
* Logistic Regression = 0.734

___
<br>
# Application <a name="modelling-application"></a>

We now have a model object, and a the required pre-processing steps to use this model for the next *delivery club* campaign.  When this is ready to launch we can aggregate the necessary customer information and pass it through, obtaining predicted probabilities for each customer signing up.

Based upon this, we can work with the client to discuss where their budget can stretch to, and contact only the customers with a high propensity to join.  This will drastically reduce marketing costs, and result in a much improved ROI.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
