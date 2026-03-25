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
