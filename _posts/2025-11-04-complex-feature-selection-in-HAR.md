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
- [06. Model Build](#model-title)
- [07. Model Test/Training Accuracy](#accuracy-summary)
- [08. Model Accuracy on New Data](#modelling-application)
- [09. Full Set Feature Performance](#fully-featured)
- [10. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, an Human Activity Recognition (HAR) study team, ran a study using HAR sensors to provide real-time feedback of weightlifting technique.  They want to classify a subject's movement on a bicep curl as using 'correct technique' or one of four common technique errors. 

For this, they collected data on six subjects using sensors strapped to their belt, forearm, arm, and a held dumbbell. The subjects performed a biceps curl with correct and common incorrect technique. 

For the next subject they would like to classify the movement they are doing, and provide this as feedback.

Based upon the data they've collected, we will look to understand the *prediction accuracy* of a model built to classify the lift, and test it on new lifters. This would allow the HAR team to provide a new lifter with technique cues to improve their movement during exercise. The HAR team has access to a much larger set of measures and can train their own models on this, whereas we only have access to the small subset of training data that they've sent to us to use in our Machine Learning builds.

Let's use Machine Learning to take on this task!

<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from each of the 6 subjects, gathering key sensor data that may help predict *class* of the movement.

Within our dataset from the 6 subjects, we found that 28.4% of sensor data across all collection windows indicated a correct lift (class A), and 71.6% came from incorrect lifts. 
Of the incorrect lifts, it was split by error type B-E: 19.3% : 18.4% : 17.4% : 16.4%. This tells us that while the data isn't perfectly balanced at 20:20:20:20:20 across lift class, it isn't *too* imbalanced either.

As we are predicting a class in a HAR context, with a dense feature space, we test Random Forest modelling approaches as industry standard. 

For each model, we will import the data in the same way, but will need to pre-process it based upon different strategies to give a feature set to include in our Random Forest algorithm.  We will train & test a model using each feature selection approach, to provide optimal performance, and then measure this predictive performance based on accuracy score and on its ability to successfully classify a new set of 20 observations from 20 new lifters.

<br>
<br>

### Results <a name="overview-results"></a>

The goal for the project was to convert sensor data into a useful predictor that could identify the type of a biceps curl as:
- Correctly performed (Class A/0)
- Error 1: Throw the elbows to the front (Class B/1)
- Error 2: Lift the dumbbell only halfway (Class C/2)
- Error 3: Lower the dumbbell only halfway (Class D/3)
- Error 4: Throw the hips to the front (Class E/4)

Our model trained on the data, using 6 features found by CFS calculated from this same data gave an accuracy score of: 99.9 %. An *almost* unbeatable result!
A model using 17 features found by CFS calculated from the full research data found this subset too simple for it, at an accuracy score of 100%. 

The only errors we made training on this set alone compared to the study's more powered model based on a fuller dataset are shown in the off-diagonal cells in the confusion matrix below: a difference of just 4 mis-classifications, but made using 11 fewer features in our model!

<img width="445" height="473" alt="image" src="https://github.com/user-attachments/assets/b1524ed0-95ef-4bbe-a159-026ded60eea0" />

A feature selection strategy using industry standard LinearSVC + RFECV selection proved too time-consuming to run, and could not have improved on the modelling results found using the CFS selection strategy as shown. 

On a hold out set, our chosen model gave a score of 20/20 correctly classified lifts, based on single sensor snapshots of bicep curl readings

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

Since predictive accuracy was very high - our feature selection and modelling approach could be tested on new subjects doing different types of lifts, to see if this accuracy translates to different movements.

From a data point of view, further feature engineering could be undertaken for more complex movements, with features likely to be similar to the ones included in the complete set from the original study.

<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be predicting the categorical *classe* metric from the subset *training* sample adapted from the HAR study database.

The key variables hypothesised to predict this will come from the sensor data contained in this set.

We calculate vector statistics from the repeat sensor collections in windows of different length (0.5 to 2.5 seconds) during the lift. For this training data, a lifter had completed 10 repetitions of a bicep curl exactly according to instructions specified by a weight lifting coach. 
The coach instructed to lift using a soecified technique (Class A), or to perform one of the following errors:
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
| total_accel_belt/arm/dumbbell/forearm | Independent | The total accelerometer data in 3 dimensions x,y,z at the belt, arm, dumbbell or forearm sensor |
| gyros/accel/magnet_belt_x/y/z | Independent | The gyrometer, accelerometer and magnetometer belt sensor readings in x/y/z dimensions: forwards, sideways, upwards  |
| gyros/accel/magnet_arm_x/y/z | Independent | The gyrometer, accelerometer and magnetometer armband sensor readings in x/y/z dimensions: forwards, sideways, upwards |
| gyros/accel/magnet_dumbbell_x/y/z | Independent | The gyrometer, accelerometer and magnetometer dumbbell sensor readings in x/y/z dimensions: forwards, sideways, upwards |
| gyros/accel/magnet_forearm_x/y/z | Independent | The gyrometer, accelerometer and magnetometer forearm sensor readings in x/y/z dimensions: forwards, sideways, upwards |

<br>
# Modelling Overview  <a name="modelling-overview"></a>

We will build a model that looks to accurately classify *classe*, based upon the sensor measures listed above.

If that can be achieved, we can use this model to predict movement type for future movements (future weighted bicep curl movements).  This information can be used to provide feedback to a lifter, guiding correct movement technique.

As we are predicting a categorical output using granular data from many inputs in a HAR environment, we use a two step approach: 1) calculate vector statistics from the raw sensor data and 2) pass these onto one of two feature selection approaches:

* LinearSVC + RFECV - an industry standard approach to HAR using sensor data.
* Correlation-Based Feature Selection (CFS) + RF - an elegant selector that trims highly correlated features.

From there, we include the identified features in a Random Forest model consisting of 500 Decision trees, to make our movement type class predictions.

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

$$\mathrm{magnitude}=\sqrt{x^2+y^2+z^2}$$

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
* **Explainability** - understanding & explaining outputs for stakeholders becomes much easier

There are many, many ways to apply Feature Selection.  These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

For our task in an HAR context, feature selection is extremely important. 

We attempt an industry standard variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* using a *Linear Support Vector Classifier* where we split the data into many "chunks" and iteratively train & validate models on each "chunk" separately.  This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was.  LinearSVC learns a linear decision boundary that distinguishes the five exercise‑quality classes using the accelerometer, gyroscope, and magnetometer features, combined with our calculated vector statistics features.
RFECV then uses the model’s coefficients to identify which of all of these features truly contribute to class separation and recursively removes the weakest ones.
One downside of this approach is the long run-times that LinearSVC can take, as we'll see!

We compare this to an elegant mathematical approach to feature selection called Correlation-Based feature Selection (CFS), which is also used in HAR.
CFS chooses the smallest set of features that are highly correlated with the class while being minimally correlated with each other. The core of this approach uses the merit function:

$$\mathrm{Merit_{\mathnormal{S}}}=\frac{k\cdot \bar {r}_{cf}}{\sqrt{k+k(k-1)\bar {r}_{ff}}}$$

Multicollinearity occurs when two or more input variables are *highly* correlated with each other, it is a scenario we attempt to avoid as in short, while it won't necessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics that describe how well the model is performing, and how much effect each input variable is truly having. CFS does a good job of reducing multicollinearity since it assigns more merit (defined above) to features that are not correlated.

We'll code this approach in Python and see how it works out in our model predictions!

Let's first attempt an industry standard approach to feature selection in HAR tasks.

<br>
# LinearSVC + RFECV <a name="linSVCRFECV-title"></a>

We will again utilise the scikit-learn library within Python to select features for our model.
The code section below continues with our prepared dataframe containing the suite of vector summary statistics.

<br>
### Feature Selection <a name="linSVCRFECV-select"></a>

Continuing with our df object containing our new vector features, we:
- import the packages we'll need: LinearSVC, RFECV and StratifiedKFold
- instantiate our Linear SVC estimator
- run cross-validated random forest estimation

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

```

Well, this is still running, after an hour...

While this will eventually converge, and deliver a nice set of features to use in modelling our prediction, it can take minutes to over an hour! This is enough of a reason to try something else, so let's end this line of investigation now.

We'd need something faster if we were investigating a range of new exercises.

Let's see how we go in using a simpler approach to feature selection.

We'll create some functions using NumPy in Python to define what we want our CFS to do: define merit mathematically, and add features through this algorithm as long as they continue to improve merit. 

Then we'll standardise the scaling of our selected features, and output a new dataframe for modelling containing only the scaled features found by our CFS function.  

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

There we have it. A new dataframe containing the features that are most correlated with classe, that were the least correlated with eachother.
Credit for this approach goes to: Mark A Hall, whose thesis on CFS can be found at: https://ml.cms.waikato.ac.nz/publications/1999/99MH-Thesis.pdf

# Model Build  <a name="model-title"></a>

Let's now build our Random Forest model:

```python
###############################################################
# prepare dataset for ML
###############################################################

# Re-Shuffle our data

df = shuffle(df_selected, random_state = 42)

# Check Class Balance

df["classe"].value_counts(normalize = True) # good class balance

###############################################################
# Split Input Variables and Output Variable
###############################################################

X = df.drop(["classe"], axis = 1)
y = df["classe"]

###############################################################
# Split Out Training and Test Sets -
# ensuring stratification evenly to classes
###############################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

###############################################################
# Model Training
###############################################################

clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features = 6)
clf.fit(X_train, y_train)

```

We've fit a Random Forest model containing 500 decision trees, using just 6 features calculated and selected using all of the raw sensor data.
Let's next see its Accuracy classifying movement on the 20% test set.

# Model Test / Training Accuracy  <a name="modelling-application"></a>

```python
# Assess model accuracy

y_pred_class = clf.predict(X_test) #default 50%. n of trees, that came to conclusion data point was in the positive/negative class
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use('bmh')
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

accuracy_score(y_test, y_pred_class)

# Feature Importance - based on mean decrease in the gini impurity score

feature_importance = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Permutation Importance (generally preferred) - the decrease seen when randomising each specific input variable

result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()
```
<img width="445" height="473" alt="image" src="https://github.com/user-attachments/assets/b1524ed0-95ef-4bbe-a159-026ded60eea0" />.

<img width="703" height="464" alt="image" src="https://github.com/user-attachments/assets/64ee9cb3-6e2e-4506-9dae-1833d3de956c" />.

<img width="713" height="465" alt="image" src="https://github.com/user-attachments/assets/f443c13e-683d-4645-930e-4ef30426de3b" />.

Accuracy: 0.99898 (!)

# Model Accuracy on New Data  <a name="accuracy-summary"></a>

```python
#
```

# Full Set Feature Performance  <a name="fully-featured"></a>

Let's assess how a model containing the full set of features identified using a much larger dataset (also using CFS) would go on the training subset that we received from the HAR team.

First, let's re-create the 17 features that were reported to be used in the HAR team's RF modelling.

### Recreate features found using CFS in the study <a name="fully-featured-recr"></a>
```python

import pandas as pd
import numpy as np

# import data
df = pd.read_csv("data/pml_training.csv")

# drop unnecessary columns

df.drop(["Unnamed: 0",
         "user_name",
         "raw_timestamp_part_1", 
         "raw_timestamp_part_2",
         "new_window"], axis = 1, inplace = True)

# Missingness is by column: calculated values are deleted from this set. 
# Count up missing values in each column

missing_counts = {col: df[col].isna().sum() for col in df.columns}

# In the data, we have complete data for a subset of columns (55), and missing data <= 20 obs for the remainder (100).

# View the dictionary
missing_counts

# Drop all columns with missing values above 20
features_with_miss = [col for col, count in missing_counts.items() if count >= 20]

df = df.drop(features_with_miss, axis=1)

#### View distribution of observations by num_window
counts = df.groupby('num_window').size()

#### 17 Features were selected by the paper using the method: ####
## A good feature subset contains features that are highly correlated with the class but uncorrelated with each other. ##
  
feat_seln_method_at = "https://ml.cms.waikato.ac.nz/publications/1999/99MH-Thesis.pdf"

# Let's recreate these measures on our training data using standard HAR (Human Activity Recognition) practice #

features = []
mags = []

#                               In BELT

# A) the mean and variance of the roll (all vars by sliding window-frame: num_window)

df['roll_belt_mean'] = (
    df.groupby('num_window')['roll_belt']
      .transform('mean')
)

df['roll_belt_var'] = (
    df.groupby('num_window')['roll_belt']
      .transform('var')   # sample variance (ddof=1)
)

features.extend([
    'roll_belt_mean',
    'roll_belt_var'
])


# B) maximum, range and variance of the belt accelerometer vector: 

# 1. Compute the vector magnitude for each row
df['accel_belt_mag'] = np.sqrt(
    df['accel_belt_x']**2 +
    df['accel_belt_y']**2 +
    df['accel_belt_z']**2
)

# 2. Compute window-level features and broadcast to each row
df['accel_belt_mag_max'] = (
    df.groupby('num_window')['accel_belt_mag']
      .transform('max')
)

df['accel_belt_mag_range'] = (
    df.groupby('num_window')['accel_belt_mag']
      .transform(lambda s: s.max() - s.min())
)

df['accel_belt_mag_var'] = (
    df.groupby('num_window')['accel_belt_mag']
      .transform('var')   # sample variance (ddof=1)
)

mags.extend(['accel_belt_mag'])

features.extend([
    'accel_belt_mag_max',
    'accel_belt_mag_range',
    'accel_belt_mag_var'
])


# C) variance of the gyro: 

# 1. Compute the vector magnitude for each row
df['gyros_belt_mag'] = np.sqrt(
    df['gyros_belt_x']**2 +
    df['gyros_belt_y']**2 +
    df['gyros_belt_z']**2
)

df['gyros_belt_mag_var'] = (
    df.groupby('num_window')['gyros_belt_mag']
      .transform('var')   # sample variance (ddof=1)
)

mags.extend(['gyros_belt_mag'])

features.extend([
    'gyros_belt_mag_var'
])

# D) variance of the magnetometer.

# 1. Compute magnetometer vector magnitude
df['magnet_belt_mag'] = np.sqrt(
    df['magnet_belt_x']**2 +
    df['magnet_belt_y']**2 +
    df['magnet_belt_z']**2
)

# 2. Compute window-level variance and broadcast to each row
df['magnet_belt_mag_var'] = (
    df.groupby('num_window')['magnet_belt_mag']
      .transform('var')   # sample variance (ddof=1)
)

mags.extend(['magnet_belt_mag'])

features.extend([
    'magnet_belt_mag_var'
])

#                                   In ARM:
    
# A) variance of the accelerometer vector
# B) maximum and minimum of the magnetometer

# A.
df['accel_arm_mag'] = np.sqrt(
    df['accel_arm_x']**2 +
    df['accel_arm_y']**2 +
    df['accel_arm_z']**2
)

df['accel_arm_mag_var'] = (
    df.groupby('num_window')['accel_arm_mag']
      .transform('var')   # sample variance (ddof=1)
)

mags.extend(['accel_arm_mag'])

features.extend([
    'accel_arm_mag_var'
])

# B.
df['magnet_arm_mag'] = np.sqrt(
    df['magnet_arm_x']**2 +
    df['magnet_arm_y']**2 +
    df['magnet_arm_z']**2
)

# 2. Compute window-level features and broadcast to each row

# maximum:
df['magnet_arm_mag_max'] = (
    df.groupby('num_window')['magnet_arm_mag']
      .transform('max')
)

# minimum:
df['magnet_arm_mag_min'] = (
    df.groupby('num_window')['magnet_arm_mag']
      .transform('min')
)

mags.extend(['magnet_arm_mag'])

features.extend([
    'magnet_arm_mag_max',
    'magnet_arm_mag_min'
])

#                                   In DUMBBELL:
    
# a) maximum of the acceleration
# b) variance of the gyro and 
# c) maximum and minimum of the magnetometer

#A
df['accel_dumbbell_mag'] = np.sqrt(
    df['accel_dumbbell_x']**2 +
    df['accel_dumbbell_y']**2 +
    df['accel_dumbbell_z']**2
)

df['accel_dumbbell_mag_max'] = (
    df.groupby('num_window')['accel_dumbbell_mag']
      .transform('max')
)

mags.extend(['accel_dumbbell_mag'])

features.extend([
    'accel_dumbbell_mag_max'
])

#B

df['gyros_dumbbell_mag'] = np.sqrt(
    df['gyros_dumbbell_x']**2 +
    df['gyros_dumbbell_y']**2 +
    df['gyros_dumbbell_z']**2
)

df['gyros_dumbbell_mag_var'] = (
    df.groupby('num_window')['gyros_dumbbell_mag']
      .transform('var')   # sample variance (ddof=1)
)

mags.extend(['gyros_dumbbell_mag'])

features.extend([
    'gyros_dumbbell_mag_var'
])


#C
df['magnet_dumbbell_mag'] = np.sqrt(
    df['magnet_dumbbell_x']**2 +
    df['magnet_dumbbell_y']**2 +
    df['magnet_dumbbell_z']**2
)

# maximum:
df['magnet_dumbbell_mag_max'] = (
    df.groupby('num_window')['magnet_dumbbell_mag']
      .transform('max')
)

# minimum:
df['magnet_dumbbell_mag_min'] = (
    df.groupby('num_window')['magnet_dumbbell_mag']
      .transform('min')
)

mags.extend(['magnet_dumbbell_mag'])

features.extend([
    'magnet_dumbbell_mag_max',
    'magnet_dumbbell_mag_min'
])


#                                   In GLOVE (Fore-arm):

# a) sum of the pitch
df['pitch_forearm_sum'] = (
    df.groupby('num_window')['pitch_forearm']
        .transform('sum')
)

features.extend([
    'pitch_forearm_sum'
])

# b) the maximum and minimum of the gyro
df['gyros_forearm_mag'] = np.sqrt(
    df['gyros_forearm_x']**2 +
    df['gyros_forearm_y']**2 +
    df['gyros_forearm_z']**2
)
# maximum:
df['gyros_forearm_mag_max'] = (
    df.groupby('num_window')['gyros_forearm_mag']
      .transform('max')
)

# minimum:
df['gyros_forearm_mag_min'] = (
    df.groupby('num_window')['gyros_forearm_mag']
      .transform('min')
)

mags.extend(['gyros_forearm_mag'])

features.extend([
    'gyros_forearm_mag_max',
    'gyros_forearm_mag_min'
])

print(features)
print(mags)
print(df.columns[55:81])
print(df.drop(mags, axis=1).columns)

df.drop(mags, axis=1, inplace = True)
```
Next, let's use these in a similar Random Forest model to the one we ran on our own data, and run it on this data.
Let's keep the number of decision trees comparable to our own model (500)

### Recreate Paper Model <a name="fully-featured-model"></a>

```python
# Recreate Paper's Model - A version of it using RF and many more decision trees

# Import packages

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

###############################################################
# Import sample data
###############################################################

# Run 101_Recreate_Paper_Features.py

###############################################################
# prepare dataset for ML
###############################################################

# Shuffle and trim data

df = shuffle(df, random_state = 42)
df = df[features+["classe"]]

# Class Balance

df.["classe"]value_counts(normalize = True) # good class balance

###############################################################
# Split Input Variables and Output Variable
###############################################################

X = df.drop(["classe"], axis = 1)
y = df["classe"]

###############################################################
# Split Out Training and Test Sets
###############################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

###############################################################
# Model Training
###############################################################

clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features = 17)
clf.fit(X_train, y_train)

# Assess model accuracy

y_pred_class = clf.predict(X_test) #default 50%. n of trees, that came to conclusion data point was in the positive/negative class
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use('bmh')
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

accuracy_score(y_test, y_pred_class)

# Feature Importance - based on mean decrease in the gini impurity score

feature_importance = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Permutation Importance (generally preferred) - the decrease seen when randomising each specific input variable

result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()
```
<img width="439" height="474" alt="image" src="https://github.com/user-attachments/assets/02e87961-9581-4e16-973f-d69df47d0f6e" />.

<img width="720" height="472" alt="image" src="https://github.com/user-attachments/assets/5a427523-b5ab-4b22-bf8e-af28dabc9632" />.

<img width="733" height="463" alt="image" src="https://github.com/user-attachments/assets/48ee91d7-fb76-4516-b136-a953912358ad" />.

Accuracy: 1.0

This training set has not challenged the model containing 17 features. With 500 Decision Trees, it achieved perfect prediction!

# Growth & Next Steps  <a name="growth-next-steps"></a>


