---
layout: post
title: Feature Selection For Classification Accuracy
image: "/posts/classification-title-img.png"
tags: [Human Activity Recognition, Machine Learning, Classification, Python]
---

Our client, a Human Activity Recognition (HAR) research team, wants to utilise Machine Learning to predict a movement activity type!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Vector and Window Calculations](#veccalc-title)
- [04. Linear SVC + RFECV Feature Selection](#linSVCRFECV-title)
- [05. CFS Feature Selection](#cfs-title)
- [06. Model Build](#model-title)
- [07. Model Test/Training Accuracy](#accuracy-summary)
- [08. Full Set Feature Performance](#fully-featured)
- [09. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, an Human Activity Recognition (HAR) study team, ran a study using HAR sensors to provide real-time feedback of weightlifting technique.  They want to classify a subject's movement on a bicep curl as being either 'correct technique', or as one of four common technique errors. 

For this, they collected data on six subjects using sensors strapped to their belt, forearm, arm, and a held dumbbell. The subjects new to bicep curls, performed a biceps curl with correct and common incorrect beginner technique according to a weight-lifting coach. 

The HAR team would like to classify the movement that the next subject does, and provide this as feedback to them quickly, to guide exercise coaching.

Based upon the data they've collected, we will look to understand the *prediction accuracy* of a model built to classify the lift, and test this on a hold out subject as a 'new' lifter, in reality one that has performed their lifts earlier under the same conditions.

This would provide the HAR team with a model to develop a visual system that can provide new lifters in their lab with technique cues, to quickly improve their movement performance on the exercise. 

The HAR team has access to a much larger set of measures and can train their own models on this data, and they have provided us with access to a small subset of their training data for us to use in our Machine Learning build.

Let's use Machine Learning to take on this classification task!

<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from each of the 6 subjects, gathering key sensor data that may help predict *class* of the movement.

Within our dataset from the 6 subjects, we found that 28.4% of sensor data across all collection windows indicated a correct lift (class A), and 71.6% came from incorrect lifts. 
Of the incorrect lifts, it was split by error type B-E: 19.3% : 18.4% : 17.4% : 16.4%. This tells us that while the data isn't perfectly balanced at 20:20:20:20:20 across lift class, it isn't *too* imbalanced either. Interestingly, 2 subjects contribute to the high proportion of class A in our full data.

For this project, we need to split our training and testing data in a way that avoids 'data leakage' between test and training sets. Typically, we'd use an 80:20 (train:test) split using all clean candidate data, but to avoid data leakage in this case, we'll need to take an entire subject's data out of the training set. This is because we want to use summaries across time-windows as features in our models. If we were to split all of our data after calculating these, we would artifically increase the prediction accuracy of all models, because we'd have pre-calculated summary statistics simply predicting themselves across the train:test split.  We have just 6 subjects, so we reassess the class breakdown by subject and select a suitable holdout that will stress-test our models most.
 
We use the 5 subjects with the most balanced classes to train our model, and the 6th as a test case. This provides the greatest challenge to the model and closest approximation possible to testing our model on a new case. Statistically, the subject with the largest average L1 distance from the others is the ideal holdout.

As we are predicting a class in a HAR context, with a dense feature space, we'll test Random Forest modelling approaches as per industry standard. 

For each model, we will import the data in the same way, but will need to pre-process it based upon different strategies that give feature sets to our Random Forest algorithm.  We will train & test a model using each feature selection approach, to provide optimal performance, and then measure this predictive performance based on accuracy score in classifying the lift of the hold-out subject that was most different to the training subjects.

<br>
<br>

### Results <a name="overview-results"></a>

The goal for the project was to convert sensor data into a useful predictor that could identify the type of a biceps curl as:
- Correctly performed (Class A/0)
- Error 1: Throw the elbows to the front (Class B/1)
- Error 2: Lift the dumbbell only halfway (Class C/2)
- Error 3: Lower the dumbbell only halfway (Class D/3)
- Error 4: Throw the hips to the front (Class E/4)

Our model trained on the data, using 4 features found by Correlation-based Feature Selection (CFS), calculated from this same data, gave an accuracy score of: 50 %. 

A model using 17 features found by CFS calculated from the full research data found relationships between sensor channels and movement, but also gave an accuracy score around 50%. 

A feature selection strategy using industry standard LinearSVC + RFECV selection proved too time-consuming to run, but may have improved on the modelling results found using the CFS selection strategy as shown. 

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

Our 4‑feature model performed poorly on an independent subject, showing that data from our 5 training subjects is not rich enough to predict movement in a 6th.
Our HAR team 17-feature model improved on this somewhat, but not meaningfully.

Our data preparation is robust, but our training set is very small, based on too few movements to generalise effectively to provide feedback in real-time using these approaches. There are more rigorous summary statistics than window-level summaries to use - better aligned with the science of HAR, and more powerful models to try.

A gold-standard HAR model is needed: A Domain-Adversarial Neural Network (DANN) for HAR. 
We'll next use Copilot to help code this pipeline in our Spyder environment using PyTorch! This, if successfully built, could be expected to dramatically increase between-subject prediction!

<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be predicting the categorical *classe* metric from the subset *training* sample adapted from the HAR study database.

The key variables hypothesised to predict this will come from the sensor data contained in this set.

We calculate vector statistics from the repeat sensor collections in windows of different length (0.5 to 2.5 seconds) during the lift. For this training data, a lifter had completed 10 repetitions of a bicep curl exactly according to instructions specified by a weight lifting coach. 
The coach instructed to lift using a specified technique (Class A), or to perform one of the following errors:
- Error 1: Throw the elbows to the front (Class B)
- Error 2: Lift the dumbbell only halfway (Class C)
- Error 3: Lower the dumbbell only halfway (Class D)
- Error 4: Throw the hips to the front (Class E)

After some data pre-processing in Python, we have a dataset for modelling that contains the following fields...

<br>
<br>

| **Variable Names** | **Variable Type** | **Description** |
|---|---|---|
| classe | Dependent | A categorical variable showing the lift class |
| num_window | Independent | The time window in which the sensor data was collected - not used for modelling  |
| roll/pitch/yaw_belt/arm/forearm/dumbbell | Independent | The roll,pitch or yaw derived for a sensor collection in the time window at one of belt/arm/forearm/dumbbell sensors |
| total_accel_belt/arm/dumbbell/forearm | Independent | The total accelerometer data in 3 dimensions x,y,z at the belt, arm, dumbbell or forearm sensor |
| gyros/accel/magnet_belt_x/y/z | Independent | The gyrometer, accelerometer and magnetometer belt sensor readings in x/y/z dimensions: forwards, sideways, upwards  |
| gyros/accel/magnet_arm_x/y/z | Independent | The gyrometer, accelerometer and magnetometer armband sensor readings in x/y/z dimensions: forwards, sideways, upwards |
| gyros/accel/magnet_dumbbell_x/y/z | Independent | The gyrometer, accelerometer and magnetometer dumbbell sensor readings in x/y/z dimensions: forwards, sideways, upwards |
| gyros/accel/magnet_forearm_x/y/z | Independent | The gyrometer, accelerometer and magnetometer forearm sensor readings in x/y/z dimensions: forwards, sideways, upwards |

<br>
# Modelling Overview  <a name="modelling-overview"></a>

We will build a model that looks to accurately classify *classe*, based upon the sensor measures listed above.

If that can be achieved, we can use this model to predict movement type for future movements (future weighted bicep curl movements).  This information could be used to provide feedback to a lifter, guiding correct movement technique.

As we are predicting a categorical output using granular data from many inputs in a HAR environment, we use a two step approach: 1) calculate vector statistics from the raw sensor data and 2) pass these onto one of two feature selection approaches:

* LinearSVC + RFECV - an industry standard approach to HAR using sensor data.
* Correlation-Based Feature Selection (CFS) - an elegant selector that trims highly correlated features.

From there, we include the identified features in a Random Forest model consisting of 500 Decision trees, to make our movement class predictions.

<br>
# Vector and Window Calculations <a name="veccalc-title"></a>

We utilise the numpy and pandas libraries within Python to compute vector magnitudes for all sensors. The code sections below are broken up into 3 key sections:

* Data Import
* Test and Training Split by subject
* Data Preprocessing - vector calculations, window-level statistics.

<br>
### Data Import <a name="veccalc-import"></a>

We import the raw training measures data.  We ensure we remove id and unnecessary columns, and keep only sensor and class data.

We also investigate the class balance of our dependent variable - which is important when assessing classification accuracy.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("data/pml_training.csv")

# drop unnecessary columns - keeping subject name to split our test/train by subject.

df.drop(["Unnamed: 0",
         "raw_timestamp_part_1", 
         "raw_timestamp_part_2",
         "cvtd_timestamp",
         "new_window"], axis = 1, inplace = True)

# Missingness is by column: calculated values are deleted from this set. 
# Count up missing values in each column

missing_counts = {col: df[col].isna().sum() for col in df.columns}

# View the dictionary
missing_counts
# In the data, we have complete data for a subset of columns (55), and missing data <= 20 obs for the remainder (100).

# Drop all columns with missing values above 20
features_with_miss = [col for col, count in missing_counts.items() if count >= 20]
df = df.drop(features_with_miss, axis=1)

df["classe"].value_counts(normalize=True).mul(100).round(2)

```

<br>
From the last step in the above code, we see that **28.4% of movements were class A and 72% were an error class B-E: 19.3%, 18.4%, 17.4%, 16.4%, respectively**.  This tells us that while the data isn't perfectly balanced at 20:20:20:20:20, it isn't *too* imbalanced either.

Since we'll be identifying a suitable holdout subject by class proportion, let's view this by subject.

<br>
### Test and Training Split by Subject <a name="veccalc-split"></a>

Since all data came from 6 subjects, and we want to use one as our test case, let's assess the breakdown by subject name and then let's calculate class proportions per subject.

```python
class_props = (
    df.groupby("user_name")["classe"]
      .value_counts(normalize=True)
      .unstack(fill_value=0)
)
```
We get the proportions:
<br>

| **classe** | **A** | **B** | **C** | **D** | **E** |
|---|---|---|---|---|---|
| **adelmo**   | 0.2993 | 0.1994 | 0.1927 | 0.1323 | 0.1763 |
| **carlitos** | 0.2680 | 0.2217 | 0.1584 | 0.1562 | 0.1957 |
| **charles**  | 0.2542 | 0.2107 | 0.1524 | 0.1816 | 0.2011 |
| **eurico**   | 0.2818 | 0.1928 | 0.1593 | 0.1896 | 0.1765 |
| **jeremy**   | 0.3460 | 0.1437 | 0.1917 | 0.1534 | 0.1652 |
| **pedro**    | 0.2452 | 0.1935 | 0.1912 | 0.1797 | 0.1904 |

<br>
Let's visualise the counts that give these proportions:

```py

# Plot histograms of class counts per subject

subjects = df["user_name"].unique()
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, subject in zip(axes, subjects):
    subset = df[df["user_name"] == subject]
    ax.hist(subset["classe"], bins=5, edgecolor="black")
    ax.set_title(f"Subject: {subject}")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
```

That code gives the plot:
<br>

<img width="726" height="414" alt="image" src="https://github.com/user-attachments/assets/b5f7eaca-8d29-4493-bba1-1ec3fa758459" />

<br>
We can see that the subjects in the right-hand column are contributing most to our class discrepancy at the total level, but in general, these classes are fairly well balanced. Let's now find the least balanced subject from these to use as our modelling test subject! We can see from this plot that it will be either Jeremy, or Adelmo. But we can't tell which it is, just by looking.

We can find out using an 'L1 distance' matrix which can then tell us the mean distance across all classes, between subjects. 

The L1 distance is a statistical measure that can tell us **vector-level** differences - here taking into account *all* classes for a subject - and comparing this to all classes for another subject. 

We can then select the subject with the highest mean L1 distance, as an emprically supported choice for 'most different' subject, to hold out as our test subject. The selected subject will be the one in our data that will be most different from our training set, and therefore most likely to represent a **new** subject performing the same movement. 

Let's compute the subject-wise L1 distance matrix.

```py

distance_matrix = class_props.apply(
    lambda row: np.abs(class_props.sub(row)).sum(axis=1),
    axis=1
)
```
We'll calculate the mean subject-subject class differences for each row and add this to the table.

```py
distance_matrix["mean_distance"] = distance_matrix.mean(axis=1)
```

Let's see the output:
<br>

| **user_name** | **adelmo** | **carlitos** | **charles** | **eurico** | **jeremy** | **pedro** | **mean_distance** |
|---|---|---|---|---|---|---|---|
| **adelmo**   | 0.0000 | 0.1312 | 0.1707 | 0.1151 | 0.1355 | 0.1231 | 0.1126 |
| **carlitos** | 0.1312 | 0.0000 | 0.0615 | 0.0961 | 0.2224 | 0.1126 | 0.1040 |
| **charles**  | 0.1707 | 0.0615 | 0.0000 | 0.0848 | 0.2619 | 0.0775 | 0.1094 |
| **eurico**   | 0.1151 | 0.0961 | 0.0848 | 0.0000 | 0.1932 | 0.0929 | 0.0970 |
| **jeremy**   | 0.1355 | 0.2224 | 0.2619 | 0.1932 | 0.0000 | 0.2025 | 0.1692 |
| **pedro**    | 0.1231 | 0.1126 | 0.0775 | 0.0929 | 0.2025 | 0.0000 | 0.1014 |

<br>

and use it to select our most distinct subject:

```py

most_distinct_subject = distance_matrix["mean_distance"].idxmax()
print("Most distinct subject (ideal holdout):", most_distinct_subject)

```

Our most distinct subject is: jeremy!

Let's recap. To ensure realistic evaluation of cross‑subject generalisation in the Human Activity Recognition (HAR) model, we computed a subject‑wise class‑distribution distance matrix using the L1 distance between class‑proportion vectors for each subject. This quantified how different each subject’s class distribution was from every other subject.
After calculating the row‑wise mean distance for each subject, 'jeremy' exhibited the highest average distributional distance (0.1692), substantially larger than all other subjects (range: 0.0970–0.1126). This indicates that Jeremy’s movement patterns and class proportions are, on average, the most distinct in the dataset.

Selecting Jeremy as the holdout test subject provides the most stringent and realistic assessment of model generalisation to unseen individuals available in our data. 

This also means that the remaining five subjects form a *comparatively* homogeneous training set. Selecting train/test splits by individual like this will avoid data leakage into the test set, reduce subject‑specific bias, and improve the stability of feature selection and model fitting.

Jeremy is the empirically justified choice for the holdout set because he is the most distributionally unique subject.

OK, let's have a look at our train to test split, using Jeremy as our model test subject.
<br>

| **classe** | **train proportion** | **test proportion** |
|---|---|---|
| **A** | 0.2715 | 0.3460 |
| **B** | 0.2039 | 0.1437 |
| **C** | 0.1708 | 0.1917 |
| **D** | 0.1661 | 0.1534 |
| **E** | 0.1877 | 0.1652 |
| **ALL** | 0.8266 | 0.1734 |

<br>
The training set exhibits a relatively balanced distribution across the five lift classes, with no single class dominating. In contrast, the test set shows a noticeably different pattern: Jeremy has a substantially higher proportion of class A observations and a lower proportion of class B compared with the pooled training subjects. We saw this in the histogram, and now we know that this has been the deciding factor between Jeremy and Adelmo.

This imbalance is important because it reflects real‑world deployment conditions: new users often perform movements differently, leading to shifts in class frequencies and sensor signatures. 

Ok, we have found the test/training split. We also have a nice, approximately 80:20 split in train:test sets overall, close to a typical breakdown for developing Random Forest models. 

<br>
### Data Preprocessing <a name="veccalc-preprocessing"></a>

For 3D Sensor data, we have certain data preprocessing steps that need to be addressed, including:

* Dealing with missing values in the data
* Summarising variables on axes to vectors

<br>
##### Missing Values

There were no missing values in the raw sensor data, as this subset of the study data had been pre-cleaned. So we will just move on to the next step.

We had removed any columns containing missing data on data import, and will create new calculated fields using the raw sensor data. 
Rarely, doing this may *create* some missing data which will be re-addressed after all fields are ready for feature selection. 

<br>
##### Summarise variables on axes to vectors and group by time window

In the next code block we do four things:
1. We group our data into vector groups by sensor, then
2. Calculate their vector magnitudes,
3. Define the summary statistics that we wish to calculate to use as features in the random forest model along with the grouping we will use to aggregate to these, and
4. Run our sensor data through this to create these new variables at the vector level.

Once we have done this, we can perform feature selection using these window summaries as features, which will hopefully contain all predictive sensor information, in a much reduced feature space.

We need to be careful here. To do this, we are assuming that our test set will also contain multiple observations from the same time-window. We know this is the case, as we have split training and testing at the subject level.

Our test set contains 20% of the data, stratified by class, which gives hundreds of observations from only a few subjects and curl repetitions, so there is likely to be enough data in our test set to produce these summaries here. 

Vector magnitudes are calculated from 3 dimensions (front, sideways, upwards) as:

$$\mathrm{magnitude}=\sqrt{x^2+y^2+z^2}$$

Let's create time-window level summaries using the numpy and pandas libraries in Python. We'll create time-window means, variances, maximums and minimums, ranges, and totals, using vector magnitudes for the 3D accelerometer, magnetometer and gyrometer data.

Since we’ve separated the data by subject, and each subject has their own unique num_window sequence, we compute the window‑level summary features separately within the training and test sets. The calculations are done by num_window only — we don’t need to consider which subject the windows came from. 

<br>
```python

## Candidate feature engineering - A function to create vector magnitudes and summaries for test and train sets independently

def make_features(df, num_window_col="num_window"):
    """
    Compute vector magnitudes and window-level summary statistics.
    Applies the same feature engineering to any input DataFrame.
    """

    # --- 1. Vector magnitude definitions ---
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

    # --- 2. Compute vector magnitudes ---
    for name, cols in sensor_groups.items():
        df[f"{name}_mag"] = np.sqrt((df[cols]**2).sum(axis=1))

    # --- 3. Summary statistics to compute ---
    stats = {
        "mean": "mean",
        "var": "var",
        "max": "max",
        "min": "min",
        "range": lambda s: s.max() - s.min(),
        "sum": "sum"
    }

    # --- 4. Identify columns to summarise ---
    candidate_cols = [
        col for col in df.columns
        if any(key in col for key in ["roll", "pitch", "yaw", "_mag"])
    ]

    # --- 5. Compute window-level summaries ---
    for col in candidate_cols:
        for stat_name, func in stats.items():
            df[f"{col}_{stat_name}"] = (
                df.groupby(num_window_col)[col].transform(func)
            )

    # --- 6. Drop window ID (not a feature) ---
    df = df.drop(columns=[num_window_col])

    return df

train_df = make_features(train_df)
test_df  = make_features(test_df)

```

With these summaries available in both sets, we're now ready to select the most predictive of these as our model features!

We'll use just our training set for this, leaving Jeremy's data aside until our model is ready to test.

<br>
##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to our Machine Learning task.  It can be a very important addition or at least, consideration, in certain scenarios. For our task in an HAR context, feature selection is extremely important. The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train, and faster to make predictions
* **Explainability** - understanding & explaining outputs for stakeholders becomes much easier

There are many, many ways to apply Feature Selection.  These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

##### RFECV using LinearSVC

We attempt an industry standard variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* using a *Linear Support Vector Classifier* where we split the data into many "chunks" and iteratively train & validate these linear classifer models on each "chunk" separately.  This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was.  LinearSVC learns a linear decision boundary that distinguishes the five exercise movement classes using the accelerometer, gyroscope, and magnetometer features, combined with our calculated vector and summary statistics features.

RFECV then uses the model’s coefficients to identify which of all of these features truly contribute to class separation and recursively removes the weakest ones.
One downside of this approach is the long run-times that LinearSVC can have, as we'll see!

##### CFS

We compare this to an approach to feature selection called Correlation-Based feature Selection (CFS), which is also used in HAR.
CFS chooses the smallest set of features that are highly correlated with the class while being minimally correlated with each other. The core of this approach uses the merit function:

$\mathrm{Merit_{\mathnormal{S}}}=\frac{k\cdot \bar {r}_{cf}}{\sqrt{k+k(k-1)\bar {r}_{ff}}}$

Multicollinearity will occur when two or more input variables are *highly* correlated with each other, it is a scenario we attempt to avoid as in short, while it won't necessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics that describe how well the model is performing, and how much effect each input variable is truly having. CFS does a good job of reducing multicollinearity since it assigns more merit (as defined above) to features that are not correlated.

We'll code the CFS approach in Python and see how it works out in our model predictions!

Let's first attempt the industry standard approach to feature selection in HAR tasks and see how far we get.

<br>
# LinearSVC + RFECV <a name="linSVCRFECV-title"></a>

We will again utilise the scikit-learn library within Python to select features for our model.
The code section below continues with our prepared dataframe containing the suite of vector summary statistics.

Our first approach uses a simple linear machine‑learning model (LinearSVC) to figure out which features are genuinely useful for predicting the *classe* activity labels. A wrapper method (RFECV) then repeatedly tests smaller and smaller sets of features to find the smallest set that still gives strong performance. Using a careful, step‑by‑step trimming process, the model learns which features matter most, RFECV then removes the weakest one, and then the model is retrained to see how performance changes. This repeats until the best subset is found.

<br>
### Implementing LinearSVC + RFECV <a name="linSVCRFECV-select"></a>

Continuing with our train_df object containing our new vector features, we:
- Import the packages we'll need: LinearSVC, RFECV and StratifiedKFold, along with Pipeline to pipe our standardisation step, and StandardScaler to apply standardisation prior to fitting the Linear SVC model.
- Instantiate our Linear SVC estimator in this pipeline, that scales all candidate features prior to linear modelling - using standardised scaling.
- run cross-validated random forest estimation.

We need to set a few model hyperparameters specifying what we want the models to do: 
The LinearSVC model produces a simple linear formula that assigns a weight to each feature, making it easy to see which features help the model make decisions. For this, the regularization setting C=1.0 keeps the model balanced so it doesn’t rely too heavily on any single feature, while dual=False makes the training faster for datasets with many samples, and max_iter=5000 ensures the model has enough time to settle on a stable solution. 

RFECV then uses this model to perform feature selection by repeatedly removing the *single* least important feature (step=1) and checking how well the model performs each time. It evaluates each feature set using 5‑fold stratified cross‑validation, which means the data is split into five parts in a way that preserves the class balance, giving a fair and reliable estimate of accuracy. 'importance_getter' helps the RFECV() function in Python to understand where to find features when they've reached it through a pipeline.

The process chooses the feature set that gives the best accuracy score, and n_jobs=-1 simply means the computer uses all its available processing power to speed up this repeated testing. Together, these steps carefully identify the smallest set of features that gives strong predictive performance.

#### A note on Feature Scaling Rationale <a name="linSVCRFECV-scaling"></a>

Because the feature set combines raw sensor magnitudes with window‑level statistics such as means, variances, ranges, and sums, the resulting variables span very different numeric scales. LinearSVC is sensitive to these scale differences: its coefficient‑based optimisation can be dominated by features with larger numeric ranges, which in turn misleads RFECV during recursive feature elimination. 
Standardising all candidate features ensures that each variable contributes on a comparable footing, allowing the model to learn stable coefficients and enabling RFECV to rank features based on genuine predictive signal rather than raw magnitude. This produces a more reliable and interpretable feature‑selection process and improves the model’s ability to generalise.

#### A note on Outlier Correction <a name="linSVCRFECV-scaling"></a>

Window summaries calculated earlier will have absorbed most outliers naturally. Any remaining outliers should be kept as they signal normal behaviour during the movement, and this behaviour would be present in the kind of future movements we would be predicting. LinearSVC+RFECV is also robust to any remaining outliers!  

Let's specify all of this in Python, and attempt to train our model: 

<br>
```python

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearSVC(C=1.0, dual=False, max_iter=5000))
])

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    n_jobs=-1,
    importance_getter='named_steps.clf.coef_'
)

rfecv.fit(X, y)

```

<br>
The full LinearSVC + RFECV run was still churning away after an hour on a laptop, and there’s a chance it would hit memory limits before finishing. It **would** eventually produce a feature set, but the runtime alone is a good reason to try something faster.
To speed these models up, we can relax a few settings: reduce max_iter so LinearSVC doesn’t spend as long converging, increase the RFECV step size so it removes features in larger chunks, and drop cross‑validation from 5 folds to 2. These changes make the process dramatically quicker, but they also make the feature search a bit less precise. We’ll still uncover useful predictors, but the selection won’t be as fine‑tuned as the full version.

With that in mind, it makes sense to reset the hyperparameters and rerun the feature selection using these faster settings.


<br>
```python

estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearSVC(C=1.0, dual=False, max_iter=2000))
])

rfecv = RFECV(
    estimator=estimator,
    step=20,
    cv=StratifiedKFold(2),
    scoring='accuracy',
    n_jobs=-1,
    importance_getter='named_steps.clf.coef_'
)

rfecv.fit(X, y)

```



<br>
This has completed in 10 minutes, and has identified that the optimum number of features is 54. We can see this optimum in the next plot:

<img width="727" height="389" alt="image" src="https://github.com/user-attachments/assets/2d777b74-3963-4134-bfa6-f9cddaffcf2a" />

However, even with the faster settings, RFECV still takes a long time to run on a laptop — too slow for a workflow to support real‑time technique feedback for a new lifter. 

We’ve already relaxed the hyperparameters, which means we’re trading some precision in feature selection for speed. With the original, more thorough settings, we’d expect RFECV to settle on about 10–25 predictive features, which is typical for this kind of HAR data.

Using the faster configuration, the results are very interesting! Only 7 of the original sensor‑level features remain, while 47 come from the engineered vector‑summary features. That tells us the window‑level summaries are capturing most of the useful signal, though not all of it. The raw sensor channels still contribute a small amount of unique information.

We could take this reduced feature set and rerun RFECV with more fine‑grained settings to refine it further, but at this point the runtime becomes impractical for our use case.

So rather than pushing deeper into the heavy HAR‑style wrapper methods, it makes sense to try something more elegant and lightweight. Let's try a good alternative, Correlation‑Based Feature Selection (CFS). This filter method looks for features that are highly correlated with the target while being minimally correlated with each other.

We’ll implement a simple CFS routine in Python using NumPy in four steps:

1. define a merit function
2. add features as long as they improve that merit
3. standardise the resulting feature set
4. produce a clean modelling dataframe based on the selected subset

<br>
# CFS Feature Selection <a name="cfs-title"></a>

Let's build our CFS function. 

This code defines a custom Correlation‑based Feature Selector (CFS) that works directly in scikit‑learn. It searches for a small set of features that are both highly correlated with the target and not strongly correlated with each other. The selector evaluates candidate subsets using the classic CFS “merit” formula, adds features one at a time when they improve the merit score, and stops when no further improvement occurs. After fitting, it stores the indices of the chosen features and the transform method returns a reduced version of the dataset containing only those selected columns.

<br>
```python

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
```
<br>
With the function now built, let's standardise our candidate feature set and run it through our CFS process.

First, we standardise our data so that all candidate features are scaled, having their original values mapped to numbers between -1 and 1, with their mean at 0.
Then, we run our CFS Selector function to apply CFS and choose the most independently predictive features.

<br>
```python

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

selector = CFSSelector().fit(X_scaled, y)

selected_features = X.columns[selector.selected_indices_]
```
<br>
Let's view our resulting feature set:
<br>
```python
print("Number of selected features:", len(selected_features))
print("\nSelected features:")
for f in selected_features:
    print("-", f)
```
<br>
The Number of selected features is (only!): 4

These are:
- belt_accel_mag_range
- pitch_forearm_min
- roll_belt_range
- arm_mag_mag_range

Importantly, this was found in just a few seconds!

We've successfully run CFS on the 'training' set, the smaller set of data provided to us by the HAR team. In reality, using their full dataset, the HAR team would find 17 features using CFS, which is in the likely range of features that LinearSCV+RFECV might find if it completed successfully when prioritising discrimination over speed as we initially tried.

The 4 features here are the top subset in the feature space, and we can compare these with the 54 we found using LinearSVC+RFECV.
Again, vector summaries of the sensor data are showing as features predicting movement class. All four features are vector summaries (a minimum, and three ranges), and none are on the level of the original sensor channels.

Let's build our model train and test data set using our top 4 features...
<br>
```python

# Convert selected feature names to a list
selected_feature_list = list(selected_features)

# Build the final dataframe
train_df_selected = pd.DataFrame({
    "classe": y
})

# Add each selected feature column
for feat in selected_feature_list:
    train_df_selected[feat] = X[feat].values

print("\nFinal dataframe shape:", train_df_selected.shape)
print(train_df_selected.head())

test_df_selected = pd.DataFrame()

for feat in selected_feature_list:
    test_df_selected[feat] = test_df[feat].values

print("Test dataframe shape:", test_df_selected.shape)
print(test_df_selected.head())

```
<br>
We have created two new dataframes: *train_df_selected* containing the features that are most correlated with classe, that were the least correlated with eachother in training, and *test_df_selected*, containing the same features in Jeremy's test data.

Credit for this approach goes to: Mark A Hall, whose thesis on CFS can be found at: https://ml.cms.waikato.ac.nz/publications/1999/99MH-Thesis.pdf

CFS has reduced the feature space, and identified some top features to include in a class prediction model. Let's move on and build a predictor.

<br>
<br>
# Model Build  <a name="model-title"></a>

Let's now build our Random Forest model, following the steps in the code below...

```python

## 1. Prepare dataset for ML

# Re-Shuffle our data for training

train_df_selected = shuffle(train_df_selected, random_state = 42)

# Check Class Balance

train_df_selected["classe"].value_counts(normalize = True)


## 2. Split Input Variables and Output Variable to create test and training sets

X_train = train_df_selected.drop(["classe"], axis = 1)
y_train = train_df_selected["classe"]
X_test = test_df_selected
y_test = test_df["classe"]


## 3. Model Training - train our model!

clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features = 4)
clf.fit(X_train, y_train)

```
<br>
We've fit a Random Forest model containing 500 decision trees, using just the 4 features selected by CFS from all of the raw sensor data, and our created vector summaries.
We're now ready to assess its accuracy in classifying movement on the test set - Jeremy's data!

<br>
<br>
# Model Test / Training Accuracy  <a name="modelling-application"></a>
<br>
### Model Performance

Based on the values of our 4 features, each of the 500 decision trees in the Random Forest independently predicts one of the 5 movement classes for every row in the test set. A decision tree chooses thresholds during training by testing many possible split points and selecting the one that most reduces Gini impurity. During prediction, the tree simply compares each feature value to these learned thresholds to decide which branch to follow. Each tree routes the sample down its branches until it reaches a leaf node, and the class stored in that leaf becomes that tree’s ‘vote’. These votes represent the class each tree considers most likely, given the feature values it has seen. The Random Forest then counts all votes across the 500 trees, and the class with the most votes becomes the model’s final prediction for each row in the test set. 

We output an object y_pred_prob that views the probabilities that each decision tree would arrive at each of the 5 outcomes for each row in the test set. 

We then build a confusion matrix to show the model predictions vs actual class values, and output metrics on the importance of each feature to the predictive power of the model.

```python
# Assess model accuracy

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

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
```
<br>
<img width="425" height="466" alt="image" src="https://github.com/user-attachments/assets/a6b60b5e-069c-4509-98ba-3b6123960b75" />
<br>
The confusion matrix shows poor prediction. The Accuracy of this model is: 0.5097. 

CFS has reduced the feature space to a point where the model cannot usefully classify a new subject's movements!
The model overwhelmingly predicts A, B, or C, with classes D and E under‑represented.
This indicates that the four selected features do not capture the complexity needed to separate the five movement classes for a new subject

The CFS‑4 feature subset produced limited performance on the held‑out subject, with substantial confusion across all five classes. The model tended to collapse predictions into classes A, B, and C, with classes D and E frequently misclassified. 

A model trained on more of the HAR data might achieve better results, still using CFS. 

<br>
### Feature Importance
Our Random Forest is an ensemble model, made up of 500 Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of our 4 input variables available at each potential split point.

We have a weak model, but if it had been accurate, because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model.

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models' predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure importance by asking: How much would accuracy decrease if a specific input variable was removed or randomised?

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this. The first, often just called Feature Importance is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the gini impurity score (for a Classification problem) was before the split was made, and compare this to the gini impurity score after the split was made. We can take the average of these improvements across all Decision Trees in the Random Forest to get a score that tells us how much better we’re making the model by using that input variable.

If we do this for each of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called Permutation Importance cleverly uses some data that has gone unused at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as Out of Bag observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the Out of Bag observations are gathered and then passed through. Once all of these observations have been run through the Decision Tree, we obtain a classification accuracy score for these predictions.

In order to understand the importance, we randomise the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score. The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

Permutation Importance is often preferred over Feature Importance which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let's put them both in place, and plot the results...

```python
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

That code gives us the below plots for Feature Importance and Permutation Importance.

<br>
<img width="727" height="408" alt="image" src="https://github.com/user-attachments/assets/1882684d-0d65-4d49-8d3c-e7abbf1c4145" />
<br>
<img width="730" height="393" alt="image" src="https://github.com/user-attachments/assets/3a52cf91-2478-4b7c-9aed-906873764687" />

Feature‑importance analysis showed that the model relied heavily on one or two features, particularly arm- and belt‑based magnitude ranges, while permutation importance revealed that pitch‑forearm minima were actually the most essential predictor. This mismatch indicates redundancy and instability within the four‑feature set. 

Overall, while CFS‑4 offers interpretability and compactness, it lacks the representational capacity required for robust subject‑independent classification. Our training data had no hope of predicting an independent lifter's movements, if using CFS feature-selection trained only on our own data.

<br>
<br>
# Full Set Feature Performance  <a name="fully-featured"></a>

Let's assess how a model containing the full set of features identified using a much larger dataset (also using CFS) would perform if trained on our training subset and tested on Jeremy's data.

First, let's re-create the 17 features that were reported to be used in the HAR team's own RF modelling.

<br>
### Recreate features found using CFS in the study <a name="fully-featured-recr"></a>

```python

import numpy as np
import pandas as pd

train_df["num_window"] = df.loc[train_df.index, "num_window"]
test_df["num_window"]  = df.loc[test_df.index, "num_window"]

# Work on a copy of the training set
df_c = train_df.copy()

# -----------------------------
# Helper functions
# -----------------------------

def mag(df, cols):
    """Compute vector magnitude for 3-axis sensor."""
    return np.sqrt(df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2)

def win(df, col, func):
    """Apply window-level function and broadcast."""
    return df.groupby("num_window")[col].transform(func)

# -----------------------------
# 1. BELT features
# -----------------------------

df_c["roll_belt_mean"] = win(df_c, "roll_belt", "mean")
df_c["roll_belt_var"]  = win(df_c, "roll_belt", "var")

df_c["accel_belt_mag"] = mag(df_c, ["accel_belt_x","accel_belt_y","accel_belt_z"])
df_c["accel_belt_mag_max"]   = win(df_c, "accel_belt_mag", "max")
df_c["accel_belt_mag_range"] = win(df_c, "accel_belt_mag", lambda s: s.max() - s.min())
df_c["accel_belt_mag_var"]   = win(df_c, "accel_belt_mag", "var")

df_c["gyros_belt_mag"] = mag(df_c, ["gyros_belt_x","gyros_belt_y","gyros_belt_z"])
df_c["gyros_belt_mag_var"] = win(df_c, "gyros_belt_mag", "var")

df_c["magnet_belt_mag"] = mag(df_c, ["magnet_belt_x","magnet_belt_y","magnet_belt_z"])
df_c["magnet_belt_mag_var"] = win(df_c, "magnet_belt_mag", "var")

# -----------------------------
# 2. ARM features
# -----------------------------

df_c["accel_arm_mag"] = mag(df_c, ["accel_arm_x","accel_arm_y","accel_arm_z"])
df_c["accel_arm_mag_var"] = win(df_c, "accel_arm_mag", "var")

df_c["magnet_arm_mag"] = mag(df_c, ["magnet_arm_x","magnet_arm_y","magnet_arm_z"])
df_c["magnet_arm_mag_max"] = win(df_c, "magnet_arm_mag", "max")
df_c["magnet_arm_mag_min"] = win(df_c, "magnet_arm_mag", "min")

# -----------------------------
# 3. DUMBBELL features
# -----------------------------

df_c["accel_dumbbell_mag"] = mag(df_c, ["accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z"])
df_c["accel_dumbbell_mag_max"] = win(df_c, "accel_dumbbell_mag", "max")

df_c["gyros_dumbbell_mag"] = mag(df_c, ["gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z"])
df_c["gyros_dumbbell_mag_var"] = win(df_c, "gyros_dumbbell_mag", "var")

df_c["magnet_dumbbell_mag"] = mag(df_c, ["magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z"])
df_c["magnet_dumbbell_mag_max"] = win(df_c, "magnet_dumbbell_mag", "max")
df_c["magnet_dumbbell_mag_min"] = win(df_c, "magnet_dumbbell_mag", "min")

# -----------------------------
# 4. FOREARM (glove) features
# -----------------------------

df_c["pitch_forearm_sum"] = win(df_c, "pitch_forearm", "sum")

df_c["gyros_forearm_mag"] = mag(df_c, ["gyros_forearm_x","gyros_forearm_y","gyros_forearm_z"])
df_c["gyros_forearm_mag_max"] = win(df_c, "gyros_forearm_mag", "max")
df_c["gyros_forearm_mag_min"] = win(df_c, "gyros_forearm_mag", "min")

# -----------------------------
# Final: drop intermediate magnitude columns
# -----------------------------

mag_cols = [col for col in df_c.columns if col.endswith("_mag")]
df_veloso = df_c.drop(columns=mag_cols)


# ---------------------------------------
# Build Veloso features for the test set
# ---------------------------------------

df_t = test_df.copy()

# BELT
df_t["roll_belt_mean"] = win(df_t, "roll_belt", "mean")
df_t["roll_belt_var"]  = win(df_t, "roll_belt", "var")

df_t["accel_belt_mag"] = mag(df_t, ["accel_belt_x","accel_belt_y","accel_belt_z"])
df_t["accel_belt_mag_max"]   = win(df_t, "accel_belt_mag", "max")
df_t["accel_belt_mag_range"] = win(df_t, "accel_belt_mag", lambda s: s.max() - s.min())
df_t["accel_belt_mag_var"]   = win(df_t, "accel_belt_mag", "var")

df_t["gyros_belt_mag"] = mag(df_t, ["gyros_belt_x","gyros_belt_y","gyros_belt_z"])
df_t["gyros_belt_mag_var"] = win(df_t, "gyros_belt_mag", "var")

df_t["magnet_belt_mag"] = mag(df_t, ["magnet_belt_x","magnet_belt_y","magnet_belt_z"])
df_t["magnet_belt_mag_var"] = win(df_t, "magnet_belt_mag", "var")

# ARM
df_t["accel_arm_mag"] = mag(df_t, ["accel_arm_x","accel_arm_y","accel_arm_z"])
df_t["accel_arm_mag_var"] = win(df_t, "accel_arm_mag", "var")

df_t["magnet_arm_mag"] = mag(df_t, ["magnet_arm_x","magnet_arm_y","magnet_arm_z"])
df_t["magnet_arm_mag_max"] = win(df_t, "magnet_arm_mag", "max")
df_t["magnet_arm_mag_min"] = win(df_t, "magnet_arm_mag", "min")

# DUMBBELL
df_t["accel_dumbbell_mag"] = mag(df_t, ["accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z"])
df_t["accel_dumbbell_mag_max"] = win(df_t, "accel_dumbbell_mag", "max")

df_t["gyros_dumbbell_mag"] = mag(df_t, ["gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z"])
df_t["gyros_dumbbell_mag_var"] = win(df_t, "gyros_dumbbell_mag", "var")

df_t["magnet_dumbbell_mag"] = mag(df_t, ["magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z"])
df_t["magnet_dumbbell_mag_max"] = win(df_t, "magnet_dumbbell_mag", "max")
df_t["magnet_dumbbell_mag_min"] = win(df_t, "magnet_dumbbell_mag", "min")

# FOREARM
df_t["pitch_forearm_sum"] = win(df_t, "pitch_forearm", "sum")

df_t["gyros_forearm_mag"] = mag(df_t, ["gyros_forearm_x","gyros_forearm_y","gyros_forearm_z"])
df_t["gyros_forearm_mag_max"] = win(df_t, "gyros_forearm_mag", "max")
df_t["gyros_forearm_mag_min"] = win(df_t, "gyros_forearm_mag", "min")

# Drop intermediate magnitude columns
mag_cols_t = [col for col in df_t.columns if col.endswith("_mag")]
df_veloso_test = df_t.drop(columns=mag_cols_t)

veloso_features = [
    # BELT
    "roll_belt_mean", "roll_belt_var",
    "accel_belt_mag_max", "accel_belt_mag_range", "accel_belt_mag_var",
    "gyros_belt_mag_var",
    "magnet_belt_mag_var",

    # ARM
    "accel_arm_mag_var",
    "magnet_arm_mag_max", "magnet_arm_mag_min",

    # DUMBBELL
    "accel_dumbbell_mag_max",
    "gyros_dumbbell_mag_var",
    "magnet_dumbbell_mag_max", "magnet_dumbbell_mag_min",

    # FOREARM
    "pitch_forearm_sum",
    "gyros_forearm_mag_max", "gyros_forearm_mag_min"
]

df_veloso = df_veloso[veloso_features + ["classe"]]
df_veloso_test = df_veloso_test[veloso_features + ["classe"]]
```

<br>
Next, let's use these in the Random Forest model we built on our own data.
Let's keep the number of decision trees comparable to our own model: (500)

<br>
### Recreate Paper's Model <a name="fully-featured-model"></a>

After CFS was run on the full HAR dataset, we had the following fields as features in the RF model...

<br>
<br>

| **Variable Names** | **Variable Type** | **Description** |
|---|---|---|
| roll_belt_mean | Independent | mean over a the time window of the roll from the belt sensor (roll calculated from accelerometry) |
| **roll_belt_var** | Independent | variance over a the time window of the roll from the belt sensor (roll calculated from accelerometry) |
| accel_belt_mag_max | Independent | (accelerometry) maximum acceleration over a time window of the belt sensor |
| accel_belt_mag_range | Independent | (accelerometry) acceleration range over a time window of the belt sensor |
| accel_belt_mag_var | Independent | (accelerometry) acceleration variance over a time window of the belt sensor |
| gyros_belt_mag_var | Independent | (gyrometery) variance of the belt gyro  |
| magnet_belt_mag_var | Independent | (magnetometry) variance of the belt magnetometer  |
| accel_arm_mag_var | Independent | (accelerometry) acceleration variance over a time window of the arm sensor  |
| magnet_arm_mag_max | Independent | (magnetometry) maximum magnetometer reading over a time window of the arm sensor  |
| magnet_arm_mag_min | Independent | (magnetometry) minimum magnetometer reading over a time window of the arm sensor  |
| accel_dumbbell_mag_max | Independent | (accelerometry) maximum acceleration over a time window of the dumbbell sensor  |
| gyros_dumbbell_mag_var | Independent |  (gyrometry) maximum gyrometer reading over a time window of the dumbbell sensor |
| magnet_dumbbell_mag_max | Independent | (magnetometry) maximum magnetometer reading over a time window of the dumbbell sensor |
| magnet_dumbbell_mag_min | Independent | (magnetometry) minimum magnetometer reading over a time window of the dumbbell sensor |
| pitch_forearm_sum | Independent | sum over the time window of the pitch from the forearm sensor (pitch calculated from accelerometry) |
| gyros_forearm_mag_max | Independent | (gyrometry) maximum gyrometer reading over a time window of the forearm sensor |
| gyros_forearm_mag_min | Independent | (gyrometry) minimum gyrometer reading over a time window of the forearm sensor |

Let's see how it went!

<br>
<img width="541" height="594" alt="image" src="https://github.com/user-attachments/assets/f733553b-b74d-496d-8be6-8076cc1b755d" />

<br>
Accuracy: 0.5032

<br>
<img width="1017" height="544" alt="image" src="https://github.com/user-attachments/assets/65ab6323-364b-4eb2-a223-72bedd84436e" />

<br>
<img width="1064" height="544" alt="image" src="https://github.com/user-attachments/assets/154246e4-4589-405c-ae91-35aa2e624614" />
<br>

The 17-feature set produced substantially better performance than the CFS‑4 subset, demonstrating that domain‑engineered HAR features capture more generalisable movement structure across subjects. The model classified classes A, B, and C reasonably well, but struggled with classes D and especially E, which were frequently misclassified as lower‑numbered classes. Feature importance analysis showed a balanced contribution across belt, dumbbell, and forearm features, with roll‑belt variance, forearm gyro maxima, and cumulative forearm pitch emerging as the strongest predictors. Permutation importance confirmed that these features are genuinely essential, while several others contribute only marginally, indicating redundancy and subject‑specific sensitivity. Overall, CFS‑17 provides a meaningful improvement, but still falls short of robust subject‑independent performance.

<br>
<br>
# Growth & Next Steps  <a name="growth-next-steps"></a>

Our training set is very small, based on too few movements to generalise effectively to provide feedback in real-time using these approaches. A gold-standard HAR model is needed: A Domain-Adversarial Neural Network (DANN) for HAR. 
We'll next use Copilot to help code this pipeline in our Spyder environment using PyTorch! This, if successful, should dramatically increase between subject prediction!

