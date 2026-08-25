---
layout: post
title: Assessing Student Outcomes at Victorian Universities 
image: "/posts/deakin-proj2.png"
tags: [Tableau, EDA, Data Viz, Python]
---

In this project we build a dashboard in Tableau using publically available education data to assess student outcomes across universities in Victoria, Australia with a focus on outcomes for Deakin University.  

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Data Overview & Preparation](#data-overview)
- [02. Building The Dashboard](#tableau-application)
- [03. Analysing The Results](#tableau-results)
- [04. Discussion](#discussion)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our aim is to describe changes in student Attrition, Success, and Retention over the previous decade using the most recently available public data and to compare Deakin University's rate metrics to those of other Victorian Table A providers, alongside State and Australian averages for all providers.

We aim to describe historic patterns and trends by bachelor student cohort: domestic or overseas, both pre and post Covid impacts to Victoria, to identify any trends that may guide a deeper investigation.

<br>
<br>

### Actions <a name="overview-actions"></a>

We'll: 
* Retrieve Victorian and Australian instution and means data from: The Australian Government Department of Education Higher Education's most recent [release](https://www.education.gov.au/higher-education-statistics/resources/2024-section-15-attrition-success-and-retention)
* Decide scope, evaluate data quality, and provide caveats and outcome definitions
* Evaluate raw data and reshape our data for Tableau, using Python
* Build Attrition, Retention, and Success Charts using filters in Tableau
* Build a combined KPI measure - selectable by outcome
* Arrange a dashboard and refine tooltips to view
* Provide key insights from this data for Deakin University: describing outcomes relative to other Victorian universities.

From the higher education statistics data, we isolate our rate measures for this analysis to: 
* Victorian education providers

We compare these to average rates derived from:
* all Australian and Victorian higher education providers 
* all Australian Table A providers - i.e., all Australian Public University providers

We exclude any individual providers outside Victoria.

#### Outcomes

Our outcomes are at the **Sector** level as:

* **attrition rate (A):** Attrition Rates for a given year are the proportion of students who commenced a course in that year who neither complete in that year or the next year, nor return in the next year. For these rates, it is only those students who were no longer at _any_ institution that are counted as attrited.
   
* **success rate (S):** Success rate for _commencing_ students: The equivalent full‑time student load (EFTSL) of units passed to the EFTSL of units attempted, calculated from commencing students in a given year.
  
* **retention rate (R):** Retention Rates for a given year are the number of students who commenced a course in that year and continued in the next (retained students) as a proportion of all students who commenced a course in the same year and did not complete in that year or the next year.
  
* **% change over the data span (in A,S,R separately) : data between years 2014-2024 for each outcome respectively:** The rate in the most recent year (2023 for A and R, 2024 for S) minus the rate in the first year (2014 for A and R, 2015 for S) divided by the rate in the first year (2014 for A and R, 2015 for S).

These rates are published and have been calculated from government linked data: with StudentID linked to Commonwealth Higher Education Student Support Number (CHESSN) to track these outcomes.

<br>
<br>

### Results & Discussion <a name="overview-results"></a>

The completed dashboard can be found [here](https://public.tableau.com/views/Deakin_Proj/BachelorStudentOutcomesVictoria?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).

Deakin University shows steadily improved success rates across domestic and overseas student cohorts over time, indicating that for both cohorts, pass rates are rising in students' commencing year. 
In its domestic cohort, attrition rates reduced over the previous 10 years, consistent with the leading Victorian providers. 

However, its overseas student cohort has been affected by factors - unknown from just this analysis - that have led to increased attrition for overseas students. This attrition rate differs between Deakin and the leading Victorian providers. 

Deakin's overseas student attrition rate change pattern over this time is most similar to Federation University's over Victoria's Covid lockdown periods - though less extreme. 
This may suggest that there are external factors that have affected overseas student attrition at some Universities and not others over this time within Victoria.

It may be beneficial to segment the overseas student cohort to better assess how students are clustered with respect to attrition and success.
Methodological changes to retention calculations in 2018 may account for some of the retention differences pre and post 2018.

While these statistics provide a broad comparison across institutions, alternative statistical methodology to create these measures and further segmentation may be beneficial to investigate these outcomes at university level.

<br>
<br>

___

<br>

# Data Overview & Preparation  <a name="data-overview"></a>

In the public Australian Government Department of Education Data : Selected Higher Education Statistics Data , we have an *Attrition, Success and Retention* worksheet showing rates for all Australian private and public higher education providers.

For this task, we aim to compare Victorian institutions, so we just extract the rates for these 9 providers and exclude universities in other states.
We can also extract the Victorian and Australian average rates for each year, and the Australian Table A provider average rate - to provide state and national benchmarks for each rate type.

In the code below, we:

* Load in the Python library and define a function to filter and reshape the data (using pandas)
* Import the required data and filter it to select the measures we need using the function.
* Reshape the wide-format data, giving us a long-format dataset ready for import to Tableau 

<br>

```python

## Load library
import pandas as pd

# --------------------------------------------
# Define reshape function for a wide excel sheet
# --------------------------------------------
def reshape_sheet(df, family_keyword):
    """
    Reshape a wide-format sheet into long format.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The wide-format dataframe (one sheet).
    family_keyword : str
        One of "Attrition", "Retention", "Success".
    
    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with columns:
        Institution, Year, MeasureFamily, MeasureType, Value
    """
    
    # Identify columns belonging to this measure family
    value_cols = [c for c in df.columns if family_keyword in c]
    
    # Melt into long format
    df_long = df.melt(
        id_vars=["Institution"],
        value_vars=value_cols,
        var_name="Field",
        value_name="Value"
    )
    
    # Split YYYY_Family_Type
    df_long[["Year", "MeasureFamily", "MeasureType"]] = df_long["Field"].str.split("_", expand=True)
    
    # Convert Year to integer
    df_long["Year"] = df_long["Year"].astype(int)
    
    # Give final tidy structure
    return df_long[["Institution", "Year", "MeasureFamily", "MeasureType", "Value"]]
```

<br>

```python
# --------------------------------------------
# Load all sheets
# --------------------------------------------
df_A = pd.read_excel("ASRwide.xlsx", sheet_name="Attrition")
df_R = pd.read_excel("ASRwide.xlsx", sheet_name="Retention")
df_S = pd.read_excel("ASRwide.xlsx", sheet_name="Success")

# --------------------------------------------
# Apply reshape function 
# --------------------------------------------
sheets = {
    "Attrition": df_A,
    "Retention": df_R,
    "Success": df_S
}

long_tables = {}

for family, df in sheets.items():
    long_tables[family] = reshape_sheet(df, family)

# Extract each long-format table
df_A_long = long_tables["Attrition"]
df_R_long = long_tables["Retention"]
df_S_long = long_tables["Success"]

# --------------------------------------------
# Combine in master table
# --------------------------------------------
df_all = pd.concat([df_A_long, df_R_long, df_S_long], ignore_index=True)

# ------------------------------------------
# Export long format for Tableau
# ------------------------------------------
df_all.to_excel("ASRlong.xlsx", index=False)

```

<br>

A sample of this data (the first 10 rows) can be seen below:

<br>
<br>

| **Institution** | **Year** | **MeasureFamily** | **MeasureType** | **Value** |
|---|---|---|---|---|
| Australia_Tot_All | 2014 | Attrition | Dom | 15 |
| Australia_A | 2014 | Attrition | Dom | 15.07 |
| State_Tot_All | 2014 | Attrition | Dom | 13.38 |
| Deakin University (3030) | 2014 | Attrition | Dom | 14.41 |
| Federation University Australia (2154) | 2014 | Attrition | Dom | 22.75 |
| La Trobe University (3020) | 2014 | Attrition | Dom | 11.43 |
| Monash University (3035) | 2014 | Attrition | Dom | 6.19 |
| RMIT University (3034) | 2014 | Attrition | Dom | 10.3 |
| Swinburne University of Technology (2177) | 2014 | Attrition | Dom | 24.54 |
| The University of Melbourne (3036) | 2014 | Attrition | Dom | 3.5 |

<br>
<br>

In the DataFrame we have:

* Institution
* Year
* MeasureFamily (one of: Attrition, Success or Retention)
* MeasureType (one of: Dom, OS or All)
* Value (the rate measure outcome value)

This data is now formatted as one sheet to be read into Tableau.

___

<br>

# Building the Dashboard <a name="tableau-application"></a>

Our dashboard will consist of 4 sections from their respective sheets, with 3 filters: 
* Institution (which includes averages) 
* Student Type 
* KPI: one of A,S or R for viewing 10-year movements.

The very first thing we need to do is check Tableau's interpretation of our data types. This is a simple variable type check for our 5 fields. 

We then specify parameters and calculate fields.

<br>

#### Specify Parameters and Calculate Fields : Attrition, Success, and Retention

We create parameters as lists :

* Select KPI Family  = List : ["Attrition","Success","Retention"] displayed as : "Attrition","Success","Retention"
* Student Type = List : ["Dom","OS","All"] displayed as : "Domestic", "Overseas", "All"

We create filters using the parameters defined as:

```sql

"Measure Type":
[Measure Type] = [Student Type]

"Earliest Value":
{ FIXED [Institution], [Measure Family], [Measure Type] :
    MIN( IF [Year] = 
            { FIXED [Institution], [Measure Family], [Measure Type] : MIN([Year]) }
         THEN [Value]
         END )
}

"Latest Value": 
{ FIXED [Institution], [Measure Family], [Measure Type] :
    MAX( IF [Year] = 
            { FIXED [Institution], [Measure Family], [Measure Type] : MAX([Year]) }
         THEN [Value]
         END )
}

"Percent Difference": 
([Latest Value] - [Earliest Value]) / [Earliest Value]

"KPI Percent Change":
IF [Measure Family] = [Select KPI Family] THEN
    ([Latest Value] - [Earliest Value]) / [Earliest Value]
END

```

<br>
#### Specify Sheets

We are now ready to use Tableau's Marks and Filters to build 4 sheets:
- Attrition Chart
- Retention Chart
- Success Chart
- KPI Combined

These are laid out using grid view in a dashboard, giving us the ability to view outcomes and select our outcome to view % change over the 10 year data window.

___

<br>
# Analysing The Results <a name="tableau-results"></a>

At this point we have everything we need to view the rates for each provider, relative to state and national averages.

The dashboard is viewable in full-screen view from [here](https://public.tableau.com/views/Deakin_Proj/BachelorStudentOutcomesVictoria?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

<br>
As we can see from the dashboard, there are clear differences in Success, Attrition, and Retention rates between individual providers. We discuss these below.

___

<br>
# Discussion <a name="discussion"></a>

Deakin has an ~8% improvement in Success rate over the 10 year period between 2015 and 2024, driven by a rise in both domestic and overseas student success, and attrition rates have remained steady on average across cohorts. 
But, driving this, while attrition rates fell for the domestic student cohort, they rose significantly among overseas students, rising fastest over Covid-affected years (2020-2022), with rates not yet returned to pre-Covid levels in this cohort (as of 2023 data). 

Despite a small drop in success rate over this period among overseas students, success rates have themselves recovered, reaching a new high in this cohort post Victorian Covid-19 outbreak and lockdown. 

Therefore, we might recommend closer inspection of both success and attrition in this overseas student cohort over the 2020-2022 years, to identify the main drivers for this cpncurrent rise in both measures. 

Impacts of Covid and the economic impact of Victoria's lockdowns over this period may have disadvantaged the students that were less likely than average to succeed in their studies. 
This could lead to their attrition, and in that scenario, would account for the rise in both the attrition and success rate over this period in the broader OS student cohort. 
It may be worth investigating this over 2020-2022 further, to support or refute this, and to generate alternate hypotheses around this. 

It is worth noting that Federation University displays a similar, albeit more extreme variance pattern in both domestic and overseas student cohorts over this 2020-2022 period, indicating that any factors driving attrition in overseas student cohorts are not likely to be unique to students at Deakin. 
They may be external, affecting some universities and not others. Monash and Melbourne (Victoria's leading providers in terms of Success and Attrition) do not display this pattern.
 
Over the past decade, Deakin's success rate has steadily approached the Victorian average, which is raised by (i) Monash University, (ii) the University of Melbourne and (iii) the University of Divinity (though this is a smaller institution). 
It is a caveat to this analysis that it is not immediately clear from this data how sensitive the Victorian average is to the results of smaller institutions. Victoria University has shifted the direction of its influence on this, from lowering the Victorian average up to 2021, to raising it, since 2021, in both domestic and overseas cohorts.

The success rate for all Australian Table A providers is lower than the Victorian average across all years, and correlates highly with it. 
When compared to this average, Deakin's formerly lower success rate has improved over the decade and has reached comparable levels to the Australian Table A provider average by 2024. 
 
From this analysis, we might recommend that segmenting the overseas student population, gathering more data on this cohort, and considering further refinement of statistical methodology used to measure attrition and success, may provide us with deeper insights particularly in this group.
