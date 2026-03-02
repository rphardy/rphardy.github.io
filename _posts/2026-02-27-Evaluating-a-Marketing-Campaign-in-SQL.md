---
layout: post
title: Evaluating a Marketing Campaign using SQL
image: "/posts/primes_image.jpeg"
tags: [SQL, Statistical Assumptions, Analysis Piping]
---

In this post I'm going to write a SQL program to produce an analytical pipeline to measure the effectiveness of a mail-out marketing 
campaign. 

I'll build a single function to reproduce this for a future campaign, outputting the analysis and new datasets to check statistical 
assumptions.

Let's build this out, in SQL!

Beginning with the Objectives, which are to: 

- Write an analysis pipeline function in SQL for a grocery store's mail-out marketing campaign
- Retrieve, for each gender and mailer type in the data: number of customers, signups (successes) and sign-up percentage
(mailer success metric)
- Wrap all steps in a function to run the anaysis from a single function call, outputting a json file containing all required outputs.

We'll describe the project using a header for future reference:

```sql

/***************************************************************************************************************************************/

/*                                   MEASURING THE SUCCESS OF MAILER TYPE IN A MAIL-OUT MARKETING CAMPAIGN                          

Objective:   
. Write an analysis pipeline function in SQL for a grocery store's mail-out marketing campaign. Retrieve: 
for each gender and mailer type: the number of customers, signups (successes) and the sign-up percentage (the mailer success metric). 
. The function outputs analysis results and interim datasets for statistical quality control to evaluate the representativeness of the
sign-up percentage and to test the key assumptions in creating it. 
. Return results for the campaign used to build the example, the campaign named: "delivery_club"

*/
```
We will need to address some statistical assumptions:

```sql
/*
Assumptions: 

. Missing customer information and campaign information is missing at random (this assumption can be tested in future using our output
  under 2 possible scenarios): 1) campaign data is the source of true customer information, or
                               2) customer details is the source of true customer information).
. Data linkage does not introduce bias (unlinked customer and campaign ids are collected with the linkage to test this)
. Missing signup information is missing at random, so signup percentage should include only known signup / known non-signup 
(a binomial proportion is used to calculate the % - excluding nulls)
. The true identity of customers, between campaign data id and customer details id, is not known (either id could be the true
  response-provider - giving two scenarios).
*/
```
Now that we're clear on the analysis build and the assumptions made along the way, we can define steps and the output that we'd expect 
to see:

```sql
/*
Overview of Steps:

             1) Link required datasets using best practice data linkage: link customer_details and campaign_data
             2) Define target population for campaign and run SQL query analysis
             3) Create data for missingness analysis (if assuming source of truth: campaign_data)
             4) Create data for missingness analysis (if assuming source of truth: customer_details)
             5) Combine steps 1-4 in a function to deliver all outputs of this pipeline, with campaign name input as: "delivery_club"
*/
```
```sql
/*
Expected Output: 

              3 sets output by step 1: linkage data (2 linkage sets are expected to be empty, 1 linkage is expected to contain all linked
              data, successfully linked on customer id)
              1 set output by step 2: analysis data (contains the analysis results for the "delivery_club" mailer campaign: grouped by
              gender and mailer type)
              2 sets output by step 3 and 4, 1 set each: missingness data (provided for further testing: MAR assumptions under the 2
              potential sources of truth for customer id)
              1 .json cell containing all the above combined - in human-readable .json format
 
First Lines of Expected Json Output:

           {
              "analysis_results": [
                  {
                      "gender": "F",
                      "signups": 110,
                      "mailer_type": "Mailer2",
                      "customer_count": 191,
                      "signup_percentage": 57.59
                  }, ...etc
*/

/***************************************************************************************************************************************/
```

Let's begin the analysis with step one: linking customer details and customers-in-campaign data. We'll use best-practice data linkage to 
view a full master linkage map (MLM) of the customer population in both sets. For this data, all customers can be linked successfully to 
an id in the campaign data, so our unlinked sets should be null, and our MLM is simply all customers in the campaign linked by id key to 
their details.

```sql
   
/* STEP 1) LINKAGE BEST PRACTICE */

drop table if exists linkage;
create temp table linkage as (
  select
      coalesce(a.customer_id, b.customer_id) as customer_id, /* colaesce ensures returning the first non-null value. i.e., if first
                                                                a.customer_id = NULL and first customer_id.b exists - return it. */
      case
          when a.customer_id = b.customer_id then 'custs_in_campaign'
          when a.customer_id is not null and b.customer_id is null then 'custs_not_in_campaign'
          when a.customer_id is null and b.customer_id is not null then 'in_campaign_not_custs'
      end as cust_join_type,

      -- Campaign fields (all NULL where customer not in campaign_data)
      a.customer_id as campaign_customer_id,
      a.mailer_type,
      a.signup_flag,
      a.campaign_name,

      -- Customer details fields (all NULL where customer not in customer_details)
      b.customer_id as customer_details_customer_id,
      b.gender

  from grocery_db.campaign_data a
  full outer join grocery_db.customer_details b
      on a.customer_id = b.customer_id

  order by
      cust_join_type,
      customer_id
  );
```
We now have three sets containing all linked and unlinked customer ids from both sources.
They are:
```
/* check set 1 of 3 : This should be an empty set for the grocery_db "delivery_club" data */  
select * from linkage where cust_join_type = 'custs_not_in_campaign' order by customer_id; 

/* check set 2 of 3 : This should be an empty set for the grocery_db "delivery_club" data */
select * from linkage where cust_join_type = 'in_campaign_not_custs' order by customer_id;

/* check set 3 of 3 : Linked analysis set. This contains all data in grocery_db now linked on customer_id for the "delivery_club" campaign */
select * from linkage where cust_join_type = 'custs_in_campaign' order by customer_id;
```

Using our linked set, it's now straightforward to define our target population and run our analysis. Let's use CTE followed by a single 
query to run the analysis.

First, let's define our target population as all linked customers:

```sql
/* STEP 2) ANALYSIS QUERY ON LINKED DATA : DEFINE TARGET POPULATION AND RUN ANALYSIS */

with target_pop as (
select 
  * 
from 
  linkage 
where 
  cust_join_type = 'custs_in_campaign')
```

Then, let's create our analysis outcome variable. 

Note that we define the signup percentage outcome in best-practice statistical format: as the average of a proportion of a binary outcome 
Success/Failure, excluding nulls. This avoids having missing signup information introducing missingess bias to our estimate. We could dive 
further into missingness assumptions, depending on our linkage sets and the MLM picture. 

Generally though, we would address this by simply using the proportion of *known* successes to *known* failures to give our reported 
percentage outcome. We might want to avoid simply calculating an average of the sum of signups, to the sum of customers, but this may be 
OK to do. 

In this particular case, doing so would give the same result (since we have no customers with signup status: unknown/missing).

```sql
select
    coalesce(gender, 'Unknown') AS gender,
    coalesce(mailer_type, 'Unknown') AS mailer_type,
    
    -- Total rows in this group (should == customer_count since each row is a unique customer, and there are no missing or duplicate customer_ids)
    count(*) as customer_count,
    
    -- Number of signups (1s only)
    sum(case when signup_flag = 1 then 1 else 0 end) as signups,
    
    -- Statistically robust proportion: 1 == signed up, 0 == did not sign up, NULL == missing
    round(
        avg(
            case
                when signup_flag = 1 then 1
                when signup_flag = 0 then 0
                else null
            end
        ) * 100,
        2
    ) as signup_percentage --, 
    
    -- Should also include: Number of non-missing signup_flag values
    -- (in this case, it equals customer count as there are no missing signup
    -- flags, so have commented it out)
 -- count(signup_flag) as non_missing_signup_count

from 
  target_pop

where
  campaign_name = 'delivery_club' -- sanity check: all customers in delivery_club, none missing.

group by
  coalesce(gender, 'Unknown'),
  coalesce(mailer_type, 'Unknown')

having 
  count(*) > 5 /* would not draw meaningful inferences from smaller populations than this, also de-risks customer identification by rare
                  attribute combinations - this step drops all data where gender = 'Unknown' */
  
order by
    signup_percentage desc;  
```
Note also, we exclude gender and mailer-type combinations with populations less than 5, since: 
- A) This is too small a sample to draw meaningful inference from. 
- B) Customers in this population could more easily be identified by their characteristics!  

We're almost finished. Lets output some sets to allow future checking of any patterns in missing data that could be informative. 

Hopefully, all missing data in a future campaign will be missing at random, and missingness is not biasing our results. 

These sets will allow us to check whether this is true.

If we weren't sure, we could assume our campaign customer details are true for the majority of customers, over our customer details data
and provide a set tagging the missing data from this perspective

```sql

/* STEP 3)
  PROVIDE A SET TO ANALYSE M.A.R. ASSUMPTION FOR THE LINKAGE: EXPORT DATA FOR MISSINGNESS ANALYSIS
  (IF ASSUMING SOURCE OF TRUTH: CAMPAIGN_DATA) */

select
    a.customer_id as campaign_cust_id,
    a.mailer_type,
    a.signup_flag,

    -- Raw gender (NULL allowed)
    b.gender,

    -- Missingness indicators
    case when b.customer_id is null then 1 else 0 end as missing_customer_details,
    case when b.gender is null then 1 else 0 end as gender_missing

from grocery_db.campaign_data a
left join grocery_db.customer_details b
    on a.customer_id = b.customer_id

where
    a.campaign_name = 'delivery_club';
```
Or, we could assume the customer detail is more accurate reflection of the truth for any discrepancies with the details in the campaign.

```sql
/* STEP 4) PROVIDE A SET TO ANALYSE M.A.R. ASSUMPTION FOR THE LINKAGE: EXPORT DATA FOR MISSINGNESS ANALYSIS
          (IF ASSUMING SOURCE OF TRUTH: CUSTOMER_DATA) */

select
    b.customer_id as cust_details_id,

    -- Demographics (raw + missingness indicator)
    b.gender,
    case when b.gender is null then 1 else 0 end as gender_missing,

    -- Campaign fields (may be missing if customer never appears in campaign_data)
    a.mailer_type,
    case when a.mailer_type is null then 1 else 0 end as mailer_type_missing,

    a.signup_flag,
    case when a.signup_flag is null then 1 else 0 end as signup_flag_missing,

    -- Did this customer appear in campaign_data at all?
    case when a.customer_id is null then 1 else 0 end as missing_campaign_record

from grocery_db.customer_details b
left join grocery_db.campaign_data a on a.customer_id = b.customer_id
    and a.campaign_name = 'delivery_club';

/* END ANALYSIS */
```

We've completed our analysis. Let's now wrap this all up in a pipeline to repeat on the next campaign. 

If our data is in the same clean form, then for the next campaign type, we'll be able to repeat all this in one line of code, 
using nothing more complex than SQL.

Let's look at the input and output specs of our intended function:

```sql
/***************************************************************************************************************************************/
/* STEP 5: WRITE A FUNCTION TO REPRODUCE ALL ANALYSIS FOR THE GIVEN CAMPAIGN AND RETURN ALL OUTPUT IN HUMAN-READABLE JSON FORMAT       */

/* INPUTS: grocery_db.campaign_data, grocery_db.customer_details                                                                       */

/* OUTPUTS: JSON file containing data : 1) The analysis results for the campaign: main interest: sign-up percentage breakdowns,        */
/*                                      2) analysis dataset used, 3-4) customers in/out of campaign/details sets,                      */ 
/*                                      5) missingness from customer details for MAR analysis: source of truth - campaign data ids     */
/*                                      6) missingness from campaign data for MAR analysis: source of truth - campaign detail ids      */
```

We'll provide the user of our function with some instructions on how to access the pipeline. 
In this example, SQL was built in SQL Workbench J, and JSON files read into R using the jsonlite package.

```sql

/* TO ACCESS OUTPUTS: Save SQL Workbench J Output in cell below: 'jsonb_pretty' in Result 7 result tab as text: "YOUR_FILE.TXT".
   Then in R/Rstudio (for e.g.): copy and run      */
/*                    the following to compile and view:                                                                               */
/*
library(jsonlite)
x <- fromJSON("YOUR_FILE.TXT")
View(x$analysis_results)
View(x$custs_in_campaign)
View(x$custs_not_in_campaign)
View(x$in_campaign_not_custs)
View(x$missingness_from_campaign)
View(x$missingness_from_customer_details)                                                                                                                                     */
/***************************************************************************************************************************************/
```

Let's build the function in one 'create or replace' step.

The following CTE objects will look very familiar!

```sql

create or replace function run_campaign_analysis(target_campaign TEXT)
returns JSONB
language plpgsql
as $$
declare
    result JSONB;
begin

    with linkage as (
        select
            coalesce(a.customer_id, b.customer_id) as customer_id,

            case
                when a.customer_id = b.customer_id then 'custs_in_campaign'
                when a.customer_id is not null and b.customer_id is null then 'custs_not_in_campaign'
                when a.customer_id is null and b.customer_id is not null then 'in_campaign_not_custs'
            end as cust_join_type,

            -- Campaign fields
            a.customer_id as campaign_customer_id,
            a.mailer_type,
            a.signup_flag,
            a.campaign_name,

            -- Customer details fields
            b.customer_id as customer_details_customer_id,
            b.gender

        from grocery_db.campaign_data a
        full outer join grocery_db.customer_details b
            on a.customer_id = b.customer_id
    ),

    target_pop as (
        select *
        from linkage
        where cust_join_type = 'custs_in_campaign'
          and campaign_name = target_campaign
    ),

    analysis_results as (
        select
            coalesce(gender, 'Unknown') as gender,
            coalesce(mailer_type, 'Unknown') as mailer_type,
            count(*) as customer_count,
            sum(case when signup_flag = 1 then 1 else 0 end) as signups,
            round(
                avg(
                    case
                        when signup_flag = 1 then 1
                        when signup_flag = 0 then 0
                        else null
                    end
                ) * 100,
                2
            ) as signup_percentage
        from target_pop
        group by
            coalesce(gender, 'Unknown'),
            coalesce(mailer_type, 'Unknown')
        having count(*) > 5
        order by signup_percentage desc
    ),

    missingness_from_campaign as (
        select
            a.customer_id as campaign_cust_id,
            a.mailer_type,
            a.signup_flag,
            b.gender,
            case when b.customer_id is null then 1 else 0 end as missing_customer_details,
            case when b.gender is null then 1 else 0 end as gender_missing
        from grocery_db.campaign_data a
        left join grocery_db.customer_details b
            on a.customer_id = b.customer_id
        where a.campaign_name = target_campaign
    ),

    missingness_from_customer_details as (
        select
            b.customer_id as cust_details_id,
            b.gender,
            case when b.gender is null then 1 else 0 end as gender_missing,
            a.mailer_type,
            case when a.mailer_type is null then 1 else 0 end as mailer_type_missing,
            a.signup_flag,
            case when a.signup_flag is null then 1 else 0 end as signup_flag_missing,
            case when a.customer_id is null then 1 else 0 end as missing_campaign_record
        from grocery_db.customer_details b
        left join grocery_db.campaign_data a
            on a.customer_id = b.customer_id
           and a.campaign_name = target_campaign
    )

    select jsonb_build_object(
        'analysis_results', (select jsonb_agg(analysis_results) from analysis_results),
        /*results: signup percentages by gender and mailer type */
        'custs_in_campaign', (select jsonb_agg(linkage) from linkage where cust_join_type = 'custs_in_campaign'),
        /* linked set for analysis : customers with details in the campaign */
        'custs_not_in_campaign', (select jsonb_agg(linkage) from linkage where cust_join_type = 'custs_not_in_campaign'),
        /* customer details dropped 1 */
        'in_campaign_not_custs', (select jsonb_agg(linkage) from linkage where cust_join_type = 'in_campaign_not_custs'),
        /* customers in campaign dropped 2 */
        'missingness_from_campaign', (select jsonb_agg(missingness_from_campaign) from missingness_from_campaign),
        /* customers with details tagged as missing in campaign */
        'missingness_from_customer_details', (select jsonb_agg(missingness_from_customer_details) from missingness_from_customer_details)
        /* customer ids in campaign tagged as missing details */
    )
    into result;

    return jsonb_pretty(result)::jsonb;

end;
$$;
```
The function is built.

Next time, to assess a new campaign set up in the same fashion, and to output all diagnostics datasets, we can simply use:

```sql
/* RUN ALL AS FUNCTION */

select jsonb_pretty(run_campaign_analysis('delivery_club'));
```
And we're done!

This has been a summary of building an analysis pipeline that applies statistical rigour, simply in SQL.
