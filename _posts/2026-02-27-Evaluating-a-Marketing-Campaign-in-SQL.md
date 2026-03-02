---
layout: post
title: Evaluating a Marketing Campaign using SQL
image: "/posts/primes_image.jpeg"
tags: [SQL, Statistical Assumptions, Analysis Piping]
---

In this post I'm going to write a SQL program to produce an analytical pipeline to measure the effectiveness of a mail-out marketing campaign. 

I'll build a single function to reproduce this for a future campaign, outputting the analysis and new datasets to check statistical assumptions.

Let's build this out, in SQL!

Beginning with the Objectives, which are to: 

- Write an analysis pipeline function in SQL for a grocery store's mail-out marketing campaign
- Retrieve, for each gender and mailer type in the data: number of customers, signups (successes) and sign-up percentage (mailer success metric)
- Wrap all steps in a function to run the anaysis from a single function call, outputting a json file containing all required outputs.

We can describe the whole project using a header.

```sql

/******************************************************************************************************************************************************************/

/*                                                 MEASURING THE SUCCESS OF MAILER TYPE IN A MAIL-OUT MARKETING CAMPAIGN                          

Objective:   
. Write an analysis pipeline function in SQL for a grocery store's mail-out marketing campaign. Retrieve: 
for each gender and mailer type: the number of customers, signups (successes) and the sign-up percentage (the mailer success metric). 
. The function outputs analysis results and interim datasets for statistical quality control to evaluate the representativeness of the sign-up percentage 
and to test the key assumptions in creating it. 
. Return results for the campaign used to build the example, the campaign named: "delivery_club"
```
We will need to address some statistical assumptions:

```sql
Assumptions: 

. Missing customer information and campaign information is missing at random (this assumption can be tested in future using our output under 2 possible scenarios: 1) campaign data is the source of true customer ids, or 2) customer details is the source of true customer ids).
. Data linkage does not introduce bias (unlinked customer and campaign ids are collected with the linkage to test this)
. Missing signup information is missing at random, so signup percentage should include only known signup / known non-signup 
(a binomial proportion is used to calculate the % - excluding nulls)
. The true identity of customers, between campaign data id and customer details id, is not known (either id could be the true response-provider - giving two scenarios).

```
Now that we're clear on the build and the assumptions made on the way, we can define the steps and the output that we'd expect:

```sql
Overview of Steps:

             1) Link required datasets using best practice data linkage: link customer_details and campaign_data
             2) Define target population for campaign and run SQL query analysis
             3) Create data for missingness analysis (if assuming source of truth: campaign_data)
             4) Create data for missingness analysis (if assuming source of truth: customer_details)
             5) Combine steps 1-4 in a function to deliver all outputs of this pipeline, with campaign name input as: "delivery_club"
```
```sql
Expected Output: 

             3 sets output by step 1: linkage data (2 linkage sets are expected to be empty, 1 linkage is expected to contain all linked data, successfully linked on customer id)
             1 set output by step 2: analysis data (contains the analysis results for the "delivery_club" mailer campaign: grouped by gender and mailer type)
             2 sets output by step 3 and 4, 1 set each: missingness data (provided for further testing: MAR assumptions under the 2 potential sources of truth for customer id) 
             1 json cell containing all the above combined - in human-readable .json format
 
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

/******************************************************************************************************************************************************************/
 
```
Let's begin with step one: linking customer details and their campaign data from two sets using best practice data linkage. 
```sql
   
/* STEP 1) LINKAGE BEST PRACTICE */

drop table if exists linkage;
create temp table linkage as (
  select
      coalesce(a.customer_id, b.customer_id) as customer_id, /*colaesce ensures returning the first non-null value. i.e., if first a.customer_id = NULL and first customer_id.b exists - return it.*/
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

/* check set 1 of 3 : Should be an empty set for the grocery_db data*/  
select * from linkage where cust_join_type = 'custs_not_in_campaign' order by customer_id; 

/* check set 2 of 3 : should be an empty set for the grocery_db data*/
select * from linkage where cust_join_type = 'in_campaign_not_custs' order by customer_id;

/* check set 3 of 3 : Should contain all data in grocery_db now linked in customer_id */
select * from linkage where cust_join_type = 'custs_in_campaign' order by customer_id;



/* STEP 2) ANALYSIS QUERY ON LINKED DATA : DEFINE TARGET POPULATION AND RUN ANALYSIS */

with target_pop as (
select 
  * 
from 
  linkage 
where 
  cust_join_type = 'custs_in_campaign')

select
    coalesce(gender, 'Unknown') AS gender,
    coalesce(mailer_type, 'Unknown') AS mailer_type,
    
    -- Total rows in this group (should equal customer_count since each row is a unique customer, and there are no missing or duplicate customer_ids)
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
    
    -- Should also include: Number of non-missing signup_flag values (in this case, it equals customer count as there are no missing signup flags, so have commented it out)
 -- count(signup_flag) as non_missing_signup_count

from 
  target_pop

where
  campaign_name = 'delivery_club' -- sanity check: all customers in delivery_club, none missing.

group by
  coalesce(gender, 'Unknown'),
  coalesce(mailer_type, 'Unknown')

having 
  count(*) > 5 /* would not draw meaningful inferences from smaller populations than this, also de-risks customer identification by rare attribute combinations - this step drops all data where gender = 'Unknown' */
  
order by
    signup_percentage desc;  



/* STEP 3) PROVIDE A SET TO ANALYSE M.A.R. ASSUMPTION FOR THE LINKAGE: EXPORT DATA FOR MISSINGNESS ANALYSIS (IF ASSUMING SOURCE OF TRUTH: CAMPAIGN_DATA) */

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



/* STEP 4) PROVIDE A SET TO ANALYSE M.A.R. ASSUMPTION FOR THE LINKAGE: EXPORT DATA FOR MISSINGNESS ANALYSIS (IF ASSUMING SOURCE OF TRUTH: CUSTOMER_DATA) */

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



/*****************************************************************************************************************************************************************************/
/* STEP 5: WRITE A FUNCTION TO REPRODUCE ALL ANALYSIS FOR THE GIVEN CAMPAIGN AND RETURN ALL OUTPUT IN HUMAN-READABLE JSON FORMAT                                             */

/* INPUTS: grocery_db.campaign_data, grocery_db.customer_details                                                                                                             */

/* OUTPUTS: JSON file containing data : 1) The analysis results for the campaign: main interest: sign-up percentage breakdowns,                                              */
/*                                      2) analysis dataset used, 3-4) customers in/out of campaign/details sets,                                                            */ 
/*                                      5) missingness from customer details for MAR analysis: source of truth - campaign data ids                                           */
/*                                      6) missingness from campaign data for MAR analysis: source of truth - campaign detail ids                                            */

/* TO ACCESS OUTPUTS: Save SQL Workbench J Output in cell below: 'jsonb_pretty' in Result 7 result tab as text: "YOUR_FILE.TXT". Then in Rstudio for e.g.: copy and run      */
/*                    the following to compile and view:                                                                                                                     */
/*
library(jsonlite)
x <- fromJSON("YOUR_FILE.TXT")
View(x$analysis_results)
View(x$custs_in_campaign)
View(x$custs_not_in_campaign)
View(x$in_campaign_not_custs)
View(x$missingness_from_campaign)
View(x$missingness_from_customer_details)                                                                                                                                     */
/******************************************************************************************************************************************************************************/

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
        'analysis_results', (select jsonb_agg(analysis_results) from analysis_results), /*results: signup percentages by gender and mailer type */
        'custs_in_campaign', (select jsonb_agg(linkage) from linkage where cust_join_type = 'custs_in_campaign'), /* linked set for analysis : customers with details in the campaign */
        'custs_not_in_campaign', (select jsonb_agg(linkage) from linkage where cust_join_type = 'custs_not_in_campaign'), /* customer details dropped 1 */
        'in_campaign_not_custs', (select jsonb_agg(linkage) from linkage where cust_join_type = 'in_campaign_not_custs'), /* customers in campaign dropped 2 */
        'missingness_from_campaign', (select jsonb_agg(missingness_from_campaign) from missingness_from_campaign), /* customers with details tagged as missing in campaign */
        'missingness_from_customer_details', (select jsonb_agg(missingness_from_customer_details) from missingness_from_customer_details) /* customer ids in campaign tagged as missing details */
    )
    into result;

    return jsonb_pretty(result)::jsonb;

end;
$$;

/* RUN ALL AS FUNCTION */

select jsonb_pretty(run_campaign_analysis('delivery_club'));
```
