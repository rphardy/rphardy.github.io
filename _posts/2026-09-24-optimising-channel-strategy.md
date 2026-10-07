---
layout: post
title: Optimising App & Web Channel Strategy Using GA4 & Smartcard Data
image: /posts/ga4-pt-app-v-web-title-img2.png
tags: [GA4,Transport,Decision Framework,Python, BigQuery]
---

Our client, a state transport department, runs its journey planning, ticketing, and disruption alert services on both a mobile app and a website. With limited development capacity, they needed to know where to invest — and where to hold off.

# Table of contents

- [00. Project Overview](#project-overview)
  * [Context](#overview-context)
  * [Actions](#overview-actions)
  * [Results](#overview-results)
  * [Growth/Next Steps](#overview-growth)
  * [Key Definitions](#overview-definition)
- [01. Data & Instrumentation Overview](#data-overview)
- [02. Methodology Overview](#methodology-overview)
- [03. Phase 0: Instrumentation](#phase-0)
- [04. Phase 1: GA4 Baseline](#phase-1)
- [05. Stakeholder Checkpoint](#checkpoint)
- [06. Phase 2: Inferring User Intent](#phase-2)
- [07. Phase 3: Synthesis & Confidence Tiers](#phase-3)
- [08. Scenario B: Smartcard Linkage](#scenario-b)
- [09. Decision Summary](#decision-summary)
- [10. Application](#application)
- [11. Growth & Next Steps](#growth-next-steps)

---

# Project Overview {#project-overview}

### Context {#overview-context}

In mid-2025, Transport Victoria (formerly Public Transport Victoria) retired its standalone PTV website, combining journey planning, real-time information, and myki services in the single transport.vic.gov.au domain, citing the old site's end-of-life state and the ongoing cost of maintaining duplicate platforms. The PTV app, however, remains a separately maintained product, with its own release cycle that addresses real-time on-app accuracy and journey planning reliability issues as they arise.

This case study uses that real situation as its motivating context — to illustrate how a structured, phased GA4 analysis can find what work remains: where the app itself still needs investment, and where it may not. All data, figures, and dashboards below are mock — built to represent the kind of GA4 and myki data we could access under this engagement, and to demonstrate the analytical method against a known and verifiable real-world backdrop. The mock data does not represent Transport Victoria's actual results.

Transport Victoria's digital team maintains four core *features* across their app and website: real-time departures, saved trips, disruption alerts, and the journey planner itself. Each feature had usage on both platforms. For this analysis we assume that no one could say with confidence if this usage showed real need for both platforms, or usage that nobody had looked at closely enough to justify investment in across both platforms.

Transport Victoria needed to answer three specific questions before committing the next development cycle:

1. **Platform exclusivity** — should a feature exist on the app only, the website only, or both?
2. **Development priority** — within available development time, what should the team build next?
3. **Retirement candidates** — which feature costs money to maintain but has low usage on either platform?


### Actions {#overview-actions}

Rather than running a single usage-split analysis, we built a phased decision framework. The framework is structured so that the team can make easy decisions quickly, ensuring only genuinely ambiguous cases would take further analytical effort.

- **Phase 0** checked that a single GA4 property covered both the app and the website, so cross-platform comparison was possible
- **Phase 1** used the raw usage split to make a fast decision on any feature with a clear gap between app use and web use
- **A stakeholder checkpoint** changed the scope of the analysis for a feature where usage data alone wasn't the right focus. For example, where a feature had a legal requirement for communication
- **Phase 2** added context about device type, time of day, and user type to the features that weren't resolved by the raw split during Phase 1
- **Phase 3** combined every finding into one recommendation list. Each recommendation carries a confidence label. The label shows which findings came from direct observation and which were inferred
- **Scenario B** linked web sessions to physical smartcard touch-on and touch-off data once a privacy impact assessment (PIA) cleared. We submitted the PIA during Phase 0. Before the PIA cleared, we planned this step under 'Scenario A'. This step (once PIA did clear) confirmed one recommendation that had rested on an inference. The team did not simply assume that inference was correct.


### Results {#overview-results}

Every one of the three questions above now has an answer, supported by evidence:

**Platform exclusivity**

- One feature (saved_trips) is moving toward app-only status. The department will phase in this change over 6 months
- One feature (journey_planner) is confirmed as a genuine dual-platform feature. A physical movement data link supports this finding
- One feature (disruption_alerts) stays on both platforms by design. The team reframed the question away from a simple usage-share comparison

**Development priority**

- Highest priority: a channel-effectiveness fix. Measured user-action data supports this fix directly: to add a "replan trip" button to the web alert banner
- Next priority: a new feature for the web journey planner aimed at increasing web downloads, backed by the smartcard linkage: to add an app download link at the point where a user plans a trip on the web
- We do not recommend further development for the two lower-priority features this cycle.

**Retirement candidates**

- No feature looked like a clear case for immediate removal
- We flagged the web version of one feature (saved_trips) for a formal retirement review in 6 months


### Growth/Next Steps {#overview-growth}

We originally deferred the real_time_departures feature due to limited development capacity, not due to weak evidence. This question remains open. The case for continued investment in this feature is already strong, but was not formally re-examined once development resources were redirected elsewhere. We can reuse the smartcard linkage pipeline built for this project for future features.


### Key Definitions {#overview-definition}

This report refers to a recommendation's **confidence tier**. We define three tiers:

- **Observed** — We built this tier directly from a measured event. Examples include a click, a usage rate, or an action taken. The conclusion needs no interpretation
- **Directional** — We built this tier from a real pattern. The meaning of the pattern needed an inference. For example: "this usage pattern probably shows planning ahead, not idle browsing"
- **Confirmed** — this tier started as a directional finding. We then checked the finding against an independent, stronger form of evidence. The finding held up under this check

This distinction proved important, since while two of the three decisions rested entirely on observed evidence, one decision required the Confirmed tier.

We also distinguish two **scenarios**, depending on whether the pending privacy approval had cleared at the time a recommendation was delivered:

- **Scenario A** — this is the report we could deliver using GA4 data alone, before the privacy impact assessment (PIA) cleared. This version is complete and actionable on its own. Any directional finding would ship with its confidence tier clearly labelled as 'Directional'.
- **Scenario B** — this is the same report, updated after the PIA cleared. Smartcard touch-on and touch-off data became available at this point. The team used this data to test the one Directional finding, strengthening its evidence-base to Confirmed.

---

# Data & Instrumentation Overview {#data-overview}

We tracked usage across app and web using one custom GA4 event. We used parameters on this event, instead of many separate event names. This approach keeps every platform and feature directly comparable.

| **Field Name** | **Scope** | **Description** |
|---|---|---|
| feature_engaged | Event | Fires whenever a user meaningfully interacts with one of the four core features |
| feature_name | Event parameter | Names the feature: real_time_departures, saved_trips, disruption_alerts, or journey_planner |
| interaction_depth | Event parameter | How far the user got: viewed, interacted, or completed |
| lookup_lead_time_min | Event parameter | For real-time departures: the minutes between the lookup and the actual departure. This value is a proxy for imminent travel versus planned travel |
| alert_channel | Event parameter | For disruption alerts: push, in-app banner, or web banner |
| alert_action | Event parameter | Shows what the user did with an alert: dismissed, viewed detail, or replanned their trip. Each alert interaction fires one event, so events map one-to-one to outcomes |
| platform | Native (GA4 export) | GA4's own field — ANDROID, IOS, or WEB — collapsed to APP/WEB throughout this analysis |
| device.category | Native (GA4 export) | GA4's own field. We use this field to split web traffic into mobile web and desktop web in Phase 2 |
| user_id | Native (GA4 export) | GA4's login-only identity field. The department sets this field from a hashed internal account ID at sign-in. This field links to smartcard data in Scenario B and is absent for anonymous sessions |
| touch-on / touch-off | External (smartcard system) | Physical boarding and alighting records. We link these records in Scenario B to confirm one finding |
| account_card_bridge | External (ticketing system) | Maps each account_id to its linked card_id_hashed. One account may hold multiple cards. We join this table to user_id in Scenario B's Gate 2B and Tier 1 checks |

---

GA4 collects the platform, user_id, and device.category fields automatically. These don't require custom dimension registration, whereas the five event parameters above them are custom registered.


# Methodology Overview {#methodology-overview}

We answer three related but distinct questions (platform exclusivity, development priority, and retirement). Instead of running three separate analyses, we use one phased evidence pipeline to answer all three questions.

The evidence in this analysis ranges from directly observed usage splits to physical movement data. For this reason, we structured the work as a sequence of gated phases. Each phase only moves to the next phase when the current evidence cannot resolve the question:

- Phase 0: Instrumentation
- Phase 1: GA4 baseline (Gate 1 — fast-track check)
- Stakeholder checkpoint
- Phase 2: Context layer: Inferring user intent
- Phase 3: Synthesis (Gate 3A — confidence tiering)
- Scenario B: Smartcard linkage (Gate 2B — sample-size check)

Each phase breaks down into two kinds of work. a **query** produces a number from the data. A **judgement** applies a threshold that the data alone cannot set. Several of the gates (Gate 1, Gate 2B) exist to hand a decision to a person instead of automating the decision. This report keeps queries and judgements separate for this reason, shown here in the sequence they occur:

| # | Phase | Type | Step |
|---|---|---|---|
| 1 | Phase 1 | Query | Baseline usage-rate query — all four features, both platforms |
| 2 | Phase 1 | Judgement | Apply Gate 1's threshold — saved_trips fast-tracked, others carried forward |
| 3 | Phase 1 | Query | Interaction-depth breakdown for saved_trips, supporting the fast-track call |
| 4 | Stakeholder Checkpoint | Judgement | Stakeholder checkpoint — scope set: escalate, reframe, or defer each remaining feature |
| 5 | Phase 2 | Query | Device, timing, and new-vs-returning cuts on the escalated feature |
| 6 | Phase 2 | Query | Channel/outcome breakdown for the reframed feature |
| 7 | Phase 3 | Judgement | Synthesise findings into confidence tiers; apply Gate 3A |
| 8 | Between Phase 3 and Scenario B | — | Wait for privacy approval — independent of the analysis itself |
| 9 | Scenario B | Query | Gate 2B sample-size check — join via GA4's user_id, bridged to the ticketing system's account-card mapping |
| 10 | Scenario B | Judgement | Assess whether the linked sample clears the bar for individual-level analysis |
| 11 | Scenario B | Query | Deterministic (Tier 1) join and cohort (Tier 2) correlation, run independently |
| 12 | Scenario B | Judgement | Compare tiers, close Gate 3A, and package the final decision matrix |

Step 8 is a wait step. We could already deliver the Scenario A recommendation. This recommendation was complete and actionable before we knew if the smartcard linkage would become possible.

---

# Phase 0: Instrumentation {#phase-0}

Before any comparison between app and web is meaningful, both platforms need to report into the same place, in the same format.


### Setup

We checked the existing gtag and Firebase settings to confirm that both data streams report into the same GA4 property. We also checked the custom dimensions listed in the Data Overview table above. Most dimensions were already registered and mapped correctly (a small number were missing and added at this stage). We then validated event delivery for all platforms in DebugView, before we let the baseline window run.

The `lookup_lead_time_min` dimension was missing from the existing setup and needed to be derived. The department manages its tagging through Google Tag Manager. For this reason, we pushed the raw timestamps to the dataLayer at the point of lookup (Block 1): 

```javascript
// Pushed to the dataLayer by the existing lookup handler,
// at the moment the real-time departure result is returned
dataLayer.push({
  event: 'real_time_lookup',
  departure_time: departureTime.toISOString(),
  lookup_time: new Date().toISOString(),
  stop_id: stopId,
  route_id: routeId
});
```

then configured the minutes-until-departure calculation as a GTM variable (Block 2):

{% raw %}
```javascript
// GTM Custom JavaScript variable, referenced by the
// feature_engaged tag as the lookup_lead_time_min parameter
function() {
  var departure = new Date({{DLV - departure_time}});
  var lookup = new Date({{DLV - lookup_time}});
  return Math.round((departure - lookup) / 60000);
}
```
{% endraw %}

Block 2 is a GTM Custom JavaScript Variable — a small function GTM runs on demand.

{% raw %}{{DLV - departure_time}}{% endraw %} and {% raw %}{{DLV - lookup_time}}{% endraw %} are GTM's syntax for "Data Layer Variable" — they reach back into the dataLayer and pull out the two values Block 1 pushed.

new Date(...) converts each of those ISO text strings back into actual JavaScript date objects, so they can be subtracted.

departure - lookup : subtracting two Date objects in JavaScript gives the difference in milliseconds.

/ 60000 converts milliseconds to minutes (60,000 ms in a minute).

Math.round(...) rounds that to a whole minute.

return hands this final number back to GTM, which then attaches it to the feature_engaged event as the lookup_lead_time_min parameter — ready for use in GA4.

In block 1: routeId is available but should be read from the specific departure result being logged, in case one stop maps to more than one route.


### Validation

A platform breakdown check confirmed that both app platforms and the website reported data consistently. We ran this check before the baseline window began. This check would have caught a version-drift issue, if one had existed. For example, one platform's software build might predate a schema update.


### Outcome

Once we had confirmed the instrumentation was correct, the 21-day baseline window began.

---

# Phase 1: GA4 Baseline {#phase-1}

### Evidence Gathered

Raw `feature_engaged` usage rate by platform, across all four features, over the 21-day window.

This query works in three stages, each building on the last:


#### Stage 1

Pull the raw event data, and sort every user into "app" or "web." GA4 records the app's two platforms (iOS and Android) separately, but we report app to web as a whole. The first step collapses iOS and Android into a single APP group, keeping web as its own group. Because every phase from here on depends on this same definition, it's created once as a standalone table:

```sql
-- platform_group collapses ANDROID + IOS into APP here, and this
-- exact definition must be reused unchanged everywhere else.
-- Source: GA4 BigQuery export (events_* daily tables)

CREATE OR REPLACE TABLE `project.analytics_derived.baseline_events_21d` AS
SELECT
  event_date,
  user_pseudo_id,
  user_id,
  event_name,
  event_params,
  device.category AS device_category,
  event_timestamp,
  user_first_touch_timestamp,
  CASE
    WHEN platform IN ('ANDROID', 'IOS') THEN 'APP'
    WHEN platform = 'WEB' THEN 'WEB'
  END AS platform_group
FROM 
 `project.analytics_XXXXXXX.events_*`
WHERE 
 _TABLE_SUFFIX BETWEEN '20260811' AND '20260831';
```


For illustration, 5 randomly selected rows from this table might be viewed as:

| event_date | user_pseudo_id | user_id | event_name | feature_name | interaction_depth | device_category | platform_group | event_timestamp | user_first_touch_timestamp |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-12 | 8841029.552 | null | feature_engaged | journey_planner | viewed | mobile | WEB | 1786523400000000 | 1786523400000000 |
| 2026-08-14 | 2210984.117 | acct_9931 | feature_engaged | real_time_departures | interacted | mobile | APP | 1786701600000000 | 1786259700000000 |
| 2026-08-15 | 5567321.884 | acct_4410 | feature_engaged | saved_trips | completed | desktop | WEB | 1786782600000000 | 1784711100000000 |
| 2026-08-18 | 9042731.209 | null | feature_engaged | disruption_alerts | — | mobile | APP | 1787040000000000 | 1786995600000000 |
| 2026-08-20 | 5567321.884 | acct_4410 | feature_engaged | journey_planner | viewed | desktop | WEB | 1787212800000000 | 1784711100000000 |


note that the column event_name has been scoped (in next steps) to events where a feature was engaged. The column feature_name has also been defined as the value.string_value for the event_params field where the key = 'feature_name': giving one of the four event types (as strings) that we are interested in. 


#### Stage 2 

Count two different things, side by side. From that pool of events, the query counts:
* Active users — anyone who did anything at all on each platform in the 21-day window (the denominator)
* Engaged users — of those, anyone who specifically interacted with one of the four features being studied, broken out feature by feature (the numerator)

```sql
-- Feature usage rate by platform (app vs web), 21-day baseline window
-- Source: analytics_derived.baseline_events_21d

WITH active_users AS (
  SELECT
    platform_group,
    COUNT(DISTINCT user_pseudo_id) AS active_users
  FROM
   `project.analytics_derived.baseline_events_21d`
  GROUP BY
   platform_group
),

feature_users AS (
  SELECT
    platform_group,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') AS feature_name,
    COUNT(DISTINCT user_pseudo_id) AS engaged_users
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   event_name = 'feature_engaged'
  GROUP BY
   platform_group,
   feature_name
)
```

#### Stage 3 

Divide the two, per feature and per platform. The final step joins those two counts together and calculates what share of each platform's active users actually engaged with each feature. This percentage is the usage rate shown as bars in the Phase 1 chart.

```sql

SELECT
  f.feature_name,
  f.platform_group AS platform,
  f.engaged_users,
  a.active_users,
  ROUND(f.engaged_users / a.active_users * 100, 1) AS usage_rate_pct
FROM
 feature_users f
JOIN active_users a USING (platform_group)
ORDER BY
 f.feature_name,
 f.platform_group;

```

### Baseline Feature Usage Dashboard

![alt text](/img/posts/phase1-feature-usage-baseline.png "Feature Usage Baseline by Platform")


### Gate 1 — Fast-Track Check

A feature with a clear platform gap needs no further analysis. We define a clear gap as roughly under 10% usage on one platform against over 50% usage on the other platform. We resolve this type of feature immediately.

**saved_trips** met this criteria directly. The feature had 9% engagement on the website against 61% engagement on the app. Thus we resolved this feature at this point, to be developed in future cycles as app-only.

The interaction depth data on the website strengthens this case further. Within that 9% figure, most users only viewed the feature. These users did not create a saved trip.

A user who completes a saved trip typically also triggers a `viewed` and an `interacted` event along the way, so counting every event at every depth would count the same person three times. Each user is instead classified by the deepest stage they reached, ranking `completed` above `interacted` above `viewed`, before counting:

```sql
WITH active_users AS (
  SELECT
   COUNT(DISTINCT user_pseudo_id) AS active_users
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB'
),

user_max_depth AS (
  SELECT
    user_pseudo_id,
    MAX(
      CASE (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'interaction_depth')
        WHEN 'completed' THEN 3
        WHEN 'interacted' THEN 2
        WHEN 'viewed' THEN 1
      END
    ) AS depth_rank
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB' AND 
   event_name = 'feature_engaged' AND 
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'saved_trips'
  GROUP BY
   user_pseudo_id
)

SELECT
  CASE depth_rank WHEN 3 THEN 'completed' WHEN 2 THEN 'interacted' WHEN 1 THEN 'viewed' END AS interaction_depth,
  COUNT(DISTINCT user_pseudo_id) AS engaged_users,
  a.active_users,
  ROUND(COUNT(DISTINCT user_pseudo_id) / a.active_users * 100, 1) AS usage_rate_pct
FROM
 user_max_depth
CROSS JOIN
 active_users a
GROUP BY
 depth_rank,
 a.active_users
ORDER BY
 depth_rank DESC;
```

This returns 7% viewed, 1.5% interacted, and 0.5% completed. These three figures sum to the 9% headline rate. No user is counted twice.


### Outcome

We resolved one of the four features using Observed evidence alone. The remaining three features carried forward to the next phase.

---

# Stakeholder Checkpoint {#checkpoint}

Not every open question is best answered by more usage data. At this point, we reviewed the remaining three features with the client team directly. We adjusted the scope as follows:

- **journey_planner** — the client team escalated this feature as the clear development priority, since its raw usage pattern actively contradicted the working assumption about how app and web were being used. The working assumption was that the app would see the heaviest use of core features, such as trip planning. The assumption predicted that the website would serve as a secondary, occasional-use channel. However, the raw usage split showed the opposite result: 88% of the website's active users engaged with journey_planner, against only 45% on the app - the opposite of the prediction. This unexpected result made the feature the clear priority for closer analysis
- **disruption_alerts** was reframed entirely — from a platform-investment question to a channel-effectiveness question. The communications team flagged a consistency requirement across all alert channels. A simple usage-share comparison could not capture this requirement
- **real_time_departures** was deferred, with the client team accepting its already-strong usage gap, 82% against 34%, as sufficient evidence for continued app-first development. The team chose not to spend further analysis time on this feature, because journey_planner had already become the priority


### Outcome

**journey_planner** proceeds to Phase 2, under its original terms. **disruption_alerts** proceeds to Phase 2, under reframed terms. **real_time_departures** is deferred, exiting the active analysis by a team decision, not because of weak evidence.

---

# Phase 2: Inferring User Intent {#phase-2}

### Evidence Gathered

For the escalated feature, **journey_planner**, we added three layers of context: device category (mobile web versus desktop web), time-of-day pattern, and the share of new users against returning users. A platform split could not directly show these factors on its own.


### First Look to Formal Query

Each cut started with a simple pass at the data, before deciding what — if anything — needed to be measured more precisely. All three queries below that form this analysis read from the baseline table created in Phase 1. For this reason, each query uses the same platform and window definitions.

**Device category** started with a plain count of engaged users on mobile web against desktop web. This first count favoured mobile web.

```sql
SELECT
 device_category,
 COUNT(DISTINCT user_pseudo_id) AS engaged_users
FROM
 `project.analytics_derived.baseline_events_21d`
WHERE
 platform_group = 'WEB' AND 
 event_name = 'feature_engaged' AND
 (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
GROUP BY
 device_category;
```

We then divided each device's engaged users by its own active-user base. This calculation told a different story: desktop web engages at 93%, mobile web at 81%. Both rates are high, and much closer together than the raw counts suggested.

```sql
WITH active_users AS (
  SELECT
   device_category,
   COUNT(DISTINCT user_pseudo_id) AS active_users
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB'
  GROUP BY
   device_category
),

feature_users_by_device AS (
  SELECT
   device_category,
   COUNT(DISTINCT user_pseudo_id) AS engaged_users
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB' AND event_name = 'feature_engaged'
    AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
  GROUP BY
   device_category
)

SELECT
  f.device_category,
  f.engaged_users,
  a.active_users,
  ROUND(f.engaged_users / a.active_users * 100, 1) AS usage_rate_pct
FROM
 feature_users_by_device f
JOIN active_users a USING (device_category)
ORDER BY
 f.device_category;
```

**Timing** showed a clear pattern early. A simple per-hour count of `journey_planner` activity, split by platform, produced two distinct patterns: app usage spiked sharply around the morning and evening commute hours. Website usage stayed comparatively flat, with a tilt toward evenings.

```sql
SELECT
  platform_group,
  EXTRACT(HOUR FROM TIMESTAMP_MICROS(event_timestamp) AT TIME ZONE 'Australia/Melbourne') AS local_hour,
  COUNT(*) AS events
FROM
 `project.analytics_derived.baseline_events_21d`
WHERE
 event_name = 'feature_engaged'
  AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
GROUP BY
 platform_group,
 local_hour
ORDER BY
 platform_group,
 local_hour;
```

This pattern set the peak and off-peak boundary for the rest of the analysis: 7–9am and 4–6pm count as peak hours. We then split this data further by device, within the web platform. This split sharpened the comparison: mobile web showed 27% peak usage and 73% off-peak usage. Desktop web showed 19% peak usage and 81% off-peak usage. These two web figures sit far closer to each other than either figure sits to the app's 72% peak share.

```sql
WITH classified_events AS (
  SELECT
    platform_group,
    device_category,
    CASE
      WHEN EXTRACT(HOUR FROM TIMESTAMP_MICROS(event_timestamp) AT TIME ZONE 'Australia/Melbourne') BETWEEN 7 AND 9
        OR EXTRACT(HOUR FROM TIMESTAMP_MICROS(event_timestamp) AT TIME ZONE 'Australia/Melbourne') BETWEEN 16 AND 18
      THEN 'peak' ELSE 'off_peak'
    END AS time_window
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   event_name = 'feature_engaged'
    AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
)

SELECT
  platform_group,
  device_category,
  time_window,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY platform_group, device_category), 1) AS usage_rate_pct
FROM
 classified_events
GROUP BY
 platform_group,
 device_category,
 time_window
ORDER BY
 platform_group,
 device_category,
 time_window;
```

**New versus returning** users showed a large, clear gap from the start.

```sql
-- Baseline window start (2026-08-11) must match the _TABLE_SUFFIX
-- start date used to create baseline_events_21d
SELECT
  platform_group,
  TIMESTAMP_MICROS(user_first_touch_timestamp) >= TIMESTAMP('2026-08-11') AS is_new,
  COUNT(*) AS engagement_events
FROM
 `project.analytics_derived.baseline_events_21d`
WHERE
 event_name = 'feature_engaged'
  AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
GROUP BY
 platform_group,
 is_new;
```

We counted each user once, instead of once per engagement event. This method held the split at 81% returning users on the app, against 64% new users on the website. These two figures are close to the inverse of each other.

```sql
-- Baseline window start (2026-08-11) must match the _TABLE_SUFFIX
-- start date used to create baseline_events_21d
WITH engaged_users AS (
  SELECT DISTINCT
    platform_group,
    user_pseudo_id,
    user_first_touch_timestamp
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   event_name = 'feature_engaged' AND
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
)

SELECT
  platform_group,
  TIMESTAMP_MICROS(user_first_touch_timestamp) >= TIMESTAMP('2026-08-11') AS is_new,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY platform_group), 1) AS usage_rate_pct
FROM
 engaged_users
GROUP BY
 platform_group,
 is_new;
```

Together, these three cuts rule out three different alternative explanations for the same question. The device category cut rules out the explanation "it's simply a desktop tool" by the engagement pattern shown. The timing cut rules out the explanation "web usage happens throughout the day the same way app usage does, just less often" by the peak/off-peak use pattern. The new-versus-returning cut rules out the explanation "the same core group of people just prefer using the website sometimes" by the difference in usage between new and returning users. 

What's left, once each of those is set aside, is the interpretation carried into the next section: web usage clusters ahead of travel, largely independent of device, and this usage disproportionately belongs to people who have not yet installed the app.


### Disruption Alerts: Channel and Outcome


For the reframed feature, no first look was needed: alert channel and the action taken are both recorded directly on each engagement event, so both cuts are simple shares of `disruption_alerts` engagement events (one event per alert interaction, as defined in the Data Overview).

Push accounts for 68% of alert engagement events, against 22% for the in-app banner and 10% for the web banner.

```sql
SELECT
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'alert_channel') AS alert_channel,
  COUNT(*) AS engagement_events,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS usage_rate_pct
FROM
  `project.analytics_derived.baseline_events_21d`
WHERE
  event_name = 'feature_engaged'
  AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'disruption_alerts'
GROUP BY 
  alert_channel
ORDER BY 
  engagement_events DESC;
```

The outcome that follows each alert differs sharply by channel: a push alert leads to a replanned trip 41% of the time, against 18% for the web banner, which is dismissed 74% of the time.

```sql
WITH alert_events AS (
  SELECT
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'alert_channel') AS alert_channel,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'alert_action') AS alert_action
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE event_name = 'feature_engaged'
    AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'disruption_alerts'
)

SELECT
  alert_channel,
  alert_action,
  COUNT(*) AS engagement_events,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY alert_channel), 1) AS usage_rate_pct
FROM
 alert_events
GROUP BY
 alert_channel,
 alert_action
ORDER BY
 alert_channel,
 alert_action;
```

### User Intent Dashboard

![alt text](/img/posts/phase2-proxy-context-analysis.png "Proxy-Context Analysis")


### Outcome

We reframed the escalated feature's usage pattern, from an app-versus-web comparison to a pre-trip-versus-in-transit comparison. Three findings support this reframe: web engagement rates are close between mobile and desktop (81% vs 93%), ruling out a simple desktop-tool explanation; web usage clusters off-peak (73–81%) while app usage clusters in commute peaks (72%), pointing to a difference in when people act rather than where; and web skews toward new users (64%) while the app skews toward returning users (81%), suggesting the two platforms serve different moments in a traveller's journey, not just different audiences. Together, these findings support a directional conclusion, that people use the website to plan ahead and the app to travel in the moment, but this remains an inference about user intent, since the data alone cannot fully confirm it. 

Based on this, we recommend reframing the feature's fix directly: redesign the web alert's call to action.

---

# Phase 3: Synthesis & Confidence Tiers {#phase-3}

### Evidence Gathered

This step reorganises every finding from Phases 1 and 2 into the three confidence tiers defined earlier: classified as one of Observed, Directional, or Confirmed.

![alt text](/img/posts/phase3-synthesis.png "Confidence-Tiered Synthesis")


### Gate 3A — Which Findings Need Further Confirmation?

Only a recommendation that rests on an inferred interpretation needs further confirmation. Of the three resolved features at this point, only one recommendation qualified. The reframed alert-channel fix and the fast-tracked feature from Phase 1 were both already Observed. No further data source could add precision to these two findings.


### Outcome

We shipped two recommendations as final at this stage (for **saved_trips** - stop new development investment on web, and for **disruption_alerts** - implement the 'replan trip' CTA on the web banner). We flagged one recommendation explicitly as directional (for **journey_planner** - don't deprioritise web development, account for usage pattern on web). This recommendation is worth revisiting if stronger evidence becomes available.

---

# Scenario B: Smartcard Linkage {#scenario-b}

Once a pending privacy impact assessment cleared, we gained access to physical smartcard touch-on/touch-off records — a genuinely independent form of evidence for the one flagged recommendation. These data are stored in two external tables (account_card_bridge, touch_events) which appear as:

transport_core.account_card_bridge

| account_id | card_id_hashed |
|---|---|
| acct_4410 | 7f3a9c1e |
| acct_4410 | b12de08a |
| acct_9931 | 44c7f0d2 |
| acct_2207 | 9a01ee3c |
| acct_5583 | e620b7f1 |

smartcard_derived.touch_events

| card_id_hashed | stop_id | touch_type | event_timestamp |
|---|---|---|---|
| 7f3a9c1e | stop_2291 | on | 2026-08-20 07:42:00 |
| 7f3a9c1e | stop_4410 | off | 2026-08-20 08:15:00 |
| b12de08a | stop_2291 | on | 2026-08-21 18:03:00 |
| 44c7f0d2 | stop_1187 | on | 2026-08-14 17:52:00 |
| 9a01ee3c | stop_3302 | on | 2026-08-16 09:10:00 |


### Sample-Size Check (Gate 2B)

Only a small share of web journey_planner users could be linked to a smartcard, matching the new-user skew that Phase 2 already found. A first-touch audience is far less likely to already be logged in.

A link runs through GA4's user_id field. This field is set only at login and links to the ticketing system's account-to-card mapping. Where one account holds more than one card, the query still counts the user only once.

We quantified: web-based journey planner users, logged in sessions, linked accounts and users, then calculated the percentage of distinct *users* that could be linked to a live session.

```sql
WITH web_journey_planner_users AS (
  SELECT
   DISTINCT user_pseudo_id
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB' AND
   event_name = 'feature_engaged' AND
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
),

logged_in_sessions AS (
  SELECT
   DISTINCT
    user_pseudo_id, 
    user_id
   FROM
    `project.analytics_derived.baseline_events_21d`
   WHERE
    user_id IS NOT NULL
),

linked_accounts AS (
  SELECT
   DISTINCT account_id
  FROM
   `project.transport_core.account_card_bridge`
),

linked_users AS (
  SELECT
   DISTINCT l.user_pseudo_id
  FROM
   logged_in_sessions l
  JOIN linked_accounts a ON l.user_id = a.account_id
)

SELECT
  COUNT(DISTINCT w.user_pseudo_id) AS journey_planner_web_users,
  COUNT(DISTINCT lu.user_pseudo_id) AS linked_users,
  ROUND(COUNT(DISTINCT lu.user_pseudo_id) / COUNT(DISTINCT w.user_pseudo_id) * 100, 1) AS linked_pct
FROM
 web_journey_planner_users w
LEFT JOIN linked_users lu USING (user_pseudo_id);
```

Of 36,260 web `journey_planner` users, we linked 2,176 users to a smartcard - 6.0%. 
This sample is too small to support further segments in the next step: a deterministic check, for example by device or by time-of-day. 
It is large enough however, for one aggregate figure, to next check touch-on events against planned origin.


### Tier 1 — Deterministic Check


For the 2,176 linked users, each web `journey_planner` session is checked against a touch-on at that same session's planned origin stop, within 24 hours. An account with more than one linked card counts as matched if *any* of its cards touch on. Matching is at the person level, not the card level.

```sql
WITH logged_in_sessions AS (
  SELECT
   DISTINCT user_pseudo_id,
   user_id
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   user_id IS NOT NULL
),

user_cards AS (
  SELECT
   l.user_pseudo_id,
   b.card_id_hashed
  FROM
   logged_in_sessions l
  JOIN `project.transport_core.account_card_bridge` b ON l.user_id = b.account_id
),

planning_sessions AS (
  SELECT
    e.user_pseudo_id,
    e.event_timestamp AS planned_at,
    (SELECT value.string_value FROM UNNEST(e.event_params) WHERE key = 'origin_stop_id') AS origin_stop_id
  FROM
   `project.analytics_derived.baseline_events_21d` e
  JOIN user_cards uc USING (user_pseudo_id)
  WHERE
   e.platform_group = 'WEB' AND
   e.event_name = 'feature_engaged' AND
   (SELECT value.string_value FROM UNNEST(e.event_params) WHERE key = 'feature_name') = 'journey_planner'
  GROUP BY
   e.user_pseudo_id,
   e.event_timestamp,
   origin_stop_id
),

matched AS (
  SELECT
    p.user_pseudo_id,
    p.planned_at,
    MIN(t.event_timestamp) AS first_touch_on
  FROM
   planning_sessions p
  JOIN user_cards uc ON uc.user_pseudo_id = p.user_pseudo_id
  LEFT JOIN `project.smartcard_derived.touch_events` t
    ON t.card_id_hashed = uc.card_id_hashed
    AND t.stop_id = p.origin_stop_id
    AND t.touch_type = 'on'
    AND t.event_timestamp BETWEEN p.planned_at AND TIMESTAMP_ADD(p.planned_at, INTERVAL 24 HOUR)
  GROUP BY
   p.user_pseudo_id,
   p.planned_at
)

SELECT
  COUNT(*) AS planning_sessions,
  COUNTIF(first_touch_on IS NOT NULL) AS matched_within_24h,
  COUNTIF(TIMESTAMP_DIFF(first_touch_on, planned_at, HOUR) <= 3) AS matched_within_3h,
  ROUND(COUNTIF(first_touch_on IS NOT NULL) / COUNT(*) * 100, 1) AS match_rate_24h_pct
FROM
 matched;
```

Across 2,910 web `journey_planner` sessions from linked users, a touch-on at the planned origin followed 39% of sessions within 24 hours. A touch-on followed 22% of sessions within 3 hours.

The 39% figure only has meaning when compared against a baseline. We did not compare this figure to an unrelated stop, or to a user's single most-frequent stop. Either choice would distort the comparison in a different direction. Instead, each of the same 2,176 users receives one randomly chosen day on which they did not plan a trip. We pair this day with one of the user's own real origin stops and a random reference time. We then check this pairing against the same 24-hour window.

```sql
WITH linked_users AS (
  SELECT
   DISTINCT l.user_pseudo_id
  FROM (
    SELECT
     DISTINCT user_pseudo_id, user_id
    FROM
     `project.analytics_derived.baseline_events_21d`
    WHERE
     user_id IS NOT NULL
  ) l
  JOIN `project.transport_core.account_card_bridge` b ON l.user_id = b.account_id
),

user_planning_days AS (
  SELECT
   DISTINCT user_pseudo_id,
   DATE(TIMESTAMP_MICROS(event_timestamp), 'Australia/Melbourne') AS planning_date
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB' AND
   event_name = 'feature_engaged' AND
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner' AND
   user_pseudo_id IN (SELECT user_pseudo_id FROM linked_users)
),

user_origin_stops AS (
  SELECT
   DISTINCT user_pseudo_id,
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'origin_stop_id') AS origin_stop_id
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   platform_group = 'WEB' AND
   event_name = 'feature_engaged' AND
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner' AND
   user_pseudo_id IN (SELECT user_pseudo_id FROM linked_users)
),

baseline_days AS (
  SELECT
   u.user_pseudo_id,
   d AS candidate_date
  FROM
   linked_users u
  CROSS JOIN UNNEST(GENERATE_DATE_ARRAY('2026-08-11', '2026-08-31')) AS d
  LEFT JOIN user_planning_days p
    ON p.user_pseudo_id = u.user_pseudo_id AND p.planning_date = d
  WHERE p.user_pseudo_id IS NULL
),

baseline_day_pick AS (
  SELECT
   user_pseudo_id,
   candidate_date
  FROM
   baseline_days
  -- The next line keeps exactly one randomly chosen row per user, from their
  -- set of candidate dates. QUALIFY filters on a window function's
  -- result, after it's computed — WHERE can't do this, since it
  -- filters before window functions run. BigQuery, Snowflake, and
  -- Databricks support QUALIFY; standard PostgreSQL, MySQL, and
  -- SQL Server do not.
  QUALIFY 
   ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY RAND()) = 1
),

baseline_stop_pick AS (
  SELECT
   user_pseudo_id,
   origin_stop_id
  FROM
   user_origin_stops
  QUALIFY 
   ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY RAND()) = 1
),

baseline_reference AS (
  SELECT
   d.user_pseudo_id,
   s.origin_stop_id,
   TIMESTAMP_ADD(
     TIMESTAMP(d.candidate_date, 'Australia/Melbourne'),
     INTERVAL CAST(RAND() * 86400 AS INT64) SECOND
   ) AS pseudo_planned_at
  FROM
   baseline_day_pick d
  JOIN baseline_stop_pick s USING (user_pseudo_id)
),

user_cards AS (
  SELECT
   DISTINCT l.user_pseudo_id,
   b.card_id_hashed
  FROM (
   SELECT DISTINCT user_pseudo_id, user_id
   FROM `project.analytics_derived.baseline_events_21d`
   WHERE user_id IS NOT NULL
  ) l
  JOIN
   `project.transport_core.account_card_bridge` b ON l.user_id = b.account_id
),

baseline_matched AS (
  SELECT
   r.user_pseudo_id,
   r.pseudo_planned_at,
   MIN(t.event_timestamp) AS first_touch_on
  FROM
   baseline_reference r
  JOIN user_cards uc ON uc.user_pseudo_id = r.user_pseudo_id
  LEFT JOIN `project.smartcard_derived.touch_events` t
   ON t.card_id_hashed = uc.card_id_hashed
   AND t.stop_id = r.origin_stop_id
   AND t.touch_type = 'on'
   AND t.event_timestamp BETWEEN r.pseudo_planned_at AND TIMESTAMP_ADD(r.pseudo_planned_at, INTERVAL 24 HOUR)
  GROUP BY
   r.user_pseudo_id, r.pseudo_planned_at
)

SELECT
  COUNT(*) AS baseline_instances,
  COUNTIF(first_touch_on IS NOT NULL) AS matched_within_24h,
  ROUND(COUNTIF(first_touch_on IS NOT NULL) / COUNT(*) * 100, 1) AS baseline_match_rate_pct
FROM
 baseline_matched;
```

The baseline rate is 11%, placing the actual sessions' 39% at roughly a 3.5× lift. One caveat to this lift figure is worth noting clearly: the actual sessions' planning timestamps are real, but the baseline's is a synthetic stand-in: a randomly chosen moment on a day when the user did not plan a trip. This baseline is the closest symmetric comparison available, but it is not a perfect comparison. Thus the 39% and 11% figures shouldn't be read as equally precise measurements.


### Tier 2 — Cohort Check


Tier 1 only reaches the 6% of `journey_planner` web users who are logged in and linked to myki. This population skews toward habitual users. Phase 2 found earlier that web usage as a whole over-represents the opposite group: new users. Tier 2 checks the same question at population scale instead, comparing hourly GA4 session volume against hourly touch-on volume, with no individual-level matching between the two.

For this, we did not assume where a web planning session and a resulting trip would line up in time. Instead, this check tests a full range of time lags and lets the data show where the relationship is strongest.


```sql
-- First look: correlation between journey_planner session volume and
-- touch-on volume, at a spread of lags, per platform and device.
-- Aggregate counts only, no individual identity is used.

WITH hourly_sessions AS (
  SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(event_timestamp), HOUR) AS session_hour,
    platform_group,
    device_category,
    COUNT(DISTINCT user_pseudo_id) AS sessions
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   event_name = 'feature_engaged' AND 
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
  GROUP BY
   session_hour,
   platform_group,
   device_category
),

hourly_touch_ons AS (
  SELECT
    TIMESTAMP_TRUNC(event_timestamp, HOUR) AS touch_hour,
    COUNT(*) AS touch_ons
  FROM
   `project.smartcard_derived.touch_events`
  WHERE
   touch_type = 'on'
  GROUP BY
   touch_hour
)

SELECT
  s.platform_group,
  s.device_category,
  lag_hours,
  ROUND(CORR(s.sessions, t.touch_ons), 2) AS correlation
FROM
 hourly_sessions s
CROSS JOIN UNNEST([0, 1, 2, 24, 27, 30]) AS lag_hours
JOIN hourly_touch_ons t
  ON t.touch_hour = TIMESTAMP_ADD(s.session_hour, INTERVAL lag_hours HOUR)
GROUP BY
 s.platform_group,
 s.device_category,
 lag_hours
ORDER BY
 s.platform_group,
 s.device_category,
 lag_hours;
```

| platform_group | device_category | lag_hours | correlation |
|---|---|---|---|
| APP | mobile | 0 | 0.68 |
| APP | mobile | 1 | 0.74 |
| APP | mobile | 2 | 0.70 |
| APP | mobile | 24 | 0.21 |
| WEB | mobile | 1 | 0.19 |
| WEB | mobile | 24 | 0.52 |
| WEB | mobile | 27 | 0.58 |
| WEB | mobile | 30 | 0.54 |
| WEB | desktop | 1 | 0.15 |
| WEB | desktop | 24 | 0.61 |
| WEB | desktop | 27 | 0.57 |
| WEB | desktop | 30 | 0.50 |

Each group shows a clear rise and fall, around a different point in time. The app peaks and falls off within a couple of hours, while both web device types stay elevated across a much wider band a day later. Instead of finding the peak by eye from this short list, the next query sweeps the full range automatically. The query then keeps only the single best lag value (by strongest correlation) for each group:

```sql
-- Full scan: sweep every hourly lag from 0 to 36h and keep only
-- the peak correlation for each platform and device.

WITH hourly_sessions AS (
  SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(event_timestamp), HOUR) AS session_hour,
    platform_group,
    device_category,
    COUNT(DISTINCT user_pseudo_id) AS sessions
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   event_name = 'feature_engaged' AND 
   (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
  GROUP BY
   session_hour,
   platform_group,
   device_category
),

hourly_touch_ons AS (
  SELECT
    TIMESTAMP_TRUNC(event_timestamp, HOUR) AS touch_hour,
    COUNT(*) AS touch_ons
  FROM
   `project.smartcard_derived.touch_events`
  WHERE
   touch_type = 'on'
  GROUP BY
   touch_hour
),

lag_correlations AS (
  SELECT
    s.platform_group,
    s.device_category,
    lag_hours,
    CORR(s.sessions, t.touch_ons) AS correlation
  FROM
   hourly_sessions s
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, 36)) AS lag_hours
  JOIN hourly_touch_ons t
    ON t.touch_hour = TIMESTAMP_ADD(s.session_hour, INTERVAL lag_hours HOUR)
  GROUP BY
   s.platform_group,
   s.device_category,
   lag_hours
)

SELECT
 platform_group,
 device_category,
 lag_hours,
 ROUND(correlation, 2) AS correlation
FROM
 lag_correlations
QUALIFY 
 ROW_NUMBER() OVER (
  PARTITION BY platform_group, device_category ORDER BY correlation DESC
  ) = 1
ORDER BY
 platform_group,
 device_category;
```

| platform_group | device_category | lag_hours | correlation |
|---|---|---|---|
| APP | mobile | 1 | 0.74 |
| WEB | desktop | 24 | 0.61 |
| WEB | mobile | 27 | 0.58 |

The peak lag value is itself part of the finding, not only the correlation strength at that point. App `journey_planner` engagement peaks at 1 hour (r=0.74), desktop web at 24 hours (r=0.61), mobile web at 27 hours (r=0.58). App usage links to a trip almost immediately. Both web device types sit roughly a day ahead of a trip. The two web device types sit close enough to each other that device type is not the cause of the difference from the app.

One limitation is worth noting. A correlation computed over roughly 500 hourly buckets, 21 days times 24 hours, per group, carries a wider margin of error than the two-decimal figures suggest. We can trust the shape and the ordering across groups but we should not treat the difference between 0.58 and 0.61 as meaningful between the two device types.

![alt text](/img/posts/scenario-b-confirmation.png "Smartcard Linkage Confirmation")


### Outcome

Tier 1's 39% touch-on rate, against an 11% baseline, is a real lift. This result rests on a thin, wide-margin sample of only 2,176 linked users. Thus, the result is precise about direction and less precise about size. Tier 2 answers a different question, using the full population with no identity linkage. It finds that a `journey_planner` session's timing link to a touch-on peaks at 24 to 27 hours later on the website, regardless of device. On the app, this link peaks almost immediately, at 1 hour.

Neither check alone could have closed Gate 3A. For Tier 1: a small deterministic sample can show the correct direction. However, this sample may have had a 6% linked population that is unrepresentative. For Tier 2: A population-scale correlation can show the correct timing pattern. However, this correlation cannot confirm that any single trip was actually planned in advance. Together, the two checks cover each other's weak point. Tier 1 confirms the *size* of the effect for the subset of users who logged in. Tier 2 confirms that the *pattern* holds across everyone, including the 94% of users that Tier 1 could never see.

Both checks point in the same direction. For this reason, journey_planner moves from Directional to Confirmed. The recommendation itself does not change: keep the feature on both platforms, and add an app-download prompt at the point of web planning. The recommendation now rests on two independent forms of evidence, instead of one inferred pattern. The smartcard linkage confirmed the Phase 2 finding, and Gate 3A is now closed for the last remaining feature: journey_planner.

---

# Decision Summary {#decision-summary}

| **Feature** | **Recommendation** | **Confidence** |
|---|---|---|
| saved_trips | Stop new development on web; keep live, review for retirement in 6 months | Observed |
| journey_planner | Keep dual-platform; add an app-download prompt at the point of web planning | Confirmed |
| disruption_alerts | Redesign the web alert with an explicit "replan trip" call to action | Observed |
| real_time_departures | Continue app-first investment; no new analysis this cycle | Deferred |

Two of the three project decisions, platform exclusivity and retirement candidates, were resolved almost entirely on Observed evidence. The physical-movement linkage mattered for exactly one recommendation, which is itself a useful finding: the higher-governance data source was worth requesting for a narrow, specific reason, not as a blanket assumption that more data is always better.

---

# Application {#application}

The client's development team now has a ranked, evidence-based list of work, instead of a flat feature list. The disruption alert redesign leads this list, backed by directly measured action-rate data. The journey planner's app-download prompt follows, backed by the smartcard-confirmed conversion case. We recommend no further web investment for the saved trips feature, pending its 6-month retirement review.

---

# Growth & Next Steps {#growth-next-steps}

The client's development team deferred one feature due to limited capacity: `real_time_departures`. This feature still has an open, evidence-backed case for continued app investment that was never formally revisited once the team's attention moved elsewhere. The smartcard linkage pipeline built for this project remains available for any future feature whose inferred recommendation a physical movement record could actually confirm or refute.
