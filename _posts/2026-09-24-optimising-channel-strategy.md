---
layout: post
title: WIP - Optimising App & Web Channel Strategy Using GA4 & Smartcard Data
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
  * [Key Definition](#overview-definition)
- [01. Data & Instrumentation Overview](#data-overview)
- [02. Methodology Overview](#methodology-overview)
- [03. Phase 0: Instrumentation](#phase-0)
- [04. Phase 1: GA4 Baseline](#phase-1)
- [05. Stakeholder Checkpoint](#checkpoint)
- [06. Phase 2: Proxy-Context Analysis](#phase-2)
- [07. Phase 3: Synthesis & Confidence Tiers](#phase-3)
- [08. Scenario B: Smartcard Linkage](#scenario-b)
- [09. Decision Summary](#decision-summary)
- [10. Application](#application)
- [11. Growth & Next Steps](#growth-next-steps)

---

# Project Overview {#project-overview}

### Context {#overview-context}

In mid-2025, Transport Victoria (formerly Public Transport Victoria) retired its standalone PTV website, combining journey planning, real-time information, and myki services in the single transport.vic.gov.au domain, citing the old site's end-of-life state and the ongoing cost of maintaining duplicate platforms. The PTV app, however, remains a separately maintained product, with its own release cycle that addresses real-time on-app accuracy and journey planning reliability issues as they arise.

This case study uses that real, already-partly-resolved situation as its motivating context — to illustrate how a structured, phased GA4 analysis could approach what remains open: where the app itself still warrants investment, and where it may not. All data, figures, and dashboards below are mock — built to represent the kind of GA4 and myki data we would have direct access to under this engagement, and to demonstrate the analytical method against a known and verifiable real-world backdrop, not to represent Transport Victoria's actual reported results.

Transport Victoria's digital team maintains four core features across their app and website: real-time departures, saved trips, disruption alerts, and the journey planner itself. Each feature had usage on both platforms, but no one could say with confidence whether that reflected genuine need for both, or just usage nobody had looked at closely enough to challenge.

Transport Victoria needed to answer three specific questions before committing the next development cycle:

1. **Platform exclusivity** — should a feature exist on app only, web only, or both?
2. **Dev priority** — within available capacity, what gets built next?
3. **Retirement candidates** — what's costing maintenance but barely used, on either platform?

### Actions {#overview-actions}

Rather than running a single usage-split analysis, we built a phased decision framework: deliberately structured so that easy calls could be made quickly, and only genuinely ambiguous cases would take further analytical effort.

- **Phase 0** audited and found a single GA4 property across app and web, so cross-platform comparison was possible
- **Phase 1** used the raw usage split to fast-track a decision on any feature with an unambiguous app-web use gap
- **A stakeholder checkpoint** redirected scope where usage data alone wasn't the right focus — for example, a feature with a legislative communication obligation
- **Phase 2** added device, timing, and user-type context to the features that weren't resolved by the raw split
- **Phase 3** synthesised every finding into a confidence-tiered recommendation, flagging which conclusions were directly observed and which were inferred
- **Scenario B** (once a privacy impact assessment cleared) linked web sessions to physical smartcard touch-on/off data, to confirm (rather than assume) the one recommendation that had rested on an inference. The PIA was submitted for this during Phase 0

### Results {#overview-results}

Every one of the three questions above now has an evidence-backed answer:

**Platform exclusivity**

- One feature is moving toward app-exclusive, phased over 6 months
- One feature was confirmed as genuinely dual-platform, backed by a physical-movement data linkage
- One feature stays dual-platform by design, once reframed away from a pure usage-share question

**Dev priority**

- Highest priority: a channel-effectiveness fix, backed directly by measured user-action data: implement a "replan trip" call to action in the web banner.
- Next: a conversion-focused addition to the web journey planner, backed by the smartcard linkage: add an app download link at the point of web planning.
- No further build recommended for the two lower-priority features this cycle.

**Retirement candidates**

- One feature flagged for a formal retirement review in 6 months: saved_trips
- No feature met the bar for immediate removal

### Growth/Next Steps {#overview-growth}

The real_time_departures feature originally deferred by capacity, rather than by evidence, remains open. The case for continued investment there is already strong, but was not formally re-examined once development resources were redirected elsewhere. The smartcard linkage pipeline built for this project is reusable for future features.

### Key Definition {#overview-definition}

Throughout this write-up we refer to a recommendation's **confidence tier**:

- **Observed** — built directly from a measured event (a click, a usage rate, an action taken). Nothing left open to interpretation between the data and the conclusion.
- **Directional** — built from a pattern that's real, but whose *meaning* required an inference (e.g., "this usage pattern probably reflects planning ahead, rather than idle browsing").
- **Confirmed** — a directional finding that was subsequently checked against an independent, harder form of evidence and held up.

This distinction is important, since two of our three headline decisions rested entirely on Observed evidence and only one ever needed the Confirmed tier.

We also distinguish two **scenarios**, depending on whether the pending privacy approval had cleared at the time a recommendation was delivered:

- **Scenario A** — the report as it could be delivered using GA4 data alone, before the privacy impact assessment (PIA) cleared. Complete and actionable on its own; any Directional finding would ship with its confidence explicitly labelled.
- **Scenario B** — the same report, updated once the PIA cleared and smartcard touch-on/touch-off data became available to test the one Directional finding against.

---

# Data & Instrumentation Overview {#data-overview}

We tracked usage across app and web using a single custom GA4 event, parameterised rather than split into many separate event names — this keeps every platform and feature directly comparable in reporting.

| **Field Name** | **Scope** | **Description** |
|---|---|---|
| feature_engaged | Event | Fires whenever a user meaningfully interacts with one of the four core features |
| feature_name | Event parameter | Which feature: real_time_departures, saved_trips, disruption_alerts, or journey_planner |
| interaction_depth | Event parameter | How far the user got: viewed, interacted, or completed |
| lookup_lead_time_min | Event parameter | For real-time departures: minutes between the lookup and actual departure — a proxy for imminent vs. planned travel |
| alert_channel | Event parameter | For disruption alerts: push, in-app banner, or web banner |
| alert_action | Event parameter | What the user did with an alert: dismissed, viewed detail, or replanned their trip. Each alert interaction fires one event, so events map one-to-one to outcomes |
| platform | Native (GA4 export) | GA4's own field — ANDROID, IOS, or WEB — collapsed to APP/WEB throughout this analysis |
| device.category | Native (GA4 export) | GA4's own field, used to split web traffic into mobile web vs desktop web in Phase 2 |
| user_id | Native (GA4 export) | GA4's login-only identity field, set from a hashed internal account ID at sign-in; bridges to smartcard data in Scenario B — absent for anonymous sessions |
| touch-on / touch-off | External (smartcard system) | Physical boarding/alighting records, linked in Scenario B to confirm one finding |
| account_card_bridge | External (ticketing system) | Maps each account_id to its linked card_id_hashed; one account may hold multiple cards. Joined to user_id in Scenario B's Gate 2B and Tier 1 checks |

---

platform, user_id and device.category are collected automatically by GA4 and don't require custom dimension registration, whereas the five event parameters above them are custom registered.

# Methodology Overview {#methodology-overview}

We are answering three related but distinct questions (platform exclusivity, dev priority, retirement) using a single phased evidence pipeline, rather than treating each as a separate analysis.

As the underlying evidence ranges from directly observed usage splits through to physical movement data, we structured the work as a sequence of gated phases, each one only escalating to the next when the current evidence genuinely couldn't resolve the question:

- Phase 0: Instrumentation
- Phase 1: GA4 baseline (Gate 1 — fast-track check)
- Stakeholder checkpoint
- Phase 2: Proxy-context layer
- Phase 3: Synthesis (Gate 3A — confidence tiering)
- Scenario B: Smartcard linkage (Gate 2B — sample-size check)

Each phase breaks down into two kinds of work: a **query**, which produces a number from the data, and a **judgement**, where a human applies a threshold the data alone can't set. Several of the gates (Gate 1, Gate 2B) exist specifically to hand a decision to a person rather than automate it, which is why keeping the two apart matters.

| # | Type | Step |
|---|---|---|
| 1 | Query | Baseline usage-rate query — all four features, both platforms |
| 2 | Judgement | Apply Gate 1's threshold — saved_trips fast-tracked, others carried forward |
| 3 | Query | Interaction-depth breakdown for saved_trips, supporting the fast-track call |
| 4 | Judgement | Stakeholder checkpoint — scope set: escalate, reframe, or defer each remaining feature |
| 5 | Query | Device, timing, and new-vs-returning cuts on the escalated feature |
| 6 | Query | Channel/outcome breakdown for the reframed feature |
| 7 | Judgement | Synthesise findings into confidence tiers; apply Gate 3A |
| 8 | — | Wait for privacy approval — independent of the analysis itself |
| 9 | Query | Gate 2B sample-size check — join via GA4's user_id, bridged to the ticketing system's account-card mapping |
| 10 | Judgement | Assess whether the linked sample clears the bar for individual-level analysis |
| 11 | Query | Deterministic (Tier 1) join and cohort (Tier 2) correlation, run independently |
| 12 | Judgement | Compare tiers, close Gate 3A, and package the final decision matrix |

Step 8 is a wait — the Scenario A recommendation was already delivered and actionable before anyone knew the smartcard linkage would become possible.

---

# Phase 0: Instrumentation {#phase-0}

Before any comparison between app and web is meaningful, both platforms need to report into the same place, in the same shape.

### Setup

We audited the existing gtag and Firebase configurations to confirm both data streams report into the same GA4 property, checked that the custom dimensions listed in the Data Overview above were already registered and mapped correctly (a small number were missing and added at this stage), and validated event delivery for all platforms in DebugView before letting any baseline window run.

`lookup_lead_time_min` was one of the dimensions missing from the existing setup, and needed to be derived rather than just passed through — the department's tagging is managed via Google Tag Manager, so the raw timestamps were pushed to the dataLayer at the point of lookup, with the actual minutes-until-departure calculation configured as a GTM variable rather than redeployed in application code:

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

```javascript
// GTM Custom JavaScript variable, referenced by the
// feature_engaged tag as the lookup_lead_time_min parameter
function() {
  var departure = new Date({{DLV - departure_time}});
  var lookup = new Date({{DLV - lookup_time}});
  return Math.round((departure - lookup) / 60000);
}
```

### Validation

A platform breakdown check confirmed both app platforms and web were reporting consistently before the baseline window began — this is also the check that would have caught a version-drift issue (e.g., one platform's build predating a schema update) had one existed.

![alt text](/img/posts/phase0-ga4-status.png "GA4 Instrumentation Status")

### Outcome

Instrumentation confirmed clean. The 21-day baseline window began.

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
Divide the two, per feature and per platform. The final step joins those two counts together and calculates what share of each platform's active users actually engaged with each feature — this is the usage rate percentage that appears as the bars in the Phase 1 chart.

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

Any feature with an unambiguous platform gap (roughly, under 10% usage on one platform against over 50% on the other) is resolved immediately, without waiting on further analysis.

**saved_trips** cleared this outright — 9% engagement on web against 61% on app — and was resolved here, permanently, regardless of anything that followed.

Interaction depth on web sharpens the case further: of that 9%, most only viewed the feature without creating a saved trip.

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
   platform_group = 'WEB' AND event_name = 'feature_engaged' AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'saved_trips'
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

This returns 7% viewed, 1.5% interacted, and 0.5% completed — summing to the 9% headline rate, with no user counted twice.

### Outcome

One of four features resolved on Observed evidence alone. The remaining three carried forward.

---

# Stakeholder Checkpoint {#checkpoint}

Not every open question is best answered by more usage data. At this point the remaining three features were reviewed with the client team directly, and the scope was adjusted:

- **journey_planner** was escalated as the clear development priority, since its raw usage pattern actively contradicted the working assumption about how app and web were being used. The working assumption going in was that the app would see the heaviest use of core features like trip planning, with web serving as a secondary, occasional-use channel. The raw usage split inverted this: 88% of web's active users engaged with journey_planner, against only 45% on app, the opposite of what the assumption predicted. This unexpected engagement direction prioritised the feature for closer analysis.
- **disruption_alerts** was reframed entirely — from a platform-investment question to a channel-effectiveness question — after the communications team flagged a consistency obligation across all alert channels that usage share alone couldn't capture
- **real_time_departures** was deferred, with the team accepting its already-strong usage gap (82% vs 34%) as sufficient for app-first development, rather than spending further analytical effort on this split given that **journey_planner** had been escalated.

### Outcome

**journey_planner** proceeds to Phase 2 on its original terms; **disruption_alerts** proceeds on reframed terms; **real_time_departures** exits the active analysis by decision, not by evidence.

---

# Phase 2: Proxy-Context Analysis {#phase-2}

### Evidence Gathered

For the escalated feature, **journey_planner**, we layered in device category (mobile web vs. desktop web), time-of-day clustering, and new-vs-returning user share — none of which are visible in a simple platform split.

### First Look to Formal Query

Each cut started with a simple pass at the data, before deciding what — if anything — needed to be measured more precisely. All three read from the baseline table created in Phase 1, so every query below inherits identical platform and window definitions.

**Device category** began with a plain count of engaged users on mobile web against desktop web, which favoured mobile web.

```sql
SELECT
 device_category,
 COUNT(DISTINCT user_pseudo_id) AS engaged_users
FROM
 `project.analytics_derived.baseline_events_21d`
WHERE
 platform_group = 'WEB' AND event_name = 'feature_engaged'
  AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
GROUP BY
 device_category;
```

Dividing each device's engaged users by its own active-user base told a different story: desktop web engages at 93%, mobile web at 81% — both high, and much closer together than the raw counts suggested.

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

**Timing** showed its shape early. A simple per-hour count of `journey_planner` activity, split by platform, produced two distinct patterns: app usage spiking sharply around the AM and PM commute windows, web usage sitting comparatively flat and tilted toward evenings.

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

That shape set the peak/off-peak boundary used for the rest of the analysis (7–9am, 4–6pm). Splitting further by device within web sharpened the comparison: 27% peak / 73% off-peak on mobile, 19% peak / 81% off-peak on desktop — both far closer to each other than either is to the app's 72% peak share.

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

**New versus returning** users showed a large, clear gap from the outset.

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

Counting once per user rather than once per engagement event held the split at 81% returning on app against 64% new on web — close to the inverse of each other.

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
   event_name = 'feature_engaged'
    AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
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

Together, these three cuts rule out three different alternative explanations for the same question. Device category rules out "it's simply a desktop tool." Timing rules out "web usage happens throughout the day the same way app usage does, just less often." New-vs-returning rules out "the same core group of people just prefer using web sometimes." What's left, once each of those is set aside, is the interpretation carried into the next section: that web usage clusters ahead of travel, largely independent of which device it happens on, and disproportionately belongs to people who haven't installed the app yet.

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

### Proxy-Context Dashboard

![alt text](/img/posts/phase2-proxy-context-analysis.png "Proxy-Context Analysis")

### Outcome

The escalated feature's usage pattern was reframed from an app-vs-web story to a pre-trip-vs-in-transit one — directionally supported, but resting on an inference about user intent that the data alone couldn't fully confirm. The reframed feature's fix (redesigning the web alert's call to action) was resolved directly, with no inference involved.

---

# Phase 3: Synthesis & Confidence Tiers {#phase-3}

### Evidence Gathered

Every finding from Phases 1 and 2 was reorganised into the three confidence tiers defined above, rather than presented as a single undifferentiated recommendation list.

![alt text](/img/posts/phase3-synthesis.png "Confidence-Tiered Synthesis")

### Gate 3A — Which Findings Need Further Confirmation?

Only a recommendation resting on an inferred interpretation is a candidate for further confirmation. Of the three resolved features at this point, only one qualified — the reframed alert-channel fix and the fast-tracked feature from Phase 1 were both already Observed, with nothing a further data source could sharpen.

### Outcome

Two recommendations shipped as final at this stage. One was flagged, explicitly, as directional and worth revisiting if stronger evidence became available.

---

# Scenario B: Smartcard Linkage {#scenario-b}

Once a pending privacy impact assessment cleared, we had access to physical smartcard touch-on/touch-off records — a genuinely independent form of evidence for the one flagged recommendation. These data are stored in two external tables (account_card_bridge, touch_events) which appear as:

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

Only a small share of web journey_planner users could be linked to a smartcard — consistent with the new-user skew already found in Phase 2, since a first-touch audience is far less likely to already be logged in.

The link runs through GA4's user_id field, set only at login, bridged to the ticketing system's account-to-card mapping; where an account holds more than one card, the user is still counted just once.

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

Of 36,260 web `journey_planner` users, 2,176 could be linked — 6.0%. That is too thin to support segmenting the deterministic check any further (by device, by time-of-day, and so on), but still enough for one aggregate figure.

### Tier 1 — Deterministic Check

For the 2,176 linked users, each web `journey_planner` session is checked against a touch-on at that same session's planned origin stop, within 24 hours. An account with more than one linked card counts as matched if *any* of its cards touch on — matching is at the person level, not the card level.

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

Across 2,910 web `journey_planner` sessions from linked users, 39% were followed by a touch-on at the planned origin within 24 hours; 22% within 3 hours.

That 39% only means something against a baseline. Rather than comparing to an unrelated stop or a user's single most-frequented one — either of which would distort the comparison in a different direction — each of the same 2,176 users gets one randomly chosen day on which they *didn't* plan a trip, paired with one of their own real origin stops and a random reference time, then checked against the identical 24-hour window.

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
  QUALIFY ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY RAND()) = 1
),

baseline_stop_pick AS (
  SELECT
   user_pseudo_id,
   origin_stop_id
  FROM
   user_origin_stops
  QUALIFY ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY RAND()) = 1
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

The baseline rate comes back at 11% — putting the treatment's 39% at roughly a 3.5× lift. One asymmetry is worth stating plainly here: the treatment's planning timestamp is real, but the baseline's is a synthetic stand-in — a uniformly random moment on a day the user didn't plan a trip. It's the closest symmetric comparison available, not a perfect one, and the 39% and 11% shouldn't be read as equally precise measurements of the same kind.

### Tier 2 — Cohort Check

Tier 1 only reaches the 6% of `journey_planner` web users who are logged in and myki-linked — a population skewed toward habitual users, which is the opposite of the group Phase 2 found web usage actually over-represents. Tier 2 checks the same question at population scale instead, using only aggregate hourly volume on both sides, with no individual matching at all.

Rather than assuming where a web planning session and a resulting trip would line up in time, this checks a full range of lags and lets the data show where the relationship is strongest.

```sql
-- First look: correlation between journey_planner session volume and
-- touch-on volume, at a spread of lags, per platform and device.
-- Aggregate counts only — no individual identity is used.

WITH hourly_sessions AS (
  SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(event_timestamp), HOUR) AS session_hour,
    platform_group,
    device_category,
    COUNT(DISTINCT user_pseudo_id) AS sessions
  FROM
   `project.analytics_derived.baseline_events_21d`
  WHERE
   event_name = 'feature_engaged' AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
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

Each group shows a clear rise and fall around a different point — the app peaks and falls off within a couple of hours, while both web device types stay elevated across a much wider band a day later. Rather than reading the peak off this shortlist by eye, the full scan is swept automatically and the single best lag per group is kept:

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
   event_name = 'feature_engaged' AND (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'feature_name') = 'journey_planner'
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
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY platform_group, device_category ORDER BY correlation DESC
) = 1
ORDER BY
 platform_group,
 device_category;
```

The peak lag itself is part of the finding, not just the correlation strength at it: app `journey_planner` engagement peaks at 1 hour (r=0.74), desktop web at 24 hours (r=0.61), mobile web at 27 hours (r=0.58). App usage is tied almost immediately to a trip; both web device types sit roughly a day ahead of one, and land close enough to each other that device isn't what's driving the difference from app.

One limitation worth noting: a correlation computed over roughly 500 hourly buckets (21 days × 24 hours) per group carries a wider margin than the two-decimal figures suggest — enough to trust the shape and ordering across groups, not enough to treat 0.58 versus 0.61 as a meaningful difference between the two device types.

![alt text](/img/posts/scenario-b-confirmation.png "Smartcard Linkage Confirmation")

### Outcome

Tier 1's 39% touch-on rate against an 11% baseline is a real lift, but it rests on a thin, wide-margin sample of only 2,176 linked users — precise about direction, less precise about magnitude. Tier 2 answers a different question entirely, using the full population with no identity linkage at all: it finds that a `journey_planner` session's timing relationship to a touch-on peaks at 24–27 hours later on web, regardless of device, and peaks almost immediately, at 1 hour, on app.

Neither check alone would have closed Gate 3A. A small deterministic sample can show the right direction without ruling out that its 6% linked population isn't representative; a population-scale correlation can show the right timing pattern without ever confirming that any single trip was actually planned in advance. Together, they cover each other's blind spot — Tier 1 confirms the *magnitude* for the subset who logged in, Tier 2 confirms the *pattern* holds across everyone, including the 94% Tier 1 could never see.

With both pointing the same direction, journey_planner moves from Directional to Confirmed. The recommendation itself doesn't change — dual-platform, with an app-download prompt at the point of web planning — but it now rests on two independent forms of evidence rather than one inferred pattern.

---

# Decision Summary {#decision-summary}

| **Feature** | **Recommendation** | **Confidence** |
|---|---|---|
| saved_trips | Stop new development on web; keep live, review for retirement in 6 months | Observed |
| journey_planner | Keep dual-platform; add an app-download prompt at the point of web planning | Confirmed |
| disruption_alerts | Redesign the web alert with an explicit "replan trip" call to action | Observed |
| real_time_departures | Continue app-first investment; no new analysis this cycle | Deferred |

Two of the three project decisions — platform exclusivity and retirement candidates — were resolved almost entirely on Observed evidence. The physical-movement linkage mattered for exactly one recommendation, which is itself a useful finding: the higher-governance data source was worth requesting for a narrow, specific reason, not as a blanket assumption that more data is always better.

---

# Application {#application}

The client's development team now has a prioritised, evidence-ranked backlog rather than a flat feature list: the disruption alert redesign leads, backed by directly measured action-rate data; the journey planner's app-download prompt follows, backed by the smartcard-confirmed conversion case; and no further web investment is planned for the saved trips feature pending its 6-month retirement review.

---

# Growth & Next Steps {#growth-next-steps}

The one feature deferred by capacity — `real_time_departures` — still has an open, evidence-backed case for continued app investment that was never formally revisited once the team's attention moved elsewhere. The smartcard linkage pipeline built for this project remains available for any future feature whose recommendation rests on an inference rather than a direct measurement.
