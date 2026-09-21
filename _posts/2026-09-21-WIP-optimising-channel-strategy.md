---
layout: post
title: Optimising App & Web Channel Strategy Using GA4 & Smartcard Data
image: /img/posts/channel-strategy-title-img.png
tags: [Digital Analytics,GA4,Transport,Decision Framework,Python]
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
- [08. Scenario: PIA Passed: Smartcard Linkage](#scenario-b)
- [09. Decision Summary](#decision-summary)
- [10. Application](#application)
- [11. Growth & Next Steps](#growth-next-steps)

---

# Project Overview {#project-overview}

### Context {#overview-context}

Our client's digital team maintains four core features across their app and website: real-time departures, saved trips, disruption alerts, and the journey planner itself. Each feature had usage on both platforms, but no one could say with confidence whether that reflected genuine need for both, or just usage nobody had looked at closely enough to challenge.

The client needed to answer three specific questions before committing the next development cycle:

1. **Platform exclusivity** — should a feature exist on app only, web only, or both?
2. **Dev priority** — within available capacity, what gets built next?
3. **Retirement candidates** — what's costing maintenance but barely used, on either platform?

### Actions {#overview-actions}

Rather than running a single usage-split analysis, we built a phased decision framework — deliberately structured so that easy calls could be made quickly, and only genuinely ambiguous cases absorbed further analytical effort.

- **Phase 0** instrumented a single GA4 property across app and web, so cross-platform comparison was possible at all. The GA4 property takes app and web input as data streams. A PIA to link myki touch-on/touch-off data at this step was initiated.
- **Phase 1** used the raw usage split to fast-track a decision on any feature with an unambiguous gap
- **A stakeholder checkpoint** redirected scope where usage data alone wasn't the only consideration — for example, a feature with a legislative communication obligation
- **Phase 2** added device, timing, and user-type context to the features that weren't resolved by the raw split
- **Phase 3** synthesised every finding into a confidence-tiered recommendation, flagging which conclusions were directly observed and which were inferred
- **Scenario: PIA Passed** — once a privacy impact assessment cleared — linked web sessions to physical smartcard touch-on/off data, to confirm (rather than assume) the one recommendation that had rested on an inference. The PIA was submitted for this during Phase 0.

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

The real_time_departures feature originally deferred by capacity (its app use vs web use), rather than by evidence, remains open — the case for continued investment there is already strong, but was never formally re-examined once resources were redirected elsewhere. The smartcard linkage pipeline built for this project is reusable for future features without repeating the governance approval process.

### Key Definition {#overview-definition}

Throughout this write-up we refer to a recommendation's **confidence tier**:

- **Observed** — built directly from a measured event (a click, a usage rate, an action taken). Nothing left open to interpretation between the data and the conclusion.
- **Directional** — built from a pattern that's real, but whose *meaning* required an inference (e.g. "this usage pattern probably reflects planning ahead, rather than idle browsing").
- **Confirmed** — a directional finding that was subsequently checked against an independent, harder form of evidence and held up.

This distinction is important as: two of our three headline decisions rested entirely on Observed evidence, and only one ever needed the Confirmed tier at all.

---

# Data & Instrumentation Overview {#data-overview}

We tracked usage across app and web using a single custom GA4 event, parameterised rather than split into many separate event names — this keeps every platform and feature directly comparable in reporting.

| **Field Name** | **Scope** | **Description** |
|---|---|---|
| feature_engaged | Event | Fires whenever a user meaningfully interacts with one of the four core features |
| feature_name | Event parameter | Which feature: real_time_departures, saved_trips, disruption_alerts, or journey_planner |
| platform | Event parameter | iOS app, Android app, or web (further split into mobile web / desktop web) |
| interaction_depth | Event parameter | How far the user got: viewed, interacted, or completed |
| lookup_lead_time_min | Event parameter | For real-time departures: minutes between the lookup and actual departure — a proxy for imminent vs. planned travel |
| alert_channel | Event parameter | For disruption alerts: push, in-app banner, or web banner |
| alert_action | Event parameter | What the user did with an alert: dismissed, viewed detail, or replanned their trip |
| touch-on / touch-off | External (smartcard system) | Physical boarding/alighting records, linked in Scenario: PIA passed - to confirm one finding |

---

# Methodology Overview {#methodology-overview}

We are answering three related but distinct questions (platform exclusivity, dev priority, retirement) using a single phased evidence pipeline, rather than treating each as a separate analysis.

As the underlying evidence ranges from directly observed usage splits through to physical movement data, we structured the work as a sequence of gated phases — each one only escalating to the next when the current evidence genuinely couldn't resolve the question:

- Phase 0: Instrumentation
- Phase 1: GA4 baseline (Gate 1 — fast-track check)
- Stakeholder checkpoint
- Phase 2: Proxy-context layer
- Phase 3: Synthesis (Gate 3A — confidence tiering)
- Scenario PIA Passed: Smartcard linkage (Gate 2B — sample-size check)

---

# Phase 0: Instrumentation {#phase-0}

Before any comparison between app and web is meaningful, both platforms need to report into the same place, in the same shape.

### Setup

We audited the existing gtag and Firebase configurations to confirm both data streams report into the same GA4 property, checked that the custom dimensions listed in the Data Overview above were already registered and mapped correctly (a small number were missing and added at this stage), and validated event delivery for all platforms in DebugView before letting any baseline window run.

`lookup_lead_time_min` was one of the dimensions missing from the existing setup, and needed to be derived rather than just passed through. If the department's tagging were managed via Google Tag Manager, the raw timestamps could be pushed to the dataLayer at the point of lookup, with the actual minutes-until-departure calculation configured as a GTM variable:

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

A platform breakdown check confirmed both app platforms and web were reporting consistently before the baseline window began — this is also the check that would have caught a version-drift issue (e.g. one platform's build predating a schema update) had one existed.

![alt text](/img/posts/phase0-ga4-status.png "GA4 Instrumentation Status")

### Outcome

Instrumentation confirmed clean. The 21-day baseline window began.

---

# Phase 1: GA4 Baseline {#phase-1}

### Evidence Gathered

Raw `feature_engaged` usage rate by platform, across all four features, over the 21-day window.

```
# TODO: insert the GA4 Data API / BigQuery export query
# used to compute feature usage rate by platform
```

![alt text](/img/posts/phase1-feature-usage-baseline.png "Feature Usage Baseline by Platform")

### Gate 1 — Fast-Track Check

Any feature with an unambiguous platform gap (roughly, under 10% usage on one platform against over 50% on the other) is resolved immediately, without waiting on further analysis.

**saved_trips** cleared this outright — 9% engagement on web against 61% on app — and was resolved here, permanently, regardless of anything that followed.

### Outcome

One of four features resolved on Observed evidence alone. The remaining three carried forward.

---

# Stakeholder Checkpoint {#checkpoint}

Not every open question is best answered by more usage data. At this point the remaining three features were reviewed with the client team directly, and the scope was adjusted:

- One feature was escalated as the clear priority, since its raw usage pattern actively contradicted the working assumption about how app and web were being used
- One feature was reframed entirely — from a platform-investment question to a channel-effectiveness question — after the communications team flagged a consistency obligation across all alert channels that usage share alone couldn't capture
- One feature was deferred, with the team accepting its already-strong directional case rather than spending further analytical effort on a foregone conclusion

### Outcome

Two features proceed to Phase 2 on their original terms; one proceeds on reframed terms; one exits the active analysis by decision, not by evidence.

---

# Phase 2: Proxy-Context Analysis {#phase-2}

### Evidence Gathered

For the escalated feature, we layered in device category (mobile web vs. desktop web), time-of-day clustering, and new-vs-returning user share — none of which are visible in a simple platform split.

```
# TODO: insert the GA4 Explore / BigQuery query used to
# compute the device, timing, and new-vs-returning cuts
```

![alt text](/img/posts/phase2-proxy-context-analysis.png "Proxy-Context Analysis")

### Analysis

Mobile web and desktop web turned out to behave almost identically to each other, and neither resembled the app — ruling out "it's just a desktop tool" as the explanation. Timing told the real story: web usage clustered off-peak regardless of device, while app usage clustered tightly around commute peaks. New-user share on web was also disproportionately high, suggesting web frequently serves as a first-touch surface rather than a habitual one.

For the reframed feature, we broke down alert delivery channel against the outcome that followed each alert (dismissed, viewed in detail, or trip replanned) — a direct measurement, requiring no interpretation layer at all.

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

Once a pending privacy impact assessment cleared, we had access to physical smartcard touch-on/touch-off records — a genuinely independent form of evidence for the one flagged recommendation.

### Sample-Size Check (Gate 2B)

Only a small share of web sessions belonged to logged-in, card-linked users — consistent with the new-user skew already found in Phase 2. This ruled out a fully segmented individual-level analysis, but still supported one aggregate figure.

```
# TODO: insert the join logic used to match web sessions
# to smartcard touch-on records within a time window
```

### Two Independent Checks

**Deterministic check** (small, precise): among linked users, a web planning session was followed by a touch-on at the planned origin within 24 hours at roughly 3.5 times the rate of a matched baseline with no session.

**Cohort check** (large, less precise): across the full population, web session volume correlated with touch-on volume at a next-day lag — for both mobile and desktop web alike — while app usage correlated almost immediately, consistent with in-transit use.

![alt text](/img/posts/scenario-b-confirmation.png "Smartcard Linkage Confirmation")

### Outcome

Both checks pointed the same direction. The flagged recommendation moved from Directional to Confirmed — the recommendation itself didn't change, but its evidentiary basis did.

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

The one feature deferred by capacity — real-time departures — still has an open, evidence-backed case for continued app investment that was never formally revisited once the team's attention moved elsewhere. The smartcard linkage pipeline built for this project required no new governance approval to reuse, and remains available for any future feature whose recommendation rests on an inference rather than a direct measurement.
