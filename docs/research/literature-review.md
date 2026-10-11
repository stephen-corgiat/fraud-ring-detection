# Literature review

The design decisions in this repository were made against a reading of about thirty papers and
industry sources on coordinated fraud, bot and agent detection, label scarcity, and adversarial
learning. This page records, for each one, **what it claims** and **what this project took from
it, or deliberately didn't**. Where a source shaped a specific decision, the entry links the ADR or
module.

A note on evidence: vendor reports and single-testbed preprints are read as *directional*. Their
numbers come from their own traffic, labels and month, and are never treated as load-bearing here.
Every claim in this repository's own results comes from the seeded simulator.

**Contents**

1. [Dense subgraphs, lock-step and coordination](#1-dense-subgraphs-lock-step-and-coordination)
2. [Graph neural networks for fraud](#2-graph-neural-networks-for-fraud)
3. [Abusive-account classification and labels](#3-abusive-account-classification-and-labels)
4. [Adversaries that respond](#4-adversaries-that-respond)
5. [Bots, agents and the network layer](#5-bots-agents-and-the-network-layer)
6. [Evaluation discipline](#6-evaluation-discipline)

---

## 1. Dense subgraphs, lock-step and coordination

**CopyCatch** — Beutel, Xu, Guruswami, Palow, Faloutsos. *CopyCatch: Stopping Group Attacks by
Spotting Lockstep Behavior in Social Networks.* WWW 2013.
Finds groups of users who act on the same set of pages within the same time window — *lock-step*
behavior — as near-bipartite cores in a time-annotated graph.
*In this project:* the inspiration for testing **serialization**, lock-step's mirror image: a
finite operator's members take turns rather than act together. The result was negative;
[ADR 0004](../adr/0004-lockstep-serialization-is-not-recoverable.md) records why.

**SynchroTrap** — Cao, Yang, Yu, Palow. *Uncovering Large Groups of Active Malicious Accounts in
Online Social Networks.* CCS 2014.
Clusters accounts by the similarity of their timestamped actions; malicious accounts controlled by
one operator act in loosely synchronized bursts that organic users don't.
*In this project:* the same family as CopyCatch. Its key assumption — synchrony survives because
operators economize — is exactly what the evasion dial's sophisticated level attacks.

**fBox** — Shah, Beutel, Gallagher, Faloutsos. *Spotting Suspicious Link Behavior with fBox: An
Adversarial Perspective.* ICDM 2014. [arXiv:1410.3915](https://arxiv.org/abs/1410.3915)
Shows that spectral methods catch *large* dense attacks but miss small ones by construction, and
flags the "stealth" attacker that stays below the spectral threshold via low reconstruction degree.
*In this project:* the reason the seedless ranker has **three entry axes**, not one. A detector
keyed on one shape invites the adversary to choose the other.
[ADR 0006](../adr/0006-seedless-concentration-as-the-front-door.md).

**FRAUDAR** — Hooi, Song, Beutel, Shah, Shin, Faloutsos. *FRAUDAR: Bounding Graph Fraud in the
Face of Camouflage.* KDD 2016. [doi:10.1145/2939672.2939747](https://doi.org/10.1145/2939672.2939747)
A greedy peeling algorithm for the densest bipartite subgraph, with a provable bound that holds even
when fraudsters add camouflage edges to legitimate items. Column weighting by inverse degree keeps
popular items from dominating.
*In this project:* concentration is the cheap, interpretable cousin of dense-block detection. FRAUDAR
taught a sharper question to ask first: **is this economy one where fraud is forced into width?** If
a ring's payout is capped per account, it must spread across many accounts, which favors block
detection. If volume per account is uncapped, a few "stars" dominate and blocks are thin. The
`booster` and `n_eff` axes cover both cases.

**HoloScope** — Liu, Hooi, Faloutsos. *HoloScope: Topology-and-Spike Aware Fraud Detection.* CIKM 2017.
Combines dense-subgraph topology with temporal spikes and rating deviation, so an attacker must evade
several signals at once.
*In this project:* the same instinct behind combining concentration with `burst7` (temporal
concentration) and the **campaign** tag.

**FlowScope** — Li, Liu, Li, Han, Shi, Hooi, Huang, Cheng. *FlowScope: Spotting Money Laundering
Based on Graphs.* AAAI 2020. [code](https://github.com/aplaceof/FlowScope)
Detects laundering as dense *multi-partite flows* (source → middle → destination) where middle
accounts pass money through and retain nearly none.
*In this project:* the argument for moving the unit of detection **up** — from accounts to the
beneficiary of the money — once account-level signals are evadable. The seedless ranker scores
*creators* for the same reason.

**CubeFlow** — Sun, Zhang, Zhao, Liu, Chen, Zhuang, Shen, Cheng. *CubeFlow: Money Laundering
Detection with Coupled Tensors.* PAKDD 2021. [arXiv:2103.12411](https://arxiv.org/abs/2103.12411)
Extends flow detection to coupled tensors with time and amount, encoding "fast in, fast out,
zero balance" behavior of laundering middles.
*In this project:* a design reference for a payee-level detector, which this repository leaves as
future work.

**Coordinated activity** — Luceri, Pantè, Burghardt, Ferrara. *Unmasking the Web of Deceit:
Uncovering Coordinated Activity to Expose Information Operations on Twitter.* WWW 2024.
Compares behavioral traces for detecting coordinated accounts (co-retweeting, shared URLs, hashtag
sequences, fast retweets, text similarity) and finds that no single trace captures every operation;
combining them surfaces more.
*In this project:* support for the multi-axis Stage 2 score (linkage, behavioral, co-target, action
mix) over any single coordination signal.

## 2. Graph neural networks for fraud

**CARE-GNN** — Dou, Liu, Sun, Deng, Peng, Yu. *Enhancing Graph Neural Network-based Fraud
Detectors against Camouflaged Fraudsters.* CIKM 2020.
Fraudsters camouflage by connecting to benign nodes; CARE-GNN learns a similarity-aware neighbor
selector per relation to filter camouflaged neighbors before aggregation.

**GAGA** — Wang et al. *Label Information Enhanced Fraud Detection against Low Homophily in Graphs.*
WWW 2023.
Group-aggregation embeddings that remain informative when fraud nodes mostly neighbor benign nodes
(low homophily), a common real-world condition.

*In this project (both):* deliberately **not** used. A GNN needs labels the system doesn't have at
the start, and its decisions are hard to show an investigator. Arp et al.'s advice is to beat a simple
baseline before reaching for capacity. The learned rule here is logistic regression whose coefficients
*are* the rule, and the comparator gradient-boosted model overfits the small per-ring folds
([ADR 0003](../adr/0003-held-out-rings-and-a-noisy-oracle.md)).

## 3. Abusive-account classification and labels

**Deep Entity Classification** — Xu, Goossen, Cevahir, Khodeir, Jin, Li, Shan, Patel, Freeman,
Pearce. *Deep Entity Classification: Abusive Account Detection for Online Social
Networks.* USENIX Security 2021.
A production system that classifies accounts from features of the account *and its neighborhood*
(aggregates over linked entities), trained in two stages: plentiful low-precision labels first,
then a small set of high-precision human labels.
*In this project:* the template for the **portable core** — the feature spine and the harness travel
across abuse types, and a new abuse type is a new task head. Also the model for the label flywheel
the learning curve prices.

**Positive-unlabeled learning** — Elkan & Noto. *Learning Classifiers from Only Positive and
Unlabeled Data.* KDD 2008.
Under "selected completely at random" labeling, a classifier trained on labeled-vs-unlabeled is a
scaled version of the true one, and the scale factor is estimable.

**SAR-PU** — Bekker, Robberechts, Davis. *Beyond the Selected Completely at Random Assumption for
Learning from Positive and Unlabeled Data.* ECML-PKDD 2019. [arXiv:1809.03207](https://arxiv.org/abs/1809.03207)
Relaxes SCAR to "selected at random" given features, via a propensity score.

**nnPU** — Kiryo, Niu, du Plessis, Sugiyama. *Positive-Unlabeled Learning with Non-Negative Risk
Estimator.* NeurIPS 2017.
Fixes the overfitting of unbiased PU risk estimators with deep models by clipping the negative-class
risk at zero.

**PU survey** — Bekker & Davis. *Learning from Positive and Unlabeled Data: A Survey.* Machine
Learning, 2020.

*In this project (PU family):* the vocabulary for what a fraud label is *allowed* to mean.
Investigator labels are never a random sample: they come from whatever the ranker surfaced. That
selection mechanism is known, because you built it, so it can be logged rather than estimated after
the fact. Concretely, the label stores record the rank and suspicion at labeling time.

**Confident Learning** — Northcutt, Jiang, Chuang. *Confident Learning: Estimating Uncertainty in
Dataset Labels.* JAIR 2021. [doi:10.1613/jair.1.12125](https://doi.org/10.1613/jair.1.12125) ·
[cleanlab](https://github.com/cgnorthcutt/cleanlab)
Estimates the joint distribution of noisy and true labels from out-of-sample predicted probabilities,
and ranks likely label errors.
*In this project:* the reason the oracle has **two separate error rates**
([ADR 0003](../adr/0003-held-out-rings-and-a-noisy-oracle.md)). Missed fraud and false confirmations
damage different metrics. At small label counts, confident learning works as a *prioritizer* for
human re-review, not as an automatic cleaner.

## 4. Adversaries that respond

**Strategic classification** — Hardt, Megiddo, Papadimitriou, Wootters. *Strategic Classification.*
ITCS 2016. [arXiv:1506.06980](https://arxiv.org/abs/1506.06980)
Models classification as a Stackelberg game against agents who pay a cost to change their features.
Robust classifiers exist when the cost structure is known.

**Performative prediction** — Perdomo, Zrnic, Mendler-Dünner, Hardt. *Performative Prediction.*
ICML 2020; and the survey by Hardt & Mendler-Dünner, *Performative Prediction: Past and Future*,
Statistical Science 2025 ([arXiv:2310.16608](https://arxiv.org/abs/2310.16608)).
When predictions change the distribution they predict, the right target is a performatively stable
point, not the original distribution.

*In this project (both):* the frame for the **evasion dial**
([ADR 0002](../adr/0002-adversary-is-a-dial.md)). Detection is a *tax* on fraud, not a wall: every
signal the adversary routes around costs them something — more accounts, more cover traffic, aged
accounts, a faked funnel. The `fixed-yield split` knob makes that cost explicit. The platform also
sets part of the adversary's price list (payout caps, qualification thresholds), which is where
robustness becomes tractable.

**Recommender poisoning** — Nguyen, Nguyen, Nguyen, Huynh, Nguyen, Weidlich, Yin. *Manipulating
Recommender Systems: A Survey of Poisoning Attacks and Countermeasures.* ACM Computing Surveys
2024. [arXiv:2404.14942](https://arxiv.org/abs/2404.14942)
Maps attacks that inject fake users with crafted interaction histories to promote or demote items,
and the detection/robust-training defenses against them.
*In this project:* the adjacent adversary. Shilling profiles that copy real users' histories are the
literature's version of **cover traffic** — the reason a sophisticated ring's members look like
diffuse organic viewers.

## 5. Bots, agents and the network layer

**BOTracle** — Kadel, Sinha, See, Fischer. *BOTracle: A Framework for Discriminating Bots and
Humans.* 2024. [arXiv:2412.02266](https://arxiv.org/abs/2412.02266)
A cost cascade — heuristics, then a per-request classifier, then a graph network over a session's
website-traversal graph — run passively on analytics logs. Bot labels are proxies (datacenter IPs),
so its numbers measure agreement with the proxy.
*In this project:* behavior from *where* a session goes, not when. Faking a human walk forces human
pace, which destroys a bot's economics ([ADR 0005](../adr/0005-what-over-when.md)).

**Known By Their Actions** — Lugoloobi, Marro, Magomere, Wright, Russell. *Known By Their Actions:
Fingerprinting LLM Browser Agents via UI Traces.* 2026. [arXiv:2605.14786](https://arxiv.org/abs/2605.14786)
Identifies *which* model drives a browser agent from 41 summary statistics of its UI event log, with
a stock gradient-boosted classifier. Behavioral identity is a property of the model, not the harness.
*In this project:* the clearest outside evidence for **what over when** — action composition carries
identity that timing alone doesn't ([ADR 0005](../adr/0005-what-over-when.md)).

**FP-Agent** — Wang, Shafiq, Vekaria. *FP-Agent: Fingerprinting AI Browsing Agents.* 2026.
[arXiv:2605.01247](https://arxiv.org/abs/2605.01247)
Fifty behavioral input-dynamics features beat 418 browser-fingerprint features at telling agents from
humans. Agents don't produce the physical event cascades a human body does: keystroke timing, cursor
trajectories, scroll inertia.
*In this project:* **benign-cohort census first.** Before trusting any client signal, measure the
honest population that mimics the positive class — password managers on logins, accessibility tools
on input. The lurker and superfan populations are this rule applied to the simulator.

**Whose Agent Are You? (MARK)** — Kang, Jeong, Sheffey, Datta, Houmansadr. *Multi-Layer
Fingerprinting and Attribution of Autonomous Web Agents.* 2026. [arXiv:2606.20910](https://arxiv.org/abs/2606.20910)
Fingerprints agents at four layers — TLS handshake, HTTP grammar, request timing, in-page behavior.
The layers are controlled by different parts of the stack, so a disguise at one leaves the others
contradicting it.
*In this project:* **coherence as a collection-design question**: which independently emitted claims
can you require to agree? The simulator's spoofable fields (`reported_position_pct`, `skip_flag`) are
modeled as client claims a detector should distrust, not facts.

**Action-time intervals** — Kang & Kim. *Quick and Easy Game Bot Detection Based on Action Time
Interval Estimation.* ETRI Journal 2023. [doi:10.4218/etrij.2022-0089](https://doi.org/10.4218/etrij.2022-0089)
A fast, near-zero-false-positive game-bot detector from per-action repetition regularity, validated
on operator labels. The authors note that a random delay defeats it.
*In this project:* the **precision-first floor**. A cheap detector tuned for zero false positives at
modest recall is worth running: it taxes the naive tier and forces evasion code to be written. It
just isn't a strategy against the sophisticated tier ([ADR 0004](../adr/0004-lockstep-serialization-is-not-recoverable.md)).

**Mobile proxy networks** — Mi, Tang, Li, Liao, Wang, Qian. *Your Phone is My Proxy: Detecting and
Understanding Mobile Proxy Networks.* NDSS 2021. [doi:10.14722/ndss.2021.24008](https://doi.org/10.14722/ndss.2021.24008)
Residential and mobile proxy exits are real subscribers' devices, recruited through monetization
SDKs in ordinary apps. Their IPs are, by construction, indistinguishable from honest users'.
*In this project:* why the sophisticated ring rides **residential infra**, and why infra linkage dies
at that level. IP evidence can discount, never convict.

**CGNAT and collateral damage** — Giotsas & Fayed, Cloudflare. *One IP address, many users:
detecting CGNAT to reduce collateral effects.* Cloudflare blog, 2025.
[link](https://blog.cloudflare.com/detecting-cgn-to-reduce-collateral-damage/)
Carrier-grade NAT puts many subscribers behind one IPv4 address. IP-denominated rules (rate limits,
reputation) punish crowds for individuals, with the burden concentrated in regions with high
user-to-IP ratios.
*In this project:* the **rarity gate**. An identifier shared by hundreds of accounts is not
evidence; the simulator's large shared mobile ASNs exist to make that gate necessary. Also: price
IP evidence by type, never act on it alone.

**Active VPN detection** — Schwartz, Manor, Otung. *SNITCH: Leveraging IP Geolocation for Active
VPN Detection.* NDSS MADWeb 2025. [doi:10.14722/madweb.2025.23008](https://doi.org/10.14722/madweb.2025.23008)
Detects geolocation lies from the server side by checking claimed location against network latency
physics. You can cheaply hide *who* you are, but not *where* you are.
*In this project:* the general principle of **making the adversary pay a physics tax** — signals bound
to unforgeable costs outlast signals bound to choices.

**Text near-duplicates** — Zhang, Vallis, Bumin, Vakharia, Bursztein. *RETSim: Resilient and
Efficient Text Similarity.* ICLR 2024. [arXiv:2311.17264](https://arxiv.org/abs/2311.17264) ·
[code](https://github.com/google/unisim)
A small embedding model for near-duplicate text that survives typos and adversarial mutation, for
spam-campaign clustering and deduplication.
*In this project:* noted as the content-side complement this repository doesn't cover. Coordinated
abuse in messaging and reviews needs *what was said*, not just who acted when.

**Industry census** — Imperva. *2025 Bad Bot Report.* 2025.
[PDF](https://www.imperva.com/resources/wp-content/uploads/sites/6/reports/2025-Bad-Bot-Report.pdf)
An annual vendor census of bot traffic by sophistication tier and industry, from blocked requests
across the vendor's network.
*In this project:* read directionally. Its tiering (simple / moderate / advanced) matches the
evasion dial's levels. Its taxonomy has no category for **payout-side metric fraud**, which is
invisible at request time and visible only in aggregate economics — the gap this repository's
creator-level view addresses.

## 6. Evaluation discipline

**Dos and Don'ts of ML in security** — Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger,
Cavallaro, Rieck. *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security 2022
(Distinguished Paper). [site](https://dodo-mlsec.org/)
Ten common pitfalls in security-ML research — sampling bias, label inaccuracy, data snooping,
spurious correlations, biased parameter selection, inappropriate baselines and performance measures,
base-rate fallacy, lab-only evaluation, inappropriate threat model — and how often they appear in
top-venue papers.
*In this project:* the spine of the method. Ground truth physically separated from detector input,
held-out rings, honest baselines, base-rate-aware metrics (operational precision and legit-FP rate
rather than balanced AUC), and an explicit, adjustable threat model.
See [method.md](method.md) and [lessons.md](lessons.md).
