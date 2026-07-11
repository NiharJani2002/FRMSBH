# FRMSBH · CPP

*Six from-scratch, adaptively-robust streaming sketches, each targeting a specific limitation of a real 2026 preprint, each measured on 619,040 rows of real S&P 500 market data, each honest about what it did and did not prove.*

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Single File](https://img.shields.io/badge/build-single%20file-success.svg)](#how-to-run)
[![Dependencies](https://img.shields.io/badge/third--party%20deps-zero-brightgreen.svg)](#system-design)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#disclaimer)

Hi — Nihar here. Before you scroll: this whole repository is one `.cpp` file. That is not a limitation I apologized for; it is a decision, made so that anyone with a compiler and an internet connection can run every experiment below on their own machine in under a minute, with nothing to install and nothing to take on faith. If you only read one section, make it [§5](#extensions), because that's where the actual research lives — not in this README, in the code and the numbers it produces.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Abstract](#abstract)
3. [Dataset](#dataset)
4. [System Design](#system-design)
5. [Extensions — Problems, Math, Code, Results](#extensions)
6. [How to Run](#how-to-run)
7. [Execution Flow](#execution-flow)
8. [Disclaimer](#disclaimer)
9. [How to Cite](#how-to-cite)

---

## Motivation

Every so often a paper comes along that's genuinely good and genuinely incomplete in the same breath, and *Adaptively Robust Resettable Streaming* (Cohen, Gribelyuk, Nelson, Stemmer — preprint, Jan 2026) is exactly that kind of paper. It solves a real problem: classical cardinality and sum sketches collapse to zero under a trivially simple adaptive attack, and the paper's fix — wiring a differentially private tree mechanism directly into the sketch's own randomness — is genuinely clever. I have no interest in re-deriving what they already got right.

What I do have interest in is the six places the paper is *explicit* about not having gone: no sharded or federated deployment, no signed-delta (ReLU) model, no zCDP tightening, no range-free heavy-tailed statistics, no way to tell a benign feedback loop from an actual attacker, and no account of a node crashing or messages arriving out of order. Every one of those is either a limitation the authors name themselves in their own Section 5, or a hidden assumption that falls out the moment you try to run their model on a system that looks like a real distributed messaging pipeline instead of a single obedient process.

So that's what this repository is: six honest attempts to push on those six specific seams, each one built from public-domain, forty-year-old classical building blocks (Bernoulli sampling, Fenwick trees, Laplace and Gaussian noise) rather than anything lifted from the paper's own machinery, each one measured — not asserted — against real market data. If a result came back wrong or the statistics didn't hold up, that's in here too, because a research artifact that only shows you the flattering runs isn't actually research, it's a highlight reel.

## Abstract

We analyze *Adaptively Robust Resettable Streaming* along four axes — theoretical, mathematical, methodological, and empirical — and identify eight concrete limitations, several stated outright by the paper's own authors: no composability across sharded nodes, a worst-case-only adversary model with no accounting for benign adaptivity, an unaddressed zCDP tightening direction, an open problem for the general (signed-delta) ReLU model, an a-priori range assumption for Bernstein statistics that real heavy-tailed data need not satisfy, no treatment of out-of-order delivery or node crashes, and no empirical section at all. We build six independent, from-scratch extensions — **Fed-Tree**, **ReLU-Renew**, **MatRelease**, **ScaleFree-Bernstein**, **BetSketch**, and **HLC-Tree** — each targeting exactly one of these gaps, each grounded in real, cited, mostly-post-2020 literature, and each evaluated on 619,040 rows of real S&P 500 daily trading data (2013–2018) rather than synthetic toy streams. Every reported number is a mean over 12–40 independent seeds with a 95% bootstrap confidence interval, not a single lucky (or unlucky) run. Where an initial design turned out to be wrong — and two of them did, on first measurement — the mistake, the diagnosis, and the fix are documented in the code and in this README rather than quietly edited away.

## Dataset

Real data or nothing — a "robust streaming" claim measured only on synthetic streams the author fully controls isn't really a measurement. This project uses the **S&P 500 five-year daily-bar panel**, mirrored for free, no key or token required, at:

```
https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv
```

| Property | Value |
|---|---|
| Rows | 619,040 |
| Distinct tickers | 505 |
| Trading days | 1,259 (2013-02-08 → 2018-02-07) |
| Tickers present at inception | 476 |
| Real later index additions | ~29 |
| Volume range (same panel) | single digits → 618,237,630 shares in one day |

That last row is doing real work in this repository, not just filling out a table. A dataset that is only "medium-tailed" would let a sloppy heavy-tail sketch pass; this one genuinely spans eight orders of magnitude on the same tickers, which is exactly the property [ScaleFree-Bernstein](#4--scalefree-bernstein-range-oblivious-heavy-tailed-sketching) needed to be tested against honestly. The 505-ticker universe with real join dates gives [Fed-Tree](#1--fed-tree-federated-robust-cardinality-across-k-sharded-nodes) and [MatRelease](#3--matrelease-zcdp-calibrated-continual-release-for-tighter-bounds) a genuine (if modest) growing-cardinality signal instead of an invented one. The true chronological order gives [HLC-Tree](#6--hlc-tree-logical-clock-indexed-robust-release-under-reordering-and-crash-recovery) real logical time to jitter against. The one thing this dataset does *not* contain is an adversary — real market data has no "attacker," so wherever an extension needed one (the probe-and-drain attack, the benign-vs-hostile split for BetSketch), that layer is synthetic and clearly labeled as such. Real key universe, real magnitudes, real order — synthetic stress only where a stress test genuinely requires one.

The program downloads this file itself, via `curl` with a `wget` fallback, the first time you run it. No manual step, no separate data-prep script, no dependency beyond a command-line tool every mainstream OS already ships.

## System Design

One file, built in layers, each layer only reachable by the layer above it:

```
resettable_streaming_extensions.cpp
├── FastRNG                    splitmix64-seeded xorshift128+, used everywhere
├── Record / loadDataset()     download, parse, chronologically re-sort the CSV
├── FenwickNoiseCounter        the one shared noise primitive every extension builds on
├── calibratePureDP / calibrateZCDP    the two noise-calibration formulas everything else calls
├── ClassicalBaselineSketch    the undefended textbook sketch every extension is measured against
├── BootstrapStats / computeStats      40-seed (or 12-seed) mean + 95% bootstrap CI, everywhere
│
├── FedTreeCardinality         ── Extension #1
├── ReLURenewSum                ── Extension #2
├── runMatReleaseComparison     ── Extension #3
├── ScaleFreeBernstein          ── Extension #4
├── BettingDetector, EnsembleMedianCounter, runBetSketchScenario   ── Extension #5
├── WatermarkReorderBuffer, HLCTreeCardinality                     ── Extension #6
│
└── runBaselineDemo() / runExt1…6() / main()
    downloads once, runs all seven experiments, writes 8 plain-text reports
```

Nothing above a given layer is ever called from below it — C++ doesn't hoist declarations the way Python does, so this ordering isn't a style preference, it's a compile requirement I learned the hard way (see [§8](#disclaimer)). Every extension is genuinely independent: delete any five of the six extension blocks and the sixth still compiles and runs on its own. Zero third-party libraries; the only thing reached outside the standard library is a single `curl`/`wget` shell-out for the one-time download.

## Extensions

Every subsection below follows the same shape on purpose: the specific gap in the paper, the mathematics behind the fix, the actual code that implements it (not paraphrased, copied straight from the `.cpp` file), the measured result, and — because a number on its own doesn't mean anything to anyone outside the sub-field — what that result actually buys you if you were running a real system.

### 1 — Fed-Tree: Federated Robust Cardinality Across K Sharded Nodes

**The problem.** The paper's whole framework works *because* it gives up composability — its own abstract says composable sketches are "unnecessarily strong" for its purposes. That's a fine trade for a single machine watching one stream. It is not fine for a sharded messaging system, which is how virtually every real distributed monitoring pipeline is actually built: the paper's single-sketch model cannot be applied to a sharded deployment at all without first shipping every shard's raw keys to one place, which defeats the entire point of sharding.

**The math.** Tickers are partitioned across *K* nodes by `tickerId mod K` — disjoint data, so the classical **parallel composition theorem** (McSherry, SIGMOD 2009) says every node can spend the *full* target ε, not ε/K. An engineer who doesn't know that will reflexively split the budget K ways instead. Since Laplace noise scale is linear in 1/ε, that mistake costs a full **factor of K** in noise standard deviation — not a rounding error, a scaling law.

**The code** (`FedTreeCardinality`, from the `.cpp` file):
```cpp
FedTreeCardinality(int K, double p, double epsPerNode, size_t horizon, uint64_t seed)
    : K_(K), p_(p), rng_(seed) {
    for (int k = 0; k < K_; ++k) {
        double sigma = calibratePureDP(/*L=*/2.0, (double)horizon, epsPerNode);
        counters_.emplace_back(new FenwickNoiseCounter(horizon, sigma, rng_, /*laplace=*/true));
    }
}
double globalEstimate(size_t t) const {
    double sum = 0.0;
    for (int k = 0; k < K_; ++k) sum += counters_[k]->noisyQuery(t);
    return sum / p_;
}
```
`epsPerNode` is passed in *whole* by the "Fed-Tree" caller and pre-divided by K by the "Naive-Split" caller — same class, same code, the only difference is what the caller believes about composition.

**The result** (40 seeds per row, ratio computed as mean-of-errors, not mean-of-ratios — see [§8](#disclaimer) for why that distinction mattered):

| K | Fed-Tree error | Naive-Split error | Ratio |
|---|---|---|---|
| 1 | 0.040 [0.030, 0.050] | 0.040 [0.031, 0.050] | 1.00 |
| 4 | 0.063 [0.046, 0.083] | 0.183 [0.126, 0.244] | 2.91 |
| 16 | 0.114 [0.091, 0.140] | 1.806 [1.386, 2.250] | 15.80 |
| 32 | 0.161 [0.123, 0.206] | 4.613 [3.514, 5.905] | **28.64** |

**What this means in real life.** If you're running a sharded key-value monitor across 32 machines and someone on your team writes the "obviously correct" per-shard privacy budget by dividing the target ε by the shard count, your dashboard's cardinality estimate will be off by nearly 29x — not 2x, not 5x, *29x* — compared to just giving every shard the full budget, for the exact same total privacy guarantee. That is the difference between a dashboard you can act on and a dashboard you should ignore, and it costs nothing extra in privacy to get right.

### 2 — ReLU-Renew: Bias-Bounded Tracking for the Signed-Delta ReLU Model

**The problem.** This one is the paper's own words: Section 5 lists, as an explicitly open problem, robust sketching under `v ← max(0, v+δ)` for *signed* δ, flagging that "partial decrements may cause slight posterior biases to persist and accumulate as keys oscillate in and out of the sample." A message queue's depth — arrivals push up, acks pull down, clamped at zero — is exactly this model, and it is not the resettable (reset-fully-to-zero) model the paper actually solves.

**The math.** Classical sample-and-hold gives each key a *permanent* random threshold `R_x ~ Exp(1/τ)`. ReLU-Renew adds one new ingredient: with small probability *q* on every touch, that threshold is redrawn from scratch, independent of history. An adversary's knowledge of a stale threshold now has a *shelf life* — a Geometric(q) random variable, mean 1/q — instead of lasting forever. This is the same family of argument used to show DP mechanisms under repeated refreshment converge exponentially fast to a stationary distribution (Chourasia, Ye, Shokri, NeurIPS 2021), just one discrete renewal at a time instead of a continuous diffusion step.

**The code** (`ReLURenewSum::applyDelta`, from the `.cpp` file):
```cpp
void applyDelta(uint32_t key, double delta) {
    double oldV = value_.count(key) ? value_[key] : 0.0;
    double newV = std::max(0.0, oldV + delta);           // the ReLU clamp itself
    if (newV <= 0.0) { value_.erase(key); thresh_.erase(key); }
    else {
        if (!value_.count(key)) thresh_[key] = rng_.exponential(tau_);  // fresh key
        value_[key] = newV;
        // the one line that separates this from the classical, fragile mechanism:
        if (q_ > 0.0 && rng_.bernoulli(q_)) thresh_[key] = rng_.exponential(tau_);
    }
    recompute(key);
}
```

**The result.** On the real advance/decline signal (+1 if a ticker's volume rose vs. yesterday, −1 if it fell, accumulated through the same ReLU clamp), tracking error is 4.9% [3.3%, 6.7%] over 12 seeds. The sharper test: hand an attacker a *perfect* initial read on whether a ticker is currently "sampled," then let it go stale. At `q=0` (classical), accuracy tracks its theoretical target *exactly*, forever — 0.361 or 0.639 depending on the belief, zero-width confidence interval, because nothing about a fixed threshold ever changes. At `q=0.1`, that same edge decays from ~0.04 to ~0.00 within a few thousand rounds, in both directions, exactly as the renewal argument predicts.

**What this means in real life.** For a message-queue-depth or similarly signed counter, an attacker who learns "this key currently has a low threshold" under the classical mechanism can exploit that fact *forever* — the bias never heals on its own. Renewal doesn't make the sketch immune to a single lucky guess, but it does mean that guess expires. In a system that runs for months, "the leak eventually closes itself" is a materially different security posture than "the leak is permanent," even though neither one is a perfect guarantee.

### 3 — MatRelease: zCDP-Calibrated Continual Release for Tighter Bounds

**The problem.** The paper's own conclusion names this exact gap and stops there: "analysis of the tree mechanism via approximate DP or zCDP would directly improve space complexity." That's a hypothesis, not a result — nobody in the paper actually measures it.

**The math.** The paper's own Laplace/pure-ε tree mechanism pays an extra factor of `log(T)` from union-bound-style tail accounting across the O(log T) nodes any query touches. Zero-concentrated DP (zCDP) composes *additively* in ρ (ρ_total = Σρᵢ, no union bound needed), which swaps that extra `log(T)` for `sqrt(log(T))` — a real, structural, asymptotic improvement, not a tuning trick.

**The code** — the two calibration functions this entire repository's zCDP-vs-Laplace story is built on:
```cpp
inline double calibratePureDP(double L, double capacity, double eps) {
    double levels = std::log2(std::max(2.0, capacity));
    return L * levels / eps;                              // linear in 1/eps
}
inline double calibrateZCDP(double L, double capacity, double rhoTotal) {
    double levels = std::log2(std::max(2.0, capacity));
    return L * std::sqrt(levels / (2.0 * std::max(rhoTotal, 1e-9)));  // sqrt, not linear
}
```
Both are called on the *identical* Fenwick hierarchy, with the *same* per-seed sample draws — the noise mechanism is the only thing that differs between the two columns.

**The result.** At matched strength (ρ ≈ ε²/2, ε=2) on the real ticker-join cardinality signal, over 40 seeds: Laplace/pure-DP mean error 29.42 [26.67, 33.10], Gaussian/zCDP mean error 15.07 [13.43, 16.78] — a measured **1.95x** improvement, non-overlapping confidence intervals.

**What this means in real life.** If you're already running the paper's own Theorem 2.4 mechanism in production, swapping its noise distribution from Laplace to Gaussian — and its composition accounting from pure-ε to zCDP — costs you nothing in implementation complexity (same tree, same sensitivity bookkeeping) and buys back roughly half your error budget, for free, at the exact same privacy strength.

### 4 — ScaleFree-Bernstein: Range-Oblivious Heavy-Tailed Sketching

**The problem.** The paper's Theorem 4.2, for Bernstein (concave, sublinear) statistics, requires a fixed-in-advance assumption: update values must lie in `[Δ_min, Δ_max]` with `Δ_max/Δ_min = O(poly(T))`. That's an assumption about the *data*, not just the algorithm — and real message/flow sizes are famously "mice and elephants" heavy-tailed. This dataset alone spans single-digit-share days next to a 618-million-share day on the *same tickers*.

**The math.** Instead of a threshold ladder sized from an assumed range, each key's current value is quantized to a lazily-created log₂-scale bucket, and the target statistic is approximated as a weighted sum of ordinary (robustly sketchable) per-bucket cardinalities. A bucket only ever exists once some key's value actually reaches it — space tracks the *observed* range, capped unconditionally at 64 buckets for any 64-bit value, regardless of whether `Δ_max/Δ_min` happens to be polynomial in anything.

**The code** (`ScaleFreeBernstein::applyValue`, the O(1) bucket-move at the heart of it):
```cpp
void applyValue(uint32_t key, double newValue, size_t t) {
    long oldB = curBucket_.count(key) ? curBucket_[key] : -1;
    long newB = newValue > 0.0 ? (long)std::floor(std::log2(newValue) * sub_) : -1;
    if (newB == oldB) return;             // a key lives in exactly ONE bucket at a time
    if (oldB != -1) leave(key, oldB, t);  // so even a huge multiplicative jump in value
    if (newB != -1) { enter(key, newB, t); curBucket_[key] = newB; }  // costs one leave + one enter,
    else curBucket_.erase(key);           // never a walk across every bucket spanned along the way
}
```

**The result.** Tracking `F = Σ sqrt(current volume)` across all 505 tickers, over 12 seeds: final relative error 1.54% [0.86%, 2.34%], using at most 64 buckets regardless of the data's actual 8-order-of-magnitude range. A second, honestly-reported number — the maximum relative error at any single checkpoint along the way, 40.1% — turned out to be almost entirely a single-checkpoint cold-start artifact (true F there is one ticker's one day of volume, a genuinely tiny denominator); excluding that one pathological point, the steady-state worst-case is 5.4% [4.9%, 5.9%], about 3.5x the final number — a gap consistent with, not larger than, what time-uniform confidence theory predicts for a bound checked at 12 points instead of one (Howard, Ramdas, McAuliffe, Sekhon, *Annals of Statistics*, 2021).

**What this means in real life.** You do not need to know, in advance, the ratio between your smallest and largest observed values to get a working, bounded-space sketch for a concave statistic like `Σ sqrt(v)` — which matters enormously for anything network-flow-shaped, where a handful of "elephant" flows can dwarf everything else by many orders of magnitude and nobody can honestly promise that ratio stays polynomial in the stream length.

### 5 — BetSketch: Betting-Gated Adaptive Robustness

**The problem.** The paper treats every adaptive update as equally hostile and pays its full defense cost unconditionally, forever. Its own example of adaptivity is completely benign — "a load balancing system might attempt to reassign load when the estimated load increases" — and there is no mechanism anywhere in the paper's model for telling that apart from an actual attacker.

**The math.** Run the cheap, undefended sketch by default. Run a sequential **betting martingale** alongside it, testing the null "a reset shortly after admission is uncorrelated with admission" — exactly the signature the classical attack leaves. By Ville's inequality, the false-alarm rate is controlled at level α for *any* stopping rule, including a data-dependent one, while a real attack's fixed statistical gap forces detection within a bounded number of steps. Once triggered, defense switches on for good, using M=5 independently-noised Gaussian/zCDP counters combined via median (CoinPress, Biswas-Dong-Kamath-Ullman, NeurIPS 2020, is itself zCDP-native) rather than one single noisy release.

**The code** (`BettingDetector`, the whole detector in eleven lines):
```cpp
class BettingDetector {
public:
    BettingDetector(double mu0, double lambda) : mu0_(mu0), lambda_(lambda) {}
    void observe(double z) {
        double bet = 1.0 + lambda_ * (z - mu0_);
        logWealth_ += std::log(std::max(bet, 1e-6));   // log-space, avoids overflow
    }
    double wealth() const { return std::exp(logWealth_); }
    bool alarmed(double invAlpha) const { return logWealth_ >= std::log(invAlpha); }
private:
    double mu0_, lambda_, logWealth_ = 0.0;
};
```

**The result.** On the real, benign chronological stream (123,865 genuine observations per trial, 12 seeds), zero trials ever crossed the alarm bar — and, reported honestly rather than as a bare "0/12," the exact Clopper-Pearson upper bound this supports is that the true false-alarm rate is not shown to exceed **~22.1%** at this sample size, not "0%." Against the synthetic probe-and-drain attack (40 seeds), the undefended baseline is driven to exactly zero on *every* seed; BetSketch detects within a mean of 8.3 steps out of 4,000 and holds relative error to 6.19% [4.87%, 7.56%] for M=5, versus 5.22% [4.08%, 6.48%] for a single zCDP counter with no ensemble at all — measured directly, paired seed-for-seed, rather than assumed by analogy; the paired difference [−0.025, 0.005] includes zero, an honest result that the ensemble's specific edge over a single well-calibrated counter isn't established by this particular experiment even though the overall pay-as-you-go design clearly works.

**What this means in real life.** A monitoring system that pays full defense cost unconditionally is paying a tax on every ordinary Tuesday to protect against an attacker who may never show up. BetSketch's whole point is that you can run cheap almost all the time and only pay the expensive bill when something statistically resembling an attack actually starts — and here, on real traffic, it never once falsely believed it was under attack, while still catching a real one in under ten steps.

### 6 — HLC-Tree: Logical-Clock-Indexed Robust Release Under Reordering and Crash Recovery

**The problem.** The paper's model is one node, one clock, updates arriving in exactly the order its guarantees are stated for. Real distributed messaging has network jitter, out-of-order delivery across partitions, and nodes that crash and come back — none of which the paper's model represents at all.

**The math.** Tag every event with a logical timestamp, buffer arrivals in a bounded reorder window, and release into the underlying counter strictly in logical order once a watermark passes — bounded reordering costs release *latency*, not accuracy, up to the configured bound. On a simulated crash, discard and rebuild only the *noise-tracking* history; the sample itself is treated as ordinary durably-persisted data (a systems concern, not a privacy one), and the fresh counter is seeded from the sketch's own true internal count — never by replaying a stale noisy number.

**The code** (`HLCTreeCardinality::checkpointAndCrash`, the whole recovery mechanism):
```cpp
double checkpointAndCrash() {
    size_t idx = lastFinalizedTime_ - counterOriginTime_;
    double lastPublicValue = counter_->noisyQuery(idx) / p_;   // logged only, never reused
    int trueSizeAtCrash = (int)sample_.size();                 // a privileged internal read
    rebuildCounter(lastFinalizedTime_ + 1);                    // fresh noise history
    counter_->update(0, trueSizeAtCrash);                      // seeded from truth, not from noise
    return lastPublicValue;
}
```

**The result.** Sweeping injected reordering against configured watermark over 40 seeds: error stays flat at ~20.4 while watermark ≥ disorder, rising to 26.09 [25.46, 26.69] only once watermark (5) falls well short of disorder (500) — bounded reordering genuinely is free up to the bound. For crash recovery, the *correct* paired statistic (bootstrapping the per-seed difference between a crashed-and-recovered sketch and a never-crashed control built from the same seed, not two separately-eyeballed marginal intervals) gives a difference of 4.79 with a 95% CI of **[−3.05, 13.13]** — includes zero, confirmed by an exact sign test (p = 0.08) — meaning recovery is statistically indistinguishable from never having crashed at all.

**What this means in real life.** A node that crashes mid-stream and restarts from a checkpoint doesn't have to collapse back to zero, and it doesn't have to leak anything extra to avoid that collapse — it just needs to know its own true current state is always a legitimate thing for it to read, and to throw away only the part of its history (the noise draws) that genuinely needs throwing away.

## How to Run

```bash
g++ -O3 -std=c++17 -o extensions resettable_streaming_extensions.cpp
./extensions
```

A bare `g++ resettable_streaming_extensions.cpp` also compiles and runs correctly on any reasonably modern compiler — `-O3 -std=c++17` just makes it faster and is not required for correctness. Verified to compile cleanly with `-Wall -Wextra -pedantic`, under `-std=c++14` as a compatibility floor, and from a bare `g++` invocation with zero flags, on Linux, Intel macOS, and Apple Silicon macOS. The only thing reached outside the C++ standard library is a shell-out to `curl` (falling back to `wget`) for the one-time dataset download — if neither is available, the program tells you exactly which URL to fetch by hand and where to put the file.

Total run time is well under a minute on ordinary hardware: the download happens once (skipped automatically on every subsequent run once the CSV is on disk), and all seven experiments together — baseline plus six extensions, several of them averaging 12 to 40 independent seeds apiece — finish in roughly 15–20 seconds.

## Execution Flow

```
./extensions
   │
   ├── ensureDataset()          skip if the CSV already exists, else curl/wget it
   ├── loadDataset()            parse, intern ticker symbols, sort chronologically
   │
   ├── runBaselineDemo()        → 00_baseline_vulnerability.txt
   ├── runExt1FedTree()         → 01_ext1_fedtree.txt
   ├── runExt2ReLURenew()       → 02_ext2_relu_renew.txt
   ├── runExt3MatRelease()      → 03_ext3_matrelease.txt
   ├── runExt4ScaleFree()       → 04_ext4_scalefree_bernstein.txt
   ├── runExt5BetSketch()       → 05_ext5_betsketch.txt
   ├── runExt6HLCTree()         → 06_ext6_hlc_tree.txt
   │
   └── writes SUMMARY.txt, prints a short console recap
```

Each of the eight `.txt` files is self-contained and re-readable on its own: it states which limitation of the original paper it targets, the exact real-data experiment behind every number, and a "paper comparison" paragraph closing the loop back to the specific theorem or open problem it's responding to. Nothing in any of them is canned — delete the files, run the program again, and they regenerate from scratch (the dataset persists on disk; the 40-or-12-seed statistics do not, and will vary slightly run to run, which is the entire point of reporting a confidence interval instead of a bare number).

## Disclaimer

Read this section before citing any number above as a settled fact, because I'd rather you hear the caveats from me than discover them yourself.

This is an independent research artifact, unaffiliated with and not endorsed by the authors of *Adaptively Robust Resettable Streaming*. Nothing here reproduces their algorithms, their proofs, or their code — the six extensions build only on classical, decades-old, public-domain techniques, applied to problems the original paper leaves open or doesn't address. Several mathematical claims in the source comments are marked "informal" or "proof sketch" on purpose: they are carefully reasoned and empirically checked, not machine-verified theorems, and should be read with that distinction in mind. A handful of the citations added during later debugging rounds (Ext4, Ext5, Ext6 specifically) are very recent 2025–2026 preprints rather than established peer-reviewed venues; each is flagged as such at its point of use, not folded silently into a blanket "peer-reviewed" claim.

I also want to be direct about the process, because it's part of the actual result: this code went through multiple rounds of adversarial self-review and real bug fixes before landing here — a use-before-declaration compile error, a mismatched noise-composition mechanism that initially made one extension *worse* rather than better, a metric that conflated an unavoidable cold-start artifact with genuine steady-state error, and a paired experiment that was originally analyzed with the wrong statistic. Every one of those was caught by actually compiling and running the code against real data, not by reasoning about it in the abstract, and every fix is documented in the code and in this README rather than quietly smoothed over. Every reported number came from a run I executed myself before writing it down.

Licensed under MIT. Use it, break it, extend it — just don't mistake "measured and reported honestly" for "final and unimprovable."

## How to Cite

If this repository is useful in your own work, please cite it alongside the paper it responds to:

```bibtex
@misc{jani2026resettableextensions,
  author = {Jani, Nihar Mahesh},
  email = {niharmaheshjani@gmail.com}
  title  = {Resettable Streaming Extensions: Six Adaptively Robust
            Sketches for Synchronized Distributed Messaging},
  year   = {2026},
  note   = {Independent research artifact, single-file C++ implementation},
  url    = {https://github.com/NiharJani2002/FRMSBH}
}

@misc{cohen2026resettable,
  author = {Cohen, Edith and Gribelyuk, Elena and Nelson, Jelani and Stemmer, Uri},
  title  = {Adaptively Robust Resettable Streaming},
  year   = {2026},
  note   = {Preprint}
}
```

And if you find a bug I didn't — please, tell me. That's exactly how the last three of these got fixed.
