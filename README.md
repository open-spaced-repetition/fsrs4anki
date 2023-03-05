<p align="center">
  <a href="https://github.com/open-spaced-repetition/fsrs4anki/wiki">
    <img src="https://user-images.githubusercontent.com/32575846/210218310-4575b0f3-c570-4c5f-acec-9d35206fc920.png" width="150" height="150" alt="FSRS4Anki">
  </a>
</p>

<div align="center">

# FSRS4Anki

_✨ A modern Anki [custom scheduling](https://faqs.ankiweb.net/the-2021-scheduler.html#add-ons-and-custom-scheduling) based on [Free Spaced Repetition Scheduler](https://github.com/open-spaced-repetition/fsrs4anki/wiki/Free-Spaced-Repetition-Scheduler) algorithm ✨_  

</div>

<p align="center">
  <a href="https://raw.githubusercontent.com/open-spaced-repetition/fsrs4anki/main/LICENSE">
    <img src="https://img.shields.io/github/license/open-spaced-repetition/fsrs4anki" alt="license">
  </a>
  <a href="https://github.com/open-spaced-repetition/fsrs4anki/releases">
    <img src="https://img.shields.io/github/v/release/open-spaced-repetition/fsrs4anki?color=blueviolet&include_prereleases" alt="release">
  </a>
</p>

## Introduction

FSRS4Anki consists of two main parts: scheduler and optimizer.

The scheduler is based on a variant of the DSR (Difficulty, Stability, Retrievability) model, which is used to predict memory states. The scheduler aims to achieve the requested retention for each card and each review.

The optimizer applies *Maximum Likelihood Estimation* and *Backpropagation Through Time* to estimate the stability of memory and learn the laws of memory from time-series review logs. Then, it can find the optimal retention to minimize the repetitions via the stochastic shortest path algorithm.

For more detail on the mechanism of the FSRS algorithm, please see this paper: [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling](https://www.maimemo.com/paper/).

[FSRS4Anki Helper](https://github.com/open-spaced-repetition/fsrs4anki-helper) is an Anki add-on that supports the FSRS4Anki Scheduler. It has five features:
1. **Reschedule** cards based on their entire review histories.
2. **Postpone** due cards whose retention is higher than your target.
3. **Advance** undue cards whose retention is lower than your target.
4. **Balance** the load during rescheduling.
5. **No Anki** on Free Days (such as weekends).

## Usage

For the tutorial on FSRS4Anki scheduler, optimizer, helper, and simulator, please see: [Usage](https://github.com/open-spaced-repetition/fsrs4anki/wiki/Usage)

## FAQ

Here collect some questions from issues, forums, and others: [FAQ](https://github.com/open-spaced-repetition/fsrs4anki/wiki/FAQ)

## Compatibility

Some add-ons modify the scheduling of Anki, which would cause conflict with FSRS4Anki scheduler. Please see [Compatibility](https://github.com/open-spaced-repetition/fsrs4anki/wiki/Compatibility) for more details. I will test these add-ons. Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if I miss any add-ons.

[![Star History Chart](https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date)](https://star-history.com/#open-spaced-repetition/fsrs4anki&Date)
