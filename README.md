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

# Introduction

FSRS4Anki consists of two main parts: scheduler and optimizer.

The scheduler is based on a variant of the DSR (Difficulty, Stability, Retrievability) model, which is used to predict memory states. The scheduler aims to achieve the requested retention for each card and each review.

The optimizer applies *Maximum Likelihood Estimation* and *Backpropagation Through Time* to estimate the stability of memory and learn the laws of memory from time-series review logs. Then, it can find the optimal retention to minimize the repetitions via the stochastic shortest path algorithm.

For more detail on the mechanism of the FSRS algorithm, please see my papers: [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling](https://www.maimemo.com/paper/) and [Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory](https://doi.org/10.1109/TKDE.2023.3251721).

[FSRS4Anki Helper](https://github.com/open-spaced-repetition/fsrs4anki-helper) is an Anki add-on that supports the FSRS4Anki Scheduler. It has six features:
1. **Reschedule** cards based on their entire review histories.
2. **Postpone** due cards whose retention is higher than your target.
3. **Advance** undue cards whose retention is lower than your target.
4. **Balance** the load during rescheduling.
5. **No Anki** on Free Days (such as weekends).
6. **Disperse** Siblings (cards with the same note) to avoid interference & reminder.

# Tutorial

中文版请见：[FSRS4Anki 使用指北](https://zhuanlan.zhihu.com/p/636564830)

## 1 Quick Start

### 1.1 Enable Anki's V3 Scheduler

Preferences > Review > Enable V3 Scheduler

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/8f91fba8-9b8b-405c-8aa9-42123ba5faeb)

### 1.2 Paste FSRS Scheduler Code

In the deck options, find the Advanced Settings column, and paste the code in [fsrs4anki_scheduler.js](https://www.github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_scheduler.js) into the Custom Scheduling field:

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/5c292f91-8845-4f8c-ac42-55f9a0f2946e)

Idealy, you've now started using the FSRS4Anki Scheduler. If you're unsure, you can change this part of the code:

```javascript
const display_memory_state = false;
```

to:

```javascript
const display_memory_state = true;
```

Then open any deck for review and you'll see:

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/0a5d4561-6052-45f3-91a5-5f21dd6497b9)

This shows that your FSRS is running normally. You can then change the code back and the message will no longer display.

## 2 Advanced Usage

### 2.1 Generate Personalized Parameters

Open the [optimizer's notebook](https://www.github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_optimizer.ipynb) and click on Open in Colab to run the optimizer on Google Colab. You don't need to configure the coding environment yourself and you can use Google's machines for free (you'll need to register a Google account):

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/5f5af21b-583d-496c-9bad-0eef0b1fb7a6)

After opening it in Colab, switch to the folder tab. Once the Optimizer connects to Google's machine, you can right-click to upload your deck file/collection file. When exporting these files, make sure to tick "Include scheduling information" and "Support older Anki versions".

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/66f9e323-fca8-4553-bcb2-b2e511fcf559)

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/65da272d-7a01-4c46-a1d9-093e548f1a2d)

After it's uploaded, change the `filename` in the notebook to the name of your uploaded file. And set your `timezone` and `next_day_starts_at`.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/f344064c-4ccf-4884-94d0-fc0a1d3c3c24)

Then click "Run All".

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/77947790-6916-4a99-ba28-8da42fd5b350)

Wait for the code to finish in section 2.3, then copy the personalized parameters that were output.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/8df1d210-73c3-4194-9b3b-256279c4c2fd)

Replace the parameters in the FSRS code you copied earlier.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/70b3b45a-f014-4574-81eb-cad6d19f93d9)

⚠️Note: when replacing these parameters, be sure not to delete the comma at the end.

### 2.2 Deck Parameter Settings

You can also generate different parameters for different decks and configure them separately in the code. In the default configuration, `deckParams` already contains three groups of parameters.

The group "global config for FSRS4Anki" is global parameters.

The group "ALL::Learning::English::Reading" are the parameters applied to the deck "ALL::Learning::English::Reading" and its sub-decks.

Similarly, the third group is the parameters applied to the deck "ALL::Archive" and its sub-decks. You can replace these with the decks you want to configure. If you need more, feel free to copy and add them.

```javascript
const deckParams = [
  {
    // Default parameters of FSRS4Anki for global
    "deckName": "global config for FSRS4Anki",
    "w": [1, 1, 5, -0.5, -0.5, 0.2, 1.4, -0.12, 0.8, 2, -0.2, 0.2, 1],
    // The above parameters can be optimized via FSRS4Anki optimizer.
    // For details about the parameters, please see: https://github.com/open-spaced-repetition/fsrs4anki/wiki/Free-Spaced-Repetition-Scheduler
    // User's custom parameters for global
    "requestRetention": 0.9, // recommended setting: 0.8 ~ 0.9
    "maximumInterval": 36500,
    "easyBonus": 1.3,
    "hardInterval": 1.2,
    // FSRS only modifies the long-term scheduling. So (re)learning steps in deck options work as usual.
    // I recommend setting steps shorter than 1 day.
  },
  {
    // Example 1: User's custom parameters for this deck and its sub-decks.
    // Need to add <div id=deck deck_name="{{Deck}}"></div> to your card's front template's first line.
    "deckName": "ALL::Learning::English::Reading",
    "w": [1.1475, 1.401, 5.1483, -1.4221, -1.2282, 0.035, 1.4668, -0.1286, 0.7539, 1.9671, -0.2307, 0.32, 0.9451],
    "requestRetention": 0.9,
    "maximumInterval": 36500,
    "easyBonus": 1.3,
    "hardInterval": 1.2,
  },
  {
    // Example 2: User's custom parameters for this deck and its sub-decks.
    // Don't omit any keys.
    "deckName": "ALL::Archive",
    "w": [1.2879, 0.5135, 4.9532, -1.502, -1.0922, 0.0081, 1.3771, -0.0294, 0.6718, 1.8335, -0.4066, 0.7291, 0.5517],
    "requestRetention": 0.9,
    "maximumInterval": 36500,
    "easyBonus": 1.3,
    "hardInterval": 1.2,
  }
];
```

If there are some decks you don't want to use FSRS with, you can add their names to the `skip_decks` list.

```javascript
const skip_decks = ["ALL::Learning::ML::NNDL", "ALL::Learning::English"];
```

## 3 Using the Helper Add-on

Please see: [FSRS4Anki Helper](https://github.com/open-spaced-repetition/fsrs4anki-helper)

# FAQ

Here collect some questions from issues, forums, and others: [FAQ](https://github.com/open-spaced-repetition/fsrs4anki/wiki/FAQ)

# Compatibility

Some add-ons modify the scheduling of Anki, which would cause conflict with FSRS4Anki scheduler.

| Add-on                                                       | Compatible? | Comment |
| ------------------------------------------------------------ |-------------------| ------- |
|[Advanced Review Bottom Bar](https://ankiweb.net/shared/info/1136455830)|Yes✅|Please use the latest version.|
|[Incremental Reading v4.11.3 (unofficial clone)](https://ankiweb.net/shared/info/999215520)|No❌|It shows the interval given by Anki's built-in scheduler, not the custom scheduler.|
| [Auto Ease Factor](https://ankiweb.net/shared/info/1672712021)|Yes✅|`Ease Factor` doesn't affect the interval given by FSRS.|
| [Delay siblings](https://ankiweb.net/shared/info/1369579727) |Yes✅|Delay siblings will modify the interval give by FSRS.|
| [autoLapseNewInterval](https://ankiweb.net/shared/info/372281481) |Yes✅|`New Interval` doesn't affect the interval given by FSRS.|
| [Straight Reward](https://ankiweb.net/shared/info/957961234) |Yes✅|`Ease Factor` doesn't affect the interval given by FSRS.|
| [Pass/Fail](https://ankiweb.net/shared/info/876946123) |Yes✅| `Pass` is the equivalent of `Good`.|

Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if I miss any add-ons.

# Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date)](https://star-history.com/#open-spaced-repetition/fsrs4anki&Date)
