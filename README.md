<p align="center">
  <a href="https://github.com/open-spaced-repetition/fsrs4anki/wiki">
    <img src="https://user-images.githubusercontent.com/32575846/210218310-4575b0f3-c570-4c5f-acec-9d35206fc920.png" width="150" height="150" alt="FSRS4Anki">
  </a>
</p>

<div align="center">

# FSRS4Anki

_✨ A modern Anki [custom scheduling](https://faqs.ankiweb.net/the-2021-scheduler.html#add-ons-and-custom-scheduling) based on [Free Spaced Repetition Scheduler](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm) algorithm ✨_  

</div>

<p align="center">
  <a href="https://raw.githubusercontent.com/open-spaced-repetition/fsrs4anki/main/LICENSE">
    <img src="https://img.shields.io/github/license/open-spaced-repetition/fsrs4anki" alt="license">
  </a>
  <a href="https://github.com/open-spaced-repetition/fsrs4anki/releases/latest">
    <img src="https://img.shields.io/github/v/release/open-spaced-repetition/fsrs4anki?color=blueviolet" alt="release">
  </a>
</p>

# Table of contents

- [FSRS4Anki](#fsrs4anki)
- [Introduction](#introduction)
- [1 Quick Start](#1-quick-start)
  - [1.1 Enable Anki's V3 Scheduler](#11-enable-ankis-v3-scheduler)
  - [1.2 Paste FSRS Scheduler Code](#12-paste-fsrs-scheduler-code)
- [2 Advanced Usage](#2-advanced-usage)
  - [2.1 Generate Personalized Parameters](#21-generate-personalized-parameters)
    - [2.1a Google Colab](#21a-google-colab)
    - [2.1b Website](#21b-website)
    - [2.1c Command Line](#21c-command-line)
    - [2.1d Anki Addon](#21d-anki-addon-experimental) **EXPERIMENTAL**
  - [2.2 Deck Parameter Settings](#22-deck-parameter-settings)
- [3 Using the Helper Add-on](#3-using-the-helper-add-on)
- [FAQ](#faq)
- [Compatibility](#compatibility)
- [Contribute](#contribute)
- [Stargazers over time](#stargazers-over-time)

# Introduction

FSRS4Anki consists of two main parts: scheduler and optimizer.

The scheduler is based on a variant of the DSR (Difficulty, Stability, Retrievability) model, which is used to predict memory states. The scheduler aims to achieve the requested retention for each card and each review.

The optimizer applies *Maximum Likelihood Estimation* and *Backpropagation Through Time* to estimate the stability of memory and learn the laws of memory from time-series review logs. Then, it can find the optimal retention to minimize the repetitions via the stochastic shortest path algorithm.

For more detail on the mechanism of the FSRS algorithm, please see my papers: [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling (free access)](https://www.maimemo.com/paper/) and [Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory (submit request)](https://www.researchgate.net/publication/369045947_Optimizing_Spaced_Repetition_Schedule_by_Capturing_the_Dynamics_of_Memory).

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

In the deck options, find the Advanced Settings column, and paste the code in [fsrs4anki_scheduler.js](https://github.com/open-spaced-repetition/fsrs4anki/releases/latest) into the Custom Scheduling field:

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

You can generate parameters in a variety of ways depending on which method you prefer.  
For the most up to date methods please check the [releases](https://github.com/open-spaced-repetition/fsrs4anki/releases/tag/v3.26.2).

### 2.1a Google Colab

Open the [optimizer's notebook](https://github.com/open-spaced-repetition/fsrs4anki/releases/latest) and click on Open in Colab to run the optimizer on Google Colab. You don't need to configure the coding environment yourself and you can use Google's machines for free (you'll need to register a Google account):

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

### 2.1b Website

Simply upload your exported decks to this website and it will optimise it for you.  
https://huggingface.co/spaces/open-spaced-repetition/fsrs4anki_app

![image](https://github.com/Luc-mcgrady/fsrs4anki/assets/63685643/a03217f0-6627-4854-971f-f2bc9d14da5c)


### 2.1c Command Line 

There is a python package for the optimizer. This package has torch as a dependency, so note it might take about half a gigabyte of space.

#### Installation

Install the package with the command:

```
python -m pip install FSRS-Optimizer
```

You should upgrade regularly to make sure you have the most recent version of FSRS-Optimizer:

```
python -m pip install fsrs4anki-optimizer --upgrade
```

#### Usage

Export your deck and cd into the folder to which you exported it.  
Then you can run:

```
python -m fsrs_optimizer "package.(colpkg/apkg)"
```

You can also list multiple files, e.g.:

```
python -m fsrs4_optimizer "file1.akpg" "file2.apkg"
```

Wildcards are supported:

```
python -m fsrs4_optimizer *.apkg
```

There are certain options which are as follows:

```
options:
  -h, --help           show this help message and exit
  -y, --yes, --no-yes  If set automatically defaults on all stdin settings.
  -o OUT, --out OUT    File to APPEND the automatically generated profile to.
```

#### Expected Functionality

![image](https://github.com/Luc-mcgrady/fsrs4anki/assets/63685643/ac2e8ae0-726c-46fd-b110-0701fa87cb66)
![image](https://github.com/Luc-mcgrady/fsrs4anki/assets/63685643/1fe8b0bb-7ac0-4a31-b594-465239ea3a1e)

### 2.1d Anki Addon **EXPERIMENTAL**

Download and install [this](https://github.com/Luc-mcgrady/fsrs4anki-helper/tree/optimizer) version of the anki helper addon either by git cloning it into the anki addons folder or [downloading it as a zip](https://github.com/Luc-mcgrady/fsrs4anki-helper/archive/refs/heads/optimizer.zip) and extracting the zip into the anki addons folder.

Install the optimizer locally.  
![image](https://user-images.githubusercontent.com/63685643/236647263-b1e57db1-4ad0-441b-9abe-91cbd36c13b0.png)  
Please pay attention to the popup.  
![image](https://github.com/Luc-mcgrady/fsrs4anki/assets/63685643/ebe42eb4-f63d-4e58-b593-c173891dd29c)


After that has downloaded and installed you should be able to run the optimizer from within anki.
Press the cog next to any given deck and hit the optimize option.  
![image](https://user-images.githubusercontent.com/63685643/236647245-757ca803-b8cf-41cd-a1ae-8ed9af852ad8.png)  
Anki may then hang a small while while it loads the optimizer.

![image](https://github.com/Luc-mcgrady/fsrs4anki/assets/63685643/e160e5ba-c51f-46a9-9813-9dceb18e47ff)  
Hit yes to find the optimum retention, Hit no to not or hit cancel to pick a different deck. 

If all is well you should then get a toolbar popup which tells you the progress of the optimization.
![image](https://user-images.githubusercontent.com/63685643/236647707-38101c10-ccd2-4417-aa3f-f2e4e10bb4c3.png)

You should then get the stats in a format which is easy to copy into the javascript scheduler.
![image](https://user-images.githubusercontent.com/63685643/236647716-bfd8099a-6e7f-46e7-bce8-e18e75e75d46.png)  
These values are saved in the addons config file which can be found and edited in anki if you want to change the retention manually for example.
![image](https://user-images.githubusercontent.com/63685643/236647915-7a865bb0-f057-4404-af0f-27c81be99082.png)

If there are any issues with this please mention them on this pull request [here](https://github.com/open-spaced-repetition/fsrs4anki-helper/pull/91).

### 2.2 Deck Parameter Settings

You can also generate different parameters for different decks and configure them separately in the code. In the default configuration, `deckParams` already contains three groups of parameters.

The group "global config for FSRS4Anki" is global parameters.

The group "MainDeck1" are the parameters applied to the deck "MainDeck1" and its sub-decks.

Similarly, the third group is the parameters applied to the deck "MainDeck2::SubDeck::SubSubDeck" and its sub-decks. You can replace these with the decks you want to configure. If you need more, feel free to copy and add them.

```javascript
const deckParams = [
  {
    // Default parameters of FSRS4Anki for global
    "deckName": "global config for FSRS4Anki",
    "w": [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61],
    // The above parameters can be optimized via FSRS4Anki optimizer.
    // For details about the parameters, please see: https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm
    // User's custom parameters for global
    "requestRetention": 0.9, // recommended setting: 0.8 ~ 0.9
    "maximumInterval": 36500,
    // FSRS only modifies the long-term scheduling. So (re)learning steps in deck options work as usual.
    // I recommend setting steps shorter than 1 day.
  },
  {
    // Example 1: User's custom parameters for this deck and its sub-decks.
    "deckName": "MainDeck1",
    "w": [0.6, 0.9, 2.9, 6.8, 4.72, 1.02, 1, 0.04, 1.49, 0.17, 1.02, 2.15, 0.07, 0.35, 1.17, 0.32, 2.53],
    "requestRetention": 0.9,
    "maximumInterval": 36500,
  },
  {
    // Example 2: User's custom parameters for this deck and its sub-decks.
    // Don't omit any keys.
    "deckName": "MainDeck2::SubDeck::SubSubDeck",
    "w": [0.6, 0.9, 2.9, 6.8, 4.72, 1.02, 1, 0.04, 1.49, 0.17, 1.02, 2.15, 0.07, 0.35, 1.17, 0.32, 2.53],
    "requestRetention": 0.9,
    "maximumInterval": 36500,
  }
];
```

If there are some decks you don't want to use FSRS with, you can add their names to the `skip_decks` list.

```javascript
const skip_decks = ["MainDeck3", "MainDeck4::SubDeck"];
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

# Contribute

You can contribute to FSRS4Anki by beta testing, submitting code, or sharing your data. If you want to share your data with me, please fill this form: https://forms.gle/KaojsBbhMCytaA7h8

# Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date)](https://star-history.com/#open-spaced-repetition/fsrs4anki&Date)
