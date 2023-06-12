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

## 1 Quick Start

### 1.1 Enable Anki's V3 Scheduling Algorithm

Tick Settings>Review>Enable V3 Scheduling Algorithm

### 1.2 Paste FSRS Algorithm Code

In the deck options, find the Advanced Settings column, and paste the following code into the Custom Scheduling field:

You can update the code by visiting: 

www.github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_scheduler.js

In theory, you've now started using the FSRS4Anki algorithm. If you're unsure, you can change this part of the code:

```javascript
const display_memory_state = false;
```

to:

```javascript
const display_memory_state = true;
```

Then open any deck for review and you'll see:

This shows that your FSRS is running normally. You can then change the code back and the message will no longer display.

## 2 Advanced Usage

### 2.1 Generate Personalized Parameters

www.github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_optimizer.ipynb

Open the optimizer's notebook and click on Open in Colab to run the optimizer on Google Colab. You don't need to configure the coding environment yourself and you can use Google's machines for free (you'll need to register a Google account):

After opening it in Colab, switch to the folder tab. Once the Optimizer connects to Google's machine, you can right-click to upload your deck file/collection file. When exporting these files, make sure to tick "Include learning progress information" and "Support older versions of Anki".

After it's uploaded, change the filename in the notebook to the name of your uploaded file.

Then click "Run All"

Wait for the code to finish in section 2.3, then copy the personalized parameters that were output.

Replace the parameters in the FSRS code you copied earlier.

⚠️Note: when replacing these parameters, be sure not to delete the comma at the end.

### 2.2 Deck Parameter Settings

You can also generate different parameters for different decks and configure them separately in the code. In the default configuration, deckParams already contains three sets of parameters.

The group "global config for FSRS4Anki" is global parameters.

The group "ALL::Learning::English::Reading" are the parameters applied to the deck "ALL::Learning::English::Reading" and its sub-decks.

Similarly, the third group is the parameters applied to the deck "ALL::Archive" and its sub-decks. You can replace these with the decks you want to configure. If you need more, feel free to copy and add them.


If there are some decks you don't want to use FSRS with, you can add their names to the skip_decks list.

const skip_decks = ["ALL::Learning::ML::NNDL", "ALL::Learning::English"];

## 3 Using the Helper Plugin

The Helper plugin is purely an added bonus and is not recommended for extensive use. Installation link:

www.ankiweb.net/shared/info/759844606

### 3.1 Reschedule

Rescheduling all cards can predict the memory status based on each card's review history and arrange intervals, using the personalized parameters we filled in earlier.

Note: For cards that have been reviewed multiple times using Anki's default algorithm, rescheduling may give different intervals than the Scheduler because the Scheduler can't access the full review history when running. In this case, the intervals given by rescheduling will be more accurate. But afterward, there will be no difference between the two.

### 3.2 Stress Balance

Once the stress balance option is enabled, re-planning will make the daily review volume as consistent and smooth as possible.

Here's a comparison, the first image is rescheduling before enabling it, and the second image is after enabling:

### 3.3 Weekend Off

In fact, you can choose any days from Monday to Sunday to take off. Once enabled, the Helper will try to avoid the dates you set for review when rescheduling.

Effect:

### 3.4 Advance/Postpone

These two functions are very similar, so I'll talk about them together. You can set the number of cards to advance/postpone, and the Helper plugin will sort them in the order of relative advance/postpone, then perform the advance/postpone, ensuring that the deviation from the original review arrangement is minimized while meeting the number of cards you set.

### 3.5 Disperse Associated Cards

In Anki, some templates will generate multiple cards related in content from the same note, such as flip cards (Chinese->English, English->Chinese) and cloze cards (when you dug many blanks on the same note). If the review dates of these cards are too close, they may interfere or remind each other. Dispersing associated cards can stagger the review dates of these cards as much as possible.

### 3.6 Advanced Search

In the card browser, you can right-click on the header and click on Difficulty, Stability, Retention to display the current memory status of the card.

It also supports filtering syntax for three attributes, here are some examples:

- s<10: Cards with memory stability less than 10 days
- d=5: Cards with difficulty equal to 5
- r<0.6: Cards with memory retrievability (recall probability) less than 60%

### 3.7 Advanced Statistics

Hold down the Shift key and click "Statistics" to enter the old version of Anki's statistics interface.

Average retention, i.e., average retention rate, reflects the percentage of all cards you have reviewed that you still remember.

Average stability, i.e., average memory stability, reflects the forgetting rate of all cards you have reviewed. The greater the stability, the slower the forgetting rate.

# FAQ

Here collect some questions from issues, forums, and others: [FAQ](https://github.com/open-spaced-repetition/fsrs4anki/wiki/FAQ)

# Compatibility

Some add-ons modify the scheduling of Anki, which would cause conflict with FSRS4Anki scheduler. Please see [Compatibility](https://github.com/open-spaced-repetition/fsrs4anki/wiki/Compatibility) for more details. I will test these add-ons. Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if I miss any add-ons.

# Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date)](https://star-history.com/#open-spaced-repetition/fsrs4anki&Date)
