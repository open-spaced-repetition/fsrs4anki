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
- [Introduction](#Introduction)
- [Installation](#Installation)
  - [1. Enable the V3 scheduler in Anki](#1-Enable-the-V3-scheduler-in-Anki)
  - [2. Export your deck or collection](#2-Export-your-deck-or-collection)
  - [3. Go to the optimizer page and run the optimizer](#3-Go-to-the-optimizer-page-and-run-the-optimizer)
  - [4. Copy the optimal parameters](#4-Copy-the-optimal-parameters)
  - [5. Copy the scheduler code, paste it into Anki, and then paste the optimal parameters into it](#5-Copy-the-scheduler-code,-paste-it-into-Anki,-and-then-paste-the-optimal-parameters-into-it)
  - [6. Use the helper add-on to reschedule all cards](#6-Use-the-helper-add-on-to-reschedule-all-cards)
  - [7. Extra features](#7-Extra-features)
- [Advanced featues](#Advanced-features)
  - [Using different parameters for different decks](#Using-different-parameters-for-different-decks)
- [FAQ](#faq)
- [Compatibility](#compatibility)
- [Contribute](#contribute)
  - [Contributors](#contributors)
- [Stargazers over time](#stargazers-over-time)

# Introduction

FSRS4Anki consists of three parts: scheduler, optimizer and helper add-on.

The scheduler replaces Anki’s built in scheduler. You can find the code here: https://github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_scheduler.js

The optimizer finds parameters that provide the best fit to your review history: https://colab.research.google.com/github/open-spaced-repetition/fsrs4anki/blob/v4.5.3/fsrs4anki_optimizer.ipynb

The add-on has many useful features, you can read about it here: https://github.com/open-spaced-repetition/fsrs4anki-helper

For more detail on the mechanism of the FSRS algorithm, please see my papers: [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling (free access)](https://www.maimemo.com/paper/) and [Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory (submit request)](https://www.researchgate.net/publication/369045947_Optimizing_Spaced_Repetition_Schedule_by_Capturing_the_Dynamics_of_Memory).

# Installation

中文版请见：[FSRS4Anki 使用指北](https://zhuanlan.zhihu.com/p/636564830)


## 1. Enable the V3 scheduler in Anki

Go to Tools > Preferences > Review > Enable V3 Scheduler.

![1](https://github.com/Expertium/fsrs4anki/assets/83031600/ecf7d77e-1244-4a0a-8cfd-22a5b76e0fe1)

## 2. Export your deck or collection

Make sure to select “Include scheduling information” and “Support older Anki versions”.

![2](https://github.com/Expertium/fsrs4anki/assets/83031600/66eb57b6-81ca-4ba5-a0d8-bada9daf9a59)

## 3. Go to the optimizer page and run the optimizer

Replace "collection-2022-09-18@13-21-58.colpkg" with the name of your deck/collection. Collections have .colpkg at the end of the filename, and decks have .apkg. Replace ‘Asia/Shanghai’ with your timezone, there is a link to the list of timezones.

![3](https://github.com/Expertium/fsrs4anki/assets/83031600/0b163c3e-f6e2-458b-a4a5-f73492f22da2)

Go to Tools > Preferences > Review > Next day starts at to find out your value for next_day_starts_at.

![4](https://github.com/Expertium/fsrs4anki/assets/83031600/36cfd965-c7b1-4824-9000-c9c36feee0c3)

To run the optimizer, either press Ctrl+F9 or go to Runtime > Run all.

![5](https://github.com/Expertium/fsrs4anki/assets/83031600/2476c8bc-a327-4a4a-8de2-96efbd9da60d)

## 4. Copy the optimal parameters

Go to section 2.2 (Result), the optimal parameters will be available there. Copy them and paste them somewhere temporarily.

![6](https://github.com/Expertium/fsrs4anki/assets/83031600/469abd9b-9032-4208-ae52-0aba1d44e213)

## 5. Copy the scheduler code, paste it into Anki, and then paste the optimal parameters into it

Go to this page: https://github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_scheduler.js

Copy all of the code and paste it in the settings of any deck (it doesn’t matter which deck):

![7](https://github.com/Expertium/fsrs4anki/assets/83031600/a11f508c-9329-42db-96f6-ca39e9430bd0)

Copy the optimal parameters and paste them here:

![8](https://github.com/Expertium/fsrs4anki/assets/83031600/1261d76d-d346-4c83-a8e2-9119b0e9e31a)

Make sure that you don’t accidentally erase the square brackets or the comma after the closing bracket. The code will break without them.

Choose your requested retention and max. interval:

![9](https://github.com/Expertium/fsrs4anki/assets/83031600/168248f8-bee3-4f7d-bb55-20aa0f07e9c1)

Higher requested retention leads to more reviews/day. FSRS’s Maximum interval overrides Anki’s built in maximum interval.

Ensure that your learning and re-learning steps are **no longer than 1 day** for every single deck. Other settings, such as “Graduating interval” and “Easy interval”, don’t matter. For more details about which Anki settings matter and which are obsolete, see FAQ: https://github.com/open-spaced-repetition/fsrs4anki/wiki/FAQ

![10](https://github.com/Expertium/fsrs4anki/assets/83031600/8e3b6901-fae9-4a5d-bd32-4bd8da46c856)

## 6. Use the helper add-on to reschedule all cards

Go to Tools > FSRS4Anki Helper > Reschedule all cards.

![11](https://github.com/Expertium/fsrs4anki/assets/83031600/0fd1e427-96ad-4d59-85f2-ef9aed64d7ce)

After rescheduling, you will likely see a lot of due cards, several times more than you are used to. This is typical. You can use the Postpone feature of the add-on to help you deal with the backlog. Read more about add-on features here: https://github.com/open-spaced-repetition/fsrs4anki-helper#overview

## 7. Extra features

To check that FSRS in enabled, change this line of code:

![12](https://github.com/Expertium/fsrs4anki/assets/83031600/73872c2c-6753-4c2d-915e-e8c06c352539)

If const display_memory_state = true; then you should be able to see something similar to this when reviewing a card:

![13](https://github.com/Expertium/fsrs4anki/assets/83031600/9a172be7-64c8-427d-b949-219752511607)

If you don’t see D, S and R, and only see “FSRS enabled”, that means the card is in the “learning” or “relearning” stage, not in the “review” stage.

You can check some interesting FSRS statistics after installing the add-on and then pressing Shift+Left Mouse Button on Stats.

![15](https://github.com/Expertium/fsrs4anki/assets/83031600/8930a547-b22f-42ab-929c-90602feeac82)

Additionally, if you are worried about privacy and don’t trust Google Collab, you can do the following before running the optimizer: 
Go to Browse > Notes > Find and Replace. Type (.|\n)* in the Find field and keep the Replace With field empty. Be sure to check (✓) the "Treat input as regular expression" option. Uncheck “Selected notes only” if you want to apply this to all notes.

![14](https://github.com/Expertium/fsrs4anki/assets/83031600/a8be6994-c8b0-4df3-868f-e566f04a5f12)

**This will make all fields blank. PLEASE MAKE A BACKUP BEFORE DOING THIS!**
Then export your collection with blanked out fields. Again, this is not necessary, just an extra measure for those few people who are worried about privacy.


# Advanced features

## Using different parameters for different decks

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

# FAQ

Here collect some questions from issues, forums, and others: [FAQ](https://github.com/open-spaced-repetition/fsrs4anki/wiki/FAQ)

# Compatibility

Some add-ons modify the scheduling of Anki, which would cause conflict with FSRS4Anki scheduler.

| Add-on                                                       | Compatible? | Comment |
| ------------------------------------------------------------ |-------------------| ------- |
|[Advanced Review Bottom Bar](https://ankiweb.net/shared/info/1136455830)|Yes✅|Please use the latest version.|
|[Incremental Reading v4.11.3 (unofficial clone)](https://ankiweb.net/shared/info/999215520)|No❌|It shows the interval given by Anki's built-in scheduler, not the custom scheduler.|
| [Auto Ease Factor](https://ankiweb.net/shared/info/1672712021)|No❌|`Ease Factor` doesn't affect the interval given by FSRS, so you won't benefit from using this add-on.|
| [Delay siblings](https://ankiweb.net/shared/info/1369579727) |No❌|Delay siblings will modify the intervals given by FSRS. However, FSRS already has similar functionality, so you don't need to use this add-on.|
| [autoLapseNewInterval](https://ankiweb.net/shared/info/372281481) |No❌|`New Interval` doesn't affect the interval given by FSRS, so you won't benefit from using this add-on.|
| [Straight Reward](https://ankiweb.net/shared/info/957961234) |No❌|`Ease Factor` doesn't affect the interval given by FSRS, so you won't benefit from using this add-on.|
| [Pass/Fail](https://ankiweb.net/shared/info/876946123) |Yes✅| `Pass` is the equivalent of `Good`.|

Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if I miss any add-ons.

# Contribute

You can contribute to FSRS4Anki by beta testing, submitting code, or sharing your data. If you want to share your data with me, please fill this form: https://forms.gle/KaojsBbhMCytaA7h8

## Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Expertium"><img src="https://avatars.githubusercontent.com/u/83031600?v=4?s=100" width="100px;" alt="Expertium"/><br /><sub><b>Expertium</b></sub></a><br /><a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=Expertium" title="Tests">⚠️</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=Expertium" title="Documentation">📖</a> <a href="#data-Expertium" title="Data">🔣</a> <a href="#ideas-Expertium" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/issues?q=author%3AExpertium" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/user1823"><img src="https://avatars.githubusercontent.com/u/92206575?v=4?s=100" width="100px;" alt="user1823"/><br /><sub><b>user1823</b></sub></a><br /><a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=user1823" title="Tests">⚠️</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=user1823" title="Documentation">📖</a> <a href="#data-user1823" title="Data">🔣</a> <a href="#ideas-user1823" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/issues?q=author%3Auser1823" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

# Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date)](https://star-history.com/#open-spaced-repetition/fsrs4anki&Date)
