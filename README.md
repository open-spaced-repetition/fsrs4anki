<p align="center">
  <a href="https://github.com/open-spaced-repetition/fsrs4anki/wiki">
    <img src="https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/9efb2ca5-51bd-411d-9694-a77b09f51fa7" width="150" height="150" alt="FSRS4Anki">
  </a>
</p>

<div align="center">

# FSRS4Anki

_✨ A modern spaced-repetition scheduler for Anki based on the [Free Spaced Repetition Scheduler algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm) ✨_  

</div>

<p align="center">
  <a href="https://raw.githubusercontent.com/open-spaced-repetition/fsrs4anki/main/LICENSE">
    <img src="https://img.shields.io/github/license/open-spaced-repetition/fsrs4anki" alt="license">
  </a>
  <a href="https://github.com/open-spaced-repetition/fsrs4anki/releases/latest">
    <img src="https://img.shields.io/github/v/release/open-spaced-repetition/fsrs4anki?color=blueviolet" alt="release">
  </a>
</p>

# [611Study.ICU](https://github.com/IQAQI233/611Study.ICU)

Hi, I'm Jarrett Ye, the creator of FSRS. FSRS is a highly efficient spaced repetition algorithm used by many people worldwide, saving considerable time. It applies cognitive science and educational technology to help students learn more effectively.

However, most Chinese high school students are still forced to study "611" - from 6 AM to 11 PM, six days a week. Tragically, many students have committed suicide or developed serious mental health issues.

The [611Study.ICU](https://github.com/IQAQI233/611Study.ICU) project aims to stop this abusive study model and protect students' lives and health.

As a former Chinese student, I experienced overtime study and sleep deprivation during high school. Anki saved my life. I want to use this algorithm to save more lives, but most Chinese high schools still prevent students from using it.

I hope this project can raise awareness and help more students.

# Table of contents

- [Introduction](#introduction)
- [How to Get Started?](#how-to-get-started)
- [Add-on Compatibility](#add-on-compatibility)
- [Contribute](#contribute)
  - [Contributors](#contributors)
- [Developer Resources](#developer-resources)
- [Stargazers Over Time](#stargazers-over-time)

# Introduction

FSRS4Anki consists of two main parts: the scheduler and the optimizer.

- The scheduler replaces Anki's built-in scheduler and schedules the cards according to the FSRS algorithm.
- The optimizer uses machine learning to learn your memory patterns and finds parameters that best fit your review history. For details about the working of the optimizer, please read [the mechanism of optimization](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-mechanism-of-optimization).

For details about the FSRS algorithm, please read [the algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm). If you are interested, you can also read my papers:
- [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling](https://dl.acm.org/doi/10.1145/3534678.3539081?cid=99660547150) (free access), and
- [Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory](https://drive.google.com/file/u/0/d/1riJbkH39JB71Wj0AzESTngUM0LaeoD2l/view) (Google Scholar).

FSRS4Anki Helper is an Anki add-on that complements the FSRS4Anki Scheduler. You can read about it here: https://github.com/open-spaced-repetition/fsrs4anki-helper

# How to Get Started?

If you are using Anki v23.10 or newer, refer to [this tutorial](https://github.com/open-spaced-repetition/fsrs4anki/blob/main/docs/tutorial.md).

If you are using an older version of Anki, refer to [this tutorial](https://github.com/open-spaced-repetition/fsrs4anki/blob/main/docs/tutorial2.md).

Note that setting up FSRS is much easier in Anki v23.10 or newer.

# Add-on Compatibility

Some add-ons can cause conflicts with FSRS. As a general rule of thumb, if an add-on affects a card's intervals, it shouldn't be used with FSRS.

| Add-on                                                       | Compatible? | Comment |
| ------------------------------------------------------------ |-------------------| ------- |
| [Review Heatmap](https://ankiweb.net/shared/info/1771074083) | Yes :white_check_mark: | Doesn't affect anything FSRS-related. |
| [Advanced Browser](https://ankiweb.net/shared/info/874215009) | Yes :white_check_mark: | Please use the latest version. |
| [Advanced Review Bottom Bar](https://ankiweb.net/shared/info/1136455830) | Yes :white_check_mark: | Please use the latest version. |
| [The KING of Button Add-ons](https://ankiweb.net/shared/info/374005964) | Yes :white_check_mark: | Please use the latest version. |
| [Pass/Fail](https://ankiweb.net/shared/info/876946123) | Yes :white_check_mark: | `Pass` is the equivalent of `Good`, `Fail` is the equivalent of `Again.` |
| [AJT Card Management](https://ankiweb.net/shared/info/1021636467) | Yes :white_check_mark: | Compatible with Anki 23.12 and newer. |
| [Incremental Reading v4.11.3 (unofficial clone)](https://ankiweb.net/shared/info/999215520) | Unsure :question: | If you are using the standalone version of FSRS, it shows the interval given by Anki's built-in scheduler, not the custom scheduler. This add-on is technically compatible with built-in FSRS, but FSRS was not designed for incremental reading, and FSRS settings do not apply to IR cards because they work in a different way compared to other card types. |
| [Delay siblings](https://ankiweb.net/shared/info/1369579727) | No :x:| Delay siblings will modify the intervals given by FSRS. However, the FSRS4Anki Helper add-on has a similar feature that works better with FSRS. Please use the FSRS4Anki Helper add-on instead. |
| [Auto Ease Factor](https://ankiweb.net/shared/info/1672712021) | No :x: | The Ease Factor is no longer relevant when FSRS is enabled, therefore you won't benefit from using this add-on. |
| [autoLapseNewInterval](https://ankiweb.net/shared/info/372281481) |No :x:| The `New Interval` setting is no longer relevant when FSRS is enabled, therefore you won't benefit from using this add-on. |
| [Straight Reward](https://ankiweb.net/shared/info/957961234) | No :x: | The Ease Factor is no longer relevant when FSRS is enabled, therefore you won't benefit from using this add-on. |

Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if you want me to check compatibility between FSRS and some add-on.

# Contribute

You can contribute to FSRS4Anki by beta testing, submitting code, or sharing your data. If you want to share your data with me, please fill out this form: https://forms.gle/KaojsBbhMCytaA7h8

## Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Expertium"><img src="https://avatars.githubusercontent.com/u/83031600?v=4?s=100" width="100px;" alt="Expertium"/><br /><sub><b>Expertium</b></sub></a><br /><a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=Expertium" title="Tests">⚠️</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=Expertium" title="Documentation">📖</a> <a href="#data-Expertium" title="Data">🔣</a> <a href="#ideas-Expertium" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/issues?q=author%3AExpertium" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/user1823"><img src="https://avatars.githubusercontent.com/u/92206575?v=4?s=100" width="100px;" alt="user1823"/><br /><sub><b>user1823</b></sub></a><br /><a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=user1823" title="Tests">⚠️</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/commits?author=user1823" title="Documentation">📖</a> <a href="#data-user1823" title="Data">🔣</a> <a href="#ideas-user1823" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/open-spaced-repetition/fsrs4anki/issues?q=author%3Auser1823" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://chrislongros.com"><img src="https://avatars.githubusercontent.com/u/98426896?v=4?s=100" width="100px;" alt="Christos Longros"/><br /><sub><b>Christos Longros</b></sub></a><br /><a href="#data-chrislongros" title="Data">🔣</a> <a href="#content-chrislongros" title="Content">🖋</a></td>
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

# Developer Resources

If you're a developer considering using the FSRS algorithm in your own projects, we've curated some valuable resources for you. Check out the [Awesome FSRS](https://github.com/open-spaced-repetition/awesome-fsrs) repository, where you'll find:

- FSRS implementations in various programming languages
- Related papers and research
- Example applications using FSRS
- Other algorithms and resources related to spaced repetition systems

This carefully curated list will help you better understand FSRS and choose the right implementation for your project. We encourage you to explore these resources and consider contributing to the FSRS ecosystem.

# Stargazers Over Time

<a href="https://star-history.com/#open-spaced-repetition/fsrs4anki&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date" />
 </picture>
</a>
