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

# Table of contents

- [Introduction](#introduction)
- [How to Get Started?](#how-to-get-started)
- [Add-on Compatibility](#add-on-compatibility)
- [Contribute](#contribute)
  - [Contributors](#contributors)
- [Stargazers over time](#stargazers-over-time)

# Introduction

FSRS4Anki consists of two main parts: the scheduler and the optimizer.

- The scheduler replaces Anki's built-in scheduler and schedules the cards according to the FSRS algorithm.
- The optimizer uses machine learning to learn your memory patterns and finds parameters that best fit your review history. For details about the working of the optimizer, please read [the mechanism of optimization](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-mechanism-of-optimization).

For details about the FSRS algorithm, please read [the algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm). If you are interested, you can also read my papers:
- [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling](https://www.maimemo.com/paper/) (free access), and
- [Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory](https://www.researchgate.net/publication/369045947_Optimizing_Spaced_Repetition_Schedule_by_Capturing_the_Dynamics_of_Memory) (submit a request).

FSRS4Anki Helper is an Anki add-on that complements the FSRS4Anki Scheduler. You can read about it here: https://github.com/open-spaced-repetition/fsrs4anki-helper

# How to Get Started?

If you are using Anki v23.10 or newer, refer to [this tutorial](https://github.com/open-spaced-repetition/fsrs4anki/blob/main/docs/tutorial.md).

If you are using an older version of Anki, refer to [this tutorial](https://github.com/open-spaced-repetition/fsrs4anki/blob/main/docs/tutorial2.md).

Note that setting up FSRS is much easier in Anki v23.10 or newer.

# Add-on Compatibility

Some add-ons can cause conflicts with FSRS. As a general rule of thumb, if an add-on affects a card's intervals, it shouldn't be used with FSRS.

| Add-on                                                       | Compatible? | Comment |
| ------------------------------------------------------------ |-------------------| ------- |
| [Advanced Review Bottom Bar](https://ankiweb.net/shared/info/1136455830) | Yes :white_check_mark: | Please use the latest version. |
| [The KING of Button Add-ons](https://ankiweb.net/shared/info/374005964) | Yes :white_check_mark: | Please use the latest version. |
| [Pass/Fail](https://ankiweb.net/shared/info/876946123) | Yes :white_check_mark: | `Pass` is the equivalent of `Good`. |
| [Learn Now & Grade Now](https://ankiweb.net/shared/info/1021636467) | Yes :white_check_mark: | `Learn Now` is compatible for Anki version 23.12. `Grade Now` works fine with the scheduler, but it doesn't work with `Compute optimal retention (experimental)`. |
| [Incremental Reading v4.11.3 (unofficial clone)](https://ankiweb.net/shared/info/999215520) | :grey_question: | If you are using the standalone version of FSRS, it shows the interval given by Anki's built-in scheduler, not the custom scheduler. It is technically compatible with built-in FSRS, but FSRS was not designed for incremental reading, and FSRS settings do not apply to IR cards. |
| [Auto Ease Factor](https://ankiweb.net/shared/info/1672712021) | No :x: | The `Ease Factor` doesn't affect the interval given by FSRS. So, you won't benefit from using this add-on. |
| [Delay siblings](https://ankiweb.net/shared/info/1369579727) | No :x:| Delay siblings will modify the intervals given by FSRS. However, the FSRS4Anki Helper add-on has a similar feature that works better with FSRS. So, use the FSRS4Anki Helper add-on instead. |
| [autoLapseNewInterval](https://ankiweb.net/shared/info/372281481) |No :x:| The `New Interval` doesn't affect the interval given by FSRS. So, you won't benefit from using this add-on. |
| [Straight Reward](https://ankiweb.net/shared/info/957961234) | No :x: | The `Ease Factor` doesn't affect the interval given by FSRS. So, you won't benefit from using this add-on. |

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

# Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=open-spaced-repetition/fsrs4anki&type=Date)](https://star-history.com/#open-spaced-repetition/fsrs4anki&Date)
