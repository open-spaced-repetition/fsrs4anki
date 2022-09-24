# FSRS4Anki

FSRS4Anki is an Anki [custom scheduling](https://faqs.ankiweb.net/the-2021-scheduler.html#add-ons-and-custom-scheduling) implementing the [Free Spaced Repetition Scheduler algorithm](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler). FSRS4Anki consists of two parts: scheduler and optimizer.

The scheduler is based on a variant of the DSR (Difficulty, Stability, Retrievability) model, which is used to predict memory states. The scheduler aims to achieve the requested retention for each card and each review.

The optimizer applies *Maximum Likelihood Estimation* and *Backpropagation Through Time* to estimate the stability of memory and learn the laws of memory from time-series review logs.

For more detail on the mechanism of the FSRS algorithm, please see the paper: [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling | Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining](https://www.maimemo.com/paper/).

## Usage

For the tutorial on FSRS4Anki scheduler, optimizer, and simulator, please see: [Usage](https://github.com/open-spaced-repetition/fsrs4anki/wiki/Usage)

## FAQ

Here collect some questions from issues, forums, and others: [FAQ](https://github.com/open-spaced-repetition/fsrs4anki/wiki/FAQ)

## Compatibility

Some add-ons intervene in the scheduling of Anki, which would cause conflict with FSRS4Anki scheduler. Please see [Compatibility](https://github.com/open-spaced-repetition/fsrs4anki/wiki/Compatibility) for more details.

I will test these add-ons. Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if I miss any add-ons.
