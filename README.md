# fsrs4anki

Anki [custom scheduling](https://faqs.ankiweb.net/the-2021-scheduler.html#add-ons-and-custom-scheduling) implements a simplified [Free Spaced Repetition Scheduler algorithm](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler), which removes the adaptive module. The full function needs an add-on to support it.

Supported Anki version >= 2.1.55 (related discussion: [Some problems in implementing a state-of-the-art SRS scheduler on Anki - Scheduling - Anki Forums (ankiweb.net)](https://forums.ankiweb.net/t/some-problems-in-implementing-a-state-of-the-art-srs-scheduler-on-anki/22705))

## Usage

Please copy the code in `main.js` and paste it into the bottom of the deck options screen (need to enable the scheduler v3 in Preferences), then it works.

![deck options](images/deck_options.png)

## Future Work 

To implement the adaptive module, it needs to modify the global parameters after each feedback and store these values permanently, with an add-on or the support of Anki's infrastructure.

A possible alternative method is to develop an independent tool that optimizes the parameters based on the data in the `revlog`. But it is not friendly for average users.

I hope I will find out the best way.
