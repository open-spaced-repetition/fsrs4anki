# Table of contents
- [Step 1: Enabling the built-in FSRS Scheduler](#step-1-enabling-the-built-in-fsrs-scheduler)
- [Step 2: Configuring FSRS settings](#step-2-configuring-fsrs-settings)
- [Step 3: Finding optimal parameters](#step-3-finding-optimal-parameters)
- [Step 4: (optional) Finding optimal retention and custom scheduling](#step-4-optional-finding-optimal-retention-and-custom-scheduling)
- [Stats and browser](#stats-and-browser)


## Step 1: Enabling the built-in FSRS Scheduler

In order to enable FSRS, go to deck options, scroll down to "Advanced", and toggle FSRS. You will notice that a lot of settings, such as "Graduating interval" or "Easy bonus," have disappeared. Many old settings become irrelevant once FSRS is enabled.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/2294ecbb-91bb-45bb-8634-de36da3372a2)

## Step 2: Configuring FSRS settings

The most important setting is desired retention: the fraction of due cards recalled successfully.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/b3881b70-5e0d-4824-a54e-08fe92605252)

It can be set as high as 0.97 or as low as 0.7. Higher retention leads to more reviews per day. Users are not allowed to set desired retention to values outside of the 0.7-0.97 range, as this will lead to inefficient studying.

The maximum interval works the same way as when using the old algorithm; it is the maximum number of days that can pass until the card is shown again.

"SM-2 retention" refers to your retention before you started using FSRS. It is only necessary to more accurately schedule cards with missing or incomplete review logs. It does not affect new cards or cards with complete review logs.

Using learning and re-learning steps longer than 1 day is not recommended. It could lead to a situation when the "Hard" interval is greater than the "Good" interval. It is also advised against having too many short (re)learning steps because same-day reviews have a negligible impact on long-term memory.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/cba3ca1a-4b55-44ee-ac32-1ca3684b1692)

If you don't enable "Reschedule cards on change", only cards that you review will be rescheduled. All other cards will remain unchanged until reviewed. If you want a smooth and gradual transition from the old algorithm to FSRS, you should disable "Reschedule cards on change". Enabling it will instantly change the intervals of all cards that this preset applies to, which often results in a large backlog of due cards.

![image](https://github.com/Expertium/fsrs4anki/assets/83031600/3d14f65e-365d-4bcb-92d6-cfdeb4703b34)


## Step 3: Finding optimal parameters

FSRS is a machine-learning algorithm that requires a lot of data to fine-tune. If you have at least 1000 reviews (across all cards that this preset applies to), you can click "Optimize FSRS weights" and then click "Optimize".

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/dad5aa7b-d506-4368-a840-ec30bdd3d6a2)

The optimal parameters will replace the default parameters automatically. If you have less than 1000 reviews, please use the default parameters, it is better than using the old algorithm.

You can also click "Analyze" after the optimization is done to see metrics that tell you how well FSRS is able to adapt to your memory and your review history. Smaller numbers are better.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/c6d383f8-6131-40e0-9728-4cc823483281)

Note that log-loss and RMSE (bins) are not perfectly correlated, so it's possible that two decks will have similar values of RMSE but very different values of log-loss, and vice versa.

## Step 4: (optional) Finding optimal retention and custom scheduling

Click on "Compute optimal retention (experimental)" and then click "Compute".

![image](https://github.com/Expertium/fsrs4anki/assets/83031600/64511506-d668-428c-bef3-be58bd4d6c5b)

It will analyze how much time you spend on your cards, as well as your habits of pressing Hard/Good/Easy, and use that information to simulate different review histories to find a value of desired retention that allows you to remember the most information within given time constraints, "Minutes study/day". Simply put, it finds the value of desired retention that gives you the most efficient study plan.

You can adjust "Deck size" and "Days to simulate" to fit your needs. If you have an exam coming in 12 months, set "Days to simulate" to 365. If you are a language learner, 5 years (1825 days) is a reasonable timeframe.

"Custom scheduling" allows you to introduce new scheduling rules on top of FSRS. This feature is only for advanced users and developers. If you have previously used FSRS and have some code in this field, please delete it.
