# Table of contents
- [Step 1: Enabling the built-in FSRS Scheduler](#step-1-enabling-the-built-in-fsrs-scheduler)
- [Step 2: Configuring FSRS settings](#step-2-configuring-fsrs-settings)
- [Step 3: Finding optimal parameters](#step-3-finding-optimal-parameters)
- [Step 4: (optional) Finding optimal retention and custom scheduling](#step-4-optional-finding-optimal-retention-and-custom-scheduling)
- [Stats and browser](#stats-and-browser)


## Step 1: Enabling the built-in FSRS Scheduler

To enable FSRS, go to deck options, scroll down to "Advanced", and toggle FSRS. After enabling FSRS, several settings, such as "Graduating interval", "Easy bonus", etc. will disappear. This is because these settings are irrelevant when FSRS is enabled.

If you have previously used FSRS using the custom scheduling method, please delete the FSRS code in the custom scheduling field.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/2294ecbb-91bb-45bb-8634-de36da3372a2)

## Step 2: Configuring FSRS settings

### Desired Retention

The most important setting to configure is the desired retention: the fraction of cards recalled successfully when they are due.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/b3881b70-5e0d-4824-a54e-08fe92605252)

The permissible range for desired retention is 0.70 to 0.97. Higher retention leads to more reviews per day.

Be conservative when adjusting this setting - higher values will greatly increase your workload, and lower values can be demoralizing when you forget a lot of material.

Users are not allowed to set the desired retention outside of the 0.70-0.97 range because it will make learning inefficient.

### Maximum interval

The  Maximum interval setting works the same way as when using the default algorithm. It is the maximum number of days that a card can wait until it is shown again. For more information, see [Maximum interval](https://docs.ankiweb.net/deck-options.html#maximum-interval) in the Anki manual.

### SM-2 retention

"SM-2 retention" refers to your average retention before you started using FSRS.

You need to configure this value only if you have cards with missing or incomplete review logs. Since review logs typically won't be missing unless you have explicitly deleted them to free up space or you have used some add-ons that modify the review history, **most users will not need to adjust this value**.

### Learning and re-learning steps

When FSRS is enabled, the learning and re-learning steps should be chosen in such a way that all the learning steps can be completed on the same day. For most users, this means that steps longer than or equal to 1 day should not be used.

The reason is that FSRS can determine more optimal intervals but the use of longer (re)learning steps doesn't allow FSRS to schedule the reviews, making the scheduling less optimal. In addition, if longer steps are used, there can be cases where the "Hard" interval exceeds the "Good" interval. 

Secondly, the use of multiple short (re)learning steps is also discouraged. This is because research shows that same-day reviews have a negligible impact on long-term memory.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/cba3ca1a-4b55-44ee-ac32-1ca3684b1692)

### Reschedule cards on change

If you don't enable "Reschedule cards on change", only cards that you review will be rescheduled. All other cards will remain unchanged until reviewed. If you want a smooth and gradual transition from the old algorithm to FSRS, you should disable "Reschedule cards on change". Enabling it will instantly change the intervals of all cards that this preset applies to, which often results in a large backlog of due cards.

![image](https://github.com/Expertium/fsrs4anki/assets/83031600/3d14f65e-365d-4bcb-92d6-cfdeb4703b34)


## Step 3: Finding optimal parameters

FSRS is a machine-learning algorithm that requires a lot of data to fine-tune. If you have at least 1000 reviews (across all cards that this preset applies to), you can click "Optimize FSRS weights" and then click "Optimize".

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/dad5aa7b-d506-4368-a840-ec30bdd3d6a2)

The optimal parameters will replace the default parameters automatically. If you have less than 1,000 reviews, please use the default parameters. Even with the default parameters, FSRS is better than the default Anki algorithm (SM-2).

After the optimization is complete, you can click "Evaluate" to see metrics that tell you how well FSRS is able to adapt to your memory and your review history. Smaller numbers are better.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/c6d383f8-6131-40e0-9728-4cc823483281)

Note that log-loss and RMSE (bins) are not perfectly correlated, so it's possible that two decks will have similar RMSE values but very different log-loss values, and vice versa.

## Step 4: (optional) Finding optimal retention and custom scheduling

Click on "Compute optimal retention (experimental)" and then click "Compute".

![image](https://github.com/Expertium/fsrs4anki/assets/83031600/64511506-d668-428c-bef3-be58bd4d6c5b)

It will analyze how much time you spend on your cards, as well as your habits of pressing Hard/Good/Easy, and use that information to simulate different review histories to find a value of desired retention that allows you to remember the most information within given time constraints, "Minutes study/day". Simply put, it finds the value of the desired retention that gives you the most efficient study plan.

You can adjust "Deck size" and "Days to simulate" to fit your needs. If you have an exam in 12 months, set "Days to simulate" to 365. If you are a language learner, 5 years (1825 days) is a reasonable timeframe.

"Custom scheduling" allows you to introduce new scheduling rules on top of FSRS. This feature is only for advanced users and developers. If you have previously used FSRS and have some code in this field, please delete it.
