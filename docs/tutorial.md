# Table of contents
- [Step 1: Enable the built-in FSRS Scheduler](#step-1-enable-the-built-in-fsrs-scheduler)
- [Step 2: Configure FSRS settings](#step-2-configure-fsrs-settings)
- [Step 3: Find optimal weights](#step-3-find-optimal-weights)
- [Step 4: (optional) Evaluate the weights](#step-4-optional-evaluate-the-weights)
- [Step 5: (optional) Compute optimal retention](#step-5-optional-compute-optimal-retention)
- [Step 6: (optional) Custom Scheduling](#step-6-optional-custom-scheduling)
- [Add-on compatibility](#add-on-compatibility)

## Step 1: Enable the built-in FSRS Scheduler

To enable FSRS, go to Deck Options, scroll down to the "Advanced" section, and toggle FSRS. This setting is shared by all deck presets. Note that after enabling FSRS, several settings, such as "Graduating interval", "Easy bonus", etc. will disappear. This is because these settings are irrelevant when FSRS is enabled.

If you have previously used FSRS using the custom scheduling method, please delete the FSRS code in the custom scheduling field before enabling the native FSRS. Also, if you are using the [FSRS4Anki Helper add-on](https://ankiweb.net/shared/info/759844606), check for add-on updates to ensure that the add-on has been updated to the 23.10 version. 

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/2294ecbb-91bb-45bb-8634-de36da3372a2)

## Step 2: Configure FSRS settings

### Desired Retention

The most important setting to configure is the desired retention: the fraction of cards recalled successfully when they are due.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/b3881b70-5e0d-4824-a54e-08fe92605252)

The permissible range for desired retention is 0.70 to 0.97. Higher retention leads to more reviews per day.

Be conservative when adjusting this setting - higher values will greatly increase your workload, and lower values can be demoralizing when you forget a lot of material.

The chart bellow illustrates how the workload changes with retention. The exact shape of the curve depends on the user's parameters and learning habits.

![Workload and retention (matplotlib)](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/5218e456-5ea9-408f-8f00-a1b5cb930de4)

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

This option controls whether the due dates of cards will be changed when you enable FSRS or change the weights and/or desired retention. By default, the cards are not rescheduled. This means that future reviews will use the new scheduling, but there will be no immediate change to your workload. This allows a smooth and gradual transition from SM-2 to FSRS.

If rescheduling is enabled, the due dates of cards will be immediately changed. This often results in a large number of cards becoming due, so is not recommended when first switching from SM2.

![image](https://github.com/Expertium/fsrs4anki/assets/83031600/3d14f65e-365d-4bcb-92d6-cfdeb4703b34)

## Step 3: Find optimal weights

The FSRS optimizer uses machine learning to learn your memory patterns and find weights that best fit your review history. So, the optimizer requires several reviews to fine-tune the weights.

If you have less than 1,000 reviews, please use the default parameters that are already entered into the "Model weights" field. Even with the default parameters, FSRS is better than the default Anki algorithm (SM-2).

If you have at least 1000 reviews (across all cards that this preset applies to), you can generate the optimal parameters for your cards using the `Optimize` button under the "Optimize FSRS weights" section. The optimal parameters will replace the default parameters automatically.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/dad5aa7b-d506-4368-a840-ec30bdd3d6a2)

The weights are preset-specific. If you have decks that vary wildly in difficulty, it is recommended to use separate presets for them because the weights for easy decks and hard decks will be different.

By default, weights will be calculated from the review history of all decks using the current preset. If you want to alter which cards are used for optimizing the weights (such as excluding suspended cards), you can adjust the search before calculating the weights. The search works the same way as it does in the Browser. For details, see [Searching](https://docs.ankiweb.net/searching.html) in the Anki Manual.

## Step 4: (optional) Evaluate the weights

You can use the `Evaluate` button in the "Optimize FSRS weights" section to see metrics that tell how well the weights in the "Model weights" field fit your review history. Smaller numbers indicate a better fit to your review history.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/c6d383f8-6131-40e0-9728-4cc823483281)

Log-loss doesn't have an intuitive interpretation. RMSE (bins) can be interpreted as the average difference between the predicted probability of recalling a card (R) and the measured (from the review history) probability. For example, RMSE=0.05 means that, on average, FSRS is off by 5% when predicting R.

Note that log-loss and RMSE (bins) are not perfectly correlated, so two decks may have similar RMSE values but very different log-loss values, and vice-versa.

## Step 5: (optional) Compute optimal retention

It is an experimental tool that tries to calculate a value of desired retention that maximizes the total knowledge within given time constraints, "Minutes study/day". Simply put, it tries to find the value of the desired retention that gives you the most efficient study plan. It does so by analyzing how much time you spend on your cards, as well as your habits of pressing Hard/Good/Easy.

![image](https://github.com/Expertium/fsrs4anki/assets/83031600/64511506-d668-428c-bef3-be58bd4d6c5b)

You can adjust "Deck size" and "Days to simulate" to fit your needs. If you are preparing for an exam that is 12 months away, set "Days to simulate" to 365. If you are a language learner, 5 years (1825 days) is a reasonable timeframe.

The suggested retention will greatly depend on your inputs, and if it significantly differs from 0.9, it's a sign that the time allocated per day is either too low or too high for the number of cards you're trying to learn. 

Since the tool is experimental, it is better to use your intuition to come up with a value of desired retention. However, the suggested retention can be useful as a reference when you have no idea of what you want your retention rate to be.

## Step 6: (optional) Custom Scheduling

"Custom scheduling" allows you to introduce new scheduling rules on top of FSRS. This feature is only for advanced users and developers.

# Add-On Compatibility

Some add-ons can cause conflicts with FSRS. As a general rule of thumb, if an add-on affects a card's intervals, it shouldn't be used with FSRS.

| Add-on                                                       | Compatible? | Comment |
| ------------------------------------------------------------ |-------------------| ------- |
|[Advanced Review Bottom Bar](https://ankiweb.net/shared/info/1136455830)|Yes✅|Please use the latest version.|
|[The KING of Button Add-ons](https://ankiweb.net/shared/info/374005964)|Yes✅|Please use the latest version.|
| [Pass/Fail](https://ankiweb.net/shared/info/876946123) |Yes✅| `Pass` is the equivalent of `Good`.|
| [Auto Ease Factor](https://ankiweb.net/shared/info/1672712021)|No❌|The `Ease Factor` doesn't affect the interval given by FSRS. So, you won't benefit from using this add-on.|
| [Delay siblings](https://ankiweb.net/shared/info/1369579727) |No❌|Delay siblings will modify the intervals given by FSRS. However, the FSRS4Anki Helper add-on has a similar feature that works better with FSRS. So, use the FSRS4Anki Helper add-on instead.|
| [autoLapseNewInterval](https://ankiweb.net/shared/info/372281481) |No❌|The `New Interval` doesn't affect the interval given by FSRS. So, you won't benefit from using this add-on.|
| [Straight Reward](https://ankiweb.net/shared/info/957961234) |No❌|The `Ease Factor` doesn't affect the interval given by FSRS. So, you won't benefit from using this add-on.|

Let me know via [issues](https://github.com/open-spaced-repetition/fsrs4anki/issues) if I miss any add-ons.
