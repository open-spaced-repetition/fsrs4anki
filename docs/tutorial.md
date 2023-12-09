中文版请见：[Anki 新算法 FSRS 配置指南](https://zhuanlan.zhihu.com/p/664758200)

# Table of contents
- [The Ultra Short Version](the-ultra-short-version)
- [Step 1: Enable the built-in FSRS Scheduler](#step-1-enable-the-built-in-fsrs-scheduler)
- [Step 2: Configure FSRS settings](#step-2-configure-fsrs-settings)
- [Step 3: Find optimal parameters](#step-3-find-optimal-parameters)
- [Step 4: (optional) Evaluate the parameters](#step-4-optional-evaluate-the-parameters)
- [Step 5: (optional) Compute optimal retention](#step-5-optional-compute-optimal-retention)
- [Step 6: (optional) Custom Scheduling](#step-6-optional-custom-scheduling)
- [FAQ](#faq)

## The Ultra Short Version

Are you busy and have no time to waste? Here's a summary of the guide.

1) Go to deck options and enable FSRS under "Advanced", at the bottom of the deck options window.
2) Ensure that all your learning and re-learning steps are shorter than `1d` and that all steps can be completed on the same day.
3) Click the "Optimize" button under the "Optimize FSRS parameters" section. The optimal parameters will replace the default parameters automatically. Parameters are preset-specific. If an error message pops up, it means you have less than 1000 reviews across all cards that this preset is applied to. In that case, just use the default parameters; it's still better than using the legacy SM-2 algorithm.
4) Choose a value of desired retention: the proportion of cards recalled successfully when they are due. **This is the most important setting in FSRS. Higher retention leads to shorter intervals and more reviews per day.** 80-95% is reasonable, 90% should work fine for most people.

You are now ready to use FSRS!

## Step 1: Enable the built-in FSRS Scheduler

To enable FSRS, go to Deck Options, scroll down to the "Advanced" section, and toggle FSRS. This setting is shared by all deck presets. Note that after enabling FSRS, several settings, such as "Graduating interval", "Easy bonus", etc. will disappear. This is because these settings are irrelevant when FSRS is enabled.

If you have previously used FSRS using the custom scheduling method, please delete the FSRS code in the custom scheduling field before enabling the native FSRS. Also, if you are using the [FSRS4Anki Helper add-on](https://ankiweb.net/shared/info/759844606), check for add-on updates to ensure that the add-on has been updated to the latest version. 

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/27c9592c-d383-45f7-bdbd-223019e9fb3e)

## Step 2: Configure FSRS settings

### Desired Retention

The most important setting to configure is the desired retention: the fraction of cards recalled successfully when they are due.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/3905329f-6941-452c-97f4-074558e6f5fd)

The permissible range for desired retention is 0.70 to 0.97 (0.7 to 0.99 in Anki 23.10.1 or newer). Higher retention leads to more reviews per day.

Be conservative when adjusting this setting - higher values will greatly increase your workload, and lower values can be demoralizing when you forget a lot of material.

The following chart illustrates how the workload changes with retention. The exact shape of the curve depends on the user's parameters and learning habits.

![Workload and retention, again (small)](https://github.com/open-spaced-repetition/fsrs4anki/assets/83031600/e2b95037-593a-4633-8774-dd16cba5f48e)

Initially, users were not allowed to set the desired retention outside of the 0.70-0.97 range because it would make learning inefficient. In Anki v23.10.1, the range has been extended to 0.70-0.99 at the request of some users. However, setting the desired retention above 0.97 is still not advised for two main reasons:

- Such a high desired retention will significantly increase your workload (cards per day). The repetitions will be so frequent that you will dread doing your reviews before you even discover the power of spaced repetition.
- With such high retention, each review will contribute minimally to your overall learning. This essentially transforms the spaced repetition system into a massed repetition system, thereby undermining the advantages of the spacing effect.

### Maximum interval

The  Maximum interval setting works the same way as when using the default algorithm. It is the maximum number of days that a card can wait until it is shown again. For more information, see [Maximum interval](https://docs.ankiweb.net/deck-options.html#maximum-interval) in the Anki manual.

### SM-2 retention

"SM-2 retention" refers to your average retention before you started using FSRS.

You need to configure this value only if you have cards with missing or incomplete review logs. Since review logs typically won't be missing unless you have explicitly deleted them to free up space or you have used some add-ons that modify the review history, **most users will not need to adjust this value**.

### Learning and re-learning steps

When FSRS is enabled, the learning and re-learning steps should be chosen in such a way that all the learning steps can be completed on the same day. This means that steps longer than or **equal** to 1 day should not be used.

The reason is that FSRS can determine more optimal intervals but the use of longer (re)learning steps doesn't allow FSRS to schedule the reviews, making the scheduling less optimal. In addition, if longer steps are used, there can be cases where the "Hard" interval exceeds the "Good" interval. 

Secondly, the use of multiple short (re)learning steps is also discouraged. This is because research shows that same-day reviews have a negligible impact on long-term memory.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/a5780dca-4d0d-4382-9323-26e45cb6f002)

### Reschedule cards on change

This option controls whether the due dates of cards will be changed when you enable FSRS or change the parameters and/or desired retention. By default, the cards are not rescheduled. This means that future reviews will use the new scheduling, but there will be no immediate change to your workload. This allows a smooth and gradual transition from SM-2 to FSRS.

If rescheduling is enabled, the due dates of cards will be immediately changed. This often results in a large number of cards becoming due, so is not recommended when first switching from SM2.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/fe61aaa6-cecb-4476-9ed2-9db05b63c7de)

## Step 3: Find optimal parameters

The FSRS optimizer uses machine learning to learn your memory patterns and find parameters that best fit your review history. So, the optimizer requires several reviews to fine-tune the parameters.

If you have less than 1,000 reviews, please use the default parameters that are already entered into the "FSRS parameters" field. Even with the default parameters, FSRS is better than the default Anki algorithm (SM-2).

If you have at least 1000 reviews (across all cards that this preset applies to), you can generate the optimal parameters for your cards using the `Optimize` button under the "Optimize FSRS parameters" section. The optimal parameters will replace the default parameters automatically.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/072c42fc-41fa-4ff0-841b-3a55dd23c8a1)

The parameters are preset-specific. If you have decks that vary wildly in difficulty, it is recommended to use separate presets for them because the parameters for easy decks and hard decks will be different.

By default, parameters will be calculated from the review history of all decks using the current preset. If you want to alter which cards are used for optimizing the parameters (such as excluding suspended cards), you can adjust the search before calculating the parameters. The search works the same way as it does in the Browser. For details, see [Searching](https://docs.ankiweb.net/searching.html) in the Anki Manual.

## Step 4: (optional) Evaluate the parameters

You can use the `Evaluate` button in the "Optimize FSRS parameters" section to see metrics that tell how well the parameters in the "FSRS parameters" field fit your review history. Smaller numbers indicate a better fit to your review history.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/871bbe4d-8b05-4439-ab38-cf5c4e9f6fdf)

Log-loss doesn't have an intuitive interpretation. RMSE (bins) can be interpreted as the average difference between the predicted probability of recalling a card (R) and the measured (from the review history) probability. For example, RMSE=0.05 means that, on average, FSRS is off by 5% when predicting R.

Note that log-loss and RMSE (bins) are not perfectly correlated, so two decks may have similar RMSE values but very different log-loss values, and vice-versa.

## Step 5: (optional) Compute optimal retention

It is an experimental tool that tries to calculate a value of desired retention that maximizes the total knowledge within given time constraints, "Minutes study/day". Simply put, it tries to find the value of the desired retention that gives you the most efficient study plan. It does so by analyzing how much time you spend on your cards, as well as your habits of pressing Hard/Good/Easy.

![image](https://github.com/open-spaced-repetition/fsrs4anki/assets/32575846/da46d838-86b1-47b5-9186-d664aacb2e44)

You can adjust "Deck size" and "Days to simulate" to fit your needs. If you are preparing for an exam that is 12 months away, set "Days to simulate" to 365. If you are a language learner, 5 years (1825 days) is a reasonable timeframe.

The suggested retention will greatly depend on your inputs, and if it significantly differs from 0.9, it's a sign that the time allocated per day is either too low or too high for the number of cards you're trying to learn. 

Since the tool is experimental, it is better to use your intuition to come up with a value of desired retention. However, the suggested retention can be useful as a reference when you have no idea of what you want your retention rate to be.

## Step 6: (optional) Custom Scheduling

"Custom scheduling" allows you to introduce new scheduling rules on top of FSRS. This feature is only for advanced users and developers.

# FAQ

Q1: Does FSRS change the way the card's ease changes?

A1: Anki's built-in ease factor doesn't affect anything once FSRS is enabled. This is also why a lot of settings, such as Starting Ease, are hidden once FSRS is enabled.

***

Q2: Once I started using FSRS on my existing deck, if I ever wanted to go back to using Anki's built-in algorithm for the same deck, would that still be possible?

A2: Yes, just turn FSRS off.

***

Q3: I'm sure I have >1000 reviews, yet when I try to optimize parameters for my preset, I get an error telling me that I don't have enough reviews. Is that a bug?

A3: FSRS only takes into account one review per day. If you review a card multiple times per day, only the chronologically first review will be used by the optimizer.

***

Q4: My first interval for "Easy" is too long! Is this normal?

A4: Yes. Anki tends to give very short first intervals. Don't be surprised if your first interval for "Good" is 5-7 days and your first interval for "Easy" is several weeks long.

***

Q5: Suppose I have a parent deck with its own preset, and each subdeck has a different preset. When I click on the parent deck to review a card that came from a subdeck, will the parameters of the preset of the parent deck be applied to the card, or the parameters of the preset of the subdeck that this card came from?

A5: The latter. Simply put, if you have something like ParentDeck::SubDeck, and the card came from the subdeck, the parameters of the preset corresponding to the subdeck will be applied.

***

Q6: I only use "Again" and "Good", will FSRS work fine?

A6: Yes. In fact, FSRS is actually more accurate for people who rarely use "Hard" and "Easy" than for people who use all 4 buttons a lot. However, this doesn't mean that you should change your habits. Keep using the buttons the same way as before.

***

Q7: How can I confirm that FSRS is working?

A7: Review a new card, remember what intervals you saw above the answer buttons. Undo review. Now set the desired retention either to 0.97 (maximum) or to 0.7 (minimum), and review the card again. You should see different intervals.

***

Q8: Is it better to use the same parameters for all my cards or use different presets with different parameters?

A8: The answer to this question depends entirely on how similar your material is. For example, if you are learning Japanese and geography, it is recommended to use two different presets with different parameters. If you have two decks with Japanese vocabulary, you should use the same preset for both of them.

***

Q9: How often should I re-optimize parameters?

A9: Once per month should be more than enough.

***

Q10: What will happen if I review my cards on a device where FSRS is not supported (or disabled) and then on another device where FSRS is enabled?

A10: Your intervals will become inaccurate, but it won't corrupt your cards and make them unusable. It will just make FSRS bad at what it's supposed to do: maintain your retention at a specified level.

***

You haven't seen the answer for your question? Here are many questions asked by others: https://github.com/open-spaced-repetition/fsrs4anki/issues?q=is%3Aissue+label%3Aquestion+

If you still have trouble, please open a new issue to provide the details about that.

https://github.com/open-spaced-repetition/fsrs4anki/issues/new/choose