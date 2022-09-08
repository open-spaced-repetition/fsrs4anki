const difficultyDecay = -0.7;
const stabilityDecay = -0.2;
const increaseFactor = 60;
const defaultDifficulty = 5;
const defaultStability = 2;
const requestRetention = 0.9;
const lapsesBase = -0.3;

debugger;

if (states.current.normal.new) {
    customData.again.d = defaultDifficulty + 2;
    customData.again.s = defaultStability / 4;
    customData.hard.d = defaultDifficulty + 1;
    customData.hard.s = defaultStability / 2;
    customData.good.d = defaultDifficulty;
    customData.good.s = defaultStability;
    customData.easy.d = defaultDifficulty - 1;
    customData.easy.s = defaultStability * 2;
    states.easy.normal.review.scheduledDays = Math.round(customData.easy.s * Math.log(requestRetention) / Math.log(0.9));
} else if (states.current.normal.learning) {
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = Math.round(customData.easy.s * Math.log(requestRetention) / Math.log(0.9));
    }
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = Math.round(customData.good.s * Math.log(requestRetention) / Math.log(0.9));
    }
} else if (states.current.normal.review) {

    if (!customData.again.d) {
        const old_d = 10 / states.current.normal.review.easeFactor;
        const old_s = states.current.normal.review.scheduledDays;
        customData.again.d = old_d;
        customData.again.s = old_s;
        customData.hard.d = old_d;
        customData.hard.s = old_s;
        customData.good.d = old_d;
        customData.good.s = old_s;
        customData.easy.d = old_d;
        customData.easy.s = old_s;
    }

    const interval = states.current.normal.review.elapsedDays;
    const last_d = customData.again.d;
    const last_s = customData.again.s;
    const retrievability = Math.exp(Math.log(0.9) * interval / last_s);

    customData.again.d = Math.min(Math.max(last_d  + retrievability - 0 + 0.2, 1), 10);
    customData.again.s = defaultStability * Math.exp(lapsesBase * (states.again.normal.relearning.review.lapses + 1))
    customData.hard.d = Math.min(Math.max(last_d  + retrievability - 0.5 + 0.2, 1), 10);
    customData.hard.s = last_s * (1 + increaseFactor * Math.pow(customData.hard.d, difficultyDecay) * Math.pow(last_s, stabilityDecay) * (Math.exp(1 - retrievability) - 1));
    customData.good.d = Math.min(Math.max(last_d  + retrievability - 1 + 0.2, 1), 10);
    customData.good.s = last_s * (1 + increaseFactor * Math.pow(customData.good.d, difficultyDecay) * Math.pow(last_s, stabilityDecay) * (Math.exp(1 - retrievability) - 1));
    customData.easy.d = Math.min(Math.max(last_d  + retrievability - 2 + 0.2, 1), 10);
    customData.easy.s = last_s * (1 + increaseFactor * Math.pow(customData.easy.d, difficultyDecay) * Math.pow(last_s, stabilityDecay) * (Math.exp(1 - retrievability) - 1));

    if (states.hard.normal?.review) {
        states.hard.normal.review.scheduledDays = Math.round(customData.hard.s * Math.log(requestRetention) / Math.log(0.9));
    }
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = Math.round(customData.good.s * Math.log(requestRetention) / Math.log(0.9));
    }
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = Math.round(customData.easy.s * Math.log(requestRetention) / Math.log(0.9));
    }
}