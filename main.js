// fsrs4anki: An Anki custom scheduling based on free spaced repetition scheduler algorithm
// The latest version will be released on https://github.com/open-spaced-repetition/fsrs4anki

// Default parameters of fsrs4anki, which can be optimized via
// https://github.com/open-spaced-repetition/fsrs4anki/blob/main/fsrs4anki_optimizer.ipynb
const defaultDifficulty = 4.6179;
const defaultStability = 2.5636;
const difficultyDecay = -0.5913;
const stabilityDecay = -0.1382;
const retrievabilityFactor = 1.1951;
const increaseFactor = 3.201;
const lapsesBase = -0.0562;

// Custom parameters for user
const requestRetention = 0.9; // recommended setting: 0.8 ~ 0.9
const maximumInterval = 36500;
const easyBonus = 1.3;
const hardInterval = 1.2;

debugger;

// auto-calculate 
const intervalModifier = Math.log(requestRetention) / Math.log(0.9);

// For new cards
if (states.current.normal.new) {
    customData.again.d = defaultDifficulty + 2;
    customData.again.s = defaultStability * 0.25;
    customData.hard.d = defaultDifficulty + 1;
    customData.hard.s = defaultStability * 0.5;
    customData.good.d = defaultDifficulty;
    customData.good.s = defaultStability;
    customData.easy.d = defaultDifficulty - 1;
    customData.easy.s = defaultStability * 2;
    states.easy.normal.review.scheduledDays = constrain_interval(Math.round(customData.easy.s));
} else if (states.current.normal.learning) {
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = constrain_interval(Math.round(customData.good.s));
    }
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = constrain_interval(Math.round(customData.easy.s * easyBonus));
    }
// For review cards
} else if (states.current.normal.review) {
    // Convert the interval and factor to stability and difficulty if the card didn't contain customData
    if (!customData.again.d) {
        const old_d = constrain_difficulty(10 / states.current.normal.review.easeFactor);
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

    customData.again.d = constrain_difficulty(last_d + retrievability - 0.25 + 0.1);
    customData.again.s = defaultStability * Math.exp(lapsesBase * (states.again.normal.relearning.review.lapses));

    customData.hard.d = constrain_difficulty(last_d + retrievability - 0.5 + 0.1);
    customData.hard.s = last_s * (1 + Math.exp(increaseFactor) * Math.pow(customData.hard.d, difficultyDecay) * Math.pow(last_s, stabilityDecay) * (Math.exp((1 - retrievability) * retrievabilityFactor) - 1));

    customData.good.d = constrain_difficulty(last_d + retrievability - 1 + 0.1);
    customData.good.s = last_s * (1 + Math.exp(increaseFactor) * Math.pow(customData.good.d, difficultyDecay) * Math.pow(last_s, stabilityDecay) * (Math.exp((1 - retrievability) * retrievabilityFactor) - 1));

    customData.easy.d = constrain_difficulty(last_d + retrievability - 2 + 0.1);
    customData.easy.s = last_s * (1 + Math.exp(increaseFactor) * Math.pow(customData.easy.d, difficultyDecay) * Math.pow(last_s, stabilityDecay) * (Math.exp((1 - retrievability) * retrievabilityFactor) - 1));

    if (states.hard.normal?.review) {
        states.hard.normal.review.scheduledDays = constrain_interval(Math.round(last_s * hardInterval));
    }
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = constrain_interval(Math.round(customData.good.s));
    }
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = constrain_interval(Math.round(customData.easy.s * easyBonus));
    }
}

function constrain_difficulty(difficulty) {
    return Math.min(Math.max(difficulty, 0.1), 10);
}

function constrain_interval(interval) {
    return Math.min(Math.max(interval * intervalModifier, 1), maximumInterval);
}