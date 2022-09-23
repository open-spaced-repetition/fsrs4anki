// FSRS4Anki v1.4.1 Scheduler
// The latest version will be released on https://github.com/open-spaced-repetition/fsrs4anki

// Default parameters of FSRS4Anki for global
let defaultDifficulty = 4.6179;
let defaultStability = 2.5636;
let difficultyDecay = -0.5913;
let stabilityDecay = -0.1382;
let retrievabilityFactor = 1.1951;
let increaseFactor = 3.201;
let lapsesBase = -0.0562;
// The above parameters can be optimized via FSRS4Anki optimizer.

// Custom parameters for user
let requestRetention = 0.9; // recommended setting: 0.8 ~ 0.9
let maximumInterval = 36500;
let easyBonus = 1.3;
let hardInterval = 1.2;

debugger;

// get the name of the card's deck
// need add <div id=deck>{{Deck}}</div> to your card's front template
if (document.getElementById('deck') !== null) {
    const deck_name = document.getElementById('deck').innerHTML;
    // parameters for a specific deck
    if (deck_name == "Shape Up") {
        defaultDifficulty = 4;
        defaultStability = 3;
        difficultyDecay = -0.5;
        stabilityDecay = -0.1;
        retrievabilityFactor = 1.2;
        increaseFactor = 3;
        lapsesBase = -0.1;
        requestRetention = 0.85;
        maximumInterval = 36500;
        easyBonus = 1.3;
        hardInterval = 1.2;
    }
}

// auto-calculate intervalModifier
const intervalModifier = Math.log(requestRetention) / Math.log(0.9);

// For new cards
if (is_new()) {
    customData.again.d = defaultDifficulty + 2;
    customData.again.s = defaultStability * 0.25;
    customData.hard.d = defaultDifficulty + 1;
    customData.hard.s = defaultStability * 0.5;
    customData.good.d = defaultDifficulty;
    customData.good.s = defaultStability;
    customData.easy.d = defaultDifficulty - 1;
    customData.easy.s = defaultStability * 2;
    states.easy.normal.review.scheduledDays = constrain_interval(customData.easy.s);
// For learning/relearning cards
} else if (is_learning()) {
    const good_interval = constrain_interval(customData.good.s);
    const easy_interval = Math.max(constrain_interval(customData.easy.s * easyBonus), good_interval + 1);
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = good_interval;
    }
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = easy_interval;
    }
// For review cards
} else if (is_review()) {
    // Convert the interval and factor to stability and difficulty if the card didn't contain customData
    if (!customData.again.d) {
        const easeFactor = states.current.normal ? states.current.normal.review.easeFactor : states.current.filtered.rescheduling.originalState.review.easeFactor;
        const scheduledDays = states.current.normal ? states.current.normal.review.scheduledDays : states.current.filtered.rescheduling.originalState.review.scheduledDays;
        const old_d = constrain_difficulty(10 / easeFactor);
        const old_s = scheduledDays;
        customData.again.d = old_d;
        customData.again.s = old_s;
        customData.hard.d = old_d;
        customData.hard.s = old_s;
        customData.good.d = old_d;
        customData.good.s = old_s;
        customData.easy.d = old_d;
        customData.easy.s = old_s;
    }

    const interval = states.current.normal?.review.elapsedDays ? states.current.normal.review.elapsedDays : states.current.filtered.rescheduling.originalState.review.elapsedDays;
    const last_d = customData.again.d;
    const last_s = customData.again.s;
    const retrievability = Math.exp(Math.log(0.9) * interval / last_s);
    const lapses = states.again.normal?.relearning.review.lapses ? states.again.normal.relearning.review.lapses : states.again.filtered.rescheduling.originalState.relearning.review.lapses;

    customData.again.d = constrain_difficulty(last_d + retrievability - 0.25 + 0.1);
    customData.again.s = defaultStability * Math.exp(lapsesBase * lapses);

    customData.hard.d = constrain_difficulty(last_d + retrievability - 0.5 + 0.1);
    customData.hard.s = next_stability(customData.hard.d, last_s, retrievability);

    customData.good.d = constrain_difficulty(last_d + retrievability - 1 + 0.1);
    customData.good.s = next_stability(customData.good.d, last_s, retrievability);

    customData.easy.d = constrain_difficulty(last_d + retrievability - 2 + 0.1);
    customData.easy.s = next_stability(customData.easy.d, last_s, retrievability);

    const hard_interval = constrain_interval(last_s * hardInterval);
    const good_interval = Math.max(constrain_interval(customData.good.s), hard_interval + 1);
    const easy_interval = Math.max(constrain_interval(customData.easy.s * easyBonus), good_interval + 1);

    if (states.hard.normal?.review) {
        states.hard.normal.review.scheduledDays = hard_interval;
    }
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = good_interval;
    }
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = easy_interval;
    }
}

function constrain_difficulty(difficulty) {
    return Math.min(Math.max(difficulty, 0.1), 10);
}

function constrain_interval(interval) {
    return Math.min(Math.max(Math.round(interval * intervalModifier), 1), maximumInterval);
}

function next_stability(d, s, r) {
    return s * (1 + Math.exp(increaseFactor) * Math.pow(d, difficultyDecay) * Math.pow(s, stabilityDecay) * (Math.exp((1 - r) * retrievabilityFactor) - 1));
}

function is_new() {
    if (states.current.normal?.new !== undefined) {
        if (states.current.normal?.new !== null) {
            return true;
        }
    }
    if (states.current.filtered?.rescheduling?.originalState !== undefined) {
        if (Object.hasOwn(states.current.filtered?.rescheduling?.originalState, 'new')) {
            return true;
        }
    } 
    return false;
}

function is_learning() {
    if (states.current.normal?.learning !== undefined) {
        if (states.current.normal?.learning !== null) {
            return true;
        }
    }
    if (states.current.filtered?.rescheduling?.originalState !== undefined) {
        if (Object.hasOwn(states.current.filtered?.rescheduling?.originalState, 'learning')) {
            return true;
        }
    }
    if (states.current.normal?.relearning !== undefined) {
        if (states.current.normal?.relearning !== null) {
            return true;
        }
    }
    if (states.current.filtered?.rescheduling?.originalState !== undefined) {
        if (Object.hasOwn(states.current.filtered?.rescheduling?.originalState, 'relearning')) {
            return true;
        }
    }
    return false;
}

function is_review() {
    if (states.current.normal?.review !== undefined) {
        if (states.current.normal?.review !== null) {
            return true;
        }
    }
    if (states.current.filtered?.rescheduling?.originalState !== undefined) {
        if (Object.hasOwn(states.current.filtered?.rescheduling?.originalState, 'review')) {
            return true;
        }
    }
    return false;
}
