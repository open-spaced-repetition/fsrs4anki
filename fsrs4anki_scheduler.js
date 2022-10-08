// FSRS4Anki v3.1.1 Scheduler
set_version();
// The latest version will be released on https://github.com/open-spaced-repetition/fsrs4anki

// Default parameters of FSRS4Anki for global
var w = [1, 1, 5, -0.5, -0.5, 0.2, 1.4, -0.12, 0.8, 2, -0.2, 0.2, 1];
// The above parameters can be optimized via FSRS4Anki optimizer.

// User's custom parameters for global
let requestRetention = 0.9; // recommended setting: 0.8 ~ 0.9
let maximumInterval = 36500;
let easyBonus = 1.3;
let hardInterval = 1.2;
const ratings = {
  "again": 1,
  "hard": 2,
  "good": 3,
  "easy": 4
};

debugger;

// get the name of the card's deck
// need add <div id=deck deck_name="{{Deck}}"></div> to your card's front template
if (document.getElementById("deck") !== null) {
    const deck_name = document.getElementById("deck").getAttribute("deck_name");
    // parameters for a specific deck
    if (deck_name == "ALL::Learning::English::Reading") {
        var w = [1.1475, 1.401, 5.1483, -1.4221, -1.2282, 0.035, 1.4668, -0.1286, 0.7539, 1.9671, -0.2307, 0.32, 0.9451];
        // User's custom parameters for the specific deck
        requestRetention = 0.9;
        maximumInterval = 36500;
        easyBonus = 1.3;
        hardInterval = 1.2;
    // parameters for a deck's all sub-decks
    } else if (deck_name.startsWith("ALL::Archive")) {
        var w = [1.2879, 0.5135, 4.9532, -1.502, -1.0922, 0.0081, 1.3771, -0.0294, 0.6718, 1.8335, -0.4066, 0.7291, 0.5517];
        ;
        // User's custom parameters for sub-decks
        requestRetention = 0.9;
        maximumInterval = 36500;
        easyBonus = 1.3;
        hardInterval = 1.2;
    }
}

// auto-calculate intervalModifier
const intervalModifier = Math.log(requestRetention) / Math.log(0.9);

// For new cards
if (is_new()) {
    init_states();
    states.easy.normal.review.scheduledDays = next_interval(customData.easy.s);
// For learning/relearning cards
} else if (is_learning()) {
    // Init states if the card didn't contain customData
    if (is_empty()) {
        init_states();
    }
    const good_interval = next_interval(customData.good.s);
    const easy_interval = Math.max(next_interval(customData.easy.s * easyBonus), good_interval + 1);
    if (states.good.normal?.review) {
        states.good.normal.review.scheduledDays = good_interval;
    }
    if (states.easy.normal?.review) {
        states.easy.normal.review.scheduledDays = easy_interval;
    }
// For review cards
} else if (is_review()) {
    // Convert the interval and factor to stability and difficulty if the card didn't contain customData
    if (is_empty()) {
        convert_states();
    }

    const interval = states.current.normal?.review.elapsedDays ? states.current.normal.review.elapsedDays : states.current.filtered.rescheduling.originalState.review.elapsedDays;
    const last_d = customData.again.d;
    const last_s = customData.again.s;
    const retrievability = Math.exp(Math.log(0.9) * interval / last_s);
    const lapses = states.again.normal?.relearning.review.lapses ? states.again.normal.relearning.review.lapses : states.again.filtered.rescheduling.originalState.relearning.review.lapses;

    customData.again.d = next_difficulty(last_d, "again");
    customData.again.s = next_forget_stability(customData.again.d, last_s, retrievability);

    customData.hard.d = next_difficulty(last_d, "hard");
    customData.hard.s = next_recall_stability(customData.hard.d, last_s, retrievability);

    customData.good.d = next_difficulty(last_d, "good");
    customData.good.s = next_recall_stability(customData.good.d, last_s, retrievability);

    customData.easy.d = next_difficulty(last_d, "easy");
    customData.easy.s = next_recall_stability(customData.easy.d, last_s, retrievability);

    let hard_interval = next_interval(last_s * hardInterval);
    let good_interval = next_interval(customData.good.s);
    let easy_interval = next_interval(customData.easy.s * easyBonus)
    hard_interval = Math.min(hard_interval, good_interval)
    good_interval = Math.max(good_interval, hard_interval + 1);
    easy_interval = Math.max(easy_interval, good_interval + 1);

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
    return Math.min(Math.max(difficulty.toFixed(2), 1), 10);
}

function next_interval(stability) {
    const new_interval = stability * intervalModifier * (0.2 * Math.random() + 0.9);
    return Math.min(Math.max(Math.round(new_interval), 1), maximumInterval);
}

function next_difficulty(d, rating) {
    let next_d = d + w[4] * (ratings[rating] - 3);
    return constrain_difficulty(mean_reversion(w[2], next_d));
}

function mean_reversion(init, current) {
    return w[5] * init + (1 - w[5]) * current;
}

function next_recall_stability(d, s, r) {
    return +(s * (1 + Math.exp(w[6]) *
    (11 - d) *
    Math.pow(s, w[7]) *
    (Math.exp((1 - r) * w[8]) - 1))).toFixed(2);
}

function next_forget_stability(d, s, r) {
    return +(w[9] * Math.pow(d, w[10]) * Math.pow(
        s, w[11]) * Math.exp((1 - r) * w[12])).toFixed(2);
}

function init_states() {
    customData.again.d = init_difficulty("again");
    customData.again.s = init_stability("again");
    customData.hard.d = init_difficulty("hard");
    customData.hard.s = init_stability("hard");
    customData.good.d = init_difficulty("good");
    customData.good.s = init_stability("good");
    customData.easy.d = init_difficulty("easy");
    customData.easy.s = init_stability("easy");
}

function init_difficulty(rating) {
    return +constrain_difficulty(w[2] + w[3] * (ratings[rating] - 3)).toFixed(2);
}

function init_stability(rating) {
    return +Math.max(w[0] + w[1] * (ratings[rating] - 1), 0.1).toFixed(2);
}

function convert_states() {
    const scheduledDays = states.current.normal ? states.current.normal.review.scheduledDays : states.current.filtered.rescheduling.originalState.review.scheduledDays;
    const easeFactor = states.current.normal ? states.current.normal.review.easeFactor : states.current.filtered.rescheduling.originalState.review.easeFactor;
    const old_s = +Math.max(scheduledDays, 0.1).toFixed(2);
    const old_d = constrain_difficulty(11 - (easeFactor - 1) / (Math.exp(w[6]) * Math.pow(old_s, w[7]) * (Math.exp(0.1 * w[8]) - 1)))
    customData.again.d = old_d;
    customData.again.s = old_s;
    customData.hard.d = old_d;
    customData.hard.s = old_s;
    customData.good.d = old_d;
    customData.good.s = old_s;
    customData.easy.d = old_d;
    customData.easy.s = old_s;
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

function is_empty() {
    return !customData.again.d | !customData.again.s | !customData.hard.d | !customData.hard.s | !customData.good.d | !customData.good.s | !customData.easy.d | !customData.easy.s;
}

function set_version() {
    const version = "3.1.1";
    customData.again.v = version;
    customData.hard.v = version;
    customData.good.v = version;
    customData.easy.v = version;
}