// FSRS4Anki v2.1.0 Scheduler
// The latest version will be released on https://github.com/open-spaced-repetition/fsrs4anki

// Default parameters of FSRS4Anki for global
var f_s = [1,1];
var f_d = [1,-1,-1,0.2];
var s_w = [3,-0.8,-0.2,1.3,2.2,-0.3,0.3,1.2];
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
// need add <div id=deck style="color: rgba(0, 0, 0, 0);">{{Deck}}</div> to your card's front template
if (document.getElementById('deck') !== null) {
    const deck_name = document.getElementById('deck').innerHTML;
    // parameters for a specific deck
    if (deck_name == "ALL::Learning::English::Reading") {
        var f_s = [1.559,1.9103];
        var f_d = [1.0082,-0.9627,-1.0287,0.0316];
        var s_w = [3.1521,-0.8427,-0.1906,1.4371,2.9026,-0.0287,0.5584,1.6425];
        // User's custom parameters for the specific deck
        requestRetention = 0.85;
        maximumInterval = 36500;
        easyBonus = 1.3;
        hardInterval = 1.2;
    // parameters for a deck's all sub-decks
    } else if (deck_name.startsWith("ALL::Archive")) {
        var f_s = [1.3028,1.4602];
        var f_d = [1.011,-0.8495,-1.1868,0.0417];
        var s_w = [3.2415,-0.8428,-0.0158,1.5379,2.1647,-0.3524,0.4513,1.1748];
        // User's custom parameters for sub-decks
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
    init_states();
    states.easy.normal.review.scheduledDays = constrain_interval(customData.easy.s);
// For learning/relearning cards
} else if (is_learning()) {
    // Init states if the card didn't contain customData
    if (is_empty()) {
        init_states();
    }
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
    return Math.min(Math.max(difficulty.toFixed(2), 1), 10);
}

function constrain_interval(interval) {
    return Math.min(Math.max(Math.round(interval * intervalModifier), 1), maximumInterval);
}

function next_difficulty(d, rating) {
    let next_d = d + f_d[2] * (ratings[rating] - 3);
    return constrain_difficulty(mean_reversion(f_d[0] * (- f_d[1] + 1), next_d));
}

function mean_reversion(init, current) {
    return f_d[3] * init + (1 - f_d[3]) * current;
}

function next_recall_stability(d, s, r) {
    return +(s * (1 + Math.exp(s_w[0]) * Math.pow(d, s_w[1]) * Math.pow(s, s_w[2]) * (Math.exp((1 - r) * s_w[3]) - 1))).toFixed(2);
}

function next_forget_stability(d, s, r) {
    return +(s_w[4] * Math.pow(d, s_w[5]) * Math.pow(s, s_w[6]) * Math.exp((1 - r) * s_w[7])).toFixed(2);
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
    return +(f_d[0] * (f_d[1] * (ratings[rating] - 4) + 1)).toFixed(2);
}

function init_stability(rating) {
    return +(f_s[0] * (f_s[1] * (ratings[rating] - 1) + 1)).toFixed(2);
}

function convert_states() {
    const scheduledDays = states.current.normal ? states.current.normal.review.scheduledDays : states.current.filtered.rescheduling.originalState.review.scheduledDays;
    const easeFactor = states.current.normal ? states.current.normal.review.easeFactor : states.current.filtered.rescheduling.originalState.review.easeFactor;
    const old_s = +Math.max(scheduledDays / intervalModifier, 0.1).toFixed(2);
    const old_d = constrain_difficulty(Math.pow((easeFactor - 1) / (Math.exp(s_w[0]) * Math.pow(old_s, s_w[2]) * (Math.exp((1 - requestRetention) * s_w[3]) - 1)), 1 / s_w[1]));
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