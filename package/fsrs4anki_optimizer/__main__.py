import fsrs4anki_optimizer
import argparse
import json
import pytz
import os

def prompt(msg: str, fallback):
    default = ""
    if fallback:
        default = f"(default: {fallback})"

    response = input(f"{msg} {default}: ")
    if response == "":
        if fallback is not None:
            return fallback
        else: # If there is no fallback
            raise Exception("You failed to enter a required parameter")
    return response

def process(filename):
    try: # Try and remember the last values inputted.
        with open(config_save, "r") as f:
            remembered_fallbacks = json.load(f)
    except FileNotFoundError:
        remembered_fallbacks = { # Defaults to this if not there
            "timezone": None, # Timezone starts with no default
            "next_day": 4,
            "revlog_start_date": "2006-10-05",
            "preview": "y",
            "filter_out_suspended_cards": "n"
        }

    # Prompts the user with the key and then falls back on the last answer given.
    def remembered_fallback_prompt(key: str, pretty: str = None):
        if pretty is None:
            pretty = key
        remembered_fallbacks[key] = prompt(f"input {pretty}", remembered_fallbacks[key])

    print("The defaults will switch to whatever you entered last.\n")

    if not args.yes:
        print("Timezone list: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568")
        remembered_fallback_prompt("timezone", "used timezone")
        if remembered_fallbacks["timezone"] not in pytz.all_timezones:
            raise Exception("Not a valid timezone, Check the list for more information")

        remembered_fallback_prompt("next_day", "used next day start hour")
        remembered_fallback_prompt("revlog_start_date", "the date at which before reviews will be ignored")
        remembered_fallback_prompt("filter_out_suspended_cards", "filter out suspended cards? (y/n)")
        
        graphs_input = prompt("View graphs? (y/n)" , remembered_fallbacks["preview"])
    else:
        graphs_input = remembered_fallbacks["preview"]

    if graphs_input.lower() != 'y':
        remembered_fallbacks["preview"] = "n"

    with open(config_save, "w+") as f: # Save the settings to load next time the program is run
        json.dump(remembered_fallbacks, f)

    show_graphs = graphs_input != "n"

    optimizer = fsrs4anki_optimizer.Optimizer()
    optimizer.anki_extract(filename)
    analysis = optimizer.create_time_series(
        remembered_fallbacks["timezone"],
        remembered_fallbacks["revlog_start_date"],
        remembered_fallbacks["next_day"],
        remembered_fallbacks["filter_out_suspended_cards"] == "y"
    )
    print(analysis)

    optimizer.define_model()
    optimizer.pretrain(verbose=show_graphs)
    optimizer.train(verbose=show_graphs)

    optimizer.predict_memory_states()
    figures = optimizer.find_optimal_retention()
    if show_graphs:
        for f in figures:
            f.show()

    optimizer.preview(optimizer.optimal_retention)

    profile = \
    f"""{{
    // Generated, Optimized anki deck settings
    "deckName": "{filename}",// PLEASE CHANGE THIS TO THE DECKS PROPER NAME
    "w": {optimizer.w},
    "requestRetention": {optimizer.optimal_retention},
    "maximumInterval": 36500,
}},
"""

    print("Paste this into your scheduling code")
    print(profile)

    if args.out:
        with open(args.out, "a+") as f:
            f.write(profile)

    optimizer.evaluate()
    if show_graphs:
        for f in optimizer.calibration_graph():
            f.show()
        for f in optimizer.compare_with_sm2():
            f.show()

if __name__ == "__main__":

    config_save = os.path.expanduser(".fsrs4anki_optimizer")

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("-y","--yes",
                        action=argparse.BooleanOptionalAction,
                        help="If set automatically defaults on all stdin settings."
                        )
    parser.add_argument("-o","--out",
                        help="File to APPEND the automatically generated profile to."
                        )
    args = parser.parse_args()

    for filename in args.filenames:
        if os.path.isdir(filename):
            files = [f for f in os.listdir(filename) if f.lower().endswith('.apkg')]
            files = [os.path.join(filename, f) for f in files]
            for file_path in files:
                process(file_path)
        else:
            process(filename)

