import pandas as pd


def main():
    """
    Main procedure for filtering tweets by year and exporting the results.
    The target year is read from a configuration file.
    """

    # ------------------------------------------------------------
    # Read the target year from the configuration file
    # The configuration file contains a line in the format: year:XXXX
    # ------------------------------------------------------------
    with open("../config/cfg.txt", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("year:"):
                year = line.strip().split(":")[1]


    # ------------------------------------------------------------
    # Load the complete tweet dataset
    # keep_default_na=False and na_values=[] ensure that empty fields
    # are preserved as empty strings rather than being converted to NaN
    # ------------------------------------------------------------
    df = pd.read_csv(
        "../../data/all_tweets.csv",
        keep_default_na=False,
        na_values=[],
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Convert the "Created At" column to datetime format to enable
    # temporal filtering and time-based operations
    # ------------------------------------------------------------
    df["Created At"] = pd.to_datetime(df["Created At"])

    # ------------------------------------------------------------
    # Filter tweets that were posted in the specified year
    # ------------------------------------------------------------
    filtered_df = df[df["Created At"].dt.year == year]

    # ------------------------------------------------------------
    # Create a sequential index for the filtered dataset
    # (starting from 1 for readability)
    # ------------------------------------------------------------
    filtered_df.loc[:, "Index"] = range(1, len(filtered_df) + 1)

    # Extract the tweet text column
    tw_text = filtered_df["Text"]

    # ------------------------------------------------------------
    # Export the tweet text of the selected year to a separate file
    # Each row corresponds to one tweet
    # ------------------------------------------------------------
    with open(f"../data/tw_Text_{year}.csv", "w", encoding="UTF-8") as outfile:
        tw_text.to_csv(outfile, index=0, lineterminator="\n")

    # ------------------------------------------------------------
    # Save the full filtered dataset (including metadata) to a CSV file
    # UTF-8-SIG encoding is used to ensure compatibility with Excel
    # ------------------------------------------------------------
    filtered_df.to_csv(
        f"../data/tweets_{year}.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ------------------------------------------------------------
# Script entry point
# Ensures that the main function runs only when the script is
# executed directly, not when imported as a module
# ------------------------------------------------------------
if __name__ == "__main__":
    main()