# %%
import numpy as np
import pandas as pd


# %%
def generate_time_ranges(row, freq):
    """
    Generates time intervals for each row of the dataframe based on the specified frequency.

    Parameters:
    - row: The row of the dataframe.
    - freq: The desired frequency for the intervals (e.g., '10T', 'H', '3H', etc.).

    Returns:
    - A pd.DatetimeIndex object representing the time interval, or NaN if conditions are not met.
    """

    start = row["datetime"].floor(
        freq
    )  
    end = row["next_datetime"].floor(
        freq
    )  

    if (start == end) and (start < row["datetime"]):
        return np.nan
    elif (start == end) and (start == row["datetime"]):
        return pd.date_range(start=start, end=end, freq=freq)
    elif (start < end) and (start == row["datetime"]):
        return pd.date_range(start=start, end=end, freq=freq)
    elif (start < end) and (end == row["next_datetime"]):
        start = start + pd.Timedelta(freq) 
        return pd.date_range(start=start, end=end, freq=freq)
    elif (start < end) and (start != row["datetime"]) and (end != row["next_datetime"]):
        start = start + pd.Timedelta(freq)  
        return pd.date_range(start=start, end=end, freq=freq)


def remove_consecutive_duplicates(df):
    """
    Removes consecutive duplicate rows based on specific conditions in the dataframe.

    Parameters:
    - df: A pandas DataFrame containing the data to process.

    Returns:
    - A new DataFrame with consecutive duplicate rows (matching the conditions) removed.
    """

    mask = (
        (df["status_change"] == 1)  
        & (df["occupied"] == 1) 
        & (
            df["numeroStallo"] == df["numeroStallo"].shift()
        ) 
        & (
            df["status_change"] == df["status_change"].shift()
        )  
        & (df["occupied"] == df["occupied"].shift())  
    )

    return df[~mask].reset_index(drop=True)


def preprocess_sensor_data(KPlace_signals, slots, storico_stallo):
    """
    Preprocesses sensor data to clean and organize it.

    Parameters:
    - KPlace_signals: Dictionary containing sensor data.
    - slots: DataFrame containing slot data.
    - storico_stallo: DataFrame with historical information on slots.

    Returns:
    - df_final: A cleaned and processed DataFrame.
    """

    # Convert KPlace_signals to a DataFrame and clean data
    KPlace_signals = pd.DataFrame(KPlace_signals)
    KPlace_signals["datetime"] = pd.to_datetime(
        KPlace_signals["datetime"]
    )  
    KPlace_signals = KPlace_signals.drop_duplicates()  # Remove duplicate rows

    # Ensure columns have the correct data types
    KPlace_signals["status_change"] = KPlace_signals["status_change"].astype(int)
    KPlace_signals["occupied"] = KPlace_signals["occupied"].astype(int)
    KPlace_signals["dev_id"] = KPlace_signals["dev_id"].astype(str)

    # Sort and reset index
    KPlace_signals.sort_values(by=["dev_id", "datetime"], inplace=True)
    KPlace_signals.reset_index(drop=True, inplace=True)

    # Rename 'dev_id' to 'devID' for consistency
    KPlace_signals.rename(columns={"dev_id": "devID"}, inplace=True)

    # Split storico_stallo into rows with single and multiple entries for 'devID'
    one_count = storico_stallo["devID"].value_counts() == 1
    more_count = storico_stallo["devID"].value_counts() > 1
    storico_stallo1 = storico_stallo[
        storico_stallo["devID"].isin(one_count[one_count].index)
    ]
    storico_stallo2 = storico_stallo[
        storico_stallo["devID"].isin(more_count[more_count].index)
    ]

    # Merge KPlace_signals with storico_stallo1 and filter by datetime range
    merged_df = pd.merge(KPlace_signals, storico_stallo1, on="devID")
    filtered_df = merged_df[
        (merged_df["start"] <= merged_df["datetime"])
        & ((merged_df["end"].isna()) | (merged_df["end"] >= merged_df["datetime"]))
    ]

    # Handle rows with multiple 'devID' entries
    diff_df = KPlace_signals[KPlace_signals["devID"].isin(more_count[more_count].index)]
    diff_df.insert(1, "idStallo", np.nan)
    diff_df.insert(1, "start", pd.NaT)
    diff_df.insert(2, "end", pd.NaT)

    for index, row in diff_df.iterrows():
        stalli_ = storico_stallo2[storico_stallo2["devID"] == row["devID"]]
        for _, stallo in stalli_.iterrows():
            if (stallo["start"] <= row["datetime"]) and (
                pd.isna(stallo["end"]) or stallo["end"] >= row["datetime"]
            ):
                diff_df.loc[index, "idStallo"] = stallo["idStallo"]
                diff_df.loc[index, "start"] = stallo["start"]
                diff_df.loc[index, "end"] = stallo["end"]

    # Combine filtered data
    KPlace_signals1 = pd.concat([filtered_df, diff_df], ignore_index=True)
    KPlace_signals1["idStallo"] = KPlace_signals1["idStallo"].astype(int)

    # Merge with slot data
    KPlace_signals1 = pd.merge(
        KPlace_signals1,
        slots[["numeroStallo", "id_strada"]],
        left_on="idStallo",
        right_on="numeroStallo",
        how="left",
    )

    # Filter and sort by slot number and datetime
    df = KPlace_signals1.copy()
    df.sort_values(by=["numeroStallo", "datetime"], inplace=True)
    df.dropna(subset=["numeroStallo"], inplace=True)

    # Filter rows of type 'info_evt'
    df = df[df["type"] == "info_evt"]

    # Remove consecutive duplicates
    df_cleaned = remove_consecutive_duplicates(df)
    df_cleaned.sort_values(by=["numeroStallo", "datetime"], inplace=True)

    # Add hour column (floor to the nearest hour)
    df_cleaned["hour"] = df_cleaned["datetime"].dt.floor("H")

    # Filter rows with status_change = 1
    df_signals = df_cleaned[df_cleaned["status_change"] == 1]
    df_signals.drop(columns=["status_change"], inplace=True)

    # Identify and filter rows with meaningful changes
    df_signals["check"] = df_signals["occupied"].diff()
    df_signals = df_signals[df_signals["check"] != 0]
    df_signals.drop(columns=["check"], inplace=True)

    # Calculate time differences and filter by conditions
    df_signals["diff"] = df_signals.groupby("numeroStallo")["datetime"].diff()
    df_signals["diff_shift"] = df_signals["diff"].shift(-1)
    df_signals = df_signals[
        (df_signals["diff"] > pd.Timedelta(seconds=60))
        & (df_signals["diff_shift"] > pd.Timedelta(seconds=60))
    ]

    # Final processing
    df_signals["check"] = df_signals["occupied"].diff()
    df_signals = df_signals[df_signals["check"] != 0]

    # Prepare columns for output
    df_signals.drop(
        columns=["diff", "diff_shift", "check", "start", "end"], inplace=True
    )
    df_signals["diff"] = df_signals.groupby("numeroStallo")["datetime"].diff().shift(-1)
    df_signals["next_datetime"] = df_signals["datetime"].shift(-1)
    df_signals.dropna(subset=["next_datetime"], inplace=True)

    # Final DataFrame with relevant columns
    df_final = df_signals.copy()
    df_final = df_final[
        ["numeroStallo", "id_strada", "datetime", "next_datetime", "occupied"]
    ]

    # Floor datetime and next_datetime to the nearest hour
    df_final["datetime"] = df_final["datetime"].dt.floor("h")
    df_final["next_datetime"] = df_final["next_datetime"].dt.floor("h")

    return df_final

    
def generate_slot_data(df_final, freq="4h"):
    """
    Generates aggregated occupancy data for slots based on frequence time intervals.

    Parameters:
    - df_final: DataFrame containing cleaned and preprocessed data.
    - freq: The desired frequency for the time intervals (default: '4h').

    Returns:
    - slots_data: A pandas DataFrame where rows represent frequence time intervals, columns represent slots (`numeroStallo`),
    and values represent the sum of occupied slots for each slot at each time interval.
    """

    df = df_final.copy()
    df["time"] = df.apply(generate_time_ranges, freq=freq, axis=1)
    df = df.explode("time")

    # Define the complete time range for the dataset
    start_time = df["datetime"].min().floor(freq)  # Round down to the nearest hour
    end_time = df["next_datetime"].max().ceil(freq)  # Round up to the nearest hour
    hour_intervals = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Group by slot and time, and calculate the sum of occupancy for each group
    occupied_slots = df.groupby(["numeroStallo", "time"]).agg({"occupied": "sum"}).reset_index()

    occupied_slots = occupied_slots[occupied_slots["time"].isin(hour_intervals)]
    occupied_slots = occupied_slots.pivot(index="time", columns="numeroStallo", values="occupied")
    occupied_slots.index = pd.to_datetime(occupied_slots.index)
    occupied_slots.columns = occupied_slots.columns.astype(int)

    return occupied_slots


def generate_road_data(df_final, slots):
    """
    Generates aggregated occupancy data for roads based on hourly time intervals.

    Parameters:
    - df_final: DataFrame containing cleaned and preprocessed data.
    - slots: DataFrame containing slot data.

    Returns:
    - roads_data: A pandas DataFrame where rows represent hourly time intervals, columns represent roads (`id_strada`),
    and values represent the sum of occupied slots for each road at each time interval.
    """

    df = df_final.copy()
    freq = "1h"
    df["time"] = df.apply(generate_time_ranges, freq=freq, axis=1)
    df = df.explode("time")

    # Group by slot and time, and calculate the sum of occupancy for each group
    df1 = df.groupby(["numeroStallo", "time"]).agg({"occupied": "sum"}).reset_index()

    # Define the complete time range for the dataset
    start_time = df["datetime"].min().floor(freq)  # Round down to the nearest hour
    end_time = df["next_datetime"].max().ceil(freq)  # Round up to the nearest hour
    hour_intervals = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Create a multi-index of all slot-time combinations
    stallo_intervals = pd.MultiIndex.from_product(
        [df["numeroStallo"].unique(), hour_intervals], names=["numeroStallo", "time"]
    )
    snapshot_df = pd.DataFrame(index=stallo_intervals).reset_index()
    snapshot_df = pd.merge(snapshot_df, df1, on=["numeroStallo", "time"], how="left")
    snapshot_df = snapshot_df.fillna(0)
    snapshot_df["numeroStallo"] = snapshot_df["numeroStallo"].astype(int)

    # Merge slot metadata (to associate each slot with a road)
    snapshot_df = snapshot_df.merge(
        slots[["numeroStallo", "id_strada"]],
        left_on="numeroStallo",
        right_on="numeroStallo",
        how="left",
    )

    # Group by road and time, summing the occupancy of all slots for each road
    roads_data = (
        snapshot_df.groupby(["id_strada", "time"])
        .agg({"occupied": "sum"})
        .reset_index()
    )

    # Pivot the data so that rows are time intervals, columns are roads, and values are occupancy
    roads_data = roads_data.pivot(index="time", columns="id_strada", values="occupied")
    roads_data = roads_data.reindex(sorted(roads_data.columns), axis=1)
    roads_data.columns = roads_data.columns.astype(int)
    roads_data.index = pd.to_datetime(roads_data.index)

    return roads_data


def generate_hourly_transactions(transaction_data, data_type, freq="h"):
    """
    Preprocesses transaction data to aggregate hourly transaction counts for each 'id_parcometro'
    within a specified date range.

    Args:
    - transaction_data (DataFrame): Input transaction data.
    - data_type (str): The type of data to generate ('transactions' or 'amount').
    - freq (str): The frequency for the time intervals (default: 'h').

    Returns:
    - DataFrame: Hourly transaction counts for each 'id_parcometro' within the specified date range.

    """

    # Clean data
    transaction_data = transaction_data.drop_duplicates()

    numeric_columns = [
        "id",
        "id_parcometro",
        "id_tipopagamento",
        "stallo",
        "amount",
        "numeroTransazione",
    ]
    transaction_data[numeric_columns] = transaction_data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Drop rows where 'end_park' is NaN or 'amount' is 0
    transaction_data.dropna(subset=["end_park"], inplace=True)
    transaction_data = transaction_data[transaction_data["amount"] != 0]

    transaction_data = transaction_data[transaction_data["id_parcometro"].notna()]

    transaction_data["start_park"] = pd.to_datetime(transaction_data["start_park"])
    transaction_data["end_park"] = pd.to_datetime(transaction_data["end_park"])

    start = transaction_data["start_park"].min().floor(freq)
    end = transaction_data["start_park"].max().floor(freq)
    hour_range = pd.date_range(start=start, end=end, freq=freq)

    # Initialize an empty DataFrame for hourly transactions
    parkimeters_id = transaction_data["id_parcometro"].unique()
    hourly_transactions = pd.DataFrame(index=hour_range, columns=parkimeters_id).fillna(
        0
    )

    # Floor the start time to the nearest hour
    transaction_data["start_floor"] = transaction_data["start_park"].dt.floor(freq)

    if data_type == "transactions":
        start_group = (
            transaction_data.groupby(["start_floor", "id_parcometro"])["start_floor"]
            .count()
            .unstack(fill_value=0)
        )
    elif data_type == "amount":
        start_group = (
            transaction_data.groupby(["start_floor", "id_parcometro"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )

    # Add the start contributions to hourly_transactions
    hourly_transactions = hourly_transactions.add(start_group, fill_value=0)

    hourly_transactions.index = pd.to_datetime(hourly_transactions.index)
    hourly_transactions.columns = hourly_transactions.columns.astype(int)

    return hourly_transactions

