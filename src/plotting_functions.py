import matplotlib.pyplot as plt
import pandas as pd

########### Funktion zum Plotten der Daten pro Sensor ###########
def plot_by_sensor(df, title, ylabel, time_range=None, highlight_range=None, events=None, x0_time=None):
    if df.empty:
        return

    # Ensure datetime
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if time_range is not None:
        start, end = time_range
        df = df[(df["time"] >= start) & (df["time"] <= end)]
        if df.empty:
            return

    # Default zero point
    if x0_time is None:
        x0_time = df["time"].min()
    x0_time = pd.to_datetime(x0_time)

    # Relative time in seconds
    df["t_rel_s"] = (df["time"] - x0_time).dt.total_seconds()

    fig, ax = plt.subplots(figsize=(10, 4))
    for sid, group in df.groupby("sensorId"):
        ax.plot(group["t_rel_s"], group["value"], label=str(sid))

    # Sequence window (also relative)
    if highlight_range is not None:
        hs, he = highlight_range
        hs_rel = (pd.to_datetime(hs) - x0_time).total_seconds()
        he_rel = (pd.to_datetime(he) - x0_time).total_seconds()
        ax.axvspan(hs_rel, he_rel, color="gray", alpha=0.3)

    # Event lines (also relative)
    if events is not None and not events.empty:
        ev = events.copy()
        ev["time"] = pd.to_datetime(ev["time"], errors="coerce")
        ev["t_rel_s"] = (ev["time"] - x0_time).dt.total_seconds()
        for _, evt in ev.iterrows():
            t = evt["t_rel_s"]
            label = str(evt["value"]).replace("[SEQ] ", "")
            ax.axvline(t, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.text(t, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7, alpha=0.85)

    ax.set_xlabel("Time since sequence start [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


########### Function to plot total mass flow of system ###########
def plot_total_massflow(df, time_range=None, highlight_range=None, events=None, x0_time=None):
    if df.empty:
        return

    # Set zero to be begning of sequence    
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if time_range is not None:
        start, end = time_range
        df = df[(df["time"] >= start) & (df["time"] <= end)]
        if df.empty:
            return
    
    if x0_time is None:
        x0_time = df["time"].min()
    x0_time = pd.to_datetime(x0_time)

    df["t_rel_s"] = (df["time"] - x0_time).dt.total_seconds()

    # Plot sensor values
    fig, ax = plt.subplots(figsize=(10, 4))
    for sid, group in df.groupby("sensorId"):
        ax.plot(group["t_rel_s"], group["value"], label=str(sid))

    # Plot total mf by adding all mf together
    df_total_mf = df.dropna(subset=["time", "sensorId", "value"]).sort_values("time")
    if not df_total_mf.empty:
        # Handle duplicate timestamps and separate different mfs into columns
        df_total_mf = df_total_mf.groupby(["time", "sensorId"], as_index=False)["value"].mean()
        df_combined = df_total_mf.pivot(index="time", columns="sensorId", values="value").sort_index()        

        # Sample mf to fit sampling grid
        freq = "10ms"
        df_sampled = df_combined.resample(freq).mean()

        # Option A: interpolate to fill NaN
        df_filled = df_sampled.interpolate(method="time", limit_area="inside")
        # Option B: zero order hold
        # df_filled = df_sampled.ffill()

        # Sum massflows in each row
        total = df_filled.sum(axis=1, min_count=1).dropna()
        # print(total.reset_index().head(10))

        if not total.empty:
            t_rel_total = (total.index - x0_time).total_seconds()
            ax.plot(t_rel_total, total.values, color="black", label="Total MF", alpha=0.5)

    # Sequence window (also relative)
    if highlight_range is not None:
        hs, he = highlight_range
        hs_rel = (pd.to_datetime(hs) - x0_time).total_seconds()
        he_rel = (pd.to_datetime(he) - x0_time).total_seconds()
        ax.axvspan(hs_rel, he_rel, color="gray", alpha=0.3)

    # Event lines (also relative)
    if events is not None and not events.empty:
        ev = events.copy()
        ev["time"] = pd.to_datetime(ev["time"], errors="coerce")
        ev["t_rel_s"] = (ev["time"] - x0_time).dt.total_seconds()
        for _, evt in ev.iterrows():
            t = evt["t_rel_s"]
            label = str(evt["value"]).replace("[SEQ] ", "")
            ax.axvline(t, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.text(t, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7, alpha=0.85)

    ax.set_xlabel("Time since sequence start [s]")
    ax.set_ylabel("Total Massflow [g/s]")
    ax.set_title("Total Massflow")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


########### Function to plot equivalence ratio of system ###########
def plot_equivalence_ratio(df, time_range=None, highlight_range=None, x0_time=None):
    if df.empty:
        return

    # Set zero to be begning of sequence    
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if time_range is not None:
        start, end = time_range
        df = df[(df["time"] >= start) & (df["time"] <= end)]
        if df.empty:
            return
    
    if x0_time is None:
        x0_time = df["time"].min()
    x0_time = pd.to_datetime(x0_time)

    df["t_rel_s"] = (df["time"] - x0_time).dt.total_seconds()        

    # Plot total mf by adding all mf together
    df_total_mf = df.dropna(subset=["time", "sensorId", "value"]).sort_values("time")
    if not df_total_mf.empty:
        # Handle duplicate timestamps and separate different mfs into columns
        df_total_mf = df_total_mf.groupby(["time", "sensorId"], as_index=False)["value"].mean()
        df_combined = df_total_mf.pivot(index="time", columns="sensorId", values="value").sort_index()        

        # Sample mf to fit sampling grid
        freq = "10ms"
        df_sampled = df_combined.resample(freq).mean()

        # Option A: interpolate to fill NaN
        df_filled = df_sampled.interpolate(method="time", limit_area="inside")
        # Option B: zero order hold
        # df_filled = df_sampled.ffill()

        # Calculate equivalence ratio
        propane_molar_mass = 44.097
        oxygen_molar_mass = 32
        st_fuel_ox_ratio = propane_molar_mass / (5 * oxygen_molar_mass)        
        total = ((df_filled["FSS_MF"] / df_filled["OSS_MF"]).where(df_filled["OSS_MF"] != 0)/ st_fuel_ox_ratio).dropna()
        
        # print(total.reset_index().head(10))

        # Plot equivalence ratio
        fig, ax = plt.subplots(figsize=(10, 4))
        if not total.empty:
            t_rel_total = (total.index - x0_time).total_seconds()
            ax.plot(t_rel_total, total.values, label="Equivalence Ratio")

    # Sequence window (also relative)
    if highlight_range is not None:
        hs, he = highlight_range
        hs_rel = (pd.to_datetime(hs) - x0_time).total_seconds()
        he_rel = (pd.to_datetime(he) - x0_time).total_seconds()
        ax.axvspan(hs_rel, he_rel, color="gray", alpha=0.3)

    # Add vertical lines and sequnce text
    if highlight_range is not None:
        hs, he = highlight_range
        hs_rel = (pd.to_datetime(hs) - x0_time).total_seconds()
        he_rel = (pd.to_datetime(he) - x0_time).total_seconds()

        ax.axvline(hs_rel, color="darkred", linestyle="--", linewidth=1.1, alpha=0.9)
        ax.axvline(he_rel, color="navy", linestyle="--", linewidth=1.1, alpha=0.9)

        ax.text(
            hs_rel, 0.98, "Ignition",
            transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=8, color="darkred"
        )
        ax.text(
            he_rel, 0.98, "Initiate Shutdown",
            transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=8, color="navy"
        )

    ax.set_xlabel("Time Since Ignition [s]")
    ax.set_ylabel("Equivalence Ratio")
    ax.set_title("Equivalence Ratio")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()