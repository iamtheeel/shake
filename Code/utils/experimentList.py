###
# experimentList.py
# Joshua Mehlman
# MIC Lab
# Fall, 2025
###
# List the experiments and show the details
###

import os
import csv
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt


baseDirectory = "/Volumes/Data/thesis/"
#baseDirectory = "/Volumes/Data/thesis/exp_Track_HPC"
#baseDirectory = "/Volumes/Data/thesis/exp_Track_HPC/cMorel"

# -----------------------------
# Filename filter (no regex)
# -----------------------------
def looks_like_datatrack_summary(filename):
    """
    Return True if filename matches:
        YYYYMMDD-HHMMSS_DataTrack_Sumary.csv
    using only string checks (no regex).
    """
    suffix = "_DataTrack_Sumary.csv"
    if not filename.endswith(suffix):
        return False

    ts = filename[:-len(suffix)]  # strip suffix
    # expected: 'YYYYMMDD-HHMMSS' -> length = 8 + 1 + 6 = 15
    if len(ts) != 15:
        return False
    if ts[8] != "-":
        return False

    ymd = ts[:8]
    hms = ts[9:]

    # check all characters are digits
    return ymd.isdigit() and hms.isdigit()

# -----------------------------
# Extract experiment and run dir
# -----------------------------
def extract_path_info(fullpath):
    """
    Extract:
      - experiment number (int, from 'exp-6' -> 6)
      - run directory (YYYYMMDD-HHMMSS_something)

    Only return experiment/run_dir if file is inside an exp-X directory.
    Otherwise return (None, None) and caller will skip it.
    """
    parts = os.path.normpath(fullpath).split(os.sep)
    dirs = parts[:-1]  # ignore filename

    # Collect ALL exp-X directories in the path
    exp_nums = []
    for d in dirs:
        if d.startswith("exp-"):
            numpart = d[4:]
            if numpart.isdigit():
                exp_nums.append(int(numpart))

    # If there are no exp-X dirs, skip this file
    if not exp_nums:
        return None, None

    # Deepest exp number
    experiment_num = exp_nums[-1]

    # Detect run directory (YYYYMMDD-HHMMSS_prefix)
    run_dir = ""
    for d in dirs:
        if (
            len(d) >= 15
            and d[8] == "-"
            and d[:8].isdigit()
            and d[9:15].isdigit()
        ):
            run_dir = d
            break

    return experiment_num, run_dir

# -----------------------------
# Parse experiment fields from file
# -----------------------------
def parse_dataTrack_fields(path):
    """
    Read the DataTrack summary CSV and extract:
      - epochs
    """
    epochs = None
    print("Parsing DataTrack fields from:", path)

    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                key = row[0].strip()

                # Missing value column? Skip safely.
                value = row[1].strip() if len(row) > 1 else ""

                if key == "epochs":
                    try:
                        epochs = int(value) if value else None
                    except ValueError:
                        epochs = None
                    #print(f"  Found epochs: {epochs}")

    except Exception as e:
        print(f"Error parsing {path}: {e}")

    return epochs

def parseHelper(value):
    try:
        return float(value) if value else None
    except ValueError:
        return None

def parse_summary_fields(path):
    """
    Read the summary CSV and extract:
      - wavelet
      - wavelet_center_freq
      - wavelet_bandwidth

    Any rows with missing columns or unexpected structure are ignored.
    """
    wavelet = ""
    modelName = ""
    center_freq = None
    bandwidth = None

    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                key = row[0].strip()

                # Missing value column? Skip safely.
                value = row[1].strip() if len(row) > 1 else ""

                if key == "wavelet": wavelet = value
                elif key == "wavelet_center_freq": center_freq = parseHelper(value)
                elif key == "wavelet_bandwidth": bandwidth = parseHelper(value)

                elif key == "modelName": modelName = value
                elif key == "optimizer": optimizer = value
                elif key == "loss": lossFun = value
                elif key == "learning_rate": lr = parseHelper(value)
                elif key == "weight_decay": wd = parseHelper(value)
                

    except Exception as e:
        print(f"Error parsing {path}: {e}")

    return wavelet, center_freq, bandwidth, modelName, optimizer, lossFun, lr, wd

def getFromValidationFile(validationFile):
    try:
        with open(validationFile, "r") as f:
            return (sum(1 for _ in f)-2) # Subtract header line
    except Exception as e:
        print("Error:", e)
        return 0
    

# -----------------------------
# Collect matching files
# -----------------------------
# list of (date, time, experiment_num, run_dir, wavelet, center_freq, bandwidth, fullpath)
rows_data = []
epochs = None

for root, dirs, files in os.walk(baseDirectory):
    for filename in files:
        if not looks_like_datatrack_summary(filename):
            continue

        fullpath = os.path.join(root, filename)

        # Get the Data fields
        experiment_num, run_dir = extract_path_info(fullpath)
        if experiment_num is None:
            epochs = parse_dataTrack_fields(fullpath)
            #print(f"Got epochs: {epochs} for file: {fullpath}")
            continue

        timestamp_part = filename.split("_", 1)[0]  # 'YYYYMMDD-HHMMSS'
        date_part, time_part = timestamp_part.split("-")
        wavelet, center_freq, bandwidth, modelName, optimizer, lossFun, lr, wd = parse_summary_fields(fullpath)

        epochs_saved = getFromValidationFile(fullpath.rsplit("/", 1)[0] + "/valiResults_byEpoch.csv")

        ## Makes some data fixes
        if epochs_saved <= 1: continue
        if epochs < 5: continue
        if wavelet == "": wavelet = "Time Domain"
        if wavelet == "cmor": wavelet = "cmorl"
        if wavelet == "cmorl": # Early data had center freq/bandwidth reversed  
            if center_freq == 10:
                foo = center_freq
                center_freq = bandwidth
                bandwidth = foo

        row = {
            "path": fullpath,
            "date": date_part,
            "time": time_part,
            "experiment": str(experiment_num),
            "run_dir": run_dir,
            "wavelet": wavelet,
            "center_freq": "" if center_freq is None else str(center_freq),
            "bandwidth": "" if bandwidth is None else str(bandwidth),
            "epochs": "" if epochs is None else str(epochs),
            "epochs_saved":  str(epochs_saved),
            "modelName": modelName,
            "optimizer": optimizer,
            "lossFun": lossFun,
            "lr": "" if lr is None else str(lr),    
            "wd": "" if wd is None else str(wd),
        }
        rows_data.append(row)

#Finished collecting data
print(f"Found {len(rows_data)} summary files inside exp-x directories.")


# -----------------------------
# GUI: Plotting
# -----------------------------
def plotValidationCurves(path, date, time):
    """
    Placeholder function to plot validation curves from a summary file.
    In a real implementation, this would read the file and generate plots.
    """
    label = f"{date}-{time}"
    validationFile = path.rsplit("/", 1)[0] + "/valiResults_byEpoch.csv"
    print(f"Plotting validation curves for: {validationFile}")

    epochs = []
    losses = []

    with open(validationFile, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(float(row["Epoch Num"]))
            losses.append(float(row["Loss"]))
    
    plt.plot(epochs, losses, label=label)
    plt.ylim(0, 0.1) 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()         # ← add legend
    plt.title("Validation Loss")
    plt.grid(True)
    plt.show()


# -----------------------------
# GUI: dynamic columns + filters
# -----------------------------
def sort_by(tree, col, descending):
    data = [(tree.set(child, col), child) for child in tree.get_children("")]

    def convert(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    data.sort(key=lambda t: convert(t[0]), reverse=descending)

    for index, (_, item_id) in enumerate(data):
        tree.move(item_id, "", index)

    tree.heading(col, command=lambda: sort_by(tree, col, not descending))

# All possible fields and their labels
ALL_FIELDS = [
    ("run_dir",         "Run Dir"),
    ("date",            "Date (YYYYMMDD)"),
    ("time",            "Time (HHMMSS)"),
    ("experiment",      "Experiment #"),
    ("modelName",       "Model Name"),
    ("wavelet",         "Wavelet"),
    ("center_freq",     "CenterFreq"),
    ("bandwidth",       "Bandwidth"),
    ("epochs",          "Epochs"),
    ("epochs_saved",    "Epochs Saved"),
    ("optimizer",       "Optimizer"),
    ("lossFun",         "Loss Function"),
    ("lr",              "Learning Rate"),
    ("wd",              "Weight Decay"),
    # ("path", "Path"),  # you can enable this if you want to see paths
]

# Start with these fields visible
#visible_field_ids = {"run_dir", "date", "time", "experiment", "wavelet", "center_freq", "bandwidth"}
visible_field_ids = {"date", "time", "wavelet", "center_freq", "bandwidth", "epochs", "epochs_run"}

# Current filter value per field_id: "All" or a specific string
field_filters = {fid: "All" for fid, _ in ALL_FIELDS}

root = tk.Tk()
root.title("DataTrack Summary Files")

main_frame = ttk.Frame(root, padding=5)
main_frame.pack(fill="both", expand=True)

# Left: field selection checkboxes
fields_frame = ttk.LabelFrame(main_frame, text="Visible fields")
fields_frame.pack(side="left", fill="y", padx=(0, 5))

field_vars = {}

# Right: table + filter row
right_frame = ttk.Frame(main_frame)
right_frame.pack(side="right", fill="both", expand=True)

filters_frame = ttk.Frame(right_frame)
filters_frame.pack(fill="x")

table_frame = ttk.Frame(right_frame)
table_frame.pack(fill="both", expand=True)

tree = ttk.Treeview(table_frame, show="headings")
tree.pack(fill="both", expand=True)

filter_widgets = {}  # field_id -> Combobox


def apply_filters_to_rows():
    """Return list of rows that match current field_filters."""
    filtered = []
    for row in rows_data:
        keep = True
        for fid, selected in field_filters.items():
            if selected == "All":
                continue
            # if field not visible, we still apply filter (so filters persist even if col hidden)
            value = row.get(fid, "")
            if value != selected:
                keep = False
                break
        if keep:
            filtered.append(row)
    return filtered


def rebuild_tree_rows():
    """Rebuild only the table rows according to current filters & visible columns."""
    # Clear existing rows
    for child in tree.get_children(""):
        tree.delete(child)

    cols = [fid for fid, _ in ALL_FIELDS if fid in visible_field_ids]

    filtered_rows = apply_filters_to_rows()

    for row in filtered_rows:
        values = [row[fid] for fid in cols]
        tree.insert("", "end", values=values, 
                    tags=(row["path"], row["date"], row["time"]))  # store path in tags for double-click


def rebuild_filter_widgets():
    """Rebuild the filter comboboxes for each visible column."""
    # Clear old widgets
    for child in filters_frame.winfo_children():
        child.destroy()
    filter_widgets.clear()

    cols = [fid for fid, _ in ALL_FIELDS if fid in visible_field_ids]

    for fid in cols:
        # Label + combobox for each visible field
        frame = ttk.Frame(filters_frame)
        frame.pack(side="left", padx=2, pady=2)

        label_text = next(label for f, label in ALL_FIELDS if f == fid)
        ttk.Label(frame, text=label_text).pack(anchor="w")

        # Collect unique values for this field from all rows (not just filtered)
        values_set = set()
        for row in rows_data:
            values_set.add(row.get(fid, ""))

        values_list = sorted(v for v in values_set if v != "")
        combo_values = ["All"] + values_list

        current = field_filters.get(fid, "All")
        if current not in combo_values:
            current = "All"
            field_filters[fid] = "All"

        var = tk.StringVar(value=current)
        cb = ttk.Combobox(frame, textvariable=var, values=combo_values, state="readonly", width=15)

        def make_callback(field_id, var_ref):
            def _on_select(event=None):
                field_filters[field_id] = var_ref.get()
                rebuild_tree_rows()
            return _on_select

        cb.bind("<<ComboboxSelected>>", make_callback(fid, var))
        cb.pack(anchor="w")

        filter_widgets[fid] = cb


def rebuild_tree_structure():
    """Rebuild the table columns + headings when visible columns change."""
    cols = [fid for fid, _ in ALL_FIELDS if fid in visible_field_ids]
    tree["columns"] = cols

    for fid, label in ALL_FIELDS:
        if fid not in visible_field_ids:
            continue
        tree.heading(fid, text=label, command=lambda c=fid: sort_by(tree, c, False))
        width = 120
        if fid in ("run_dir",):
            width = 220
        tree.column(fid, anchor="center", width=width)

    rebuild_filter_widgets()
    rebuild_tree_rows()


def on_field_toggle(field_id):
    # Ensure at least one field stays visible
    if field_vars[field_id].get():
        visible_field_ids.add(field_id)
    else:
        if len(visible_field_ids) == 1 and field_id in visible_field_ids:
            # Can't hide last remaining field – re-check
            field_vars[field_id].set(True)
            return
        visible_field_ids.discard(field_id)

    rebuild_tree_structure()


# Create checkboxes for visible fields
for fid, label in ALL_FIELDS:
    var = tk.BooleanVar(value=(fid in visible_field_ids))
    field_vars[fid] = var
    cb = ttk.Checkbutton(
        fields_frame,
        text=label,
        variable=var,
        command=lambda f=fid: on_field_toggle(f),
    )
    cb.pack(anchor="w")


# Double-click row → print full path
def on_double_click(event):
    item_id = tree.focus()
    if not item_id:
        return
    tags = tree.item(item_id, "tags")
    if tags:
        print("Selected file:", tags[0])
        plotValidationCurves(tags[0], tags[1], tags[2]) #File, date, time

tree.bind("<Double-1>", on_double_click)

# Initial build
rebuild_tree_structure()

root.mainloop()