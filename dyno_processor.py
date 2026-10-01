import os
import csv
import math
from typing import List, Dict, Optional, Tuple


# ============================================================
# USER SETTINGS
# ============================================================

# Use "combined" if one CSV contains time, force, and RPM.
# Use "separate" if force data and RPM data are in separate CSV files.
INPUT_MODE = "combined"

# Combined input file
COMBINED_CSV = "raw_dyno_data.csv"

FORCE_CSV = "force_data.csv"
RPM_CSV = "rpm_data.csv"

OUTPUT_CSV = "processed_dyno_data.csv"
SUMMARY_TXT = "dyno_summary_report.txt"
PLOT_FOLDER = "dyno_plots"


TORQUE_ARM_FT = 1.0

# Use "force" if the input column is already force in lbf.
# Use "voltage" if the input column is voltage/raw signal that must be converted to force.
FORCE_INPUT_TYPE = "force"

# Calibration equation:
# force_lbf = CAL_SLOPE * raw_input + CAL_INTERCEPT
CAL_SLOPE = 1.0
CAL_INTERCEPT = 0.0

CALIBRATION_CSV = ""

ZERO_FIRST_N_SAMPLES = 3

MOVING_AVERAGE_WINDOW = 3

FORCE_UNCERTAINTY_LBF = 0.20
RPM_UNCERTAINTY_RPM = 10.0
ARM_LENGTH_UNCERTAINTY_FT = 0.005

MAKE_PLOTS = True


##Helper def

def safe_float(value) -> Optional[float]:
    """Convert a value to float safely. Return None if conversion fails."""
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def read_csv_file(filename: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{filename} is empty.")

    return rows


def normalize_column_name(name: str) -> str:
    """Normalize column names so different styles can still be recognized."""
    return (
        name.lower()
        .strip()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
    )


def find_column(rows: List[Dict[str, str]], possible_names: List[str]) -> Optional[str]:
    """Find a column in the CSV using several possible names."""
    if not rows:
        return None

    existing_columns = list(rows[0].keys())

    normalized_map = {
        normalize_column_name(col): col
        for col in existing_columns
    }

    for name in possible_names:
        key = normalize_column_name(name)

        if key in normalized_map:
            return normalized_map[key]

    return None


def extract_numeric_column(
    rows: List[Dict[str, str]],
    column_name: Optional[str]
) -> List[Optional[float]]:
    """Extract a numeric column from CSV rows."""
    if column_name is None:
        return [None for _ in rows]

    return [safe_float(row.get(column_name)) for row in rows]


def create_index_time(length: int) -> List[float]:
    """Create simple sample-based time if no time column exists."""
    return [float(i) for i in range(length)]


def clean_time_list(time_values: List[Optional[float]]) -> List[float]:
    """Clean time data. If no time data exists, use sample index."""
    if all(value is None for value in time_values):
        return create_index_time(len(time_values))

    cleaned_time = []
    last_value = 0.0

    for value in time_values:
        if value is None:
            cleaned_time.append(last_value)
        else:
            cleaned_time.append(value)
            last_value = value

    return cleaned_time


##Calibration

def linear_regression(
    x_values: List[float],
    y_values: List[float]
) -> Tuple[float, float, float]:
    """
    Calculate linear regression:
    y = slope * x + intercept
    """
    if len(x_values) != len(y_values):
        raise ValueError("Calibration x and y lengths do not match.")

    if len(x_values) < 2:
        raise ValueError("At least two calibration points are required.")

    n = len(x_values)

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    numerator = 0.0
    denominator = 0.0

    for x, y in zip(x_values, y_values):
        numerator += (x - x_mean) * (y - y_mean)
        denominator += (x - x_mean) ** 2

    if denominator == 0:
        raise ValueError("Calibration raw values cannot all be the same.")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    ss_total = 0.0
    ss_residual = 0.0

    for x, y in zip(x_values, y_values):
        predicted = slope * x + intercept
        ss_total += (y - y_mean) ** 2
        ss_residual += (y - predicted) ** 2

    if ss_total == 0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - ss_residual / ss_total

    return slope, intercept, r_squared


def load_calibration() -> Tuple[float, float, Optional[float]]:
    """
    Load calibration settings.
    If CALIBRATION_CSV is empty, use CAL_SLOPE and CAL_INTERCEPT.
    """
    if CALIBRATION_CSV == "":
        return CAL_SLOPE, CAL_INTERCEPT, None

    rows = read_csv_file(CALIBRATION_CSV)

    raw_col = find_column(rows, [
        "raw_signal",
        "voltage",
        "volt",
        "v",
        "sensor_output",
        "raw"
    ])

    force_col = find_column(rows, [
        "known_force_lbf",
        "force_lbf",
        "known_force",
        "force",
        "load_lbf"
    ])

    if raw_col is None or force_col is None:
        raise ValueError(
            "Calibration CSV must contain raw_signal and known_force_lbf columns."
        )

    raw_values = []
    force_values = []

    for row in rows:
        raw = safe_float(row.get(raw_col))
        force = safe_float(row.get(force_col))

        if raw is not None and force is not None:
            raw_values.append(raw)
            force_values.append(force)

    slope, intercept, r_squared = linear_regression(raw_values, force_values)

    return slope, intercept, r_squared

#Data processing

def interpolate_at_time(
    target_time: float,
    source_time: List[float],
    source_value: List[Optional[float]]
) -> Optional[float]:
    """Interpolate source_value at a target time."""
    if not source_time or not source_value:
        return None

    if target_time < source_time[0] or target_time > source_time[-1]:
        return None

    for i in range(len(source_time) - 1):
        t1 = source_time[i]
        t2 = source_time[i + 1]

        v1 = source_value[i]
        v2 = source_value[i + 1]

        if t1 <= target_time <= t2:
            if v1 is None or v2 is None:
                return None

            if t2 == t1:
                return v1

            fraction = (target_time - t1) / (t2 - t1)
            return v1 + fraction * (v2 - v1)

    return None


def average_first_n(values: List[Optional[float]], n: int) -> float:
    """Average the first n valid values."""
    if n <= 0:
        return 0.0

    valid_values = []

    for value in values[:n]:
        if value is not None:
            valid_values.append(value)

    if not valid_values:
        return 0.0

    return sum(valid_values) / len(valid_values)


def apply_zero_offset(
    values: List[Optional[float]],
    n: int
) -> Tuple[List[Optional[float]], float]:
    """Apply zero/tare correction based on the first n samples."""
    zero_offset = average_first_n(values, n)

    corrected_values = []

    for value in values:
        if value is None:
            corrected_values.append(None)
        else:
            corrected_values.append(value - zero_offset)

    return corrected_values, zero_offset

#filter
def moving_average(
    values: List[Optional[float]],
    window: int
) -> List[Optional[float]]:
    """Apply moving average smoothing."""
    if window <= 1:
        return values[:]

    smoothed = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start:i + 1]

        valid_segment = []

        for value in segment:
            if value is not None:
                valid_segment.append(value)

        if valid_segment:
            smoothed.append(sum(valid_segment) / len(valid_segment))
        else:
            smoothed.append(None)

    return smoothed


def calculate_torque(force_lbf: Optional[float]) -> Optional[float]:
    """Calculate torque from force and torque arm length."""
    if force_lbf is None:
        return None

    return force_lbf * TORQUE_ARM_FT


def calculate_horsepower(
    torque_lbft: Optional[float],
    rpm: Optional[float]
) -> Optional[float]:
    """Calculate horsepower from torque and RPM."""
    if torque_lbft is None or rpm is None:
        return None

    return torque_lbft * rpm / 5252.0


def calculate_torque_uncertainty(
    force_lbf: Optional[float]
) -> Optional[float]:
    """Calculate torque uncertainty."""
    if force_lbf is None:
        return None

    force_term = TORQUE_ARM_FT * FORCE_UNCERTAINTY_LBF
    arm_term = force_lbf * ARM_LENGTH_UNCERTAINTY_FT

    return math.sqrt(force_term ** 2 + arm_term ** 2)


def calculate_hp_uncertainty(
    torque_lbft: Optional[float],
    rpm: Optional[float],
    torque_uncertainty: Optional[float]
) -> Optional[float]:
    """Calculate horsepower uncertainty."""
    if torque_lbft is None or rpm is None or torque_uncertainty is None:
        return None

    if torque_lbft == 0 or rpm == 0:
        return None

    hp = calculate_horsepower(torque_lbft, rpm)

    if hp is None:
        return None

    relative_uncertainty = math.sqrt(
        (torque_uncertainty / torque_lbft) ** 2 +
        (RPM_UNCERTAINTY_RPM / rpm) ** 2
    )

    return abs(hp) * relative_uncertainty


def calculate_stats(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Calculate min, max, average, and count."""
    valid_values = []

    for value in values:
        if value is not None:
            valid_values.append(value)

    if not valid_values:
        return {
            "min": None,
            "max": None,
            "avg": None,
            "count": 0
        }

    return {
        "min": min(valid_values),
        "max": max(valid_values),
        "avg": sum(valid_values) / len(valid_values),
        "count": len(valid_values)
    }


def find_peak(
    values: List[Optional[float]],
    rpm_values: List[Optional[float]]
) -> Tuple[Optional[float], Optional[float]]:
    """Find peak value and corresponding RPM."""
    best_value = None
    best_rpm = None

    for value, rpm in zip(values, rpm_values):
        if value is None:
            continue

        if best_value is None or value > best_value:
            best_value = value
            best_rpm = rpm

    return best_value, best_rpm

#Load input

def load_combined_data():
    """Load time, force/raw signal, and RPM from one CSV file."""
    rows = read_csv_file(COMBINED_CSV)

    time_col = find_column(rows, [
        "time_s",
        "time",
        "seconds",
        "sec",
        "t"
    ])

    force_col = find_column(rows, [
        "force_lbf",
        "force",
        "load_lbf",
        "load",
        "lbf"
    ])

    raw_col = find_column(rows, [
        "raw_signal",
        "voltage",
        "volt",
        "v",
        "sensor_output",
        "raw"
    ])

    rpm_col = find_column(rows, [
        "rpm",
        "engine_rpm",
        "speed_rpm",
        "tach",
        "tachometer"
    ])

    time_values = clean_time_list(extract_numeric_column(rows, time_col))
    rpm_values = extract_numeric_column(rows, rpm_col)

    slope, intercept, r_squared = load_calibration()

    if FORCE_INPUT_TYPE == "force":
        if force_col is None:
            raise ValueError("No force column found in combined CSV.")

        raw_input = extract_numeric_column(rows, force_col)
        force_lbf = raw_input[:]

    elif FORCE_INPUT_TYPE == "voltage":
        if raw_col is None:
            raise ValueError("No voltage/raw column found in combined CSV.")

        raw_input = extract_numeric_column(rows, raw_col)

        force_lbf = []

        for raw in raw_input:
            if raw is None:
                force_lbf.append(None)
            else:
                force_lbf.append(slope * raw + intercept)

    else:
        raise ValueError("FORCE_INPUT_TYPE must be 'force' or 'voltage'.")

    return time_values, raw_input, force_lbf, rpm_values, slope, intercept, r_squared


def load_separate_data():
    """Load force/raw signal and RPM from two separate CSV files."""
    force_rows = read_csv_file(FORCE_CSV)
    rpm_rows = read_csv_file(RPM_CSV)

    force_time_col = find_column(force_rows, [
        "time_s",
        "time",
        "seconds",
        "sec",
        "t"
    ])

    rpm_time_col = find_column(rpm_rows, [
        "time_s",
        "time",
        "seconds",
        "sec",
        "t"
    ])

    force_col = find_column(force_rows, [
        "force_lbf",
        "force",
        "load_lbf",
        "load",
        "lbf"
    ])

    raw_col = find_column(force_rows, [
        "raw_signal",
        "voltage",
        "volt",
        "v",
        "sensor_output",
        "raw"
    ])

    rpm_col = find_column(rpm_rows, [
        "rpm",
        "engine_rpm",
        "speed_rpm",
        "tach",
        "tachometer"
    ])

    force_time = clean_time_list(extract_numeric_column(force_rows, force_time_col))
    rpm_time = clean_time_list(extract_numeric_column(rpm_rows, rpm_time_col))
    rpm_raw = extract_numeric_column(rpm_rows, rpm_col)

    slope, intercept, r_squared = load_calibration()

    if FORCE_INPUT_TYPE == "force":
        if force_col is None:
            raise ValueError("No force column found in force CSV.")

        raw_input = extract_numeric_column(force_rows, force_col)
        force_lbf = raw_input[:]

    elif FORCE_INPUT_TYPE == "voltage":
        if raw_col is None:
            raise ValueError("No voltage/raw column found in force CSV.")

        raw_input = extract_numeric_column(force_rows, raw_col)

        force_lbf = []

        for raw in raw_input:
            if raw is None:
                force_lbf.append(None)
            else:
                force_lbf.append(slope * raw + intercept)

    else:
        raise ValueError("FORCE_INPUT_TYPE must be 'force' or 'voltage'.")

    rpm_aligned = []

    for t in force_time:
        rpm_aligned.append(interpolate_at_time(t, rpm_time, rpm_raw))

    return force_time, raw_input, force_lbf, rpm_aligned, slope, intercept, r_squared

##Output

def round_or_blank(value: Optional[float], digits: int = 4):
    """Round value or return blank if None."""
    if value is None:
        return ""

    return round(value, digits)


def write_csv_file(
    filename: str,
    rows: List[Dict[str, object]]
) -> None:
    """Write rows to a CSV file."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(
    zero_offset: float,
    slope: float,
    intercept: float,
    r_squared: Optional[float],
    force: List[Optional[float]],
    torque: List[Optional[float]],
    rpm: List[Optional[float]],
    horsepower: List[Optional[float]],
    torque_uncertainty: List[Optional[float]],
    hp_uncertainty: List[Optional[float]]
) -> None:
    """Write a text summary report."""
    force_stats = calculate_stats(force)
    torque_stats = calculate_stats(torque)
    rpm_stats = calculate_stats(rpm)
    hp_stats = calculate_stats(horsepower)

    peak_torque, peak_torque_rpm = find_peak(torque, rpm)
    peak_hp, peak_hp_rpm = find_peak(horsepower, rpm)

    avg_torque_uncertainty = calculate_stats(torque_uncertainty)["avg"]
    avg_hp_uncertainty = calculate_stats(hp_uncertainty)["avg"]

    with open(SUMMARY_TXT, "w", encoding="utf-8-sig") as file:
        file.write("Dyno Data Summary Report\n")
        file.write("========================\n\n")

        file.write("Project Software Scope\n")
        file.write("----------------------\n")
        file.write("This software processes dynamometer data after force and RPM data are collected.\n")
        file.write("It does not replace measurement hardware, DAQ hardware, absorber control, or safety systems.\n\n")

        file.write("System Configuration\n")
        file.write("--------------------\n")
        file.write(f"Input mode: {INPUT_MODE}\n")
        file.write(f"Force input type: {FORCE_INPUT_TYPE}\n")
        file.write(f"Torque arm length: {TORQUE_ARM_FT} ft\n")
        file.write(f"Zero offset applied: {zero_offset:.4f} lbf\n")
        file.write(f"Moving average window: {MOVING_AVERAGE_WINDOW}\n\n")

        file.write("Calibration\n")
        file.write("-----------\n")
        file.write(f"Force = {slope:.6f} * raw_input + {intercept:.6f}\n")

        if r_squared is not None:
            file.write(f"Calibration R^2: {r_squared:.6f}\n")
        else:
            file.write("Calibration R^2: Not used\n")

        file.write("\n")

        file.write("Key Results\n")
        file.write("-----------\n")
        file.write(f"Peak torque: {peak_torque} lb-ft at {peak_torque_rpm} RPM\n")
        file.write(f"Peak horsepower: {peak_hp} hp at {peak_hp_rpm} RPM\n")
        file.write(f"Average torque uncertainty: {avg_torque_uncertainty} lb-ft\n")
        file.write(f"Average horsepower uncertainty: {avg_hp_uncertainty} hp\n\n")

        file.write("Statistics\n")
        file.write("----------\n")
        file.write(f"Force stats: {force_stats}\n")
        file.write(f"Torque stats: {torque_stats}\n")
        file.write(f"RPM stats: {rpm_stats}\n")
        file.write(f"Horsepower stats: {hp_stats}\n\n")

        file.write("Equations Used\n")
        file.write("--------------\n")
        file.write("Torque = Force * Torque Arm Length\n")
        file.write("Horsepower = Torque * RPM / 5252\n")
        file.write("Torque uncertainty = sqrt((r*uF)^2 + (F*ur)^2)\n")
        file.write("Horsepower relative uncertainty = sqrt((uT/T)^2 + (uRPM/RPM)^2)\n")


def make_plot_folder(folder_name: str) -> None:
    """Create plot folder if it does not exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)


def make_plots(
    folder_name: str,
    time_values: List[float],
    force: List[Optional[float]],
    torque: List[Optional[float]],
    rpm: List[Optional[float]],
    horsepower: List[Optional[float]]
) -> None:
    """Generate plots if matplotlib is installed."""
    if not MAKE_PLOTS:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping plots.")
        return

    make_plot_folder(folder_name)

    def clean_xy(x_values, y_values):
        x_clean = []
        y_clean = []

        for x, y in zip(x_values, y_values):
            if x is not None and y is not None:
                x_clean.append(x)
                y_clean.append(y)

        return x_clean, y_clean

    plot_definitions = [
        ("Force vs Time", time_values, force, "Time (s)", "Force (lbf)", "force_vs_time.png"),
        ("RPM vs Time", time_values, rpm, "Time (s)", "RPM", "rpm_vs_time.png"),
        ("Torque vs Time", time_values, torque, "Time (s)", "Torque (lb-ft)", "torque_vs_time.png"),
        ("Horsepower vs Time", time_values, horsepower, "Time (s)", "Horsepower (hp)", "horsepower_vs_time.png"),
        ("Torque vs RPM", rpm, torque, "RPM", "Torque (lb-ft)", "torque_vs_rpm.png"),
        ("Horsepower vs RPM", rpm, horsepower, "RPM", "Horsepower (hp)", "horsepower_vs_rpm.png"),
    ]

    for title, x_values, y_values, xlabel, ylabel, filename in plot_definitions:
        x_clean, y_clean = clean_xy(x_values, y_values)

        if not x_clean or not y_clean:
            continue

        plt.figure()
        plt.plot(x_clean, y_clean)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

        path = os.path.join(folder_name, filename)
        plt.savefig(path, dpi=200)
        plt.close()

    print(f"Plots saved in folder: {folder_name}")

##Main

def main():
    print("Starting dyno data processor...")

    if INPUT_MODE == "combined":
        time_values, raw_input, force_raw, rpm, slope, intercept, r_squared = load_combined_data()

    elif INPUT_MODE == "separate":
        time_values, raw_input, force_raw, rpm, slope, intercept, r_squared = load_separate_data()

    else:
        raise ValueError("INPUT_MODE must be 'combined' or 'separate'.")

    # Step 1: zero correction
    force_zeroed, zero_offset = apply_zero_offset(force_raw, ZERO_FIRST_N_SAMPLES)

    # Step 2: smoothing
    force_smoothed = moving_average(force_zeroed, MOVING_AVERAGE_WINDOW)

    # Step 3: torque calculation
    torque = []

    for force in force_smoothed:
        torque.append(calculate_torque(force))

    # Step 4: horsepower calculation
    horsepower = []

    for torque_value, rpm_value in zip(torque, rpm):
        horsepower.append(calculate_horsepower(torque_value, rpm_value))

    # Step 5: torque uncertainty calculation
    torque_uncertainty = []

    for force in force_smoothed:
        torque_uncertainty.append(calculate_torque_uncertainty(force))

    # Step 6: horsepower uncertainty calculation
    hp_uncertainty = []

    for torque_value, rpm_value, torque_unc_value in zip(
        torque,
        rpm,
        torque_uncertainty
    ):
        hp_uncertainty.append(
            calculate_hp_uncertainty(
                torque_value,
                rpm_value,
                torque_unc_value
            )
        )

    # Step 7: build output rows
    processed_rows = []

    for i in range(len(time_values)):
        processed_rows.append({
            "sample": i,
            "time_s": round_or_blank(time_values[i]),
            "raw_input": round_or_blank(raw_input[i]),
            "force_raw_lbf": round_or_blank(force_raw[i]),
            "force_zeroed_lbf": round_or_blank(force_zeroed[i]),
            "force_smoothed_lbf": round_or_blank(force_smoothed[i]),
            "torque_lbft": round_or_blank(torque[i]),
            "rpm": round_or_blank(rpm[i]),
            "horsepower": round_or_blank(horsepower[i]),
            "torque_uncertainty_lbft": round_or_blank(torque_uncertainty[i]),
            "horsepower_uncertainty_hp": round_or_blank(hp_uncertainty[i])
        })

    # Step 8: write output CSV
    write_csv_file(OUTPUT_CSV, processed_rows)

    # Step 9: write summary report
    write_summary_report(
        zero_offset=zero_offset,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        force=force_smoothed,
        torque=torque,
        rpm=rpm,
        horsepower=horsepower,
        torque_uncertainty=torque_uncertainty,
        hp_uncertainty=hp_uncertainty
    )

    # Step 10: make plots
    make_plots(
        folder_name=PLOT_FOLDER,
        time_values=time_values,
        force=force_smoothed,
        torque=torque,
        rpm=rpm,
        horsepower=horsepower
    )

    print("Processing complete.")
    print(f"Processed data saved as: {OUTPUT_CSV}")
    print(f"Summary report saved as: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
