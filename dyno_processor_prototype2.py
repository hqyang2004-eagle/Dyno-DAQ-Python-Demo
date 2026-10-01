"""
Prototype 2: Virtual Dynamometer Data-Processing Prototype

"""
import os
import csv
import math
import random
from typing import List, Dict, Optional, Tuple

# ============================================================
# USER SETTINGS
# ============================================================

GENERATE_VIRTUAL_DATA = False
PROCESS_MULTIPLE_VIRTUAL_RUNS = False
VIRTUAL_RUN_COUNT = 3
INJECT_VIRTUAL_WARNING_CASE = False

VIRTUAL_DATA_FOLDER = "virtual_input_data"
PROCESSED_DATA_FOLDER = "processed_output_data"
PLOT_ROOT_FOLDER = "dyno_plots"

INPUT_MODE = "combined"
COMBINED_CSV = "noisy_virtual_dyno_demo.csv"
FORCE_CSV = "force_data.csv"
RPM_CSV = "rpm_data.csv"

OUTPUT_CSV = "processed_dyno_data.csv"
SUMMARY_TXT = "dyno_summary_report.txt"

TORQUE_ARM_FT = 1.0
FORCE_INPUT_TYPE = "force"

CAL_SLOPE = 1.0
CAL_INTERCEPT = 0.0
CALIBRATION_CSV = ""

ZERO_FIRST_N_SAMPLES = 3
MOVING_AVERAGE_WINDOW = 7

FORCE_UNCERTAINTY_LBF = 0.20
RPM_UNCERTAINTY_RPM = 10.0
ARM_LENGTH_UNCERTAINTY_FT = 0.005

LOAD_CELL_CAPACITY_LBF = 100.0
LOAD_CELL_WARNING_RATIO = 0.90
MAX_SAFE_RPM = 11000.0
MIN_FLOW_GPM = 10.0
MAX_WATER_TEMP_C = 85.0
MAX_HP_UNCERTAINTY_HP = 0.20

MAKE_PLOTS = True


# ============================================================
# BASIC HELPER FUNCTIONS
# ============================================================


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


def ensure_folder(folder_name: str) -> None:
    """Create a folder if it does not already exist."""
    if folder_name and not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)


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


def write_csv_file(filename: str, rows: List[Dict[str, object]]) -> None:
    """Write rows to a CSV file."""
    if not rows:
        return

    folder = os.path.dirname(filename)
    ensure_folder(folder)

    fieldnames = list(rows[0].keys())

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_column_name(name: str) -> str:
    """Normalize column names so different CSV styles can still be recognized."""
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
    """Find a CSV column using several possible names."""
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
    """Create sample-index-based time if no time column exists."""
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


def round_or_blank(value: Optional[float], digits: int = 4):
    """Round a value or return blank if the value is None."""
    if value is None:
        return ""

    return round(value, digits)


def calculate_stats(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Calculate min, max, average, and count."""
    valid_values = []

    for value in values:
        if value is not None:
            valid_values.append(value)

    if not valid_values:
        return {"min": None, "max": None, "avg": None, "count": 0}

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


# ============================================================
# VIRTUAL TEST DATA GENERATOR
# ============================================================


def generate_virtual_test_data(
    filename: str,
    duration_s: float = 20.0,
    sample_rate_hz: int = 20,
    run_number: int = 1,
    inject_warning_case: bool = False
) -> None:
    """
    Generate virtual dyno test data for Prototype 2.

    Output columns:
    - time_s
    - force_lbf
    - rpm
    - water_temp_c
    - flow_gpm
    """
    folder = os.path.dirname(filename)
    ensure_folder(folder)

    random.seed(100 + run_number)
    total_samples = int(duration_s * sample_rate_hz)
    rows = []

    # Slightly change each run so multi-run comparison has realistic variation.
    rpm_start = 2200 + random.uniform(-100, 100)
    rpm_end = 9000 + random.uniform(-200, 200)
    force_peak = 62 + random.uniform(-3, 3)

    for i in range(total_samples):
        time_s = i / sample_rate_hz
        progress = time_s / duration_s

        # Simulated RPM sweep.
        rpm = rpm_start + (rpm_end - rpm_start) * progress
        rpm += random.uniform(-45, 45)

        # Simulated force curve.
        # It rises, reaches a broad peak, then slightly decreases.
        force_lbf = 12 + force_peak * math.sin(progress * math.pi)
        force_lbf += random.uniform(-1.5, 1.5)

        if force_lbf < 0:
            force_lbf = 0.0

        # Simulated water brake monitoring data.
        water_temp_c = 25 + 42 * progress + random.uniform(-0.5, 0.5)
        flow_gpm = 10.8 + random.uniform(-0.25, 0.25)

        # Optional warning case for demonstration.
        if inject_warning_case and run_number == VIRTUAL_RUN_COUNT:
            if progress > 0.70:
                flow_gpm = 9.5 + random.uniform(-0.2, 0.2)
            if progress > 0.85:
                water_temp_c = 86 + random.uniform(0, 3)

        rows.append({
            "time_s": round(time_s, 3),
            "force_lbf": round(force_lbf, 3),
            "rpm": round(rpm, 1),
            "water_temp_c": round(water_temp_c, 2),
            "flow_gpm": round(flow_gpm, 2)
        })

    write_csv_file(filename, rows)
    print(f"Virtual test data generated: {filename}")


# ============================================================
# CALIBRATION FUNCTIONS
# ============================================================


def linear_regression(
    x_values: List[float],
    y_values: List[float]
) -> Tuple[float, float, float]:
    """Calculate linear regression: y = slope * x + intercept."""
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
    """Load calibration settings from CALIBRATION_CSV or constants."""
    if CALIBRATION_CSV == "":
        return CAL_SLOPE, CAL_INTERCEPT, None

    rows = read_csv_file(CALIBRATION_CSV)

    raw_col = find_column(rows, [
        "raw_signal", "voltage", "volt", "v", "sensor_output", "raw"
    ])

    force_col = find_column(rows, [
        "known_force_lbf", "force_lbf", "known_force", "force", "load_lbf"
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

    return linear_regression(raw_values, force_values)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================


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


def load_combined_data(filename: str):
    """Load time, force/raw signal, RPM, water temperature, and flow rate from one CSV."""
    rows = read_csv_file(filename)

    time_col = find_column(rows, ["time_s", "time", "seconds", "sec", "t"])

    force_col = find_column(rows, [
        "force_lbf", "force", "load_lbf", "load", "lbf"
    ])

    raw_col = find_column(rows, [
        "raw_signal", "voltage", "volt", "v", "sensor_output", "raw"
    ])

    rpm_col = find_column(rows, [
        "rpm", "engine_rpm", "speed_rpm", "tach", "tachometer"
    ])

    water_temp_col = find_column(rows, [
        "water_temp_c", "watertemp_c", "water_temperature_c",
        "temp_c", "temperature_c", "outlet_temp_c"
    ])

    flow_col = find_column(rows, [
        "flow_gpm", "water_flow_gpm", "flowrate_gpm", "flow_rate_gpm", "gpm"
    ])

    time_values = clean_time_list(extract_numeric_column(rows, time_col))
    rpm_values = extract_numeric_column(rows, rpm_col)
    water_temp_c = extract_numeric_column(rows, water_temp_col)
    flow_gpm = extract_numeric_column(rows, flow_col)

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

    return {
        "time": time_values,
        "raw_input": raw_input,
        "force_lbf": force_lbf,
        "rpm": rpm_values,
        "water_temp_c": water_temp_c,
        "flow_gpm": flow_gpm,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared
    }


def load_separate_data(force_csv: str, rpm_csv: str):
    """Load force/raw signal and RPM from two separate CSV files."""
    force_rows = read_csv_file(force_csv)
    rpm_rows = read_csv_file(rpm_csv)

    force_time_col = find_column(force_rows, ["time_s", "time", "seconds", "sec", "t"])
    rpm_time_col = find_column(rpm_rows, ["time_s", "time", "seconds", "sec", "t"])

    force_col = find_column(force_rows, [
        "force_lbf", "force", "load_lbf", "load", "lbf"
    ])

    raw_col = find_column(force_rows, [
        "raw_signal", "voltage", "volt", "v", "sensor_output", "raw"
    ])

    rpm_col = find_column(rpm_rows, [
        "rpm", "engine_rpm", "speed_rpm", "tach", "tachometer"
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

    water_temp_c = [None for _ in force_time]
    flow_gpm = [None for _ in force_time]

    return {
        "time": force_time,
        "raw_input": raw_input,
        "force_lbf": force_lbf,
        "rpm": rpm_aligned,
        "water_temp_c": water_temp_c,
        "flow_gpm": flow_gpm,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared
    }


# ============================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================


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


def moving_average(
    values: List[Optional[float]],
    window: int
) -> List[Optional[float]]:
    """Apply moving-average smoothing."""
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


# ============================================================
# ENGINE PERFORMANCE CALCULATIONS
# ============================================================


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


def calculate_torque_uncertainty(force_lbf: Optional[float]) -> Optional[float]:
    """Calculate torque uncertainty."""
    if force_lbf is None:
        return None

    force_term = TORQUE_ARM_FT * FORCE_UNCERTAINTY_LBF
    arm_term = abs(force_lbf) * ARM_LENGTH_UNCERTAINTY_FT

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


# ============================================================
# VALIDATION AND SAFETY FUNCTIONS
# ============================================================


def create_safety_flag(
    force_lbf: Optional[float],
    rpm: Optional[float],
    water_temp_c: Optional[float] = None,
    flow_gpm: Optional[float] = None
) -> str:
    """Create a safety flag for each sample."""
    flags = []

    if force_lbf is not None:
        abs_force = abs(force_lbf)

        if abs_force > LOAD_CELL_CAPACITY_LBF:
            flags.append("FORCE_OVERLOAD")
        elif abs_force > LOAD_CELL_CAPACITY_LBF * LOAD_CELL_WARNING_RATIO:
            flags.append("FORCE_WARNING")

    if rpm is not None and rpm > MAX_SAFE_RPM:
        flags.append("RPM_OVER_LIMIT")

    if water_temp_c is not None and water_temp_c > MAX_WATER_TEMP_C:
        flags.append("WATER_TEMP_HIGH")

    if flow_gpm is not None and flow_gpm < MIN_FLOW_GPM:
        flags.append("LOW_WATER_FLOW")

    if not flags:
        return "OK"

    return "; ".join(flags)


def evaluate_validation_checks(
    force: List[Optional[float]],
    rpm: List[Optional[float]],
    hp_uncertainty: List[Optional[float]],
    water_temp: Optional[List[Optional[float]]] = None,
    flow_gpm: Optional[List[Optional[float]]] = None
) -> List[str]:
    """Return PASS/WARNING/FAIL validation messages."""
    results = []

    abs_force = []
    for value in force:
        if value is None:
            abs_force.append(None)
        else:
            abs_force.append(abs(value))

    force_stats = calculate_stats(abs_force)
    rpm_stats = calculate_stats(rpm)
    hp_unc_stats = calculate_stats(hp_uncertainty)

    max_force = force_stats["max"]
    max_rpm = rpm_stats["max"]
    max_hp_unc = hp_unc_stats["max"]

    if max_force is not None:
        usage_percent = 100.0 * max_force / LOAD_CELL_CAPACITY_LBF

        if max_force > LOAD_CELL_CAPACITY_LBF:
            results.append(
                f"FAIL: Load cell force exceeds capacity. "
                f"Max force = {max_force:.2f} lbf, capacity = {LOAD_CELL_CAPACITY_LBF:.2f} lbf."
            )
        elif max_force > LOAD_CELL_CAPACITY_LBF * LOAD_CELL_WARNING_RATIO:
            results.append(
                f"WARNING: Load cell force is above {LOAD_CELL_WARNING_RATIO * 100:.0f}% of capacity. "
                f"Usage = {usage_percent:.1f}%."
            )
        else:
            results.append(
                f"PASS: Load cell force is within range. Max usage = {usage_percent:.1f}%."
            )

    if max_rpm is not None:
        if max_rpm > MAX_SAFE_RPM:
            results.append(
                f"FAIL: RPM exceeds maximum safe limit. Max RPM = {max_rpm:.1f}, "
                f"limit = {MAX_SAFE_RPM:.1f}."
            )
        else:
            results.append(f"PASS: RPM is within safe range. Max RPM = {max_rpm:.1f}.")

    if max_hp_unc is not None:
        if max_hp_unc > MAX_HP_UNCERTAINTY_HP:
            results.append(
                f"WARNING: Horsepower uncertainty is above target limit. "
                f"Max uncertainty = +/-{max_hp_unc:.3f} hp, "
                f"target = +/-{MAX_HP_UNCERTAINTY_HP:.3f} hp."
            )
        else:
            results.append(
                f"PASS: Horsepower uncertainty is within target limit. "
                f"Max uncertainty = +/-{max_hp_unc:.3f} hp."
            )

    if water_temp is not None:
        temp_stats = calculate_stats(water_temp)
        max_temp = temp_stats["max"]

        if max_temp is not None:
            if max_temp > MAX_WATER_TEMP_C:
                results.append(
                    f"FAIL: Water temperature exceeds safety limit. Max temp = {max_temp:.2f} C, "
                    f"limit = {MAX_WATER_TEMP_C:.2f} C."
                )
            else:
                results.append(f"PASS: Water temperature is within safety limit. Max temp = {max_temp:.2f} C.")

    if flow_gpm is not None:
        flow_stats = calculate_stats(flow_gpm)
        min_flow = flow_stats["min"]

        if min_flow is not None:
            if min_flow < MIN_FLOW_GPM:
                results.append(
                    f"FAIL: Water flow is below minimum required flow rate. Min flow = {min_flow:.2f} gpm, "
                    f"minimum = {MIN_FLOW_GPM:.2f} gpm."
                )
            else:
                results.append(f"PASS: Water flow is above minimum required flow rate. Min flow = {min_flow:.2f} gpm.")

    return results


def get_overall_validation_status(validation_results: List[str]) -> str:
    """Return overall PASS, WARNING, or FAIL based on validation messages."""
    has_fail = False
    has_warning = False

    for message in validation_results:
        if message.startswith("FAIL"):
            has_fail = True
        elif message.startswith("WARNING"):
            has_warning = True

    if has_fail:
        return "FAIL"

    if has_warning:
        return "WARNING"

    return "PASS"


# ============================================================
# OUTPUT AND PLOTTING FUNCTIONS
# ============================================================


def write_summary_report(
    summary_txt: str,
    input_file: str,
    zero_offset: float,
    slope: float,
    intercept: float,
    r_squared: Optional[float],
    force_raw: List[Optional[float]],
    force_zeroed: List[Optional[float]],
    force_smoothed: List[Optional[float]],
    torque: List[Optional[float]],
    rpm: List[Optional[float]],
    horsepower: List[Optional[float]],
    torque_uncertainty: List[Optional[float]],
    hp_uncertainty: List[Optional[float]],
    water_temp_c: List[Optional[float]],
    flow_gpm: List[Optional[float]],
    validation_results: List[str]
) -> None:
    """Write a text summary report."""
    folder = os.path.dirname(summary_txt)
    ensure_folder(folder)

    peak_torque, peak_torque_rpm = find_peak(torque, rpm)
    peak_hp, peak_hp_rpm = find_peak(horsepower, rpm)

    avg_torque_uncertainty = calculate_stats(torque_uncertainty)["avg"]
    avg_hp_uncertainty = calculate_stats(hp_uncertainty)["avg"]

    overall_status = get_overall_validation_status(validation_results)

    with open(summary_txt, "w", encoding="utf-8-sig") as file:
        file.write("Prototype 2 Virtual Dynamometer Data-Processing Report\n")
        file.write("=====================================================\n\n")

        file.write("Software Scope\n")
        file.write("--------------\n")
        file.write("This software processes dynamometer data after force and RPM data are collected.\n")
        file.write("It supports virtual data generation, filtering, torque and horsepower calculation,\n")
        file.write("uncertainty estimation, safety flags, validation checks, plots, and multi-run comparison.\n")
        file.write("It does not replace measurement hardware, DAQ hardware, absorber control, or safety systems.\n\n")

        file.write("Input File\n")
        file.write("----------\n")
        file.write(f"{input_file}\n\n")

        file.write("System Configuration\n")
        file.write("--------------------\n")
        file.write(f"Input mode: {INPUT_MODE}\n")
        file.write(f"Force input type: {FORCE_INPUT_TYPE}\n")
        file.write(f"Torque arm length: {TORQUE_ARM_FT} ft\n")
        file.write(f"Load cell capacity: {LOAD_CELL_CAPACITY_LBF} lbf\n")
        file.write(f"Max safe RPM: {MAX_SAFE_RPM} RPM\n")
        file.write(f"Minimum required water flow: {MIN_FLOW_GPM} gpm\n")
        file.write(f"Maximum water temperature: {MAX_WATER_TEMP_C} C\n")
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
        file.write(f"Peak torque: {round_or_blank(peak_torque)} lb-ft at {round_or_blank(peak_torque_rpm)} RPM\n")
        file.write(f"Peak horsepower: {round_or_blank(peak_hp)} hp at {round_or_blank(peak_hp_rpm)} RPM\n")
        file.write(f"Average torque uncertainty: {round_or_blank(avg_torque_uncertainty)} lb-ft\n")
        file.write(f"Average horsepower uncertainty: {round_or_blank(avg_hp_uncertainty)} hp\n\n")

        file.write("Statistics\n")
        file.write("----------\n")
        file.write(f"Raw force stats: {calculate_stats(force_raw)}\n")
        file.write(f"Zeroed force stats: {calculate_stats(force_zeroed)}\n")
        file.write(f"Filtered force stats: {calculate_stats(force_smoothed)}\n")
        file.write(f"Torque stats: {calculate_stats(torque)}\n")
        file.write(f"RPM stats: {calculate_stats(rpm)}\n")
        file.write(f"Horsepower stats: {calculate_stats(horsepower)}\n")
        file.write(f"Water temperature stats: {calculate_stats(water_temp_c)}\n")
        file.write(f"Flow rate stats: {calculate_stats(flow_gpm)}\n\n")

        file.write("Validation Results\n")
        file.write("------------------\n")
        file.write(f"Overall validation status: {overall_status}\n")

        for message in validation_results:
            file.write(f"- {message}\n")

        file.write("\n")

        file.write("Equations Used\n")
        file.write("--------------\n")
        file.write("Torque = Force * Torque Arm Length\n")
        file.write("Horsepower = Torque * RPM / 5252\n")
        file.write("Torque uncertainty = sqrt((r*uF)^2 + (F*ur)^2)\n")
        file.write("Horsepower relative uncertainty = sqrt((uT/T)^2 + (uRPM/RPM)^2)\n\n")

        file.write("Prototype 2 Interpretation\n")
        file.write("--------------------------\n")
        file.write("This virtual prototype allows the team to test the DAQ and data-processing workflow\n")
        file.write("before the physical dynamometer is fully built. The same processing steps can later\n")
        file.write("be used with real load cell, RPM, temperature, and flow sensor data.\n")


def clean_xy_pair(x_values, y_values):
    """Remove None values from paired x-y data."""
    x_clean = []
    y_clean = []

    for x, y in zip(x_values, y_values):
        if x is not None and y is not None:
            x_clean.append(x)
            y_clean.append(y)

    return x_clean, y_clean


def make_plots(
    folder_name: str,
    time_values: List[float],
    force: List[Optional[float]],
    torque: List[Optional[float]],
    rpm: List[Optional[float]],
    horsepower: List[Optional[float]],
    water_temp_c: List[Optional[float]],
    flow_gpm: List[Optional[float]]
) -> None:
    """Generate standard dyno plots if matplotlib is installed."""
    if not MAKE_PLOTS:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping plots.")
        return

    ensure_folder(folder_name)

    plot_definitions = [
        ("Force vs Time", time_values, force, "Time (s)", "Force (lbf)", "force_vs_time.png"),
        ("RPM vs Time", time_values, rpm, "Time (s)", "RPM", "rpm_vs_time.png"),
        ("Torque vs Time", time_values, torque, "Time (s)", "Torque (lb-ft)", "torque_vs_time.png"),
        ("Horsepower vs Time", time_values, horsepower, "Time (s)", "Horsepower (hp)", "horsepower_vs_time.png"),
        ("Torque vs RPM", rpm, torque, "RPM", "Torque (lb-ft)", "torque_vs_rpm.png"),
        ("Horsepower vs RPM", rpm, horsepower, "RPM", "Horsepower (hp)", "horsepower_vs_rpm.png"),
        ("Water Temperature vs Time", time_values, water_temp_c, "Time (s)", "Water Temperature (C)", "water_temp_vs_time.png"),
        ("Flow Rate vs Time", time_values, flow_gpm, "Time (s)", "Flow Rate (gpm)", "flow_rate_vs_time.png"),
    ]

    for title, x_values, y_values, xlabel, ylabel, filename in plot_definitions:
        x_clean, y_clean = clean_xy_pair(x_values, y_values)

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


def make_raw_filtered_force_plot(
    folder_name: str,
    time_values: List[float],
    raw_force: List[Optional[float]],
    filtered_force: List[Optional[float]]
) -> None:
    """Plot raw force and filtered force together."""
    if not MAKE_PLOTS:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping raw vs filtered plot.")
        return

    ensure_folder(folder_name)

    time_clean = []
    raw_clean = []
    filtered_clean = []

    for t, raw, filtered in zip(time_values, raw_force, filtered_force):
        if t is not None and raw is not None and filtered is not None:
            time_clean.append(t)
            raw_clean.append(raw)
            filtered_clean.append(filtered)

    if not time_clean:
        return

    plt.figure()
    plt.plot(time_clean, raw_clean, label="Raw Force")
    plt.plot(time_clean, filtered_clean, label="Filtered Force")
    plt.title("Raw Force vs Filtered Force")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (lbf)")
    plt.grid(True)
    plt.legend()

    path = os.path.join(folder_name, "raw_vs_filtered_force.png")
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# SINGLE-RUN PROCESSING FUNCTION
# ============================================================


def process_dyno_run(
    input_mode: str,
    combined_csv: str,
    force_csv: str,
    rpm_csv: str,
    output_csv: str,
    summary_txt: str,
    plot_folder: str
) -> Dict[str, Optional[float]]:
    """Process one dyno run."""
    print(f"\nProcessing run from: {combined_csv if input_mode == 'combined' else force_csv}")

    if input_mode == "combined":
        data = load_combined_data(combined_csv)
    elif input_mode == "separate":
        data = load_separate_data(force_csv, rpm_csv)
    else:
        raise ValueError("INPUT_MODE must be 'combined' or 'separate'.")

    time_values = data["time"]
    raw_input = data["raw_input"]
    force_raw = data["force_lbf"]
    rpm = data["rpm"]
    water_temp_c = data["water_temp_c"]
    flow_gpm = data["flow_gpm"]
    slope = data["slope"]
    intercept = data["intercept"]
    r_squared = data["r_squared"]

    # Step 1: zero/tare correction.
    force_zeroed, zero_offset = apply_zero_offset(force_raw, ZERO_FIRST_N_SAMPLES)

    # Step 2: moving-average filtering.
    force_smoothed = moving_average(force_zeroed, MOVING_AVERAGE_WINDOW)

    # Step 3: torque calculation.
    torque = []
    for force_value in force_smoothed:
        torque.append(calculate_torque(force_value))

    # Step 4: horsepower calculation.
    horsepower = []
    for torque_value, rpm_value in zip(torque, rpm):
        horsepower.append(calculate_horsepower(torque_value, rpm_value))

    # Step 5: torque uncertainty calculation.
    torque_uncertainty = []
    for force_value in force_smoothed:
        torque_uncertainty.append(calculate_torque_uncertainty(force_value))

    # Step 6: horsepower uncertainty calculation.
    hp_uncertainty = []
    for torque_value, rpm_value, torque_unc_value in zip(torque, rpm, torque_uncertainty):
        hp_uncertainty.append(
            calculate_hp_uncertainty(torque_value, rpm_value, torque_unc_value)
        )

    # Step 7: validation checks.
    validation_results = evaluate_validation_checks(
        force=force_smoothed,
        rpm=rpm,
        hp_uncertainty=hp_uncertainty,
        water_temp=water_temp_c,
        flow_gpm=flow_gpm
    )

    # Step 8: build output rows with safety flags.
    processed_rows = []

    for i in range(len(time_values)):
        safety_flag = create_safety_flag(
            force_lbf=force_smoothed[i],
            rpm=rpm[i],
            water_temp_c=water_temp_c[i],
            flow_gpm=flow_gpm[i]
        )

        processed_rows.append({
            "sample": i,
            "time_s": round_or_blank(time_values[i]),
            "raw_input": round_or_blank(raw_input[i]),
            "force_raw_lbf": round_or_blank(force_raw[i]),
            "force_zeroed_lbf": round_or_blank(force_zeroed[i]),
            "force_filtered_lbf": round_or_blank(force_smoothed[i]),
            "torque_lbft": round_or_blank(torque[i]),
            "rpm": round_or_blank(rpm[i]),
            "horsepower": round_or_blank(horsepower[i]),
            "torque_uncertainty_lbft": round_or_blank(torque_uncertainty[i]),
            "horsepower_uncertainty_hp": round_or_blank(hp_uncertainty[i]),
            "water_temp_c": round_or_blank(water_temp_c[i]),
            "flow_gpm": round_or_blank(flow_gpm[i]),
            "safety_flag": safety_flag
        })

    # Step 9: write processed CSV.
    write_csv_file(output_csv, processed_rows)

    # Step 10: write summary report.
    input_file_name = combined_csv if input_mode == "combined" else f"{force_csv}, {rpm_csv}"

    write_summary_report(
        summary_txt=summary_txt,
        input_file=input_file_name,
        zero_offset=zero_offset,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        force_raw=force_raw,
        force_zeroed=force_zeroed,
        force_smoothed=force_smoothed,
        torque=torque,
        rpm=rpm,
        horsepower=horsepower,
        torque_uncertainty=torque_uncertainty,
        hp_uncertainty=hp_uncertainty,
        water_temp_c=water_temp_c,
        flow_gpm=flow_gpm,
        validation_results=validation_results
    )

    # Step 11: generate plots.
    make_plots(
        folder_name=plot_folder,
        time_values=time_values,
        force=force_smoothed,
        torque=torque,
        rpm=rpm,
        horsepower=horsepower,
        water_temp_c=water_temp_c,
        flow_gpm=flow_gpm
    )

    # Step 12: generate raw vs filtered plot.
    make_raw_filtered_force_plot(
        folder_name=plot_folder,
        time_values=time_values,
        raw_force=force_zeroed,
        filtered_force=force_smoothed
    )

    # Step 13: return key metrics for multi-run comparison.
    peak_torque, peak_torque_rpm = find_peak(torque, rpm)
    peak_hp, peak_hp_rpm = find_peak(horsepower, rpm)

    torque_stats = calculate_stats(torque)
    hp_stats = calculate_stats(horsepower)
    rpm_stats = calculate_stats(rpm)
    flow_stats = calculate_stats(flow_gpm)
    temp_stats = calculate_stats(water_temp_c)

    overall_status = get_overall_validation_status(validation_results)

    print(f"Processed CSV saved: {output_csv}")
    print(f"Summary report saved: {summary_txt}")
    print(f"Overall validation status: {overall_status}")

    return {
        "output_csv": output_csv,
        "summary_txt": summary_txt,
        "peak_torque_lbft": peak_torque,
        "peak_torque_rpm": peak_torque_rpm,
        "peak_hp": peak_hp,
        "peak_hp_rpm": peak_hp_rpm,
        "avg_torque_lbft": torque_stats["avg"],
        "avg_hp": hp_stats["avg"],
        "max_rpm": rpm_stats["max"],
        "min_flow_gpm": flow_stats["min"],
        "max_water_temp_c": temp_stats["max"],
        "overall_status": overall_status
    }


# ============================================================
# MULTI-RUN COMPARISON FUNCTIONS
# ============================================================


def load_processed_run(filename: str):
    """Load a processed dyno CSV for multi-run comparison."""
    rows = read_csv_file(filename)

    rpm_col = find_column(rows, ["rpm"])
    torque_col = find_column(rows, ["torque_lbft"])
    hp_col = find_column(rows, ["horsepower"])

    rpm = extract_numeric_column(rows, rpm_col)
    torque = extract_numeric_column(rows, torque_col)
    horsepower = extract_numeric_column(rows, hp_col)

    return rpm, torque, horsepower


def compare_multiple_runs(
    processed_files: List[str],
    comparison_csv: str,
    comparison_plot_folder: str
) -> None:
    """Compare multiple processed dyno runs."""
    comparison_rows = []

    for filename in processed_files:
        if not os.path.exists(filename):
            continue

        rpm, torque, horsepower = load_processed_run(filename)

        peak_torque, peak_torque_rpm = find_peak(torque, rpm)
        peak_hp, peak_hp_rpm = find_peak(horsepower, rpm)

        torque_stats = calculate_stats(torque)
        hp_stats = calculate_stats(horsepower)
        rpm_stats = calculate_stats(rpm)

        comparison_rows.append({
            "run_file": filename,
            "peak_torque_lbft": round_or_blank(peak_torque),
            "peak_torque_rpm": round_or_blank(peak_torque_rpm),
            "peak_hp": round_or_blank(peak_hp),
            "peak_hp_rpm": round_or_blank(peak_hp_rpm),
            "avg_torque_lbft": round_or_blank(torque_stats["avg"]),
            "avg_hp": round_or_blank(hp_stats["avg"]),
            "max_rpm": round_or_blank(rpm_stats["max"])
        })

    if comparison_rows:
        write_csv_file(comparison_csv, comparison_rows)
        print(f"Multi-run comparison saved: {comparison_csv}")

    if not MAKE_PLOTS:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping multi-run plots.")
        return

    ensure_folder(comparison_plot_folder)

    # Torque vs RPM comparison.
    plt.figure()

    for filename in processed_files:
        if not os.path.exists(filename):
            continue

        rpm, torque, horsepower = load_processed_run(filename)
        rpm_clean, torque_clean = clean_xy_pair(rpm, torque)

        if rpm_clean and torque_clean:
            label = os.path.splitext(os.path.basename(filename))[0]
            plt.plot(rpm_clean, torque_clean, label=label)

    plt.title("Multi-Run Torque vs RPM Comparison")
    plt.xlabel("RPM")
    plt.ylabel("Torque (lb-ft)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(comparison_plot_folder, "multi_run_torque_vs_rpm.png"), dpi=200)
    plt.close()

    # Horsepower vs RPM comparison.
    plt.figure()

    for filename in processed_files:
        if not os.path.exists(filename):
            continue

        rpm, torque, horsepower = load_processed_run(filename)
        rpm_clean, hp_clean = clean_xy_pair(rpm, horsepower)

        if rpm_clean and hp_clean:
            label = os.path.splitext(os.path.basename(filename))[0]
            plt.plot(rpm_clean, hp_clean, label=label)

    plt.title("Multi-Run Horsepower vs RPM Comparison")
    plt.xlabel("RPM")
    plt.ylabel("Horsepower (hp)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(comparison_plot_folder, "multi_run_horsepower_vs_rpm.png"), dpi=200)
    plt.close()

    print(f"Multi-run plots saved in folder: {comparison_plot_folder}")


# ============================================================
# MAIN WORKFLOW
# ============================================================


def main():
    """Main program workflow."""
    print("Starting Prototype 2 virtual dyno data processor...")

    ensure_folder(VIRTUAL_DATA_FOLDER)
    ensure_folder(PROCESSED_DATA_FOLDER)
    ensure_folder(PLOT_ROOT_FOLDER)

    processed_files = []

    # Mode A: Virtual multi-run prototype.
    if GENERATE_VIRTUAL_DATA and PROCESS_MULTIPLE_VIRTUAL_RUNS:
        for run_number in range(1, VIRTUAL_RUN_COUNT + 1):
            virtual_csv = os.path.join(
                VIRTUAL_DATA_FOLDER,
                f"virtual_dyno_data_run{run_number}.csv"
            )

            output_csv = os.path.join(
                PROCESSED_DATA_FOLDER,
                f"processed_dyno_data_run{run_number}.csv"
            )

            summary_txt = os.path.join(
                PROCESSED_DATA_FOLDER,
                f"dyno_summary_report_run{run_number}.txt"
            )

            plot_folder = os.path.join(
                PLOT_ROOT_FOLDER,
                f"run{run_number}"
            )

            generate_virtual_test_data(
                filename=virtual_csv,
                duration_s=20.0,
                sample_rate_hz=20,
                run_number=run_number,
                inject_warning_case=INJECT_VIRTUAL_WARNING_CASE
            )

            process_dyno_run(
                input_mode="combined",
                combined_csv=virtual_csv,
                force_csv=FORCE_CSV,
                rpm_csv=RPM_CSV,
                output_csv=output_csv,
                summary_txt=summary_txt,
                plot_folder=plot_folder
            )

            processed_files.append(output_csv)

        comparison_csv = os.path.join(PROCESSED_DATA_FOLDER, "multi_run_comparison.csv")
        comparison_plot_folder = os.path.join(PLOT_ROOT_FOLDER, "multi_run_comparison")

        compare_multiple_runs(
            processed_files=processed_files,
            comparison_csv=comparison_csv,
            comparison_plot_folder=comparison_plot_folder
        )

    # Mode B: Virtual single-run prototype.
    elif GENERATE_VIRTUAL_DATA and not PROCESS_MULTIPLE_VIRTUAL_RUNS:
        virtual_csv = os.path.join(VIRTUAL_DATA_FOLDER, "virtual_dyno_data_single_run.csv")

        generate_virtual_test_data(
            filename=virtual_csv,
            duration_s=20.0,
            sample_rate_hz=20,
            run_number=1,
            inject_warning_case=INJECT_VIRTUAL_WARNING_CASE
        )

        process_dyno_run(
            input_mode="combined",
            combined_csv=virtual_csv,
            force_csv=FORCE_CSV,
            rpm_csv=RPM_CSV,
            output_csv=OUTPUT_CSV,
            summary_txt=SUMMARY_TXT,
            plot_folder=PLOT_ROOT_FOLDER
        )

    # Mode C: Real or manually prepared CSV data.
    else:
        process_dyno_run(
            input_mode=INPUT_MODE,
            combined_csv=COMBINED_CSV,
            force_csv=FORCE_CSV,
            rpm_csv=RPM_CSV,
            output_csv=OUTPUT_CSV,
            summary_txt=SUMMARY_TXT,
            plot_folder=PLOT_ROOT_FOLDER
        )

    print("\nPrototype 2 processing complete.")


if __name__ == "__main__":
    main()
