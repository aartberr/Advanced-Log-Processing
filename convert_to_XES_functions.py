# .XES transformation functions
from pathlib import Path
import os
import linecache
import csv
from datetime import datetime, timedelta, timezone
from LILAC_main.benchmark.evaluation.settings import benchmark_settings
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import re
from functools import lru_cache

PLACEHOLDER_PATTERN = re.compile(r"<([^<>]+)>") # Field names are expected to be in the format <FieldName> in the log format string
OPTIONAL_BRACKET_FIELD_PATTERN = re.compile(r"\(\[<([^<>]+)>\]\)\?")
OPTIONAL_PAREN_WRAPPED_FIELD_PATTERN = re.compile(r"\(([^<>()]*)\(<([^<>]+)>\)([^<>()]*)\)\?")
LITERAL_PID_BRACKET_PATTERN = re.compile(r"\[pid\]", re.IGNORECASE)
TIME_FORMAT_CANDIDATES = [
    "%m-%d %H:%M:%S.%f",
    "%m-%d %H:%M:%S",
    "%H:%M:%S,%f",
    "%H:%M:%S.%f",
    "%H:%M:%S",
    "%H%M%S",
    "%H:%M",
]
DATE_FORMAT_CANDIDATES = [
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%y/%m/%d",
    "%y%m%d",
    "%m-%d",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%b %d",
    "%d %b",
]
MONTH_NAME_PATTERN = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
DATE_WITH_YEAR_PATTERNS = [
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%y/%m/%d",
    "%y%m%d",
]

# Convert escaped literals (e.g. \[ \] \( \) \|) to plain literals.
def normalize_inferred_log_format(log_format):
    if not log_format:
        return log_format
    return re.sub(r"\\([\[\]\(\)\|{}])", r"\1", log_format)


def infer_timestamp_from_raw_line(line, base_date, day_offset, prev_time_fragment):
    """Infer timestamp directly from raw line when format-based extraction fails."""
    month_day_time = re.search(
        rf"\b(?P<month>{MONTH_NAME_PATTERN})\s+(?P<day>\d{{1,2}})\s+(?P<time>\d{{2}}:\d{{2}}:\d{{2}}(?:[\.,]\d+)?)\b",
        line,
    )
    if month_day_time:
        inferred_year = infer_year_from_raw_line(line, base_date.year)
        event_ts = try_parse_month_day_time(
            month_day_time.group("month"),
            month_day_time.group("day"),
            month_day_time.group("time"),
            inferred_year,
        )
        if event_ts:
            return event_ts, day_offset, prev_time_fragment

    return None, day_offset, prev_time_fragment


def try_parse_month_day_time(month_token, day_token, time_token, base_year):
    if not month_token or not day_token or not time_token:
        return None

    month_day_time = f"{month_token} {day_token} {time_token}"
    month_day_formats = [
        "%b %d %H:%M:%S,%f",
        "%b %d %H:%M:%S.%f",
        "%b %d %H:%M:%S",
        "%b %d %H:%M",
    ]
    for fmt in month_day_formats:
        try:
            parsed = datetime.strptime(month_day_time, fmt)
            return parsed.replace(year=base_year)
        except ValueError:
            continue
    return None


def infer_year_from_date_text(date_text, fallback_year):
    if not date_text:
        return fallback_year

    stripped = date_text.strip()
    for fmt in DATE_WITH_YEAR_PATTERNS:
        try:
            return datetime.strptime(stripped, fmt).year
        except ValueError:
            continue
    return fallback_year


def infer_year_from_raw_line(line, fallback_year):
    for candidate in re.findall(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{2}/\d{2}/\d{2}\b|\b\d{6}\b", line):
        parsed_year = infer_year_from_date_text(candidate, None)
        if parsed_year is not None:
            return parsed_year
    return fallback_year

@lru_cache(maxsize=128)
# A function to count lines in a file, even for large files, by reading in binary mode and counting newline bytes
def count_lines(path, chunk_size=1024 * 1024):
    total = 0
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size == 0:
            return 1
        f.seek(0)

        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total += chunk.count(b'\n')

        f.seek(-1, os.SEEK_END)
        if f.read(1) != b'\n':
            total += 1
    return total

# A function to get the first and last non-empty line of a log file
def get_first_and_last_line(log_file_path):
    try:
        log_path = os.fspath(log_file_path)
        total_lines = count_lines(log_path)
        if total_lines == 0:
            print(f"{log_path} appears to be empty.")
            return

        first = 1
        # Move forward until we find the first non-empty line
        while first <= total_lines and not linecache.getline(log_path, first).strip():
            first += 1

        last = total_lines
        # Move backward until we find the last non-empty line
        while last >= first and not linecache.getline(log_path, last).strip():
            last -= 1

        if first > total_lines or last < first:
            print(f"{log_path} appears to be empty or only whitespace.")
            return

        first_line = linecache.getline(log_path, first).rstrip()
        last_line = linecache.getline(log_path, last).rstrip()  
        return first_line, last_line
    except Exception as e:
        print(f"Error reading {log_path}: {e}")

# A function to convert a log format to a regex pattern with named groups, where placeholders like <Time> become (?P<Time>.+?)
def compile_log_format_pattern(log_format):
    log_format = normalize_inferred_log_format(log_format)
    optional_tokens = {}

    # Handle optional fields that are in parentheses
    def replace_optional_paren_wrapped_field(match):
        prefix = re.escape(match.group(1))
        field = match.group(2)
        suffix = re.escape(match.group(3))
        token = f"__OPTIONAL_FIELD_{len(optional_tokens)}__"
        optional_tokens[token] = fr"(?:{prefix}\((?P<{field}>.+?)\){suffix})?"
        return token

    def replace_optional_bracket_field(match):
        field = match.group(1)
        token = f"__OPTIONAL_FIELD_{len(optional_tokens)}__"
        optional_tokens[token] = fr"(?:\[(?P<{field}>.+?)\])?"
        return token

    def replace_literal_pid_bracket(match):
        token = f"__OPTIONAL_FIELD_{len(optional_tokens)}__"
        optional_tokens[token] = r"\[(?P<pid>.+?)\]"
        return token

    rewritten_format = OPTIONAL_PAREN_WRAPPED_FIELD_PATTERN.sub(replace_optional_paren_wrapped_field, log_format)
    rewritten_format = OPTIONAL_BRACKET_FIELD_PATTERN.sub(replace_optional_bracket_field, rewritten_format)
    rewritten_format = LITERAL_PID_BRACKET_PATTERN.sub(replace_literal_pid_bracket, rewritten_format)

    parts = []
    cursor = 0
    for match in PLACEHOLDER_PATTERN.finditer(rewritten_format):
        literal = re.escape(rewritten_format[cursor:match.start()])
        for token, replacement in optional_tokens.items():
            literal = literal.replace(re.escape(token), replacement)

        field = match.group(1)
        parts.append(literal)
        if field == '*':
            parts.append(r"(?:.+?)")
        else:
            parts.append(fr"(?P<{field}>.+?)")
        cursor = match.end()

    literal = re.escape(rewritten_format[cursor:])
    for token, replacement in optional_tokens.items():
        literal = literal.replace(re.escape(token), replacement)
    parts.append(literal)

    return re.compile("^" + ''.join(parts) + "$")

# A function to extract a specific field value from a log line using the provided format
def get_component_from_log_line(log_format, field_name, log_line=None):
    sample_line = log_line
    if sample_line is None:
        print("No log line provided")
    pattern = compile_log_format_pattern(log_format)
    if field_name not in pattern.groupindex:
        print(f"WARNING: Field '{field_name}' not defined in log format.")
        print(f"Available fields: {list(pattern.groupindex.keys())}")
        return None

    match = pattern.match(sample_line)
    if not match:   
        print(f"WARNING: Log line does not match the log format. Line: {sample_line}")
        return None
    field_value = match.group(field_name)
    if field_value is None:
        print(f"WARNING: Field '{field_name}' could not be extracted.")
        return None

    return field_value.strip()

# A function that lists log fields
def list_log_fields(log_format):
    """Return all placeholder names declared in a log format string."""
    pattern = compile_log_format_pattern(log_format)
    return list(pattern.groupindex.keys())

# A function to match a field name in the log format regardless of case, returning the field name as it is in the log format, if found
def find_field_case_insensitive(log_format, desired_field_name):
    desired_lower_case = desired_field_name.lower()
    for field in list_log_fields(log_format):
        if field.lower() == desired_lower_case:
            return field
    return None

# A function to parse a time-of-day string using the known candidate formats
def parse_time_fragment(time_text):
    if time_text is None:
        raise ValueError("Time text is required for parsing")

    stripped = time_text.strip()
    for fmt in TIME_FORMAT_CANDIDATES:
        try:
            return datetime.strptime(stripped, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported time format: {time_text}")

# A function to parse combined date and time strings using the known candidate formats
def parse_datetime_components(date_text, time_text):
    if date_text is None or time_text is None:
        raise ValueError("Both date and time values are required to parse a full timestamp")

    clean_date = date_text.strip()
    clean_time = time_text.strip()
    for date_fmt in DATE_FORMAT_CANDIDATES:
        try:
            parsed_date = datetime.strptime(clean_date, date_fmt).date()
        except ValueError:
            continue

        for time_fmt in TIME_FORMAT_CANDIDATES:
            try:
                parsed_time = datetime.strptime(clean_time, time_fmt).time()
            except ValueError:
                continue

            return datetime.combine(parsed_date, parsed_time)
    raise ValueError(f"Unable to parse date/time combination: '{date_text}' '{time_text}'")

# A function to parse a single string that might contain both date and time information.
def try_parse_full_datetime(value):
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    # Unix epoch support (seconds/milliseconds).
    if stripped.isdigit():
        try:
            if len(stripped) >= 13:
                return datetime.fromtimestamp(int(stripped[:13]) / 1000.0)
            if len(stripped) >= 10:
                return datetime.fromtimestamp(int(stripped[:10]))
        except (ValueError, OSError, OverflowError):
            pass

    try:
        return datetime.fromisoformat(stripped)
    except ValueError:
        pass

    # Known dataset-specific full datetime variants.
    full_datetime_formats = [
        "%a %b %d %H:%M:%S %Y",      # Apache: Sun Dec 04 04:47:44 2005
        "%Y%m%d-%H:%M:%S:%f",        # HealthApp: 20171223-22:15:29:606
        "%Y-%m-%d-%H.%M.%S.%f",      # BGL: 2005-06-03-15.42.50.675872
        "%Y-%m-%d %H:%M:%S,%f",      # Hadoop/Zookeeper style
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%y/%m/%d %H:%M:%S",         # Spark: 17/06/09 20:10:40
    ]
    for fmt in full_datetime_formats:
        try:
            return datetime.strptime(stripped, fmt)
        except ValueError:
            continue

    # Proxifier style: 10.30 16:49:06 (month.day time) without year.
    try:
        prox_dt = datetime.strptime(stripped, "%m.%d %H:%M:%S")
        return prox_dt.replace(year=datetime.now().year)
    except ValueError:
        pass

    tokens = stripped.split()
    for split_idx in range(1, len(tokens)):
        date_candidate = " ".join(tokens[:split_idx])
        time_candidate = " ".join(tokens[split_idx:])
        try:
            return parse_datetime_components(date_candidate, time_candidate)
        except ValueError:
            continue

    return None

# A function to compute duration when only time-of-day information is available
def calculate_duration_without_date(log_file_path, log_format, time_field="Time"):
    pattern = compile_log_format_pattern(log_format)
    prev_time_of_day = None
    first_timestamp = None
    last_timestamp = None
    day_offset = 0

    try:
        with open(log_file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue

                if time_field not in pattern.groupindex:
                    break

                match = pattern.match(line)
                if not match:
                    continue

                time_value = match.group(time_field)
                if time_value is None:
                    continue

                try:
                    parsed_time = parse_time_fragment(time_value)
                except ValueError:
                    continue

                if prev_time_of_day and parsed_time < prev_time_of_day:
                    day_offset += 1

                adjusted_time = parsed_time + timedelta(days=day_offset)
                if first_timestamp is None:
                    first_timestamp = adjusted_time
                last_timestamp = adjusted_time
                prev_time_of_day = parsed_time
    except FileNotFoundError:
        print(f"WARNING: Log file {log_file_path} not found while estimating duration")
    except OSError as exc:
        print(f"WARNING: Could not read {log_file_path}: {exc}")

    if first_timestamp and last_timestamp:
        return last_timestamp - first_timestamp
    return None

# A function to compute log duration when date and time information is available
def compute_duration_with_dates(date_start, time_start, date_end, time_end):
    """Calculate duration when both date and time fields are available."""
    if None in (date_start, time_start, date_end, time_end):
        return None

    try:
        start_dt = parse_datetime_components(date_start, time_start)
        end_dt = parse_datetime_components(date_end, time_end)
        return end_dt - start_dt
    except ValueError as exc:
        print(f"WARNING: Unable to compute duration with provided dates: {exc}")
        return None

# A function to compute duration when only timestamp information is available
def compute_duration_with_timestamp(time_start, time_end):
    """Compute the elapsed time between two time-of-day strings, handling day rollover."""
    if time_start is None or time_end is None:
        return None

    try:
        start_time = parse_time_fragment(time_start)
        end_time = parse_time_fragment(time_end)
    except ValueError as exc:
        print(f"WARNING: Unable to parse timestamps: {exc}")
        return None

    if end_time < start_time:
        end_time = end_time + timedelta(days=1)

    return end_time - start_time

# A function to compute whole log duration based on available fields, trying date+time first, timestamp or time-only
def compute_log_duration(log_file_path, log_format):
    first_line, last_line = get_first_and_last_line(log_file_path)
    time_field_name = None
    timestamp_field_name = None
    date_field_name = None
    time_first_log = time_last_log = None
    timestamp_first_log = timestamp_last_log = None
    if not first_line or not last_line:
        raise RuntimeError(f"Unable to retrieve boundary lines from {log_file_path}")

    # Resolve the canonical time field from the log format
    date_field_name = find_field_case_insensitive(log_format, "Date")
    if not date_field_name:
        timestamp_field_name = find_field_case_insensitive(log_format, "Timestamp")
        if timestamp_field_name:
            timestamp_first_log = get_component_from_log_line(log_format, timestamp_field_name, first_line)
            timestamp_last_log = get_component_from_log_line(log_format, timestamp_field_name, last_line)
        if not timestamp_field_name:
            time_field_name = find_field_case_insensitive(log_format, "Time")
            if time_field_name:
                time_first_log = get_component_from_log_line(log_format, time_field_name, first_line)
                time_last_log = get_component_from_log_line(log_format, time_field_name, last_line)
    else:
        time_field_name = find_field_case_insensitive(log_format, "Time") or find_field_case_insensitive(log_format, "Timestamp")
        if time_field_name:
            time_first_log = get_component_from_log_line(log_format, time_field_name, first_line)
            time_last_log = get_component_from_log_line(log_format, time_field_name, last_line)
    
    if not date_field_name and not timestamp_field_name and not time_field_name:
        raise RuntimeError("No <Time> or <Timestamp> placeholder defined in the log format; cannot compute durations")

    log_duration = None

    if date_field_name and time_first_log and time_last_log:
        date_first_log = get_component_from_log_line(log_format, date_field_name, first_line)
        date_last_log = get_component_from_log_line(log_format, date_field_name, last_line)
        log_duration = compute_duration_with_dates(date_first_log, time_first_log, date_last_log, time_last_log)
    elif timestamp_field_name and timestamp_first_log and timestamp_last_log: #if there is no date but there are timestamps
        log_duration = compute_duration_with_timestamp(timestamp_first_log, timestamp_last_log)
        
    if log_duration is None:
        fallback_field = time_field_name or timestamp_field_name or find_field_case_insensitive(log_format, "Time") or find_field_case_insensitive(log_format, "Timestamp")
        if fallback_field:
            log_duration = calculate_duration_without_date(log_file_path, log_format, fallback_field)
        
    if log_duration is not None:
        return log_duration
    else:
        print(f"WARNING: Unable to compute log duration from available fields. Fields are: {list_log_fields(log_format)}")
        return None

# A function to compute average time between log messages based on the log duration and total number of log lines
def compute_average_time_between_logs(log_file_path, log_format):
    log_duration = compute_log_duration(log_file_path, log_format)
    if log_duration is None:
        print("Unable to compute average time between logs without log duration")
        return None

    total_lines = count_lines(log_file_path)

    average_time = log_duration / (total_lines - 1)
    print(f"Average time between logs: {average_time}")
    return average_time


def infer_event_timestamps_for_log(log_file_path, log_format):
    log_format = normalize_inferred_log_format(log_format)
    pattern = compile_log_format_pattern(log_format)
    field_names = list_log_fields(log_format)

    date_field = find_field_case_insensitive(log_format, "Date")
    month_field = find_field_case_insensitive(log_format, "Month")
    day_field = find_field_case_insensitive(log_format, "Day")
    time_field = find_field_case_insensitive(log_format, "Time")
    timestamp_field = find_field_case_insensitive(log_format, "Timestamp")

    base_date = datetime.now(timezone.utc).date()
    day_offset = 0
    prev_time_fragment = None
    inferred_timestamps = []

    with open(log_file_path, "r", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            match = pattern.match(line)
            if not match:
                event_ts, day_offset, prev_time_fragment = infer_timestamp_from_raw_line(
                    line, base_date, day_offset, prev_time_fragment
                )
                if event_ts:
                    inferred_timestamps.append(event_ts)
                continue

            field_values = {field: (match.group(field) or "").strip() for field in field_names}
            event_ts, day_offset, prev_time_fragment = resolve_event_timestamp(
                field_values,
                date_field,
                month_field,
                day_field,
                time_field,
                timestamp_field,
                base_date,
                day_offset,
                prev_time_fragment,
            )

            if event_ts:
                inferred_timestamps.append(event_ts)

    return inferred_timestamps

# A function to suggest a time window for grouping log messages into traces based on log duration and average time between messages
def suggest_time_window_for_traces(log_file_path, log_format, num_of_average_intervals=10):
    inferred_timestamps = infer_event_timestamps_for_log(log_file_path, log_format)
    if len(inferred_timestamps) < 2:
        suggested_window = timedelta(minutes=5)
        print(f"Suggested time window for traces: {suggested_window}")
        return suggested_window

    total_span = inferred_timestamps[-1] - inferred_timestamps[0]
    if total_span <= timedelta(0):
        suggested_window = timedelta(minutes=5)
        print(f"Suggested time window for traces: {suggested_window}")
        return suggested_window

    average_time_between_messages = total_span / (len(inferred_timestamps) - 1)
    baseline_window = average_time_between_messages * num_of_average_intervals
    duration_cap = total_span / 2
    suggested_window = min(baseline_window, duration_cap)

    if suggested_window <= timedelta(0):
        suggested_window = timedelta(minutes=5)
        print(f"Suggested time window for traces: {suggested_window}")
        return suggested_window

    print(f"Average time between logs: {average_time_between_messages}")

    print(f"Suggested time window for traces: {suggested_window}")
    return suggested_window


def resolve_event_timestamp(field_values, date_field, month_field, day_field, time_field, timestamp_field, base_date, day_offset, prev_time_fragment):
    if time_field and field_values.get(time_field):
        month_token = field_values.get(month_field) or (field_values.get(date_field) if day_field else "")
        day_token = field_values.get(day_field) or (field_values.get(date_field) if month_field else "")
        event_year = infer_year_from_date_text(field_values.get(date_field), base_date.year)
        event_ts = try_parse_month_day_time(month_token, day_token, field_values.get(time_field), event_year)
        if event_ts:
            return event_ts, day_offset, prev_time_fragment

    return get_event_timestamp(
        field_values,
        date_field,
        time_field,
        timestamp_field,
        base_date,
        day_offset,
        prev_time_fragment,
    )

# A function to resolve event timestamp and return updated rollover state
def get_event_timestamp(raw_values, date_field, time_field, timestamp_field, base_date, day_offset, prev_time_fragment):
    date_val = raw_values.get(date_field)
    time_val = raw_values.get(time_field)
    timestamp_val = raw_values.get(timestamp_field)

    if date_val and time_val:
        try:
            return parse_datetime_components(date_val, time_val), day_offset, prev_time_fragment
        except ValueError:
            return None, day_offset, prev_time_fragment

    if timestamp_val:
        ts = timestamp_val.strip()
        parsed_full = try_parse_full_datetime(ts)
        if parsed_full:
            return parsed_full, day_offset, prev_time_fragment
        try:
            return datetime.fromisoformat(ts), day_offset, prev_time_fragment
        except ValueError:
            pass
        try:
            frag = parse_time_fragment(ts)
            if prev_time_fragment and frag < prev_time_fragment:
                day_offset += 1
            prev_time_fragment = frag
            return datetime.combine(base_date + timedelta(days=day_offset), frag.time()), day_offset, prev_time_fragment
        except ValueError:
            return None, day_offset, prev_time_fragment

    if time_val:
        parsed_full = try_parse_full_datetime(time_val)
        if parsed_full:
            return parsed_full, day_offset, prev_time_fragment
        try:
            frag = parse_time_fragment(time_val)
        except ValueError:
            return None, day_offset, prev_time_fragment
        if prev_time_fragment and frag < prev_time_fragment:
            day_offset += 1
        prev_time_fragment = frag
        return datetime.combine(base_date + timedelta(days=day_offset), frag.time()), day_offset, prev_time_fragment

    return None, day_offset, prev_time_fragment

# A function to help in .XES creation by assigning each log line to a trace format in a csv file
# structured_log_file is the source CSV; output_csv_file_name is the new target CSV
# CSV fields: trace_id, event_id, timestamp, event, original_log_message, full_log_line, log_level (optional), + attribute fields
def assign_log_line_to_trace(log_file_path, log_format, time_window, structured_log_file, output_csv_file_name):
    log_format = normalize_inferred_log_format(log_format)
    pattern = compile_log_format_pattern(log_format)
    field_names = list_log_fields(log_format)

    date_field = find_field_case_insensitive(log_format, "Date")
    month_field = find_field_case_insensitive(log_format, "Month")
    day_field = find_field_case_insensitive(log_format, "Day")
    time_field = find_field_case_insensitive(log_format, "Time")
    timestamp_field = find_field_case_insensitive(log_format, "Timestamp")
    level_field = find_field_case_insensitive(log_format, "Level") or find_field_case_insensitive(log_format, "Severity")

    if not (date_field or time_field or timestamp_field):
        raise RuntimeError("Log format must define at least one of <Date>, <Time>, or <Timestamp> for trace assignment")

    skip_fields = {date_field, time_field, timestamp_field, level_field}
    skip_fields.discard(None)
    attribute_fields = [f for f in field_names if f not in skip_fields]
    base_header = ["trace_id", "event_id", "timestamp", "event", "original_log_message", "full_log_line"]
    if level_field:
        base_header.append("log_level")
    header = base_header + attribute_fields

    base_date = datetime.now(timezone.utc).date()
    day_offset = 0
    prev_time_fragment = None

    trace_index = 1
    line_index = 0
    trace_start_timestamp = None
    last_event_timestamp = None

    with open(structured_log_file, "r", newline="", encoding="utf-8") as structured_source:
        structured_rows = list(csv.DictReader(structured_source))

    with open(log_file_path, "r", encoding="utf-8") as source, open(output_csv_file_name, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=header)
        writer.writeheader()

        for raw_line in source:
            structured_row = structured_rows[line_index] if line_index < len(structured_rows) else None
            line_index += 1
            if structured_row is None:
                break

            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            match = pattern.match(line)
            if not match:
                event_ts, day_offset, prev_time_fragment = infer_timestamp_from_raw_line(
                    line, base_date, day_offset, prev_time_fragment
                )
                field_values = {}
            else:
                field_values = {field: (match.group(field) or "").strip() for field in field_names}
                event_ts, day_offset, prev_time_fragment = resolve_event_timestamp(
                    field_values,
                    date_field,
                    month_field,
                    day_field,
                    time_field,
                    timestamp_field,
                    base_date,
                    day_offset,
                    prev_time_fragment,
                )

            if event_ts:
                if trace_start_timestamp is None:
                    trace_start_timestamp = event_ts
                else:
                    # Split trace if there is a very large gap between consecutive events
                    if last_event_timestamp and (event_ts - last_event_timestamp) > (time_window * 10):
                        trace_index += 1
                        trace_start_timestamp = event_ts
                    # Or if the current trace span reaches the configured time window
                    elif (event_ts - trace_start_timestamp) >= time_window:
                        trace_index += 1
                        trace_start_timestamp = event_ts

                last_event_timestamp = event_ts

            row = {
                "trace_id": f"trace_{trace_index}",
                "event_id": structured_row["EventId"],
                "timestamp": event_ts.isoformat() if event_ts else "",
                "event": structured_row["EventTemplate"],
                "original_log_message": structured_row.get("Content"),
                "full_log_line": line,
            }

            if level_field:
                row["log_level"] = field_values.get(level_field, "")

            for field in attribute_fields:
                row[field] = field_values.get(field, "")

            writer.writerow(row)

# A function to convert the generated CSV file with traces into .XES format
def convert_csv_to_xes(input_csv_path, output_xes_path):
    df = pd.read_csv(input_csv_path, keep_default_na=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).copy()

    for column in df.columns:
        if column == "timestamp":
            continue
        df[column] = df[column].replace({pd.NA: "", "nan": "", "NaN": "", "NAN": ""})
        df[column] = df[column].where(pd.notna(df[column]), "")
        df[column] = df[column].astype(str)

    df = df.rename(columns={
        "trace_id": "case:concept:name",
        "event": "concept:name",
        "timestamp": "time:timestamp"
    })

    df = pm4py.format_dataframe(df, case_id="case:concept:name",activity_key="concept:name",timestamp_key="time:timestamp")

    event_log = log_converter.apply(df)

    xes_exporter.apply(event_log, str(output_xes_path))

    print(f"XES file created: {output_xes_path}")
