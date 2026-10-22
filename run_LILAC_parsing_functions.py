# LILAC Parsing Functions
from pathlib import Path
import os
import ollama
import json
import csv
import shutil
from LILAC_main.benchmark.evaluation.settings import benchmark_settings
import subprocess
import sys


def resolve_data_type(data_type=None):
    resolved = (data_type or os.environ.get("DATA_TYPE", "2k")).strip().lower()
    return "full" if resolved == "full" else "2k"


def normalize_log_file_for_data_type(log_file, data_type):
    normalized = str(log_file).replace("\\", "/")
    # Remove accidental dataset-prefix folders regardless of their position.
    for marker in ("/datasets_2k/", "/datasets_full/", "datasets_2k/", "datasets_full/"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]

    for marker in ("/2k_dataset/", "/full_dataset/", "2k_dataset/", "full_dataset/"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]

    if data_type == "full":
        normalized = normalized.replace("_2k.log", "_full.log")
    else:
        normalized = normalized.replace("_full.log", "_2k.log")

    return normalized

# LILAC Installation check and dependency installation
def install_lilac_deps():
    # Check if LILAC directory exists
    lilac_dir = Path("LILAC_main")  
    if not lilac_dir.exists() or not lilac_dir.is_dir():
        print(f"LILAC-main directory not found. Please download from https://github.com/logpai/LILAC")
        return
    # Check if LILAC requirements are installed
    SETUP_FLAGS = {
        "LILAC_dependencies": Path(".lilac_installed"),
    }
    # Install LILAC requirements from requirements.txt (first-time only)
    req_file = Path("./LILAC_main/requirements.txt")
    
    # Check if already installed
    if SETUP_FLAGS["LILAC_dependencies"].exists():
        print("LILAC dependencies already installed (skipping)")
        return
    
    print("First-time setup: Installing LILAC dependencies...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)],capture_output=True,text=True)
    
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: pip install failed with code {result.returncode}")
    
    # Create marker file to skip future installs
    SETUP_FLAGS["LILAC_dependencies"].write_text("Setup complete\n")
    print("Setup complete. Marker file created: .lilac_installed")

# A function to read sample log lines from a file
def read_sample_logs(log_file_name, num_samples = 10):
    try:
        with open(log_file_name, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f if line.strip()]
            return lines[:num_samples]
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []

# A function to read sampled examples from LILAC's sampled_examples directory for a given dataset and shot size
def read_sampled_examples(dataset_name, num_samples=10, shot_size=16, data_type=None):
    resolved_data_type = resolve_data_type(data_type)
    sample_path = Path("LILAC_main") / f"full_dataset" / "sampled_examples" / dataset_name / f"{shot_size}shot.json"
    samples = []

    def load_fallback_samples():
        configured_log = benchmark_settings[dataset_name]["log_file"]
        normalized_log = normalize_log_file_for_data_type(configured_log, resolved_data_type)
        fallback_log_path = Path("LILAC_main") / f"{resolved_data_type}_dataset" / Path(normalized_log)
        return read_sample_logs(str(fallback_log_path), num_samples=num_samples)

    if not sample_path.exists():
        print(f"Sampled examples not found at {sample_path}. Fetching first 10 samples from log file instead.")
        return load_fallback_samples()

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"json decode error for line: {line}. Fetching first 10 samples from log file instead.")
                    fallback_samples = load_fallback_samples()
                    if fallback_samples:
                        return fallback_samples
                    continue

                query = record.get("query")
                if query:
                    samples.append(query.strip())

                if len(samples) >= num_samples:
                    break
    except Exception as e:
        print(f"Error reading sampled examples for {dataset_name}: {e}. Fetching first 10 samples from log file instead.")
        fallback_samples = load_fallback_samples()
        if fallback_samples:
            return fallback_samples

    return samples


def ensure_lilac_sample_file(dataset_name, shot_size, example_size=None, data_type=None):
    resolved_data_type = resolve_data_type(data_type)
    sample_dir = Path("LILAC_main") / "full_dataset" / "sampled_examples" / dataset_name
    sample_file = sample_dir / f"{shot_size}shot.json"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if sample_file.exists():
        return example_size

    sample_full_dir = Path("LILAC_main") / "full_dataset" / "sampled_examples_full" / dataset_name
    sample_full_file = sample_full_dir / f"{shot_size}shot.json"
    if sample_full_file.exists():
        shutil.copyfile(sample_full_file, sample_file)
        print(f"Created missing sample file from sampled_examples_full: {sample_file}")
        return example_size
    
    configured_log = benchmark_settings.get(dataset_name, {}).get("log_file")
    normalized_log = normalize_log_file_for_data_type(configured_log or f"{dataset_name}/{dataset_name}_{resolved_data_type}.log", resolved_data_type)
    structured_path = Path("LILAC_main") / resolved_data_type / f"{normalized_log}_structured.csv"
    raw_log_path = Path("LILAC_main") / f"{resolved_data_type}_dataset" / normalized_log

    rows_written = 0
    if structured_path.exists():
        with open(structured_path, "r", encoding="utf-8", newline="") as source, open(sample_file, "w", encoding="utf-8") as target:
            reader = csv.DictReader(source)
            for row in reader:
                query = (row.get("Content") or "").strip()
                answer = (row.get("EventTemplate") or "").strip().replace("<*>", "{variables}")
                if not query:
                    continue
                if not answer:
                    answer = query
                target.write(json.dumps({"query": query, "answer": answer}) + "\n")
                rows_written += 1
                if rows_written >= shot_size:
                    break

        if rows_written:
            print(f"Created missing sample file from structured logs: {sample_file}")
            return 0

    fallback_logs = read_sample_logs(str(raw_log_path), num_samples=shot_size)
    if fallback_logs:
        with open(sample_file, "w", encoding="utf-8") as target:
            for line in fallback_logs:
                target.write(json.dumps({"query": line, "answer": line}) + "\n")
        print(f"Created missing sample file from raw logs: {sample_file}")
        return 0

    print(f"WARNING: Could not create sampled examples for {dataset_name} shot={shot_size}")
    return None
    
# A function to build a formatted string of known log format examples from settings.py
def build_format_examples(data_type=None):
    resolved_data_type = resolve_data_type(data_type)
    examples = "Here are examples of known log formats:\n\n"
    
    for dataset_name, settings in list(benchmark_settings.items()): 
        log_format = settings['log_format']
        regex = settings.get('regex', [])
        st = settings.get('st')
        depth = settings.get('depth')

        examples += "for this dataset:\n"
        examples += f"Dataset: {dataset_name}\n"
        examples += f"  Format: {log_format}\n"
        if regex:
            examples += f"  Regex: {regex}\n"
        examples += f"  Similarity Threshold: {st}\n"
        examples += f"  Depth: {depth}\n"
        examples += "  Sampled logs:\n"
        print(f"Reading sampled examples for {dataset_name}...")
        sample_logs = read_sampled_examples(dataset_name, num_samples=10, data_type=resolved_data_type)
        for log in sample_logs:
            examples += f"    {log}\n"  
        if not sample_logs:
            examples += "    [no sampled examples found]\n"
        examples += "\n"
    
    return examples

# A function to get log format and regex patterns from LILAC for a given dataset using LLM
def get_log_format_and_regex_from_lilac(log_path, model="llama3", data_type=None):
    # Read samples
    samples = read_sample_logs(str(log_path), num_samples=30)
    if not samples:
        return {"error": "Could not read log file"}
    
    # Build prompt
    format_examples = build_format_examples(data_type=data_type)
    
    prompt = f"""You are a log format expert. Analyze the following sample log lines and infer their format.

{format_examples}

Now analyze these NEW log samples that have the same format with each other but are from an unseen dataset:
{'\n'.join(samples)}

Based on the examples above and these new samples, determine:
1. The log format (using <Field> notation like <Date>, <Time>, <Level>, <Component>, <Content>, <PID> etc.)
2. Any regex patterns needed to extract variable values (like IPs, numbers, paths)
3. The similarity threshold (a float between 0 and 1) that best matches this dataset, by understanding how similar the logs of each other on the new dataset
4. The parsing depth (an integer) that should be used for this dataset, referencing similar datasets

Respond in this exact JSON format:
{{
  "log_format": "<inferred format here>",
  "regex": ["regex1", "regex2"],
    "similarity_threshold": similarity_threshold_value,
    "depth": depth_value,
  "reasoning": "Brief explanation"
}}

IMPORTANT:
- Respond ONLY with valid JSON, no other text before or after the JSON.
- The JSON must be directly parseable by the Python json module without extra fixes.
- Fields in the log format should always be enclosed in angle brackets <> and match correctly with the log lines
- Important: The log format contains ONLY the fields and some special characters (like brackets, colons, dashes) that appear in the log lines.
- Fields should describe a general concept, and not be system specific (e.g. <Date>, <Time>, <Level>, <Component>, <Content>, <PID>)  
- Provide numeric values for "similarity_threshold" (float) and "depth" (integer) with no surrounding quotes.
- No need to copy the exact values from the examples, but use them as a reference to determine appropriate values for the new dataset.
- Date can appear as "03-17" or "Mar 17"/"17 Mar": represent all of these as a single field named <Date>.
"""
    
    try:
        client = ollama.Client()

        print(f"Sending to Ollama model '{model}' for format inference...")
        response = client.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={"temperature": 0}
        )

        result_text = response.get("response", "").strip()
        if not result_text:
            return {"error": "Empty response from Ollama"}

        print("Received response from Ollama:")
        print(result_text)

        result = json.loads(result_text)
        return result
        
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse LLM response: {e}", "raw_response": result_text}
    except Exception as e:
        return {"error": f"Error calling Ollama: {e}"}

# A function to add inferred format and regex for the given dataset on settings.py
def add_dataset_to_settings_file(dataset_name, format_config, data_type=None):
    """
    Add inferred format and regex to settings.py file
    
    Args:
        dataset_name: Name of the dataset
        format_config: Dict with 'log_format' and 'regex' keys
    """
    # Check if dataset already exists in the imported benchmark_settings dict
    if dataset_name in benchmark_settings:
        print(f"WARNING: Dataset '{dataset_name}' already exists in settings.py")
        print(f"Existing datasets: {list(benchmark_settings.keys())}")
        return False
    
    settings_file = Path("LILAC_main/benchmark/evaluation/settings.py")
    
    if not settings_file.exists():
        print(f"ERROR: settings.py not found at {settings_file}")
        return False
    
    log_format = format_config.get('log_format', '')
    regex = format_config.get('regex', [])
    similarity_threshold = format_config.get('similarity_threshold', 0.5)
    depth = format_config.get('depth', 4)

    print(format_config)
    
    resolved_data_type = resolve_data_type(data_type)

    # Read current settings.py content
    content = settings_file.read_text()
    
    # Generate the new entry
    suffix = "full" if resolved_data_type == "full" else "2k"
    new_entry = f"""    '{dataset_name}': {{
        'log_file': '{dataset_name}/{dataset_name}_{suffix}.log',
        'log_format': '{log_format}',
        'regex': {regex},
        'st': {similarity_threshold},
        'depth': {depth}
    }}"""
    
    # Find the closing brace of benchmark_settings dict
    lines = content.split('\n')
    insert_idx = -1
    
    # Find the last "}" that closes the dict
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '}':
            insert_idx = i
            break
    
    if insert_idx == -1:
        print("ERROR: Could not find closing brace of benchmark_settings")
        return False
    
    # Ensure the existing last entry ends with a comma
    prev_idx = insert_idx - 1
    while prev_idx >= 0 and not lines[prev_idx].strip():
        prev_idx -= 1

    if prev_idx >= 0 and not lines[prev_idx].rstrip().endswith(','):
        lines[prev_idx] = lines[prev_idx].rstrip() + ','

    # Insert the new entry before the closing brace
    lines.insert(insert_idx, new_entry)
    updated_content = '\n'.join(lines)
    
    # Write back to settings.py
    settings_file.write_text(updated_content)
    
    print("\n" + "="*60)
    print(f"✓ Successfully added '{dataset_name}' to settings.py")
    print("="*60)
    print("\nNew entry added:")
    print(new_entry)
    print("="*60 + "\n")
    
    return True

# A function to update LILAC's sampling script with the new dataset entry
def update_sampling_configuration(dataset_name, log_file_name, log_format):
    """Ensure the sampling script knows about the new dataset."""
    sampling_file = Path("LILAC_main/sampling/LILAC_sampling_extended.py")

    if not sampling_file.exists():
        print(f"WARNING: Sampling script not found at {sampling_file}")
        return False

    content = sampling_file.read_text()
    lines = content.split('\n')

    try:
        datasets_start = next(i for i, line in enumerate(lines) if line.strip().startswith("datasets = ["))
    except StopIteration:
        print("WARNING: Could not locate datasets list in sampling script")
        return False

    datasets_end = datasets_start
    while datasets_end < len(lines) and ']' not in lines[datasets_end]:
        datasets_end += 1

    dataset_entry = f"'{dataset_name}'"
    dataset_already = any(dataset_entry in line for line in lines[datasets_start:datasets_end])

    if not dataset_already:
        prev_idx = datasets_end - 1
        while prev_idx >= 0 and not lines[prev_idx].strip():
            prev_idx -= 1
        if prev_idx >= 0 and not lines[prev_idx].rstrip().endswith(','):
            lines[prev_idx] = lines[prev_idx].rstrip() + ','

        lines.insert(datasets_end, f"    '{dataset_name}'")
        print(f"Added {dataset_name} to datasets list in sampling script")

    try:
        benchmark_start = next(i for i, line in enumerate(lines) if line.strip().startswith("benchmark = {"))
    except StopIteration:
        print("WARNING: Could not locate benchmark dict in sampling script")
        return False

    brace_balance = 0
    benchmark_end = None
    for idx in range(benchmark_start, len(lines)):
        brace_balance += lines[idx].count('{')
        brace_balance -= lines[idx].count('}')
        if idx > benchmark_start and brace_balance == 0:
            benchmark_end = idx
            break

    if benchmark_end is None:
        print("WARNING: Could not determine end of benchmark dict")
        return False

    benchmark_entry = f"'{dataset_name}':"
    entry_exists = any(benchmark_entry in line for line in lines[benchmark_start:benchmark_end])

    if not entry_exists:
        prev_idx = benchmark_end - 1
        while prev_idx >= 0 and not lines[prev_idx].strip():
            prev_idx -= 1
        if prev_idx >= 0 and not lines[prev_idx].rstrip().endswith(','):
            lines[prev_idx] = lines[prev_idx].rstrip() + ','

        sanitized_format = log_format.replace("'", "\\'")
        new_entry = [
            f"    '{dataset_name}': {{",
            f"        'log_file': '{dataset_name}/{log_file_name}',",
            f"        'log_format': '{sanitized_format}',",
            "    }"
        ]
        lines[benchmark_end:benchmark_end] = new_entry
        print(f"Added {dataset_name} entry to benchmark dict in sampling script")

    sampling_file.write_text('\n'.join(lines))
    return True

# Return resolved log format and regex, inferring via Ollama when missing
def get_log_format(dataset_name, log_file_name, log_path, data_type=None):
    resolved_data_type = resolve_data_type(data_type)
    log_format = None
    regex_pattern = []

    existing_config = benchmark_settings.get(dataset_name)
    if existing_config and existing_config.get("log_format") and "regex" in existing_config:
        print(f"Using existing log format and regex for dataset '{dataset_name}' from settings.py")
        log_format = existing_config["log_format"]
        regex_pattern = existing_config.get("regex", []) or []
    else:
        print(f"No existing log format and regex found for dataset '{dataset_name}' in settings.py; inferring using Ollama...")
        format_config = get_log_format_and_regex_from_lilac(log_path, data_type=resolved_data_type)
        add_dataset_to_settings_file(dataset_name, format_config, data_type=resolved_data_type)

        if isinstance(format_config, dict):
            log_format = format_config.get("log_format")
            regex_pattern = format_config.get("regex", []) or []
        else:
            print(f"WARNING: Unexpected format config returned for {dataset_name}: {format_config}")

    if log_format:
        update_sampling_configuration(dataset_name, log_file_name, log_format)
    else:
        print(f"WARNING: No inferred log_format returned for {dataset_name}; sampling script not updated")

    return log_format

# A function to temporarily remove a dataset entry from settings.py (for testing purposes) and return the original content for restoration later
def remove_dataset_from_settings_temporarily(settings_path, dataset_name):
    if not settings_path.exists():
        print(f"WARNING: settings.py not found at {settings_path}")
        return None, None

    original_content = settings_path.read_text(encoding="utf-8")
    lines = original_content.splitlines(keepends=True)

    start_idx = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"'{dataset_name}':") or stripped.startswith(f'"{dataset_name}":'):
            start_idx = index
            break

    if start_idx is None:
        return None, None

    brace_balance = 0
    end_idx = None
    for index in range(start_idx, len(lines)):
        brace_balance += lines[index].count('{')
        brace_balance -= lines[index].count('}')
        if index > start_idx and brace_balance <= 0:
            end_idx = index
            break

    if end_idx is None:
        print(f"WARNING: Could not remove dataset '{dataset_name}' from settings.py")
        return None, None

    removed_config = benchmark_settings.pop(dataset_name, None)
    del lines[start_idx:end_idx + 1]
    settings_path.write_text("".join(lines), encoding="utf-8")
    print(f"Temporarily removed dataset '{dataset_name}' from settings.py")
    return original_content, removed_config

# A function to restore the original settings.py content after temporary modification
def restore_settings_file(settings_path, original_content, dataset_name=None, removed_config=None):
    if original_content is not None:
        settings_path.write_text(original_content, encoding="utf-8")
        print("Restored settings.py to original state")

    if dataset_name and removed_config is not None and dataset_name not in benchmark_settings:
        benchmark_settings[dataset_name] = removed_config

# A function to keep only the specified dataset in LILAC_eval.py's datasets list for focused evaluation, and return original content for restoration later  
def set_lilac_eval_single_dataset(lilac_eval_path, dataset_name):
    if not lilac_eval_path.exists():
        print(f"WARNING: LILAC_eval.py not found at {lilac_eval_path}")
        return None

    original_content = lilac_eval_path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original_content else "\n"
    lines = original_content.splitlines(keepends=True)

    start_idx = None
    end_idx = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("datasets") and "=" in stripped:
            start_idx = index
            break

    if start_idx is None:
        print("WARNING: Could not find datasets block in LILAC_eval.py")
        return None

    # Handle inline list case (e.g., datasets = ["Android"])
    if "[" in lines[start_idx] and "]" in lines[start_idx]:
        list_header = lines[start_idx].split("=")[0].rstrip()
        lines[start_idx] = f"{list_header} = [\"{dataset_name}\"]{newline}"
        lilac_eval_path.write_text("".join(lines), encoding="utf-8")
        return original_content

    for index in range(start_idx + 1, len(lines)):
        if lines[index].strip().startswith("]"):
            end_idx = index
            break

    if end_idx is None:
        print("WARNING: Could not find end of datasets block in LILAC_eval.py")
        return None

    list_indent = "    "
    for index in range(start_idx + 1, end_idx):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            list_indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
            break

    list_header = lines[start_idx].split("=")[0].rstrip()
    new_block = [
        f"{list_header} = [{newline}",
        f"{list_indent}\"{dataset_name}\"{newline}",
        f"]{newline}",
    ]

    lines[start_idx:end_idx + 1] = new_block
    lilac_eval_path.write_text("".join(lines), encoding="utf-8")
    return original_content

# A function to restore the original LILAC_eval.py content after temporary modification
def restore_lilac_eval_file(lilac_eval_path, original_content):
    if original_content is not None:
        lilac_eval_path.write_text(original_content, encoding="utf-8")
