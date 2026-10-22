#Advanced Log Processing
"""
Requirements:
- download LILAC-main from https://github.com/logpai/LILAC and place it in the same directory
- rename LILAC-main to LILAC_main (to avoid import issues)
- add empty __init__.py file in LILAC_main and LILAC_main/benchmark directories to make them a package
- add 2k datasets in LILAC_main/2k_dataset/dataset_name_directory
- install Ollama and download the desired model for LLM calls
- add LILAC_sampling_extended.py in LILAC_main/sampling
- add LILAC_main/benchmark/logparser/ollama_query.py with the one in this directory if you don't have access to openai API key
- change LILAC_main/benchmark/logparser/LILAC.py l. 7 from "from .gpt_query import query_template_from_gpt_with_check" to "query_template_from_gpt_with_check"
- change LILAC_main/benchmark/evaluation/common.py l. 94 so the default is False for --full_data if you want to test on 2k dataset, or True if you want to test on full dataset. By default, it is set to False for testing on 2k dataset
- place your dataset name in LILAC_main/benchmark/evaluation/LILAC_eval.py and comment out any of the other datasets if you only want to test on that dataset. By default, it is set to test on Android dataset in the 2k setting
- if log format is provided manually in LILAC_main/benchmark/evaluation/settings.py make sure the field names are inside <>.
- download Minerful from https://github.com/Process-in-Chains/MINERful.git, place it in the same directory, and name it MINERful
- added an extended LILAC_main/benchmark/evaluation/LILAC_eval.py for generalizing on new datasets, with the right naming process (kept original version as LILAC_eval_original_as_in_LILAC_repo.py for reference)
- log format can be defined in settings.py or inferred by the program. If you add it make sure the field names are inside <> and that include at least time or timestamp field for time component.
"""

import subprocess
import sys
import csv
from pathlib import Path
import os
from datetime import datetime, timezone
from time import perf_counter
import pandas as pd
from convert_to_XES_functions import * 
from run_LILAC_parsing_functions import * 

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_2k"


def append_row_to_csv(csv_file_path, fieldnames, row):
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_file_path.exists() or csv_file_path.stat().st_size == 0
    with open(csv_file_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)

def lilac_parse_and_evaluate(dataset_name, shot, example_size, model, data_type="2k"):
    # Run LILAC's sampling script to generate samples for the new dataset, which will be used for log parsing and evaluation
    command_for_LILAC_sampling = [sys.executable, "LILAC_main/sampling/LILAC_sampling_extended.py", "--dataset", dataset_name,]
    try:
        print(f"Running sampling script for dataset '{dataset_name}'...")
        result = subprocess.run(command_for_LILAC_sampling, capture_output=True, text=True, check=False)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except Exception as exc:
            print(f"WARNING: Failed to run sampling script for {dataset_name}: {exc}")

    if shot > 0:
        example_size = ensure_lilac_sample_file(dataset_name, shot, example_size, data_type=data_type)

    # Log parsing using LILAC -> Run LILAC evaluation from its package directory
    original_cwd = Path.cwd()
    evaluation_dir = Path("LILAC_main/benchmark/evaluation")
    lilac_eval_file = (evaluation_dir / "LILAC_eval.py").resolve()
    if evaluation_dir.exists():
        original_lilac_eval_content = set_lilac_eval_single_dataset(lilac_eval_file, dataset_name)
        try:
            os.chdir(evaluation_dir)
            command_for_LILAC_eval = [sys.executable, "-m", "LILAC_eval", "--shot", str(shot), "--model", model, "--example_size", str(example_size)]
            if data_type == "full":
                command_for_LILAC_eval.append("-full")
            print(f"Running LILAC evaluation for dataset '{dataset_name}'...")
            result = subprocess.run(command_for_LILAC_eval, check=False)
            print(f"LILAC evaluator exited with code {result.returncode}")
        except Exception as exc:
            print(f"WARNING: Failed to run LILAC evaluation for {dataset_name}: {exc}")
        finally:
            os.chdir(original_cwd)
            restore_lilac_eval_file(lilac_eval_file, original_lilac_eval_content)
    else:
        print(f"WARNING: Evaluation directory not found at {evaluation_dir}")

    return example_size

def convert_log_to_xes(log_path, log_format, log_file_name, dataset_output_dir, num_of_average_intervals, dataset_name, shot, example_size, model, data_type="2k"):
    xes_timing_csv = dataset_output_dir.parent / "xes_creation_times.csv"

    # Convert to .XES
    # For a time window of logs, create traces
    # Define time window
    # Based on avg time between logs, define a time window to group log messages into traces
    time_window_suggestion = suggest_time_window_for_traces(log_path, log_format, num_of_average_intervals)

    # Create traces and save on csv file                
    structured_log_file = Path(f"LILAC_main/result/result_LILAC_{data_type}_{shot}_{example_size}_{model}/{Path(log_file_name).stem}.log_structured.csv")
    if not structured_log_file.exists():
        print(f"WARNING: Structured log file not found, skipping trace/XES/Minerful steps: {structured_log_file}")
        append_row_to_csv(
            xes_timing_csv,
            [
                "timestamp_utc",
                "dataset",
                "data_type",
                "shot",
                "avg_intervals",
                "model",
                "example_size",
                "status",
                "duration_seconds",
                "output_xes",
            ],
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name,
                "data_type": data_type,
                "shot": shot,
                "avg_intervals": num_of_average_intervals,
                "model": model,
                "example_size": example_size,
                "status": "missing_structured_log",
                "duration_seconds": 0.0,
                "output_xes": str(dataset_output_dir / f"{dataset_name}_shot{shot}_exmp{example_size}_avg{num_of_average_intervals}.xes"),
            },
        )
        return None

    output_csv_file = dataset_output_dir / f"{dataset_name}_shot{shot}_exmp{example_size}_avg{num_of_average_intervals}_traces.csv"
    assign_log_line_to_trace(log_path, log_format, time_window_suggestion, structured_log_file, output_csv_file)

    if not output_csv_file.exists() or output_csv_file.stat().st_size == 0:
        print(f"WARNING: Trace CSV is missing or empty, skipping XES/Minerful: {output_csv_file}")
        append_row_to_csv(
            xes_timing_csv,
            [
                "timestamp_utc",
                "dataset",
                "data_type",
                "shot",
                "avg_intervals",
                "model",
                "example_size",
                "status",
                "duration_seconds",
                "output_xes",
            ],
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name,
                "data_type": data_type,
                "shot": shot,
                "avg_intervals": num_of_average_intervals,
                "model": model,
                "example_size": example_size,
                "status": "missing_trace_csv",
                "duration_seconds": 0.0,
                "output_xes": str(dataset_output_dir / f"{dataset_name}_shot{shot}_exmp{example_size}_avg{num_of_average_intervals}.xes"),
            },
        )
        return None
              
    # Save .XES file
    output_xes_file = dataset_output_dir / f"{dataset_name}_shot{shot}_exmp{example_size}_avg{num_of_average_intervals}.xes"
    if output_xes_file.exists():
        output_xes_file.unlink()

    xes_start = perf_counter()
    xes_status = "success"
    try:
        convert_csv_to_xes(output_csv_file, output_xes_file)
    except Exception:
        xes_status = "failed"
        raise
    finally:
        xes_elapsed = perf_counter() - xes_start
        append_row_to_csv(
            xes_timing_csv,
            [
                "timestamp_utc",
                "dataset",
                "data_type",
                "shot",
                "avg_intervals",
                "model",
                "example_size",
                "status",
                "duration_seconds",
                "output_xes",
            ],
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name,
                "data_type": data_type,
                "shot": shot,
                "avg_intervals": num_of_average_intervals,
                "model": model,
                "example_size": example_size,
                "status": xes_status,
                "duration_seconds": round(xes_elapsed, 6),
                "output_xes": str(output_xes_file),
            },
        )

    return output_xes_file

def apply_minerful(support_threshold, output_xes_file, dataset_output_dir, dataset_name, shot, example_size, num_of_average_intervals, model, data_type):
    minerful_timing_csv = dataset_output_dir.parent / "minerful_times.csv"

    if output_xes_file is None or not Path(output_xes_file).exists():
        print(f"WARNING: XES file not found, skipping Minerful: {output_xes_file}")
        return

    minerful_dir = Path("MINERful")              # folder of Minerful
    minerful_jar = minerful_dir / "MINERful.jar"  # core Minerful jar
    minerful_bin_dir = minerful_dir / "bin"
    minerful_lib_dir = minerful_dir / "lib"       # folder containing dependency jars
    xes_log = str(output_xes_file)

    for support in support_threshold:
        # Apply Miner
        out_csv = str(dataset_output_dir / f"{dataset_name}_shot{shot}_exmp{example_size}_avg{num_of_average_intervals}_st{support}_minerful_output.csv")

        if Path(out_csv).exists():
            Path(out_csv).unlink()

        java_exe = "java"

        # Build classpath in a cross-platform way: jar + bin + each lib/*.jar.
        cp_entries = [str(minerful_jar), str(minerful_bin_dir)]
        cp_entries.extend(str(jar_path) for jar_path in sorted(minerful_lib_dir.glob("*.jar")))
        classpath = os.pathsep.join(cp_entries)

        cmd = [java_exe,"-cp", classpath,"minerful.MinerFulMinerStarter","-iLF", xes_log,"-s", f"{support}","-oCSV", out_csv,]

        miner_start = perf_counter()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            miner_status = "success"
        except subprocess.CalledProcessError as exc:
            miner_elapsed = perf_counter() - miner_start
            append_row_to_csv(
                minerful_timing_csv,
                [
                    "timestamp_utc",
                    "dataset",
                    "data_type",
                    "shot",
                    "avg_intervals",
                    "support",
                    "status",
                    "duration_seconds",
                    "input_xes",
                    "output_csv",
                ],
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset_name,
                    "data_type": data_type,
                    "shot": shot,
                    "avg_intervals": num_of_average_intervals,
                    "support": support,
                    "status": "failed_calledprocess",
                    "duration_seconds": round(miner_elapsed, 6),
                    "input_xes": xes_log,
                    "output_csv": out_csv,
                },
            )
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            print(f"WARNING: Minerful failed for {dataset_name} (shot={shot}, avg={num_of_average_intervals}, s={support}): {exc}")
            continue
        except FileNotFoundError as exc:
            miner_elapsed = perf_counter() - miner_start
            append_row_to_csv(
                minerful_timing_csv,
                [
                    "timestamp_utc",
                    "dataset",
                    "data_type",
                    "shot",
                    "avg_intervals",
                    "support",
                    "status",
                    "duration_seconds",
                    "input_xes",
                    "output_csv",
                ],
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset_name,
                    "data_type": data_type,
                    "shot": shot,
                    "avg_intervals": num_of_average_intervals,
                    "support": support,
                    "status": "failed_java_not_found",
                    "duration_seconds": round(miner_elapsed, 6),
                    "input_xes": xes_log,
                    "output_csv": out_csv,
                },
            )
            raise RuntimeError(
                f"Could not find Java executable '{java_exe}'. "
                "If Java is installed, set java_exe to its full path "
                "(e.g. r'C:\\Program Files\\Java\\jdk-17\\bin\\java.exe')."
            ) from exc
        else:
            miner_elapsed = perf_counter() - miner_start
            append_row_to_csv(
                minerful_timing_csv,
                [
                    "timestamp_utc",
                    "dataset",
                    "data_type",
                    "shot",
                    "avg_intervals",
                    "support",
                    "status",
                    "duration_seconds",
                    "input_xes",
                    "output_csv",
                ],
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset_name,
                    "data_type": data_type,
                    "shot": shot,
                    "avg_intervals": num_of_average_intervals,
                    "support": support,
                    "status": miner_status,
                    "duration_seconds": round(miner_elapsed, 6),
                    "input_xes": xes_log,
                    "output_csv": out_csv,
                },
            )



def run_for_dataset(dataset_name, log_file_name, log_path, shot, num_of_average_intervals, support_threshold=0.5, resolved_log_format=None, results_root_dir=None, data_type="2k", example_size=0, model="llama3"):
    if not log_path.exists():
        print(f"WARNING: Log file not found, skipping dataset '{dataset_name}': {log_path}")
        return

    print(f"\n=== Processing dataset '{dataset_name}' with log '{log_file_name}' ===")

    # Get log format once per run unless already resolved in main.
    log_format = resolved_log_format if resolved_log_format is not None else get_log_format(dataset_name, log_file_name, log_path)

    # Reuse dataset output directory; inferred format is persisted once per dataset in main().
    dataset_output_dir = (results_root_dir or RESULTS_DIR) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    effective_example_size = lilac_parse_and_evaluate(dataset_name, shot, example_size, model, data_type=data_type)

    output_xes_file = convert_log_to_xes(log_path, log_format, log_file_name, dataset_output_dir, num_of_average_intervals, dataset_name, shot, effective_example_size, model, data_type=data_type)

    apply_minerful(support_threshold, output_xes_file, dataset_output_dir, dataset_name, shot, effective_example_size, num_of_average_intervals, model, data_type)
    
     
def main():
    data_type = "2k"  # choose from: "2k", "full"
    example_size = 8
    model = "llama3"
    repeat_count = 5
    dataset_subset = "all"  # choose from: "all", "first_half", "last_half"
    shot = [8, 16, 32, 64, 128]  # shot size for LILAC sampling and parsing
    num_of_average_intervals = [100, 250, 500, 1000]   # number of intervals multiplied with average log time difference to calculate time windows for traces
    support_threshold = [0.3, 0.5, 0.7, 0.9]  # support threshold for Minerful constraint mining
    force_reinfer_log_format = False # set True to ignore existing settings.py config and infer from scratch
    if data_type not in {"2k", "full"}:
        raise ValueError(f"Unsupported data_type: {data_type}. Use '2k' or 'full'.")
    if dataset_subset not in {"all", "first_half", "last_half"}:
        raise ValueError(
            f"Unsupported dataset_subset: {dataset_subset}. Use 'all', 'first_half', or 'last_half'."
        )
    results_root_dir = RESULTS_DIR

    datasets_root = Path(f"LILAC_main/{data_type}_dataset")
    datasets_to_run = sorted(datasets_root.rglob("*.log"))

    if not datasets_to_run:
        print(f"WARNING: No .log files found under {datasets_root.resolve()}")
        return

    if dataset_subset != "all":
        selected_dataset_count = max(1, (len(datasets_to_run) + 1) // 2)
        if dataset_subset == "first_half":
            datasets_to_run = datasets_to_run[:selected_dataset_count]
        else:
            datasets_to_run = datasets_to_run[-selected_dataset_count:]

    print(f"Found {len(datasets_to_run)} dataset log file(s) under {datasets_root}")
    
    # Check if LILAC directory exists and install dependencies
    install_lilac_deps()
    
    # Run the whole pipeline for each dataset/ 8 examples/ llama3   
    for run_index in range(1, repeat_count + 1):
        print(f"Starting run {run_index}/{repeat_count} across all datasets")

        for log_path in datasets_to_run:
            log_file_name = log_path.name
            relative_parent = log_path.parent.relative_to(datasets_root)
            dataset_name = relative_parent.name if relative_parent != Path(".") else log_path.stem

            settings_file = Path("LILAC_main/benchmark/evaluation/settings.py")
            original_settings_content = None
            removed_settings_config = None
            if force_reinfer_log_format:
                original_settings_content, removed_settings_config = remove_dataset_from_settings_temporarily(settings_file, dataset_name)

            dataset_log_format = get_log_format(dataset_name, log_file_name, log_path)
            dataset_output_dir = results_root_dir / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            format_output_file = dataset_output_dir / f"{dataset_name}_inferred_format.txt"
            format_output_file.write_text((dataset_log_format or "") + "\n", encoding="utf-8")

            for shot_size in shot:
                for num_intervals in num_of_average_intervals:
                    run_for_dataset(dataset_name, log_file_name, log_path, shot_size, num_intervals, support_threshold, resolved_log_format=dataset_log_format, results_root_dir=results_root_dir, data_type=data_type, example_size=example_size, model=model)

            if force_reinfer_log_format:
                restore_settings_file(settings_file, original_settings_content, dataset_name, removed_settings_config)
    
    # Run the whole pipeline for each dataset/ full examples/ llama3
    for run_index in range(1, repeat_count + 1):
        print(f"Starting run {run_index}/{repeat_count} across all datasets")

        for log_path in datasets_to_run:
            log_file_name = log_path.name
            relative_parent = log_path.parent.relative_to(datasets_root)
            dataset_name = relative_parent.name if relative_parent != Path(".") else log_path.stem

            settings_file = Path("LILAC_main/benchmark/evaluation/settings.py")
            original_settings_content = None
            removed_settings_config = None
            if force_reinfer_log_format:
                original_settings_content, removed_settings_config = remove_dataset_from_settings_temporarily(settings_file, dataset_name)

            dataset_log_format = get_log_format(dataset_name, log_file_name, log_path)
            dataset_output_dir = results_root_dir / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            format_output_file = dataset_output_dir / f"{dataset_name}_inferred_format.txt"
            format_output_file.write_text((dataset_log_format or "") + "\n", encoding="utf-8")
            
            example_size = 0
            for shot_size in shot:
                if shot_size == 8 or example_size == shot_size:
                    continue  # skip 8-shot for full dataset since it's already done in the 2k runs
                for num_intervals in num_of_average_intervals:
                    example_size = shot_size
                    run_for_dataset(dataset_name, log_file_name, log_path, shot_size, num_intervals, support_threshold, resolved_log_format=dataset_log_format, results_root_dir=results_root_dir, data_type=data_type, example_size=example_size, model=model)

            if force_reinfer_log_format:
                restore_settings_file(settings_file, original_settings_content, dataset_name, removed_settings_config)

    
    # Run the whole pipeline for each dataset/ no examples/ llama3
    for run_index in range(1, repeat_count + 1):
        print(f"Starting run {run_index}/{repeat_count} across all datasets")

        for log_path in datasets_to_run:
            log_file_name = log_path.name
            relative_parent = log_path.parent.relative_to(datasets_root)
            dataset_name = relative_parent.name if relative_parent != Path(".") else log_path.stem

            settings_file = Path("LILAC_main/benchmark/evaluation/settings.py")
            original_settings_content = None
            removed_settings_config = None
            if force_reinfer_log_format:
                original_settings_content, removed_settings_config = remove_dataset_from_settings_temporarily(settings_file, dataset_name)

            dataset_log_format = get_log_format(dataset_name, log_file_name, log_path)
            dataset_output_dir = results_root_dir / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            format_output_file = dataset_output_dir / f"{dataset_name}_inferred_format.txt"
            format_output_file.write_text((dataset_log_format or "") + "\n", encoding="utf-8")

            for shot_size in shot:
                if shot_size == 8:
                    example_size = 0  # to distinguish from 8-shot case in the results
                    for num_intervals in num_of_average_intervals:
                        run_for_dataset(dataset_name, log_file_name, log_path, shot_size, num_intervals, support_threshold, resolved_log_format=dataset_log_format, results_root_dir=results_root_dir, data_type=data_type, example_size=example_size, model=model)
                else:
                    continue  # skip 8-shot for full dataset since it's already done in the 2k runs
                
            if force_reinfer_log_format:
                restore_settings_file(settings_file, original_settings_content, dataset_name, removed_settings_config)
    
if __name__ == "__main__":
    main()
