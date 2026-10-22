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


def append_row_to_csv(csv_file_path, fieldnames, row):
    """Append one row to CSV, creating file and header if needed."""
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_file_path.exists() or csv_file_path.stat().st_size == 0
    with open(csv_file_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def run_for_dataset(dataset_name, log_file_name, log_path, shot, num_of_average_intervals, support_threshold=0.5, resolved_log_format=None, results_root_dir=None, data_type="2k", example_size=0, model="llama3"):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found for dataset '{dataset_name}': {log_path}")

    print(f"\n=== Processing dataset '{dataset_name}' with log '{log_file_name}' ===")

    # Get log format once per run unless already resolved in main.
    log_format = resolved_log_format if resolved_log_format is not None else get_log_format(dataset_name, log_file_name, log_path, data_type=data_type)

    # Reuse dataset output directory; inferred format is persisted once per dataset in main().
    dataset_output_dir = results_root_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    xes_timing_csv = results_root_dir / "xes_creation_times.csv"
    minerful_timing_csv = results_root_dir / "minerful_times.csv"

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

    # Log parsing using LILAC -> Run LILAC evaluation from its package directory
    original_cwd = Path.cwd()
    evaluation_dir = Path("LILAC_main/benchmark/evaluation")
    if evaluation_dir.exists():
        try:
            os.chdir(evaluation_dir)
            command_for_LILAC_eval = [sys.executable, "-m", "LILAC_eval", "--shot", str(shot), "--model", model, "--example_size", str(example_size)]
            if data_type == "full": command_for_LILAC_eval.append("-full")
            print(f"Running LILAC evaluation for dataset '{dataset_name}'...")
            eval_env = os.environ.copy()
            eval_env["LILAC_DATASET"] = dataset_name
            result = subprocess.run(command_for_LILAC_eval, check=False, env=eval_env)
            print(f"LILAC evaluator exited with code {result.returncode}")
            if result.returncode != 0: raise RuntimeError(f"LILAC evaluator failed for dataset '{dataset_name}' with exit code {result.returncode}.")
        except Exception as exc:
            print(f"WARNING: Failed to run LILAC evaluation for {dataset_name}: {exc}")
            raise
        finally:
            os.chdir(original_cwd)
    else:
        print(f"WARNING: Evaluation directory not found at {evaluation_dir}")

    # Convert to .XES
    # For a time window of logs, create traces
    # Define time window
    # Based on avg time between logs, define a time window to group log messages into traces
    time_window_suggestion = suggest_time_window_for_traces(log_path, log_format, num_of_average_intervals)

    # Create traces and save on csv file
    output_xes_file = dataset_output_dir / f"{dataset_name}_shot{shot}_avg{num_of_average_intervals}.xes"
    if not output_xes_file.exists():
        xes_start = perf_counter()
        xes_status = "success"
        thesis_result_root = BASE_DIR / f"result_{data_type}"
        typed_result_dir = thesis_result_root / f"result_LILAC_{data_type}_{shot}_{example_size}_{model}"
        legacy_result_dir = thesis_result_root / f"result_LILAC_{shot}_{example_size}_{model}"
        structured_log_candidates = [
            typed_result_dir / f"{Path(log_file_name).stem}.log_structured.csv",
            legacy_result_dir / f"{Path(log_file_name).stem}.log_structured.csv",
        ]
        structured_log_file = next((candidate for candidate in structured_log_candidates if candidate.exists()), None)

        if structured_log_file is None:
            fallback_matches = []
            for result_dir in (typed_result_dir, legacy_result_dir):
                if result_dir.exists():
                    fallback_matches.extend(sorted(result_dir.glob(f"{dataset_name}*.log_structured.csv")))
            if fallback_matches: structured_log_file = fallback_matches[0]
        if structured_log_file is None:
            xes_status = "missing_structured_log"
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
            raise FileNotFoundError(
                "Structured log file not found for trace/XES/Minerful steps. "
                f"Checked: {structured_log_candidates}"
            )

        output_csv_file = dataset_output_dir / f"{dataset_name}_shot{shot}_avg{num_of_average_intervals}_traces.csv"
        assign_log_line_to_trace(log_path, log_format, time_window_suggestion, structured_log_file, output_csv_file)

        if not output_csv_file.exists() or output_csv_file.stat().st_size == 0: raise RuntimeError(f"Trace CSV is missing or empty: {output_csv_file}")

        # Save .XES file
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
    else:
        print(f"Skipping trace/XES generation, file already exists: {output_xes_file}")
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
                "status": "skipped_existing",
                "duration_seconds": 0.0,
                "output_xes": str(output_xes_file),
            },
        )

    for support in support_threshold:
        # Apply Miner
        minerful_dir = Path("MINERful")              # folder of Minerful
        minerful_jar = minerful_dir / "MINERful.jar"  # core Minerful jar
        minerful_bin_dir = minerful_dir / "bin"
        minerful_lib_dir = minerful_dir / "lib"       # folder containing dependency jars
        xes_log = str(output_xes_file)
        out_csv = str(dataset_output_dir / f"{dataset_name}_shot{shot}_avg{num_of_average_intervals}_st{support}_minerful_output.csv")
        if Path(out_csv).exists():
            print(f"Skipping Minerful, output already exists: {out_csv}")
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
                    "status": "skipped_existing",
                    "duration_seconds": 0.0,
                    "input_xes": xes_log,
                    "output_csv": out_csv,
                },
            )
            continue
        java_exe = "java"

        # Build classpath similar to MINERful launcher: jar + bin + each lib/*.jar.
        cp_entries = [str(minerful_jar), str(minerful_bin_dir)]
        cp_entries.extend(str(jar_path) for jar_path in sorted(minerful_lib_dir.glob("*.jar")))
        classpath = os.pathsep.join(cp_entries)

        cmd = [java_exe,"-cp", classpath,"minerful.MinerFulMinerStarter","-iLF", xes_log,"-s", f"{support}","-oCSV", out_csv,]

        miner_start = perf_counter()
        try:
            subprocess.run(cmd, check=True)
            miner_status = "success"
        except subprocess.CalledProcessError as exc:
            miner_status = "failed_calledprocess"
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
            print(f"WARNING: Minerful failed for {dataset_name} (shot={shot}, avg={num_of_average_intervals}, s={support}): {exc}")
            continue
        except FileNotFoundError as exc:
            miner_status = "failed_java_not_found"
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

def main():
    data_type = "2k"  # choose from: "2k", "full"
    example_size = 0
    model = "llama3"
    shot = [8, 16, 32, 64, 128]  # shot size for LILAC sampling and parsing
    num_of_average_intervals = [10, 100, 500, 1000, 1500]   # number of intervals multiplied with average log time difference to calculate time windows for traces
    support_threshold = [0.3, 0.5, 0.7, 0.9]  # support threshold for Minerful constraint mining
    force_reinfer_log_format = False # set True to ignore existing settings.py config and infer from scratch
    results_root_dir = BASE_DIR / f"results_{data_type}"

    if data_type not in {"2k", "full"}:
        raise ValueError(f"Unsupported data_type: {data_type}. Use '2k' or 'full'.")

    datasets_root = Path(f"LILAC_main/{data_type}_dataset")
    datasets_to_run = sorted(datasets_root.rglob("*.log"))

    if not datasets_to_run:
        print(f"WARNING: No .log files found under {datasets_root.resolve()}")
        return

    print(f"Found {len(datasets_to_run)} dataset log file(s) under {datasets_root}")
    
    # Check if LILAC directory exists and install dependencies
    install_lilac_deps()

    for log_path in datasets_to_run:
        log_file_name = log_path.name
        relative_parent = log_path.parent.relative_to(datasets_root)
        dataset_name = relative_parent.name if relative_parent != Path(".") else log_path.stem

        settings_file = Path("LILAC_main/benchmark/evaluation/settings.py")
        original_settings_content = None
        removed_settings_config = None
        if force_reinfer_log_format:
            original_settings_content, removed_settings_config = remove_dataset_from_settings_temporarily(settings_file, dataset_name)

        dataset_log_format = get_log_format(dataset_name, log_file_name, log_path, data_type=data_type)
        dataset_output_dir = results_root_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        format_output_file = dataset_output_dir / f"{dataset_name}_inferred_format.txt"
        format_output_file.write_text((dataset_log_format or "") + "\n", encoding="utf-8")

        for shot_size in shot:
            for num_intervals in num_of_average_intervals:
                run_for_dataset(dataset_name, log_file_name, log_path, shot_size, num_intervals, support_threshold, resolved_log_format=dataset_log_format, results_root_dir=results_root_dir, data_type=data_type, example_size=example_size, model=model)
        
        if force_reinfer_log_format:
            restore_settings_file(settings_file, original_settings_content, dataset_name, removed_settings_config)

if __name__ == "__main__":
    main()
