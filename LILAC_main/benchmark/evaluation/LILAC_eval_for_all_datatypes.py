"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os

sys.path.append('../')

from logparser.LILAC import LogParser
from evaluation.settings import benchmark_settings
from evaluation.utils.common import common_args
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average


datasets = ["Mac"]


def normalize_log_file_for_data_type(log_file, data_type):
    normalized = log_file.replace("\\", "/")
    # Remove accidental dataset-prefix folders regardless of their position.
    for marker in ("/datasets_2k/", "/datasets_full/", "datasets_2k/", "datasets_full/"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]

    for marker in ("/2k_dataset/", "/full_dataset/", "2k_dataset/", "full_dataset/"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]

    if data_type == "full": normalized = normalized.replace("_2k.log", "_full.log")
    elif data_type == "2k": normalized = normalized.replace("_full.log", "_2k.log")

    return normalized


if __name__ == "__main__":
    args = common_args()
    env_data_type = os.environ.get("DATA_TYPE", "").strip().lower()
    data_type = env_data_type if env_data_type in {"2k", "full"} else ("full" if args.full_data else "2k")
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_LILAC_{data_type}_{args.shot}_{args.example_size}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    result_file = prepare_results(output_dir=output_dir, otc=args.oracle_template_correction, complex=args.complex, frequent=args.frequent)

    # Optional runtime override to run exactly one dataset without rewriting this file.
    dataset_filter = os.environ.get("LILAC_DATASET", "").strip()
    run_datasets = [dataset_filter] if dataset_filter else datasets

    for dataset in run_datasets:
        if dataset not in benchmark_settings:
            raise KeyError(f"Dataset '{dataset}' is not defined in benchmark_settings")
        setting = benchmark_settings[dataset]
        log_file = normalize_log_file_for_data_type(setting['log_file'], data_type)

        # In full-data mode, prefer the *_full.log variant when available.
        if args.full_data:
            full_variant = log_file.replace("_2k.log", "_full.log")
            full_variant_path = os.path.join(input_dir, full_variant)
            if full_variant != log_file and os.path.exists(full_variant_path):
                log_file = full_variant

        indir = os.path.join(input_dir, os.path.dirname(log_file))
        structured_stem = os.path.splitext(os.path.basename(log_file))[0]
        if os.path.exists(os.path.join(output_dir, f"{structured_stem}.log_structured.csv")): parser = None; print("parseing result exist.")
        else: parser = LogParser
        # run evaluator for a dataset
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                # 'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex'],
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': [],
                'data_type': data_type,
                'shot': args.shot, 'example_size': args.example_size,
                'model': args.model, 'selection_method': args.selection_method,
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file,
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"LILAC_complex={args.complex}_frequent={args.frequent}_{args.shot}_{args.example_size}_{args.model}", args.complex, args.frequent)
