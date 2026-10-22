# Advanced Processing of Log Files

This project was developed as part of my **Integrated Master's Thesis in Electrical and Computer Engineering** at the **Aristotle University of Thessaloniki (AUTH)**, under the supervision of **Professor Anastasios Gounaris**.

The project presents a **unified, end-to-end framework for the automated and holistic processing of system log files**, integrating LLM-based log parsing, custom event log construction, and declarative process mining into a single workflow. By connecting these distinct processing techniques, the pipeline enables the transformation of raw, unstructured log data into representations suitable for advanced behavioral analysis.

The main objective is to **bridge the gap between log parsing and process mining** through a custom methodology for constructing process-oriented event traces and XES event logs. Designed to accommodate heterogeneous datasets, the framework enables the **systematic discovery of behavioral patterns, event relationships, and process constraints** through a generalized, automated workflow.

## **Processing Pipeline**

The pipeline integrates three complementary stages to transform raw system logs into actionable insights about system behavior:

1. **Log Parsing (LILAC)** – Applies LLM-based parsing to extract structured events and identify recurring log templates from unstructured log data.
2. **Event Log Generation** – Introduces a custom transformation methodology that reconstructs process-oriented event traces from parsed logs, establishing the structure required for **XES representation** and enabling their integration into process-mining workflows.
3. **Process Mining (MINERful)** – Analyzes the generated event logs to discover **declarative process constraints**, revealing behavioral patterns and relationships between events.

**Pipeline Overview**

```text
Raw System Logs
       ↓
LLM-Based Log Parsing (LILAC)
       ↓
Structured Events & Templates
       ↓
Custom Event Log Construction & XES Conversion
       ↓
Process Mining (MINERful)
       ↓
Discovered Behavioral Constraints
```

## **Key Features**

- **Automated Workflow Integration** – Connects distinct processing tools and automates intermediate transformations, significantly reducing manual intervention.
- **Custom Event Log Construction** – Implements a methodology for grouping related events into meaningful process traces, reconstructing the event-log structure required for process mining.
- **Dataset Generalizability** – Supports the processing of heterogeneous system log datasets through a configurable framework.
- **Experimental Flexibility** – Enables systematic evaluation across different parsing configurations, trace-construction parameters, and constraint-mining thresholds.
- **Behavioral Knowledge Discovery** – Extracts declarative process constraints to reveal underlying event relationships and system behavior.

## **Files**

- **`Advanced_Log_Processing.py`** – Main pipeline script integrating and automating the complete workflow, from raw log parsing and event log construction to declarative process mining.
- **`Advanced_Log_Processing_for_2k_datasets_final.py`** – Experimental runner designed for systematic evaluation across heterogeneous datasets using 2k-entry subsets, supporting repeated executions and configurable experimental parameters.
- **`run_LILAC_parsing_functions.py`** – LILAC integration module that streamlines parsing across different datasets, supports configurable language models, and coordinates sampling, log-format inference, and parsing operations.
- **`convert_to_XES_functions.py`** – Custom event log generation module responsible for constructing process-oriented traces and transforming structured log data into standardized XES event logs.
- **`LILAC_main/`** – Integrated LILAC framework for LLM-based log parsing, including sampling, evaluation, and supporting components.
- **`MINERful/`** – Integrated Java-based process-mining framework for discovering declarative behavioral constraints.
- **`Vernadou_Artemis_Thesis.pdf`** – Complete Master's thesis documenting the methodology, implementation, experimental evaluation, and findings.

## **Dataset Requirements**

The pipeline supports heterogeneous system log datasets in `.log` format, organized into two categories: **smaller dataset samples (2k)** and **full datasets**.

Each dataset should be placed in its own subdirectory, following the structure:

```text
LILAC_main/
├── 2k_dataset/
│   └── dataset_name/
│       └── dataset_name_2k.log
└── full_dataset/
    └── dataset_name/
        └── dataset_name_full.log
```

Dataset selection and processing parameters are configured through the corresponding pipeline scripts.

## **Usage**

### **1. Environment Setup**

Ensure **Python**, **Java**, **Ollama**, and the required Python dependencies are installed. The repository includes the adapted **LILAC** and **MINERful** frameworks.

### **2. Dataset Preparation**

Organize the input `.log` files according to the structure described in **Dataset Requirements**. The pipeline automatically discovers and processes datasets within the selected directory.

### **3. Configuration**

Customize the execution parameters in the `main()` function of the corresponding script, including dataset selection, language model, parsing configurations, trace-construction settings, and process-mining thresholds.

### **4. Execution**

Run the main pipeline:

```bash
python Advanced_Log_Processing.py
```

For experimental evaluation using smaller dataset samples:

```bash
python Advanced_Log_Processing_for_2k_datasets_final.py
```

The pipeline coordinates the processing stages and generates the corresponding intermediate and final outputs.

### **5. Results**

Results are organized into dataset-specific directories under `results_2k/` or `results_full/`, depending on the configuration.

Generated artifacts include **structured parsing results, event traces, XES event logs and discovered process constraints**.
## **Thesis**

The complete methodology, experimental evaluation, and findings are presented in the document:

**`Vernadou_Artemis_Thesis.pdf`**

## **Acknowledgments**

This project was developed as part of my Integrated Master's Thesis at the **Aristotle University of Thessaloniki**, under the supervision of **Professor Anastasios Gounaris**, whom I sincerely thank for his guidance and support throughout the research.

The project builds on the open-source [LILAC](https://github.com/logpai/LILAC) and [MINERful](https://github.com/Process-in-Chains/MINERful) frameworks, combining them with **custom-developed methods for event log construction and workflow automation** to create a more comprehensive approach to system log analysis. code developed to integrate the different processing stages.
