import json
import os
import pandas as pd
import re
import string
from sklearn.utils import shuffle
import textdistance
import random
import heapq
from collections import Counter, defaultdict, deque, OrderedDict
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import time
import calendar
import argparse
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

datasets = [
    "Apache",
    "BGL",
    "Hadoop",
    "HDFS",
    "HealthApp",
    "HPC",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Zookeeper",
    'Android',
    'Hadoop',
    'Spark',
    'Zookeeper',
    'BGL',
    'HPC',
    'Thunderbird',
    'Windows',
    'Linux',
    'Apache',
    'HDFS',
    'HealthApp',
    'Mac',
    'OpenSSH',
    'OpenStack',
    'Proxifier'
]

benchmark = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_full.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_full.log',
        'log_format': '<SessionId> <Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
    },

    'Spark': {
        'log_file': 'Spark/Spark_full.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_full.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    },

    'BGL': {
        'log_file': 'BGL/BGL_full.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    },

    'HPC': {
        'log_file': 'HPC/HPC_full.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_full.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    },

    'Windows': {
        'log_file': 'Windows/Windows_full.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    },

    'Linux': {
        'log_file': 'Linux/Linux_full.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_full.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    },

    'Apache': {
        'log_file': 'Apache/Apache_full.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_full.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_full.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_full.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    },

    'Mac': {
        'log_file': 'Mac/Mac_full.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    },
    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Month> <Date> <Time> <PID> <Level> <Component>: <Content>',
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample log queries for selected LILAC datasets")
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Dataset name(s) to process. Defaults to all predefined datasets if omitted."
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
        help="Shot counts to generate (e.g., --shots 10 20)."
    )
    parser.add_argument(
        "--data-dir",
        default=str(BASE_DIR / "full_dataset" / "datasets_2k"),
        help="Base directory containing dataset logs and structured CSV files."
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="Fraction of structured logs to consider during sampling."
    )
    return parser.parse_args()


def generate_logformat_regex(log_format):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, log_format):
    """ Function to transform log file to dataframe
    """
    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    line_count = 0
    with open(log_file, 'r', encoding='utf8', errors='ignore') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                line_count += 1
            except Exception as _:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(line_count)]
    return logdf


def lcs_distance(x, y):
    seq1 = x.split()
    seq2 = y.split()
    lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    return 1 - 2 * lengths[-1][-1] / (len(seq1) + len(seq2))


def lev_distance(x, y):
    return textdistance.levenshtein.normalized_distance(x, y)


def euc_distance(x, y):
    return textdistance.cosine.normalized_distance(x, y)


def jaccard_distance(x, y):
    return textdistance.jaccard.normalized_distance(x.split(), y.split())


def ratcliff_distance(x, y):
    return textdistance.ratcliff_obershelp.normalized_distance(x, y)


def min_distance(c_set, t_set):
    D = []
    for c_inst in c_set:
        min_candidate_distance = 1e10
        for t_inst in t_set:
            min_candidate_distance = min(min_candidate_distance, jaccard_distance(c_inst, t_inst))
        D.append(min_candidate_distance)
    return D


def adaptive_random_sampling(logs, k, n_candidate=128):
    sample_set = []
    T = []
    for r in range(k):
        print("r: ", r)
        if len(sample_set) == 0:
            i = max(range(0, len(logs)), key=lambda x: (len(logs[x][0].split()), logs[x][2]))
            T.append(logs[i][0])
            sample_set.append(logs[i][1])
            #del logs[i]
            continue
        candidate_set = [(x, logs[x]) for x in range(len(logs)) if x in random.sample(range(len(logs)), n_candidate)]
        candidate_set = sorted(candidate_set, key=lambda x: x[1][2], reverse=True)
        candidate_distance = min_distance([x[1][0] for x in candidate_set], T)
        best_candidate = max(range(len(candidate_distance)), key=candidate_distance.__getitem__)
        T.append(candidate_set[best_candidate][1][0])
        sample_set.append(candidate_set[best_candidate][1][1])
        #del logs[candidate_set[best_candidate][0]]
    return sample_set
# shot * (candidate + candidate*log_candidate, candidate * shot * distance)


class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = [
            "a",
            "an",
            "and",
            "i",
            "ie",
            "so",
            "to",
            "the",

        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))
        #print(self.__filter_stopwords(['LDAP', 'Built', 'with']))

    def build(self, sequences):
        print("Build vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            #print(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token]) for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]


def clean(s):
    log_format = re.sub(r'[0-9A-Za-z, ]+', '', s)
    unique_chars = list(set(log_format))
    sorted_string = ''.join(sorted(unique_chars))
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.?!', ' ', s)
    s = " ".join([word for word in s.strip().split() if not bool(re.search(r'\d', word))])
    # trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    return s, sorted_string


def hierichical_clustering(contents):
    t1 = time.time()
    vocab = Vocab()
    vocab.build([v[0].split() for v in contents.values()])
    t2 = time.time()
    # print("Build time: ", t2 - t1)

    # hierichical clustering
    hierichical_clusters = {}
    for k, v in contents.items():
        frequent_token = tuple(sorted(vocab.topk_tokens(v[0].split(), 3))) 
        log_format = v[1]
        if frequent_token not in hierichical_clusters:
            hierichical_clusters[frequent_token] = {"size": 1, "cluster": {log_format: [k]}}
        else:
            hierichical_clusters[frequent_token]["size"] = hierichical_clusters[frequent_token]["size"] + 1
            if log_format not in hierichical_clusters[frequent_token]["cluster"]:
                hierichical_clusters[frequent_token]["cluster"][log_format] = [k]
            else:
                hierichical_clusters[frequent_token]["cluster"][log_format].append(k)
    print("Number of coarse-grained clusters: ", len(hierichical_clusters.keys()))
    total_fine_clusters = 0
    for k, v in hierichical_clusters.items():
        total_fine_clusters += len(hierichical_clusters[k]["cluster"])
    print("Number of fine-grained clusters: ", total_fine_clusters)
    return hierichical_clusters


def hierichical_distribute(hierichical_clusters, shot, labelled_logs=[]):
    # hierichical distribution
    candidate_samples = []
    coarse_clusters = hierichical_clusters.keys()
    coarse_clusters = shuffle(list(coarse_clusters))
    # coarse_clusters = sorted(coarse_clusters, key=lambda x: hierichical_clusters[x]["size"], reverse=True)
    corase_size = len(coarse_clusters)
    for coarse_id, coarse_key in enumerate(coarse_clusters):
        coarse_quota = int(shot // corase_size) + (coarse_id < shot % corase_size)
        if coarse_quota == 0:
            break
        # print("Coarse quota: ", coarse_quota)
        # coarse cluster of coarse_key has been allocated {coarse_quota}
        fine_clusters = hierichical_clusters[coarse_key]["cluster"].keys()
        fine_clusters = sorted(fine_clusters, key=lambda x: len(hierichical_clusters[coarse_key]["cluster"][x]), reverse=True)
        fine_size = len(fine_clusters)
        # print("Fine size: ", fine_size)
        for fine_id, fine_key in enumerate(fine_clusters):
            fine_quota = int(coarse_quota // fine_size) + (fine_id < coarse_quota % fine_size)
            if fine_quota == 0:
                break
            # print("Fine quota: ", fine_quota)
            # fine cluster of fine_key has been allocated {fine_quota}
            # print("Coarse key: ", coarse_key, " Fine key: ", fine_key, " Fine quota: ", fine_quota, " Corase quota: " , coarse_quota, len(hierichical_clusters[coarse_key]["cluster"][fine_key]))
            samples = random.choices(hierichical_clusters[coarse_key]["cluster"][fine_key], k=fine_quota)
            candidate_samples.extend(samples)

    return candidate_samples


def sample_dataset(dataset, shots, data_dir, sample_fraction):
    if dataset not in benchmark:
        print(f"Dataset '{dataset}' missing from benchmark configuration. Skipping.")
        return None

    log_file = benchmark[dataset]['log_file']
    structured_path = Path(data_dir) / f"{log_file}_structured.csv"

    if not structured_path.exists():
        print(f"Structured log file not found: {structured_path}. Skipping {dataset}.")
        return None

    sampled_full_dir = BASE_DIR / "full_dataset" / "sampled_examples_full" / dataset
    sampled_dir = BASE_DIR / "full_dataset" / "sampled_examples" / dataset
    sampled_full_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    existing = True
    for shot in shots:
        shot_path = sampled_full_dir / f"{shot}shot.json"
        if not shot_path.exists():
            existing = False
            break
    if existing:
        print(f"All requested shots already exist for {dataset}. Skipping generation.")
        return None

    print(dataset)
    labelled_logs = pd.read_csv(structured_path)
    length = int(sample_fraction * len(labelled_logs))
    labelled_logs = labelled_logs[:length]
    print("Original logs:", len(labelled_logs))
    print("Unique templates:", labelled_logs['EventTemplate'].nunique())

    begin_time = time.time()
    contents = {}
    for i, x in enumerate(labelled_logs['Content'].to_list()):
        x, fx = clean(x)
        if len(x.split()) > 1:
            contents[i] = (x, fx)

    hierichical_clusters = hierichical_clustering(contents)
    clustering_time = time.time() - begin_time
    print("Hierichical clustering time:", clustering_time)

    metrics = {"times": {}, "coverage": {}}

    for shot in shots:
        full_target = sampled_full_dir / f"{shot}shot.json"
        simple_target = sampled_dir / f"{shot}shot.json"
        if full_target.exists():
            continue

        shot_start = time.time()
        sampled_ids = hierichical_distribute(hierichical_clusters, shot, labelled_logs['Content'].to_list())
        sampled_templates = set([row['EventTemplate'] for _, row in labelled_logs.take(sampled_ids).iterrows()])
        shot_time = (time.time() - shot_start) + clustering_time

        coverage = float(len(sampled_templates) / min(labelled_logs['EventTemplate'].nunique(), shot))
        print(f"{shot}-shot sampling covers sampled templates: {len(sampled_templates)}")
        print(f"{shot}-shot sampling cover rate: {coverage}")
        print("Hierichical sampling + clustering time:", shot_time)

        metrics['times'][shot] = shot_time
        metrics['coverage'][shot] = coverage

        candidate_samples = [(row['Content'], row['EventTemplate']) for _, row in labelled_logs.take(sampled_ids).iterrows()]
        candidate_samples = [{"query": x[0], "answer": x[1].replace('<*>', '{variables}')} for x in candidate_samples]

        with open(full_target, "w", encoding="utf-8") as f_full:
            for s in candidate_samples[:shot]:
                f_full.write(json.dumps(s) + "\n")

        with open(simple_target, "w", encoding="utf-8") as f_simple:
            for s in candidate_samples[:shot]:
                f_simple.write(json.dumps(s) + "\n")

    return metrics


if __name__ == '__main__':
    args = parse_args()

    selected_datasets = args.dataset if args.dataset else datasets
    shots = sorted(set(args.shots))
    data_dir = args.data_dir
    sample_fraction = args.sample_fraction

    average_times = {shot: [] for shot in shots}
    coverage_rates = {shot: [] for shot in shots}

    for dataset in selected_datasets:
        metrics = sample_dataset(dataset, shots, data_dir, sample_fraction)
        if not metrics:
            continue

        for shot in shots:
            if shot in metrics['times']:
                average_times[shot].append(metrics['times'][shot])
            if shot in metrics['coverage']:
                coverage_rates[shot].append(metrics['coverage'][shot])

    for shot in shots:
        if average_times[shot]:
            print(f"{shot}-shot average time: {np.mean(average_times[shot])}")
        if coverage_rates[shot]:
            print(f"{shot}-shot coverage rate: {np.mean(coverage_rates[shot])}")