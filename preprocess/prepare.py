"""Split raw benchmarks before fitting encoders, PCA, StatAvg, or SMOTE."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from configuration import LABELS
from preprocess.schema import CIC_COLUMNS, UNSW_COLUMNS
from preprocess.statavg import aggregate_statistics, local_statistics, normalize
from utils import sha256_file, write_csv, write_json

CATEGORICAL = ('proto', 'state', 'service')
CIC_FILES = {'ddos': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
             'portscan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
             'dos_hulk': 'Wednesday-workingHours.pcap_ISCX.csv'}


def load_unsw(directory):
    paths = [directory / f'UNSW-NB15_{i}.csv' for i in range(1, 5)]
    frame = pd.concat([pd.read_csv(p, header=None, low_memory=False) for p in paths], ignore_index=True)
    if frame.shape[1] != len(UNSW_COLUMNS):
        raise ValueError('UNSW-NB15 raw CSVs must contain 49 columns without a header.')
    frame.columns = UNSW_COLUMNS
    attack = frame['attack_cat'].astype('string').str.strip().str.lower()
    binary = pd.to_numeric(frame['label'], errors='coerce')
    missing = attack.isna() | attack.isin(('', 'nan', 'none', 'null', 'missing', '-'))
    if (missing & (binary != 0)).any():
        raise ValueError('Missing attack category on a non-normal UNSW-NB15 record.')
    attack.loc[missing] = 'normal'
    frame['attack_cat'] = attack.replace({'backdoors': 'backdoor'}).astype(str)
    frame = frame.drop(columns=['srcip', 'dstip'])
    for column in CATEGORICAL:
        frame[column] = frame[column].map(lambda x: 'missing' if pd.isna(x) else str(x).strip().lower())
    numeric = [c for c in frame if c not in (*CATEGORICAL, 'attack_cat')]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[~frame['attack_cat'].isin(('analysis', 'backdoor', 'shellcode', 'worms'))]
    parts = [shuffle(frame.loc[frame['attack_cat'] == label], random_state=0).head(50000)
             for label in ('normal', 'generic')]
    parts.append(frame.loc[~frame['attack_cat'].isin(('normal', 'generic'))])
    frame = shuffle(pd.concat(parts, ignore_index=True), random_state=0).reset_index(drop=True)
    return frame, paths


def load_cic(directory):
    paths = {k: directory / v for k, v in CIC_FILES.items()}
    raw = {k: pd.read_csv(p, low_memory=False) for k, p in paths.items()}
    for frame in raw.values():
        if frame.shape[1] != len(CIC_COLUMNS):
            raise ValueError('CIC-IDS2017 input must use the 79-column MachineLearningCSV schema.')
        frame.columns = CIC_COLUMNS
        frame['Label'] = frame['Label'].astype(str).str.strip()
    parts = [raw['portscan'].loc[raw['portscan']['Label'] == 'BENIGN'],
             raw['ddos'].loc[raw['ddos']['Label'] == 'DDoS'],
             shuffle(raw['portscan'].loc[raw['portscan']['Label'] == 'PortScan'], random_state=8).head(130000),
             shuffle(raw['dos_hulk'].loc[raw['dos_hulk']['Label'] == 'DoS Hulk'], random_state=8).head(130000)]
    frame = shuffle(pd.concat(parts, ignore_index=True), random_state=8).reset_index(drop=True)
    frame['Label'] = frame['Label'].replace({'DoS Hulk': 'DoS_Hulk'})
    features = [c for c in frame if c != 'Label']
    frame[features] = frame[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame, list(paths.values())


def features_and_labels(frame, dataset):
    target = 'attack_cat' if dataset == 'UNSW-NB15' else 'Label'
    excluded = [target, 'label'] if dataset == 'UNSW-NB15' else [target]
    return frame.drop(columns=excluded), frame[target].astype(str)


def fit_transformer(training, dataset):
    numeric = [c for c in training if c not in CATEGORICAL] if dataset == 'UNSW-NB15' else list(training)
    imputer = SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)
    imputer.fit(training[numeric])
    artifacts = {}; output = list(numeric)
    if dataset == 'UNSW-NB15':
        for column in CATEGORICAL:
            try:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
            encoded = encoder.fit_transform(training[[column]].astype(str))
            pca = PCA(n_components=1 if encoded.shape[1] == 1 else .98,
                      svd_solver='full', random_state=0)
            pca.fit(encoded)
            names = [f'pca_{column}_{i+1}' for i in range(pca.n_components_)]
            artifacts[column] = {'encoder': encoder, 'pca': pca}
            output.extend(names)
    return {'dataset': dataset, 'numeric': numeric, 'imputer': imputer,
            'categorical': artifacts, 'features': output,
            'fit_scope': 'union_of_local_training_partitions_only'}


def transform(frame, transformer):
    blocks = [transformer['imputer'].transform(frame[transformer['numeric']]).astype(np.float32)]
    for column, item in transformer['categorical'].items():
        blocks.append(item['pca'].transform(item['encoder'].transform(frame[[column]].astype(str))).astype(np.float32))
    result = np.concatenate(blocks, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Feature transformation produced nonfinite values.')
    return result


def split(frame, fraction, seed, target):
    return train_test_split(frame, test_size=fraction, random_state=seed,
                            shuffle=True, stratify=frame[target])


def partition_clients(frame, count, seed, target, classes):
    buckets = [[] for _ in range(count)]
    for class_index, name in enumerate(classes):
        group = shuffle(frame.loc[frame[target] == name], random_state=seed + class_index)
        for client_id, positions in enumerate(np.array_split(np.arange(len(group)), count)):
            buckets[client_id].append(group.iloc[positions])
    return [shuffle(pd.concat(parts), random_state=seed+1000+i) for i, parts in enumerate(buckets)]


def encode_labels(values, classes):
    mapping = {name: i for i, name in enumerate(classes)}
    try:
        indices = np.array([mapping[v] for v in values], dtype=int)
    except KeyError as exc:
        raise ValueError(f'Unknown class: {exc}') from exc
    return np.eye(len(classes), dtype=np.float32)[indices]


def prepare_selected(selected, dataset, client_counts, out_root, raw_paths=()):
    classes = LABELS[dataset]; target = 'attack_cat' if dataset == 'UNSW-NB15' else 'Label'
    base = 0 if dataset == 'UNSW-NB15' else 8
    fraction = .1 if dataset == 'UNSW-NB15' else .3
    unknown = set(selected[target].unique()) - set(classes)
    if unknown:
        raise ValueError(f'Unexpected target classes: {unknown}')
    development, independent = split(selected, fraction, base, target)
    for count in client_counts:
        out = Path(out_root) / dataset / f'clients_{count}'
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Refusing to overwrite existing partitions: {out}')
        out.mkdir(parents=True, exist_ok=True)
        pool, validation = split(development, 1 / (count + 1), base+count, target)
        client_subsets = partition_clients(pool, count, base+10*count, target, classes)
        training, local_test = [], []
        for i, subset in enumerate(client_subsets):
            offset = i if dataset == 'UNSW-NB15' else i+1
            tr, te = split(subset, fraction, base+100*count+offset, target)
            training.append(tr); local_test.append(te)
        fitting, _ = features_and_labels(pd.concat(training), dataset)
        transformer = fit_transformer(fitting, dataset)
        arrays = [transform(features_and_labels(tr, dataset)[0], transformer) for tr in training]
        records = [local_statistics(a) for a in arrays]
        statistics = aggregate_statistics(records)
        memberships, audit = [], []
        def save(name, features, labels, raw_frame, original_count=None):
            ys = encode_labels(labels, classes)
            data = pd.DataFrame(features, columns=transformer['features'])
            data[list(classes)] = ys
            path = out / name; data.to_csv(path, index=False)
            audit.append({'name': name, 'sha256': sha256_file(path), 'rows': len(data),
                          'original_rows': len(raw_frame) if original_count is None else original_count})
            memberships.extend({'row_id': int(idx), 'partition': name} for idx in raw_frame.index)
        for i, (tr, te, unscaled) in enumerate(zip(training, local_test, arrays), start=1):
            y = features_and_labels(tr, dataset)[1].reset_index(drop=True)
            x = normalize(unscaled, statistics)
            if dataset == 'UNSW-NB15':
                from imblearn.over_sampling import SMOTE
                counts = y.value_counts()
                if len(counts) >= 2 and counts.min() >= 2:
                    x, y = SMOTE(random_state=base+10000*count+i,
                                 k_neighbors=min(5, int(counts.min())-1)).fit_resample(
                                     pd.DataFrame(x, columns=transformer['features']), y)
                    x = np.asarray(x, dtype=np.float32)
            save(f'client_{i:02d}_train.csv', x, y, tr)
            xt, yt = features_and_labels(te, dataset)
            save(f'client_{i:02d}_test.csv', normalize(transform(xt, transformer), statistics), yt, te)
        for name, frame in [('Val.csv', validation), ('Test.csv', independent)]:
            x, y = features_and_labels(frame, dataset)
            save(name, normalize(transform(x, transformer), statistics), y, frame)
        ids = [r['row_id'] for r in memberships]
        if len(set(ids)) != len(ids) or set(ids) != set(selected.index):
            raise AssertionError('Partition overlap or missing rows.')
        write_csv(out / 'split_membership.csv', memberships)
        write_csv(out / 'split_manifest.csv', audit)
        write_csv(out / 'statavg_global_statistics.csv',
                  [{'feature': name, 'global_mean': statistics['mean'][j],
                    'global_variance': statistics['variance'][j], 'normalization_scale': statistics['scale'][j],
                    'zero_variance': bool(statistics['zero_variance'][j]), 'total_training_samples': statistics['count']}
                   for j, name in enumerate(transformer['features'])])
        write_csv(out / 'statavg_local_statistics.csv',
                  [{'client': i+1, 'feature': name, 'sample_count': record['count'],
                    'local_mean': record['mean'][j], 'local_variance': record['variance'][j]}
                   for i, record in enumerate(records) for j, name in enumerate(transformer['features'])])
        joblib.dump({'transformer': transformer, 'statistics': statistics}, out / 'preprocessing_artifacts.joblib')
        write_json(out / 'preprocessing_manifest.json', {
            'dataset': dataset, 'K': count, 'preprocessing_seed': base, 'features': transformer['features'],
            'class_order': classes, 'selected_class_counts': selected[target].value_counts().to_dict(),
            'raw_sources': [{'name': Path(p).name, 'sha256': sha256_file(p)} for p in raw_paths],
            'fit_scope': 'local_training_partitions_only', 'held_out_used_for_fitting': False,
            'smote': dataset == 'UNSW-NB15', 'files': audit})
        print(f'{dataset} K={count}: wrote {len(audit)} partitions to {out}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(LABELS), required=True)
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--output-root', default='data/processed')
    parser.add_argument('--clients', nargs='+', type=int, default=[10, 20], choices=[10, 20])
    args = parser.parse_args(argv)
    directory = Path(args.raw_root) / args.dataset
    selected, sources = load_unsw(directory) if args.dataset == 'UNSW-NB15' else load_cic(directory)
    prepare_selected(selected, args.dataset, list(dict.fromkeys(args.clients)), args.output_root, sources)


if __name__ == '__main__':
    main()
