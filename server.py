"""Synchronous FL orchestration, screening, and final independent evaluation."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from aggregators import (aggregate_models, cma_weights, coordinate_aggregate,
                         ddaba_weights, fedavg_weights, trust_aggregate)
from client import set_seed, train_client
from configuration import LABELS, MODULES
from data_process import load_data
from metrics import classify, mqv_score
from model import build_model
from mpdd import MPDD, first_layer_update
from utils import (digest, runtime_metadata, sha256_file, source_hashes, weights_hash,
                   write_csv, write_json)


class FederatedRunner:
    def __init__(self, config, data=None, model_factory=build_model, seed_function=set_seed):
        self.config = config.resolved()
        self.data = data if data is not None else load_data(self.config)
        self.factory = model_factory; self.seed_function = seed_function
        self.global_model = None; self.local_models = []; self.output_owned = False
        self.state = [{'client_id': i, 'removed': False, 'removed_by': None,
                       'removed_round': None, 'mqv_streak': 0, 'mqv_ever_rejected': False,
                       'mpdd_ever_warned': False} for i in range(self.config.K)]
        self.mpdd = MPDD(self.config.K, self.config.tau_r, self.config.warmup_rounds,
                         self.config.confirmation_rounds, self.config.minimum_survivors,
                         self.config.monitor_only, self.config.record_pairs)

    def initialize(self):
        cfg = self.config
        self.seed_function(cfg.seed)
        dim, classes = len(self.data.validation.features), len(LABELS[cfg.dataset])
        self.global_model = self.factory(dim, classes, cfg.learning_rate, 'global')
        self.local_models = [self.factory(dim, classes, cfg.learning_rate, f'client_{i+1:02d}')
                             for i in range(cfg.K)]
        initial = self.global_model.get_weights()
        for model in self.local_models:
            model.set_weights(initial)
        return weights_hash(initial)

    def run(self):
        cfg = self.config; out = Path(cfg.output)
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError(f'Output directory must be empty: {out}')
        self.output_owned = True
        write_json(out / 'config.json', cfg.as_dict())
        write_json(out / 'data_manifest.json', self.data.manifest)
        sources = source_hashes(Path(__file__).resolve().parent)
        write_json(out / 'source_hashes.json', sources)
        initial_hash = self.initialize()
        environment = runtime_metadata(); write_json(out / 'environment.json', environment)
        write_json(out / 'initialization.json', {'global_weights_sha256': initial_hash})
        mqv_enabled, mpdd_enabled, cma_enabled = cfg.module_flags
        mqv_records, cluster_records, client_records = [], [], []
        pair_records, attack_records, round_records, aggregate_records = [], [], [], []
        completed_rounds = 0; stopping_reason = 'completed'; start = time.perf_counter()
        for round_id in range(1, cfg.rounds + 1):
            active = [i for i in range(cfg.K) if not self.state[i]['removed']]
            if not active:
                stopping_reason = 'no_active_clients'; break
            round_start = time.perf_counter(); dispatched = self.global_model.get_weights()
            uploads = {}; local_start = time.perf_counter()
            for i in active:
                self.local_models[i].set_weights(dispatched)
                events = train_client(self.local_models[i], self.data.train[i], i, cfg,
                                      seed_function=self.seed_function)
                for event in events:
                    attack_records.append({'round': round_id, 'client_id': i, **event})
                uploads[i] = self.local_models[i].get_weights()
            local_seconds = time.perf_counter() - local_start
            screen_start = time.perf_counter(); quality, validation_acc = {}, {}
            needs_scores = mqv_enabled or cma_enabled or cfg.method == 'ddaba'
            if needs_scores:
                for i in active:
                    p = self.local_models[i].predict(self.data.validation.x,
                                                     batch_size=cfg.batch_size, verbose=0)
                    quality[i] = mqv_score(self.data.validation.y, p)
                    validation_acc[i] = float(np.mean(p.argmax(axis=1) == self.data.validation.y.argmax(axis=1)))
            passed, rejected, mqv_removed = [], [], []
            for i in active:
                failed = bool(mqv_enabled and quality[i] < cfg.tau_mqv)
                self.state[i]['mqv_streak'] = self.state[i]['mqv_streak'] + 1 if failed else 0
                self.state[i]['mqv_ever_rejected'] |= failed
                if failed:
                    rejected.append(i)
                    if self.state[i]['mqv_streak'] >= cfg.mqv_confirmation_rounds:
                        self.state[i].update(removed=True, removed_by='MQV', removed_round=round_id)
                        mqv_removed.append(i)
                else:
                    passed.append(i)
                if needs_scores:
                    mqv_records.append({'round': round_id, 'client_id': i, 'KS': quality[i],
                                        'validation_accuracy': validation_acc[i], 'tau_mqv': cfg.tau_mqv,
                                        'rejection_enabled': mqv_enabled, 'rejected': failed,
                                        'consecutive_rejections': self.state[i]['mqv_streak'],
                                        'removed_this_round': i in mqv_removed})
            mqv_seconds = time.perf_counter() - screen_start
            if mqv_removed:
                self.mpdd.notify_removal()
            mpdd_start = time.perf_counter(); removed, warning = [], []
            retained = passed
            if mpdd_enabled and passed:
                updates = [first_layer_update(uploads[i], dispatched) for i in passed]
                retained, removed, warning, row, pairs = self.mpdd.inspect(passed, updates, round_id)
                cluster_records.append(row); pair_records.extend(pairs)
                for i in warning:
                    self.state[i]['mpdd_ever_warned'] = True
                for i in removed:
                    self.state[i].update(removed=True, removed_by='MPDD', removed_round=round_id)
            mpdd_seconds = time.perf_counter() - mpdd_start
            root_seconds = 0.0; aggregation_seconds = 0.0
            if retained:
                models = [uploads[i] for i in retained]; info = {}
                root_weights = None
                if cfg.method in ('fltrust', 'flest'):
                    t0 = time.perf_counter()
                    self.seed_function(cfg.seed + round_id)
                    root = self.factory(len(self.data.validation.features), len(LABELS[cfg.dataset]),
                                        cfg.learning_rate, f'root_{round_id}')
                    root.set_weights(dispatched)
                    root.fit(self.data.validation.x, self.data.validation.y, epochs=cfg.local_epochs,
                             batch_size=cfg.batch_size, verbose=cfg.fit_verbose, shuffle=False)
                    root_weights = root.get_weights(); root_seconds = time.perf_counter() - t0
                    del root
                t0 = time.perf_counter()
                weights = np.full(len(retained), 1 / len(retained))
                if cfg.method in MODULES:
                    if cma_enabled:
                        weights = cma_weights(retained, quality, cfg.beta, cfg.lam, cfg.cma_rounding)
                    new_weights = aggregate_models(models, weights)
                elif cfg.method == 'fedavg':
                    weights = fedavg_weights([self.data.train[i].used_rows for i in retained])
                    new_weights = aggregate_models(models, weights)
                elif cfg.method in ('median', 'trimmed_mean'):
                    weights = None
                    new_weights = coordinate_aggregate(models, cfg.method, cfg.trim_ratio)
                elif cfg.method == 'ddaba':
                    weights, info = ddaba_weights([validation_acc[i] for i in retained])
                    new_weights = aggregate_models(models, weights)
                else:
                    new_weights, weights, info = trust_aggregate(models, dispatched, root_weights,
                                                                 cfg.method, cfg.seed + round_id)
                self.global_model.set_weights(new_weights)
                aggregation_seconds = time.perf_counter() - t0
                aggregate_records.append({'round': round_id, 'client_ids': retained,
                                          'weights': None if weights is None else weights.tolist(),
                                          'method': cfg.method, 'diagnostics': info})
                completed_rounds += 1
            else:
                stopping_reason = 'all_mqv_rejected' if len(rejected) == len(active) else 'no_models_to_aggregate'
            for i, state in enumerate(self.state):
                client_records.append({'round': round_id, **state,
                                       'trained_this_round': i in active, 'mqv_rejected': i in rejected,
                                       'mpdd_warning': i in warning, 'aggregated': i in retained})
            round_records.append({'round': round_id, 'active': len(active), 'mqv_passed': len(passed),
                                  'aggregated': len(retained), 'local_training_seconds': local_seconds,
                                  'mqv_seconds': mqv_seconds, 'mpdd_seconds': mpdd_seconds,
                                  'root_training_seconds': root_seconds,
                                  'aggregation_seconds': aggregation_seconds,
                                  'round_seconds': time.perf_counter() - round_start})
            print(f'round={round_id}/{cfg.rounds} active={len(active)} retained={len(retained)}', flush=True)
            if not retained:
                break
        if cfg.monitor_only:
            scores = {'accuracy': None, 'macro_f1': None}
            per_class, cm, final_val_acc = [], None, None
        else:
            if self.data.test is None:
                raise ValueError('The independent test partition is required for evaluation.')
            prediction = self.global_model.predict(self.data.test.x, batch_size=cfg.batch_size, verbose=0)
            scores, per_class, cm = classify(self.data.test.y, prediction, LABELS[cfg.dataset])
            val_prediction = self.global_model.predict(self.data.validation.x, batch_size=cfg.batch_size, verbose=0)
            final_val_acc = 100 * float(np.mean(val_prediction.argmax(axis=1) == self.data.validation.y.argmax(axis=1)))
        final_rows = []
        for state in self.state:
            final_rows.append({**state, 'is_malicious': state['client_id'] in cfg.malicious_ids})
        bad = [r for r in final_rows if r['is_malicious']]; good = [r for r in final_rows if not r['is_malicious']]
        result = {'dataset': cfg.dataset, 'client_setting': cfg.client_setting, 'K': cfg.K,
                  'method': cfg.method, 'scenario': cfg.scenario, 'seed': cfg.seed, **scores,
                  'validation_accuracy': final_val_acc, 'completed_rounds': completed_rounds,
                  'planned_rounds': cfg.rounds, 'status': stopping_reason,
                  'mqv_removed_malicious': sum(r['removed_by'] == 'MQV' for r in bad),
                  'mpdd_removed_malicious': sum(r['removed_by'] == 'MPDD' for r in bad),
                  'removed_benign': sum(r['removed'] for r in good), 'initial_malicious': len(bad),
                  'initial_benign': len(good),
                  'mqv_ever_tpr': 100 * sum(r['mqv_ever_rejected'] for r in bad) / len(bad) if bad else None,
                  'mqv_ever_frr': 100 * sum(r['mqv_ever_rejected'] for r in good) / len(good) if good else None,
                  'elapsed_seconds': time.perf_counter() - start,
                  'initial_weights_sha256': initial_hash, 'data_fingerprint': self.data.manifest['fingerprint'],
                  'source_fingerprint': digest(sources), 'environment_fingerprint': digest(environment),
                  'tag': cfg.tag, 'monitor_only': cfg.monitor_only,
                  'independent_test_evaluated': not cfg.monitor_only,
                  'model_backend': type(self.global_model).__module__ + '.' + type(self.global_model).__name__}
        for name, rows in [('mqv_client_round_records.csv', mqv_records),
                           ('cluster_round_summary.csv', cluster_records),
                           ('client_status_by_round.csv', client_records),
                           ('client_final_status.csv', final_rows), ('attack_events.csv', attack_records),
                           ('round_metrics.csv', round_records), ('aggregation_rounds.csv', aggregate_records),
                           ('per_class_metrics.csv', per_class)]:
            write_csv(out / name, rows)
        if cfg.record_pairs:
            write_csv(out / 'mpdd_pairwise_distance_by_round.csv', pair_records)
        if cm is not None:
            np.savetxt(out / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
        if cfg.save_model:
            self.global_model.save_weights(str(out / 'global_final.weights.h5'))
        write_json(out / 'result.json', result)
        write_json(out / 'DONE.json', {'status': stopping_reason,
                                      'output_hashes': {p.name: sha256_file(p) for p in sorted(out.iterdir())
                                                        if p.is_file() and p.name != 'DONE.json'}})
        return result
