"""First-layer update-distance clustering and consecutive-round confirmation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import AgglomerativeClustering

EPSILON = 1e-12


def first_layer_update(local_weights, global_weights):
    if not local_weights or not global_weights:
        raise ValueError('Missing model weights.')
    local = np.asarray(local_weights[0], dtype=np.float64)
    reference = np.asarray(global_weights[0], dtype=np.float64)
    if local.ndim != 2 or local.shape != reference.shape:
        raise ValueError('The first tensor must be a matching Dense kernel.')
    delta = (local - reference).ravel()
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite first-layer update.')
    return delta


def cosine_distance(a, b, epsilon=EPSILON):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if not a.size or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Invalid update vectors.')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= epsilon and nb <= epsilon:
        return 0.0
    if na <= epsilon or nb <= epsilon:
        return 1.0
    return float(1.0 - np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def pairwise_distance(updates):
    n = len(updates)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            result[i, j] = result[j, i] = cosine_distance(updates[i], updates[j])
    return result


def within_median(distance, positions):
    block = distance[np.ix_(positions, positions)]
    return float(np.median(block[np.triu_indices(len(positions), 1)])) if len(positions) > 1 else 0.0


@dataclass
class Partition:
    valid: bool
    reason: str
    reference: tuple
    suspicious: tuple
    within_reference: float
    within_suspicious: float
    between: float
    R: float
    q: dict


def partition(distance, client_ids):
    ids = tuple(int(i) for i in client_ids)
    d = np.asarray(distance, dtype=np.float64)
    if d.shape != (len(ids), len(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Invalid distance dimensions or duplicate client IDs.')
    if not np.isfinite(d).all() or not np.allclose(d, d.T, rtol=0, atol=1e-8):
        raise ValueError('Distance matrix must be finite and symmetric.')
    d = np.maximum((d + d.T) * .5, 0.0); np.fill_diagonal(d, 0.0)
    def invalid(reason):
        return Partition(False, reason, (), (), np.nan, np.nan, np.nan, np.nan, {})
    if len(ids) < 3:
        return invalid('fewer_than_three_candidates')
    kwargs = dict(n_clusters=2, linkage='average')
    try:
        estimator = AgglomerativeClustering(metric='precomputed', **kwargs)
    except TypeError:
        estimator = AgglomerativeClustering(affinity='precomputed', **kwargs)
    labels = estimator.fit_predict(d)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2:
        return invalid('not_two_clusters')
    if counts[0] == counts[1]:
        return invalid('equal_size_clusters')
    ref_pos = np.flatnonzero(labels == unique[np.argmax(counts)])
    sus_pos = np.flatnonzero(labels != unique[np.argmax(counts)])
    wr, ws = within_median(d, ref_pos), within_median(d, sus_pos)
    between = float(np.median(d[np.ix_(ref_pos, sus_pos)]))
    ratio = between / max(wr, ws, EPSILON)
    relative = {}
    for pos, client_id in enumerate(ids):
        peers = [j for j in ref_pos if j != pos]
        relative[client_id] = float(np.median(d[pos, peers]) / max(wr, EPSILON)) if peers else 0.0
    return Partition(bool(np.isfinite(ratio)), 'valid_structure',
                     tuple(ids[i] for i in ref_pos), tuple(ids[i] for i in sus_pos),
                     wr, ws, between, float(ratio), relative)


class MPDD:
    def __init__(self, total_clients, tau_r, warmup_rounds=2, confirmation_rounds=2,
                 minimum_survivors=True, monitor_only=False, record_pairs=False):
        if total_clients < 3 or not np.isfinite(tau_r) or tau_r <= 1:
            raise ValueError('Invalid MPDD configuration.')
        self.K = int(total_clients); self.tau_r = float(tau_r)
        self.warmup = int(warmup_rounds); self.confirmation = int(confirmation_rounds)
        self.minimum = self.K // 2 + 1 if minimum_survivors else 0
        self.monitor_only = bool(monitor_only); self.record_pairs = bool(record_pairs)
        self.sums = np.zeros((self.K, self.K), dtype=np.float64)
        self.counts = np.zeros((self.K, self.K), dtype=np.int64)
        self.hits = np.zeros(self.K, dtype=np.int64)
        self.members = None; self.episode = 0; self.episode_round = 0
        self.pending_reset = None

    def _reset(self, ids):
        self.sums.fill(0); self.counts.fill(0); self.hits.fill(0)
        self.members = tuple(ids); self.episode += 1; self.episode_round = 0
        self.pending_reset = None

    def notify_removal(self):
        self.pending_reset = 'client_removed'

    def inspect(self, client_ids, updates, round_id):
        ids = tuple(int(i) for i in client_ids)
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.K for i in ids):
            raise ValueError('Invalid MPDD client IDs.')
        if len(ids) != len(updates):
            raise ValueError('Update coverage mismatch.')
        reason = self.pending_reset
        if self.members is None:
            reason = 'initial_episode'
        elif ids != self.members:
            reason = 'mqv_passing_membership_changed'
        if reason is not None:
            self._reset(ids)
        self.episode_round += 1
        current = pairwise_distance(updates)
        averaged = np.zeros_like(current)
        pairs = []
        for row, i in enumerate(ids):
            for col in range(row + 1, len(ids)):
                j = ids[col]
                self.sums[i, j] += current[row, col]; self.sums[j, i] = self.sums[i, j]
                self.counts[i, j] += 1; self.counts[j, i] = self.counts[i, j]
                avg = self.sums[i, j] / self.counts[i, j]
                averaged[row, col] = averaged[col, row] = avg
                if self.record_pairs:
                    pairs.append({'round': round_id, 'episode': self.episode,
                                  'client_i': i, 'client_j': j,
                                  'distance': current[row, col], 'average_distance': avg,
                                  'observations': int(self.counts[i, j])})
        split = partition(averaged, ids)
        warmup = self.episode_round <= self.warmup
        valid = bool(split.valid and not warmup and split.R >= self.tau_r)
        gate_reason = ('episode_warmup' if warmup else split.reason if not split.valid else
                       'separation_below_tau_R' if split.R < self.tau_r else 'valid_cluster_split')
        proposed = []
        for i in ids:
            self.hits[i] = self.hits[i] + 1 if valid and i in split.suspicious else 0
            if self.hits[i] >= self.confirmation:
                proposed.append(i)
        blocked = bool(proposed and len(ids) - len(proposed) < self.minimum)
        removed = [] if blocked or self.monitor_only else proposed
        warnings = [i for i in ids if self.hits[i] > 0 and i not in removed]
        retained = [i for i in ids if i not in removed]
        row = {'round': int(round_id), 'episode': self.episode, 'episode_round': self.episode_round,
               'reset_reason': reason, 'warmup': warmup, 'structural_valid': split.valid,
               'gate_valid': valid, 'gate_reason': gate_reason, 'R': split.R,
               'tau_r': self.tau_r, 'within_reference': split.within_reference,
               'within_suspicious': split.within_suspicious, 'between': split.between,
               'reference_ids': list(split.reference), 'suspicious_ids': list(split.suspicious),
               'passing_ids': list(ids), 'warning_ids': warnings, 'proposed_ids': proposed,
               'removed_ids': removed, 'safety_blocked': blocked, 'q': split.q}
        if removed:
            self.notify_removal()
        return retained, removed, warnings, row, pairs
