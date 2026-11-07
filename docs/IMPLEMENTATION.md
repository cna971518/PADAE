# Implementation notes

## Data and training

Start from the official raw CSV files using the preprocessing commands in the
README. Base split seeds are 0 for UNSW-NB15 and 8 for CIC-IDS2017. Independent
test fractions are 0.1 and 0.3; the server validation fraction of development
data is 1/(K+1). Local test fractions are 0.1 and 0.3. Classes are distributed
approximately evenly across clients. These partitions are reused across
training seeds.

UNSW-NB15 fits a separate one-hot encoder and 98%-variance PCA for each of
proto, state, and service on the offline union of local training partitions.
StatAvg combines training-only counts, means, and population variances,
including between-client mean differences. Local SMOTE follows normalization
only for UNSW-NB15. Test and validation data do not fit transformations.
This is not a claim of decentralized or privacy-guaranteed encoder fitting.

Training uses the five-hidden-layer DNN, Adam, and fixed sample order. Global
weights are dispatched each round while client optimizer states persist.
Local seeds are training_seed + zero-based client ID. Training, local-test,
and validation rows are truncated to full batches by default; the independent
test set is evaluated in full. `--no-trim-batches` changes this protocol.

## MQV

One-hot targets and softmax probabilities are flattened for the ROC-derived
max(abs(TPR-FPR)) statistic. This equals the empirical two-sample K–S distance
between true-class and non-true-class probabilities, not a normal/attack
binary classification score. A score below tau_MQV rejects the current
upload; equality passes. Two consecutive MQV rejections remove the client.
Its streak is separate from MPDD. If all uploads are rejected, the last
available global model is evaluated and the boundary outcome is recorded.

## MPDD

Only the first Dense kernel is compared, using local minus dispatched global
weights. Pairwise cosine distances are averaged within a stable membership
episode and clustered by two-group average linkage. Two zero-norm updates
have distance 0; exactly one zero-norm update gives distance 1 (epsilon 1e-12).

The group separation ratio is:

```text
R = median(between-group distances)
    / max(median(within larger group), median(within smaller group), 1e-12)
```

Self-pairs are excluded; a singleton's within-group median is 0. At least
three candidates, unequal group sizes, and a finite R are required. The
larger group is a provisional reference, not a known benign set. Individual
q ratios are diagnostics and do not determine removals.

Episode rounds 1 and 2 are observation-only. From round 3 onward, a valid
partition with R >= tau_R contributes a hit to the smaller group. The same
client must receive two consecutive hits before removal. Warning clients
remain eligible when MQV passes. Membership changes or removals reset the
remaining distance history and streaks. A removal is blocked if it would
leave fewer than floor(initial_K/2)+1 of the current MPDD candidates; this
guard does not override MQV rejection. Attack identities are never used to
choose the reference group or make screening decisions.

## Computing the MPDD threshold

`calibrate_mpdd.py --execute` generates clean, MPDD-only monitoring runs.
MQV rejection and CMA are disabled; all local updates are uniformly aggregated.
MPDD records distances and partitions but never removes a client. Its temporary
monitoring threshold cannot change training or the calibration evidence.
Neither local test files nor the independent Test.csv are loaded in this mode.

For each dataset, after the two observation rounds, let E contain pairs of
consecutive rounds in the same episode where the same clean client appears
in the smaller group and both partitions are structurally valid. The rule is:

```text
critical_R = max over E of min(R_previous, R_current)
tau_R      = critical_R + margin
```

The default workflow explicitly uses K in {10,20}, development seeds 42–46,
and margin 0.000001. These are declared calibration settings, not inputs
recovered from previous runs. The margin places tau_R strictly above all
eligible two-round critical ratios in the calibration sample; it is not a
guarantee of zero false removal on unseen data. No test metric or poisoned
trajectory is used to select the threshold. Missing evidence, incomplete
runs, duplicate seeds, mixed protocols, or incompatible environments cause
an error instead of a fabricated threshold.

The output records each dataset's threshold, margin, eligible evidence,
client-count/seed coverage, training settings, and hashes of the newly used
training and validation files. `--calibration-file` loads those values into
single and batch experiments and checks coverage, settings, and data. It
never rewrites the fixed parameter defaults or existing experiment outputs.
Calibration seeds are separate from the default evaluation seeds 100–104.

The fixed experiment defaults remain 1.0387229896327386 (UNSW-NB15) and
1.1887096500465608 (CIC-IDS2017). They are parameter values, not required
historical logs. A new calibration is not forced to reproduce those decimals.

## CMA

The high-contribution group uses the experiment's ceiling rule:

```text
s = number of retained models
v = min(s, max(1, ceil(beta * s)))
```

Models are stably sorted by descending K–S score. When v < s, each high-group
model receives lambda/v and each remaining model receives (1-lambda)/(s-v).
When v = s, weights are uniform. The empty-set case is handled before
aggregation; a singleton model is unchanged. With s=6 and beta=0.2, v=2.
`floor` is not an alternative runtime setting. Both kernels and biases are
aggregated. CMA-only computes K–S scores without enabling MQV rejection.

## Reference methods

Reference methods bypass PADAE screening. FedAvg weights the effective
training counts, whereas Uniform ablation uses equal weights. Median and
Trimmed-mean operate coordinate-wise on all uploaded tensors; the latter
trims floor(0.2*n) values from each tail.

DDaBA uses validation-accuracy gaps and the text-consistent-plus rule: the
top cutoff is ln(10/9)*mu and the outlier cutoff is (ln(4)+1.5*ln(3))*mu.
Top clients have twice the per-client weight of other retained clients;
equal accuracies yield uniform weights. This follows the textual 2:1 rule,
not the minus sign printed in Eq.(9). It is an explicit implementation
choice rather than an official-author implementation claim.

FLTrust and FLEST use all-tensor local-minus-global updates. The root model
trains on Val.csv with a fresh Adam optimizer each round. Updates are scaled
to the root-update norm; zero root norm or zero total trust preserves the
global model. FLEST uses Euclidean KMeans (two clusters, n_init=10), the larger
cluster reference, positive cosine confidence, and middle-50%-rank means.
Equal-size clusters fall back to root trust; identical updates use their
common mean. Root training and KMeans are seeded by training_seed+round_id.

## Resource measurement and testing

`benchmark.py` measures synthetic-upload server load, not end-to-end FL
training or large-population defense effectiveness. It reports actual MQV
pass counts separately from the deliberately imposed MPDD candidate count.
Repetitions are summarized by process. Local training and network time are
excluded.

The software generates its own outputs; it does not prefill manuscript
results. Fresh raw-data runs need no historical records. Exact numerical
agreement with manuscript tables still requires validation in the target
training environment. See TEST_REPORT.md for checks actually performed.
