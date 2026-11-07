# Test report

Date: 2026-10-22

- Automated checks: **116 passed, 1 skipped, 0 failed**.
- Test interpreter: Python 3.13.5.
- All 29 Python files passed Python 3.9 syntax parsing; this is not execution under Python 3.9.
- Eight command-line help entry points passed.
- Python comment tokens: 0. Short functional docstrings are retained.

Tests cover numerical MQV/MPDD/CMA operations, all reference methods and
module configurations, threshold boundaries, observation and confirmation
rules, membership resets, batch plans, and result integrity.

The clean-calibration tests generate fresh monitoring trajectories using a
deterministic model test double, calculate the maximum two-round critical
ratio plus margin, and load the output into single/batch configurations.
They also check held-out data exclusion, data/protocol mismatch handling,
threshold-file changes, incomplete trajectories, duplicate rounds, and
preservation of the ceiling rule.

Both raw CSV preprocessing entry points were executed on small synthetic
files with the expected dataset schemas. The tests generated K=10 and K=20
partitions, checked held-out separation and complete independent-test use,
and verified that existing data are not overwritten. These are not the
full official datasets.

The optional TensorFlow architecture/training test was skipped because
TensorFlow is not installed in this environment. No real TensorFlow FL
training, full official-dataset preprocessing, manuscript table replication,
or GPU benchmark replication was run. The end-to-end control-flow tests
use a model test double and are not classification-performance evidence.

Detailed per-test outcomes are in `verification_results.json`.
