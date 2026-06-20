:# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hands-on workshop (instructional content in **Korean**) demonstrating an end-to-end ML
workflow on **Amazon SageMaker Unified Studio** with **MLflow** experiment tracking. It is a
sequence of Jupyter notebooks, not an installable package — there is no build, lint, or test
suite. The notebooks are the source of truth.

## The notebooks are the only versioned code

`code/`, `processing/`, `training/`, `tmp/`, `data/`, `bank-additional/`, `mlruns/`, and model
artifacts (`*.tar.gz`, `xgboost-model`) are **all gitignored** — they are generated at runtime.
The `.py` scripts in `code/`/`processing/`/`training/` are produced by `%%writefile` cells inside
the notebooks; `bank-additional/` and `data/` are downloaded by the notebooks. **Never edit those
generated files directly** — edit the `%%writefile` cell in the owning notebook, since a re-run
overwrites them. To inspect a notebook as plain Python:

```bash
jupyter nbconvert --to script --stdout 2-training.ipynb
```

## Two independent tracks — do not conflate them

- **Notebooks 0→4** — Bank Marketing dataset, **binary classification** (predict term-deposit
  subscription, `binary:logistic`). These run **sequentially and share state** (see `%store` below).
- **Notebook 5** (`5-pipelines_...ipynb`) — UCI **Abalone**, **regression** (`reg:...`, MSE). A
  **standalone** SageMaker Pipelines demo that does NOT consume any output of 0→4 and can run alone.

The two tracks have separate preprocessing/training/evaluation scripts with different schemas
(`processing/preprocessing.py` + `training/train.py` for bank; `code/preprocessing.py` +
`code/evaluation.py` for abalone). When changing data shape, only touch the track you mean.

## Cross-notebook state via `%store`

Notebook 0 computes `bucket`, `prefix`, `role`, `region`, `mlflow_arn`, `mlflow_name`,
`experiment_name` and persists them with `%store`. Notebooks 1→4 begin with `%store -r ...` to
restore them. **Notebook 0 must be run first**, in the same Jupyter profile, or the later
notebooks fail at restore.

## Environment bootstrap (notebook 0, cell 1) — handle with care

The kernel image ships only modular/v3 SageMaker packages, so notebook 0 installs the **classic
SageMaker Python SDK v2** (`sagemaker>=2.220,<3`) needed for `FrameworkProcessor`, `Estimator`,
and `Pipelines`. Two pins matter:
- `setuptools<81` — setuptools 81+ removes `pkg_resources`, which breaks `import mlflow` (2.13.2).
- `mlflow==2.13.2` with `sagemaker-mlflow`.

The install is wrapped in an `importlib.util.find_spec("sagemaker.sklearn")` guard that skips
reinstall + kernel restart when already present. **Preserve this guard** — without it, "Run All"
triggers an infinite install→restart loop.

## How config is discovered (not hardcoded)

- `project = Project()` (from `sagemaker_studio`) yields the S3 root (`project.s3.root`), IAM role
  (`project.iam_role`), domain/project IDs, and Glue database.
- The **MLflow tracking ARN is looked up via the DataZone API** by paginating environments and
  partial-matching the name `"MLExperiments"` — `project.mlflow_tracking_server_arn` is NOT used
  because the real environment name may be prefixed (e.g. `"OnCreate MLExperiments"`).

## SageMaker job conventions

- Training scripts read SageMaker env vars (`SM_MODEL_DIR`, `SM_CHANNEL_TRAIN`,
  `SM_CHANNEL_VALIDATION`, `SM_HOSTS`, etc.) and read MLflow config from `MLFLOW_TRACKING_ARN` /
  `MLFLOW_EXPERIMENT_NAME` / `REGION` passed in as environment.
- Containers install their own deps at job start (`pip install` inside the script) and treat
  MLflow as **optional** — every MLflow call is guarded so a tracking failure never kills the job.
- CSV convention: **label is the first column, headerless**. The booster is pickled as a file
  literally named `xgboost-model`.
- Notebook 2 trains two models — built-in XGBoost (`xgb1`/`xgb2`) and Script Mode (`train.py`) —
  then `mlflow.register_model(...)`. Logging must succeed *before* registration, or an empty
  artifact gets registered.
