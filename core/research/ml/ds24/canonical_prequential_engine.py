from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


TARGET = "forward_return_60m__decision_5m"
CORE_FAMILIES = ("ridge", "rff_ridge", "pca_ridge", "spline_additive_ridge")
METADATA_COLUMNS = {
    "asset_id",
    "canonical_symbol",
    "provider_symbol",
    "timestamp_utc",
    "session_date",
    "session_type",
    "decision_timestamp",
    "target_value",
    "target_available_timestamp",
    "target_is_trainable",
    "target_resolution_classification",
}


def mean_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(keep_empty_features=True)
    except TypeError:
        return SimpleImputer()


@dataclass(frozen=True)
class PredictorManifest:
    predictors: list[str]
    stock_predictors: list[str]
    context_predictors: list[str]
    manifest_hash: str

    @property
    def predictor_count(self) -> int:
        return len(self.predictors)


@dataclass(frozen=True)
class PartitionRow:
    asset_id: str
    year: int
    feature_partition: str
    target_partition: str


@dataclass(frozen=True)
class FitPredictionResult:
    family: str
    config: str
    decision_timestamp: pd.Timestamp
    training_rows: int
    scoring_rows: int
    fit_wall_seconds: float
    artifact_path: Path
    artifact_hash: str
    prediction_path: Path
    prediction_hash: str


@dataclass
class ExpandingPreprocessorMoments:
    predictors: list[str]
    n_rows: int
    sum_y: float
    obs_count: np.ndarray
    sum_x: np.ndarray
    sum_x2: np.ndarray
    sum_xy_obs: np.ndarray
    sum_y_obs: np.ndarray
    pair_count_both: np.ndarray
    pair_sum_left_both: np.ndarray
    pair_cross_both: np.ndarray

    @classmethod
    def empty(cls, predictors: list[str]) -> "ExpandingPreprocessorMoments":
        d = len(predictors)
        return cls(
            predictors=list(predictors),
            n_rows=0,
            sum_y=0.0,
            obs_count=np.zeros(d, dtype=np.float64),
            sum_x=np.zeros(d, dtype=np.float64),
            sum_x2=np.zeros(d, dtype=np.float64),
            sum_xy_obs=np.zeros(d, dtype=np.float64),
            sum_y_obs=np.zeros(d, dtype=np.float64),
            pair_count_both=np.zeros((d, d), dtype=np.float64),
            pair_sum_left_both=np.zeros((d, d), dtype=np.float64),
            pair_cross_both=np.zeros((d, d), dtype=np.float64),
        )

    def update_frame(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        x = frame[self.predictors].to_numpy(dtype=np.float64, copy=True)
        y = frame["target_value"].to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(x)
        x0 = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask_f = mask.astype(np.float64)
        self.n_rows += int(len(frame))
        self.sum_y += float(np.sum(y))
        self.obs_count += mask_f.sum(axis=0)
        self.sum_x += x0.sum(axis=0)
        self.sum_x2 += (x0 * x0).sum(axis=0)
        self.sum_xy_obs += (x0 * y[:, None]).sum(axis=0)
        self.sum_y_obs += (mask_f * y[:, None]).sum(axis=0)
        self.pair_count_both += mask_f.T @ mask_f
        self.pair_sum_left_both += x0.T @ mask_f
        self.pair_cross_both += x0.T @ x0

    def imputer_mean(self) -> np.ndarray:
        return np.divide(self.sum_x, self.obs_count, out=np.zeros_like(self.sum_x), where=self.obs_count > 0)

    def scaler_std(self) -> np.ndarray:
        mu = self.imputer_mean()
        missing = self.n_rows - self.obs_count
        second = self.sum_x2 + missing * mu * mu
        var = np.maximum(second / max(self.n_rows, 1) - mu * mu, 0.0)
        std = np.sqrt(var)
        std[std == 0.0] = 1.0
        return std

    def standardized_moments(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        if self.n_rows <= 0:
            raise RuntimeError("EMPTY_EXPANDING_STATE")
        mu = self.imputer_mean()
        std = self.scaler_std()
        d = len(self.predictors)
        x_cross = np.empty((d, d), dtype=np.float64)
        for j in range(d):
            for k in range(d):
                both = self.pair_count_both[j, k]
                sum_j_both = self.pair_sum_left_both[j, k]
                sum_k_both = self.pair_sum_left_both[k, j]
                missing_both = self.n_rows - self.obs_count[j] - self.obs_count[k] + both
                x_cross[j, k] = (
                    self.pair_cross_both[j, k]
                    + mu[j] * (self.sum_x[k] - sum_k_both)
                    + mu[k] * (self.sum_x[j] - sum_j_both)
                    + mu[j] * mu[k] * missing_both
                )
        xy_imp = self.sum_xy_obs + mu * (self.sum_y - self.sum_y_obs)
        ztz = (x_cross - self.n_rows * np.outer(mu, mu)) / np.outer(std, std)
        zty = (xy_imp - mu * self.sum_y) / std
        return ztz, zty, self.sum_y / self.n_rows, mu, std

    def score_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        mu = self.imputer_mean()
        std = self.scaler_std()
        x = frame[self.predictors].to_numpy(dtype=np.float64, copy=True)
        x = np.where(np.isfinite(x), x, mu)
        return (x - mu) / std

    def predict_ridge(self, frame: pd.DataFrame, alpha: float = 10.0) -> np.ndarray:
        ztz, zty, intercept, _, _ = self.standardized_moments()
        coef = np.linalg.solve(ztz + alpha * np.eye(ztz.shape[0]), zty)
        return self.score_matrix(frame) @ coef + intercept

    def predict_pca_ridge(self, frame: pd.DataFrame, alpha: float = 5.0) -> np.ndarray:
        ztz, zty, intercept, _, _ = self.standardized_moments()
        components = min(24, max(2, len(self.predictors) // 3))
        values, vectors = np.linalg.eigh((ztz + ztz.T) / (2.0 * max(self.n_rows, 1)))
        order = np.argsort(values)[::-1][:components]
        basis = vectors[:, order].T
        latent_xtx = basis @ ztz @ basis.T
        latent_xty = basis @ zty
        coef = np.linalg.solve(latent_xtx + alpha * np.eye(components), latent_xty)
        return (self.score_matrix(frame) @ basis.T) @ coef + intercept

    def content_hash(self) -> str:
        payload = {
            "predictors": self.predictors,
            "n_rows": self.n_rows,
            "sum_y": self.sum_y,
            "obs_count": self.obs_count.tolist(),
            "sum_x": self.sum_x.tolist(),
            "sum_x2": self.sum_x2.tolist(),
            "sum_xy_obs": self.sum_xy_obs.tolist(),
            "sum_y_obs": self.sum_y_obs.tolist(),
            "pair_count_both": self.pair_count_both.tolist(),
            "pair_sum_left_both": self.pair_sum_left_both.tolist(),
            "pair_cross_both": self.pair_cross_both.tolist(),
        }
        return stable_hash(payload)

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path) -> "ExpandingPreprocessorMoments":
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, cls):
            raise TypeError("EXPANDING_PREPROCESSOR_STATE_TYPE_MISMATCH")
        return state


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictor_manifest(path: Path) -> PredictorManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictors = list(payload["predictors"])
    stock_count = int(payload["stock_predictor_count"])
    stock_predictors = predictors[:stock_count]
    context_predictors = predictors[stock_count:]
    if len(stock_predictors) != stock_count:
        raise ValueError("STOCK_PREDICTOR_COUNT_MISMATCH")
    if len(context_predictors) != int(payload["context_predictor_count"]):
        raise ValueError("CONTEXT_PREDICTOR_COUNT_MISMATCH")
    if len(predictors) != int(payload["predictor_count"]):
        raise ValueError("PREDICTOR_COUNT_MISMATCH")
    observed_hash = stable_hash(predictors)
    legacy_hash = hashlib.sha256(json.dumps(predictors, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    if observed_hash != payload["hash"] and legacy_hash != payload["hash"]:
        raise ValueError("PREDICTOR_MANIFEST_HASH_MISMATCH")
    return PredictorManifest(
        predictors=predictors,
        stock_predictors=stock_predictors,
        context_predictors=context_predictors,
        manifest_hash=str(payload["hash"]),
    )


def read_partition_manifest(path: Path) -> list[PartitionRow]:
    rows = pd.read_csv(path)
    return [
        PartitionRow(
            asset_id=str(row.asset_id),
            year=int(row.year),
            feature_partition=str(row.feature_partition),
            target_partition=str(row.target_partition),
        )
        for row in rows.itertuples(index=False)
    ]


def model_for_family(family: str, predictors: list[str]) -> BaseEstimator:
    if family == "ridge":
        return Pipeline(
            [
                ("imputer", mean_imputer()),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if family == "rff_ridge":
        return Pipeline(
            [
                ("imputer", mean_imputer()),
                ("scaler", StandardScaler()),
                ("rff", RBFSampler(gamma=0.1, n_components=96, random_state=17)),
                ("model", Ridge(alpha=5.0)),
            ]
        )
    if family == "pca_ridge":
        n_components = min(24, max(2, len(predictors) // 3))
        return Pipeline(
            [
                ("imputer", mean_imputer()),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=17)),
                ("model", Ridge(alpha=5.0)),
            ]
        )
    if family == "spline_additive_ridge":
        selected = predictors[: min(12, len(predictors))]
        selector = ColumnTransformer([("first_features", "passthrough", selected)], remainder="drop")
        return Pipeline(
            [
                ("selector", selector),
                ("imputer", mean_imputer()),
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
                ("model", Ridge(alpha=25.0)),
            ]
        )
    raise ValueError(f"Unknown core family {family}")


class CanonicalPrequentialEngine:
    def __init__(
        self,
        *,
        root: Path,
        feature_root: Path,
        predictor_manifest: PredictorManifest,
        partitions: Iterable[PartitionRow],
    ) -> None:
        self.root = root
        self.feature_root = feature_root
        self.predictor_manifest = predictor_manifest
        self.partitions = list(partitions)

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def context_path(self, year: int) -> Path:
        return self.feature_root / "shared-context" / f"year={year}" / "context.parquet"

    def assemble_partitions(
        self,
        *,
        rows: Iterable[PartitionRow],
        decision_dates: set[str] | None = None,
        decision_timestamps: set[pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        context_cache: dict[int, pd.DataFrame] = {}
        for row in rows:
            feature_cols = [
                "asset_id",
                "canonical_symbol",
                "decision_timestamp",
                "session_date",
                *self.predictor_manifest.stock_predictors,
            ]
            target_cols = [
                "asset_id",
                "canonical_symbol",
                "decision_timestamp",
                "target_value",
                "target_is_trainable",
                "target_available_timestamp",
            ]
            context_cols = ["decision_timestamp", *self.predictor_manifest.context_predictors]
            features = pd.read_parquet(self.resolve(row.feature_partition), columns=feature_cols)
            features["decision_timestamp"] = pd.to_datetime(features["decision_timestamp"], utc=True)
            if decision_dates is not None:
                features = features[features["session_date"].astype(str).isin(decision_dates)]
            if decision_timestamps is not None:
                features = features[features["decision_timestamp"].isin(decision_timestamps)]
            if features.empty:
                continue
            targets = pd.read_parquet(self.resolve(row.target_partition), columns=target_cols)
            targets["decision_timestamp"] = pd.to_datetime(targets["decision_timestamp"], utc=True)
            targets["target_available_timestamp"] = pd.to_datetime(targets["target_available_timestamp"], utc=True)
            if decision_dates is not None:
                targets = targets[targets["decision_timestamp"].dt.date.astype(str).isin(decision_dates)]
            if decision_timestamps is not None:
                targets = targets[targets["decision_timestamp"].isin(decision_timestamps)]
            if targets.empty:
                continue
            if row.year not in context_cache:
                context = pd.read_parquet(self.context_path(row.year), columns=context_cols)
                context["decision_timestamp"] = pd.to_datetime(context["decision_timestamp"], utc=True)
                context_cache[row.year] = context.drop_duplicates("decision_timestamp")
            joined = features.merge(
                context_cache[row.year],
                on="decision_timestamp",
                how="left",
                validate="many_to_one",
            )
            if len(joined) != len(features):
                raise RuntimeError("CONTEXT_JOIN_CARDINALITY_VIOLATION")
            joined = joined.merge(
                targets,
                on=["asset_id", "canonical_symbol", "decision_timestamp"],
                how="inner",
                validate="one_to_one",
            )
            if not joined.empty:
                frames.append(joined)
        if not frames:
            return pd.DataFrame(columns=self.package_columns())
        data = pd.concat(frames, ignore_index=True)
        data[self.predictor_manifest.predictors] = data[self.predictor_manifest.predictors].replace([np.inf, -np.inf], np.nan)
        return data.sort_values(["decision_timestamp", "asset_id"]).reset_index(drop=True)

    def package_columns(self) -> list[str]:
        return [
            "asset_id",
            "canonical_symbol",
            "decision_timestamp",
            "session_date",
            *self.predictor_manifest.predictors,
            "target_value",
            "target_is_trainable",
            "target_available_timestamp",
        ]

    def decision_spine(self, reference_asset: str = "SPY") -> pd.Series:
        frames: list[pd.Series] = []
        for row in self.partitions:
            if row.asset_id != reference_asset:
                continue
            target = pd.read_parquet(
                self.resolve(row.target_partition),
                columns=["decision_timestamp", "target_is_trainable"],
            )
            target = target[target["target_is_trainable"].astype(bool)]
            frames.append(pd.to_datetime(target["decision_timestamp"], utc=True))
        if not frames:
            raise RuntimeError("REFERENCE_ASSET_SPINE_NOT_FOUND")
        return pd.concat(frames, ignore_index=True).drop_duplicates().sort_values().reset_index(drop=True)

    def training_and_score_frames(self, panel: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        timestamp = pd.Timestamp(timestamp).tz_convert("UTC")
        train = panel[
            (panel["decision_timestamp"] < timestamp)
            & (panel["target_is_trainable"].astype(bool))
            & (panel["target_available_timestamp"] <= timestamp)
        ].copy()
        score = panel[panel["decision_timestamp"] == timestamp].copy()
        return train, score

    def fit_predict_persist(
        self,
        *,
        family: str,
        config: str,
        timestamp: pd.Timestamp,
        train: pd.DataFrame,
        score: pd.DataFrame,
        output_root: Path,
        run_id: str,
    ) -> FitPredictionResult:
        if train.empty:
            raise RuntimeError("EMPTY_TRAINING_ESTATE")
        if score.empty:
            raise RuntimeError("EMPTY_SCORE_ESTATE")
        predictors = self.predictor_manifest.predictors
        model = model_for_family(family, predictors)
        started = time.perf_counter()
        model.fit(train[predictors], train["target_value"].astype(float))
        fit_wall = time.perf_counter() - started
        predictions = pd.DataFrame(
            {
                "family": family,
                "config": config,
                "decision_timestamp": pd.Timestamp(timestamp).isoformat(),
                "asset_id": score["asset_id"].astype(str),
                "prediction": np.asarray(model.predict(score[predictors]), dtype=float),
            }
        )
        stamp = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = output_root / "model_artifacts" / f"{family}_{config}_{run_id}_{stamp}.pkl"
        prediction_path = output_root / "oof_predictions" / f"{family}_{config}_{run_id}_{stamp}.csv"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as handle:
            pickle.dump(
                {
                    "family": family,
                    "config": config,
                    "decision_timestamp": pd.Timestamp(timestamp).isoformat(),
                    "predictors": predictors,
                    "predictor_manifest_hash": self.predictor_manifest.manifest_hash,
                    "training_rows": len(train),
                    "model": model,
                },
                handle,
            )
        predictions.to_csv(prediction_path, index=False)
        return FitPredictionResult(
            family=family,
            config=config,
            decision_timestamp=pd.Timestamp(timestamp),
            training_rows=int(len(train)),
            scoring_rows=int(len(score)),
            fit_wall_seconds=float(fit_wall),
            artifact_path=artifact_path,
            artifact_hash=sha256_file(artifact_path),
            prediction_path=prediction_path,
            prediction_hash=sha256_file(prediction_path),
        )
