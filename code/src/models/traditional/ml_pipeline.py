"""sklearn Pipeline factory for the classical ML baselines."""

from __future__ import annotations

import logging

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def build_ml_pipeline(cfg: dict) -> Pipeline:
    """Build a sklearn Pipeline for the configured model."""
    ml_model = str(cfg.get("ml_model", "lda")).lower()
    pca_components = cfg.get("pca_components", None)
    seed = int(cfg.get("seed", 42))

    # ── Scaler ───────────────────────────────────────────────────────────
    steps: list = [("scaler", StandardScaler())]

    # ── Optional PCA ─────────────────────────────────────────────────────
    if pca_components is not None:
        n = int(pca_components)
        steps.append(("pca", PCA(n_components=n, random_state=seed)))
        logger.info(f"PCA: n_components={n}")

    # ── Classifier ───────────────────────────────────────────────────────
    if ml_model == "lda":
        solver = str(cfg.get("lda_solver", "svd"))
        shrinkage = cfg.get("lda_shrinkage", None)

        # 'svd' solver does not support shrinkage
        if solver == "svd" and shrinkage is not None:
            logger.warning(
                "LDA solver='svd' does not support shrinkage. "
                "Set lda_solver='lsqr' or 'eigen' to use shrinkage. Ignoring shrinkage."
            )
            shrinkage = None

        classifier = LDA(solver=solver, shrinkage=shrinkage)
        logger.info(f"Classifier: LDA(solver={solver}, shrinkage={shrinkage})")

    elif ml_model == "svm":
        kernel = str(cfg.get("svm_kernel", "linear"))
        C = float(cfg.get("svm_C", 1.0))
        gamma = cfg.get("svm_gamma", "scale")
        # SVM with probability=True enables predict_proba (needed for AUROC)
        classifier = SVC(
            kernel=kernel, C=C, gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
        logger.info(f"Classifier: SVM(kernel={kernel}, C={C}, gamma={gamma})")

    elif ml_model == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for ml_model='xgb'. "
                "Install it with: pip install xgboost"
            ) from exc
        n_est = int(cfg.get("xgb_n_estimators", 300))
        max_depth = int(cfg.get("xgb_max_depth", 4))
        lr = float(cfg.get("xgb_learning_rate", 0.05))
        classifier = XGBClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            learning_rate=lr,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        logger.info(
            f"Classifier: XGBoost(n_estimators={n_est}, max_depth={max_depth}, lr={lr})"
        )

    else:
        raise ValueError(
            f"Unknown ml_model='{ml_model}'. "
            "Supported: 'lda', 'svm', 'xgb'"
        )

    steps.append(("classifier", classifier))
    pipeline = Pipeline(steps)

    total_steps = " → ".join(n for n, _ in pipeline.steps)
    logger.info(f"Pipeline: {total_steps}")
    return pipeline
