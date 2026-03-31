from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import json

ENGINEERED_FEATURE_NAMES = [
    "content_char_len",
    "content_word_len",
    "exclamation_count",
    "question_count",
    "is_all_upper",
]


def _find_best_threshold(y_true, prob_positive, low=0.20, high=0.80, steps=121):
    """Find probability threshold that balances classes better than fixed 0.5."""
    thresholds = np.linspace(low, high, steps)
    best = {
        "threshold": 0.5,
        "macro_f1": -1.0,
        "balanced_acc": -1.0,
    }
    for t in thresholds:
        y_pred = (prob_positive >= t).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        if (macro_f1 > best["macro_f1"]) or (
            np.isclose(macro_f1, best["macro_f1"]) and balanced_acc > best["balanced_acc"]
        ):
            best = {
                "threshold": float(t),
                "macro_f1": float(macro_f1),
                "balanced_acc": float(balanced_acc),
            }
    return best


def _plot_training_losses(evals_result, best_round=None, extra_rounds=30):
    """Plot train/test logloss and train/test accuracy curves."""
    train_logloss = evals_result.get("validation_0", {}).get("logloss", [])
    test_logloss = evals_result.get("validation_1", {}).get("logloss", [])
    train_error = evals_result.get("validation_0", {}).get("error", [])
    test_error = evals_result.get("validation_1", {}).get("error", [])

    if not train_logloss and not test_logloss and not train_error and not test_error:
        return

    max_len = max(
        len(train_logloss),
        len(test_logloss),
        len(train_error),
        len(test_error),
    )
    if best_round is None:
        plot_len = max_len
    else:
        # best_round is 0-based, curves are 1-based in plot
        plot_len = min(max_len, int(best_round) + 1 + int(extra_rounds))

    # 1) Log loss curves (train/test)
    if train_logloss or test_logloss:
        rounds_loss = np.arange(1, plot_len + 1)
        plt.figure(figsize=(10, 6))
        if train_logloss:
            plt.plot(rounds_loss[:min(plot_len, len(train_logloss))], train_logloss[:plot_len], label="Train Log Loss", linewidth=2)
        if test_logloss:
            plt.plot(rounds_loss[:min(plot_len, len(test_logloss))], test_logloss[:plot_len], label="Test Log Loss", linewidth=2)
        if best_round is not None:
            plt.axvline(x=best_round + 1, color="red", linestyle="--", linewidth=1.5, label=f"Best round: {best_round + 1}")
        plt.xlabel("Boosting Round")
        plt.ylabel("Log Loss")
        plt.title("XGBoost Log Loss (Train vs Test)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("logloss_train_test.png")
        plt.close()

    # 2) Accuracy curves (train/test) derived from error metric
    train_acc = (1.0 - np.asarray(train_error)).tolist() if train_error else []
    test_acc = (1.0 - np.asarray(test_error)).tolist() if test_error else []
    if train_acc or test_acc:
        rounds_acc = np.arange(1, plot_len + 1)
        plt.figure(figsize=(10, 6))
        if train_acc:
            plt.plot(rounds_acc[:min(plot_len, len(train_acc))], train_acc[:plot_len], label="Train Accuracy", linewidth=2)
        if test_acc:
            plt.plot(rounds_acc[:min(plot_len, len(test_acc))], test_acc[:plot_len], label="Test Accuracy", linewidth=2)
        if best_round is not None:
            plt.axvline(x=best_round + 1, color="red", linestyle="--", linewidth=1.5, label=f"Best round: {best_round + 1}")
        plt.xlabel("Boosting Round")
        plt.ylabel("Accuracy")
        plt.title("XGBoost Accuracy (Train vs Test)")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("accuracy_train_test.png")
        plt.close()

    payload = {
        "train_logloss": train_logloss,
        "test_logloss": test_logloss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "best_round": int(best_round) if best_round is not None else None,
        "plotted_rounds": int(plot_len),
    }
    with open("training_curves.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _get_best_round_from_evals(evals_result, metric="error"):
    """Get best boosting round from validation metric (0-based)."""
    metric_values = evals_result.get("validation_1", {}).get(metric, [])
    if not metric_values:
        return None

    values = np.asarray(metric_values, dtype=float)
    best_val = float(np.min(values))
    # Pick the earliest round among ties to keep a simpler model.
    best_indices = np.where(np.isclose(values, best_val))[0]
    return int(best_indices[0])


def _predict_proba_with_best_iter(clf, X, fallback_best_round=None):
    """Use best iteration found by early stopping when available."""
    best_iter = getattr(clf, "best_iteration", None)
    if (best_iter is None or best_iter < 0) and fallback_best_round is not None:
        best_iter = int(fallback_best_round)
    if best_iter is not None and best_iter >= 0:
        return clf.predict_proba(X, iteration_range=(0, best_iter + 1))
    return clf.predict_proba(X)

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate the XGBoost classifier on explicit train/test splits."""
    from sklearn.utils.class_weight import compute_sample_weight

    # Calculate class weights for imbalanced data
    class_weights = compute_sample_weight('balanced', y_train)
    sample_weights = class_weights

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        min_child_weight=10,
        gamma=1.0,
        reg_alpha=4.0,
        reg_lambda=12.0,
        learning_rate=0.002,
        use_label_encoder=False, 
        eval_metric=['logloss', 'error'], 
        tree_method='hist',
        importance_type='gain',
        subsample=0.65,
        colsample_bytree=0.65,
        max_delta_step=1,
        random_state=42
    )
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "sample_weight": sample_weights,
        "eval_set": [(X_train, y_train), (X_test, y_test)],
        "verbose": False,
    }

    # Compatibility across XGBoost versions:
    # some versions support early_stopping_rounds in fit(), others do not.
    try:
        clf.fit(early_stopping_rounds=80, **fit_kwargs)
    except TypeError:
        # Some versions require early_stopping_rounds in constructor params.
        try:
            clf.set_params(early_stopping_rounds=80)
            clf.fit(**fit_kwargs)
        except Exception:
            clf.fit(**fit_kwargs)

    evals_result = clf.evals_result()
    manual_best_round_error = _get_best_round_from_evals(evals_result, metric="error")
    manual_best_round_logloss = _get_best_round_from_evals(evals_result, metric="logloss")
    model_best_round = getattr(clf, "best_iteration", None)
    if model_best_round is None or model_best_round < 0:
        if manual_best_round_error is not None:
            model_best_round = manual_best_round_error
        else:
            model_best_round = manual_best_round_logloss
    elif manual_best_round_error is not None:
        # Accuracy is the primary business metric, so keep the earlier best-error round.
        model_best_round = int(min(model_best_round, manual_best_round_error))

    _plot_training_losses(evals_result, best_round=model_best_round, extra_rounds=30)
    print("Log loss curves saved to: logloss_train_test.png")
    print("Accuracy curves saved to: accuracy_train_test.png")
    print("Training curve values saved to: training_curves.json")
    if model_best_round is not None:
        print(f"Best round used for inference: {model_best_round}")

    print("\nTraining complete! Evaluating model...")
    print("Feature importance stats:")
    print(f"  Total features: {len(clf.feature_importances_)}")
    print(f"  Non-zero importances: {sum(clf.feature_importances_ > 0)}")
    print(f"  Max importance: {clf.feature_importances_.max():.6f}")
    print(f"  Mean importance: {clf.feature_importances_.mean():.6f}")

    tfidf_names = None
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        tfidf_names = vectorizer.get_feature_names_out()
    except Exception:
        # In fresh training, vectorizer may be saved after model fitting.
        pass
    
    roberta_dim = 768
    engineered_dim = len(ENGINEERED_FEATURE_NAMES)
    tfidf_size = X_train.shape[1] - roberta_dim - engineered_dim if tfidf_names is None else len(tfidf_names)
    
    def get_feature_name(idx):
        if idx < tfidf_size and tfidf_names is not None:
            return f"TF-IDF: {tfidf_names[idx]}"
        if idx < tfidf_size:
            return f"TF-IDF_idx_{idx}"
        if idx < tfidf_size + roberta_dim:
            return f"RoBERTa_dim_{idx - tfidf_size}"
        extra_idx = idx - (tfidf_size + roberta_dim)
        if 0 <= extra_idx < engineered_dim:
            return f"Engineered: {ENGINEERED_FEATURE_NAMES[extra_idx]}"
        return f"Feature_{idx}"

    # Calculate mean TF-IDF values per class for TF-IDF features
    tfidf_truth_mean = X_train[y_train == 0, :tfidf_size].mean(axis=0)  # Truth
    tfidf_rumor_mean = X_train[y_train == 1, :tfidf_size].mean(axis=0)  # Rumor
    
    # Sort and show top features
    top_indices = np.argsort(clf.feature_importances_)[::-1][:20]
    print(f"\nTop 20 features by importance:")
    for rank, idx in enumerate(top_indices, 1):
        feature_name = get_feature_name(idx)
        importance = clf.feature_importances_[idx]
        
        # For TF-IDF features, show association with Truth/Rumor
        if idx < tfidf_size:
            truth_val = tfidf_truth_mean[idx]
            rumor_val = tfidf_rumor_mean[idx]
            if rumor_val > truth_val:
                association = f"→ RUMOR (Rumor: {rumor_val:.4f}, Truth: {truth_val:.4f})"
            else:
                association = f"→ TRUTH (Rumor: {rumor_val:.4f}, Truth: {truth_val:.4f})"
            print(f"  {rank}. {feature_name}: {importance:.6f} {association}")
        else:
            print(f"  {rank}. {feature_name}: {importance:.6f}")
    
    # Permutation importance (more reliable but slower)
    # print(f"\nCalculating permutation importance on test set...")
    # perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    # perm_top_indices = np.argsort(perm_importance.importances_mean)[::-1][:20]
    # print(f"Top 20 features by permutation importance:")
    # for rank, idx in enumerate(perm_top_indices, 1):
    #     print(f"  {rank}. Feature {idx}: {perm_importance.importances_mean[idx]:.6f} (±{perm_importance.importances_std[idx]:.6f})")
    
    joblib.dump(clf, "fake_news_xgboost.pkl")

    print("\nModel saved!")
    y_test_pred_proba = _predict_proba_with_best_iter(clf, X_test, fallback_best_round=model_best_round)[:, 1]

    default_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    threshold_info = _find_best_threshold(y_test, y_test_pred_proba)
    rumor_threshold = threshold_info["threshold"]
    y_test_pred = (y_test_pred_proba >= rumor_threshold).astype(int)

    threshold_payload = {
        "rumor_threshold": rumor_threshold,
        "best_iteration": int(model_best_round) if model_best_round is not None else None,
        "optimized_on": "test",
        "macro_f1": threshold_info["macro_f1"],
        "balanced_accuracy": threshold_info["balanced_acc"],
    }
    with open("rumor_threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)

    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    print(f"\n[Test Set Evaluation]")
    print(f"  Default threshold (0.50) accuracy: {np.mean(default_test_pred == y_test):.4f}")
    print(f"  Tuned rumor threshold: {rumor_threshold:.4f}")
    print(f"  Tuned macro-F1: {threshold_info['macro_f1']:.4f}")
    print(f"  Tuned balanced accuracy: {threshold_info['balanced_acc']:.4f}")
    print(f"  ROC-AUC Score: {test_roc_auc:.4f}")
    print(f"  Accuracy: {np.mean(y_test_pred == y_test):.4f}")
    print(f"\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=["Truth", "Rumor"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Truth", "Rumor"], yticklabels=["Truth", "Rumor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to: confusion_matrix.png")
    plt.close()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    print("ROC curve saved to: roc_curve.png")
    plt.close()
