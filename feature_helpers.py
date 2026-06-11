"""
feature_helpers.py
High-level feature-preparation wrappers used by model_evaluation.py.
Adapted from helper_prepare_input_v2.py (standalone — local import only).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack

import data_helpers as hdp     # local import — same folder


def prepare_X_train(df_input, text_mode="dense", num_mode="raw",
                    text_attributes=None, cat_attributes=None,
                    boolean_attributes=None, num_attributes=None,
                    max_categories=10):
    """
    Fit all feature transformers on training data and return feature matrix.

    Parameters
    ----------
    text_mode : "dense"  → TF-IDF + TruncatedSVD (300-d, normalised)
                "sparse" → raw TF-IDF sparse matrix
    num_mode  : "raw"    → imputed only
                "std"    → imputed + standardised
    num_attributes : list of numeric columns to include (e.g. salary + LLM score)

    Returns
    -------
    X_train     : csr_matrix
    transformers: dict of fitted transformers (pass to prepare_X_test)
    """
    if text_mode not in {"dense", "sparse"}:
        raise ValueError("text_mode must be 'dense' or 'sparse'")
    if num_mode not in {"raw", "std"}:
        raise ValueError("num_mode must be 'raw' or 'std'")

    # Text
    if text_mode == "dense":
        X_text_dense, vec, svd, norm = hdp.make_denseTFIDF_SVD(df_input, text_attributes)
        X_text = csr_matrix(X_text_dense)
        text_tfms = {"tfidf_vectorizer": vec, "svd": svd, "normalizer": norm}
    else:
        X_text, vec = hdp.make_sparseTFIDF(df_input, text_attributes)
        text_tfms = {"tfidf_vectorizer": vec}

    # Numeric
    if num_mode == "raw":
        X_num, imp_num = hdp.make_rawNum(df_input, num_attributes)
        num_tfms = {"num_imputer": imp_num}
    else:
        X_num, imp_num, scl_num = hdp.make_stdNum(df_input, num_attributes)
        num_tfms = {"num_imputer": imp_num, "num_scaler": scl_num}

    # Boolean
    X_bool, imp_bool = hdp.make_bool(df_input, boolean_attributes)

    # Categorical
    X_cat, cat_encoders, cat_metadata = hdp.make_onehot_topk_train(
        df_input, cat_attributes, max_categories)

    X_train = hstack([X_text, X_num, X_bool, X_cat], format="csr")

    transformers = {
        **text_tfms, **num_tfms,
        "bool_imputer":  imp_bool,
        "cat_encoders":  cat_encoders,
        "cat_metadata":  cat_metadata,
        "text_mode":     text_mode,
        "num_mode":      num_mode,
        "text_attributes":    text_attributes,
        "cat_attributes":     cat_attributes,
        "boolean_attributes": boolean_attributes,
        "num_attributes":     num_attributes,
    }
    return X_train, transformers


def prepare_X_test(df_input, transformers):
    """Apply fitted transformers to test data (no re-fitting)."""
    text_mode = transformers["text_mode"]
    num_mode  = transformers["num_mode"]
    text_attributes    = transformers.get("text_attributes")
    cat_attributes     = transformers.get("cat_attributes")
    boolean_attributes = transformers.get("boolean_attributes")
    num_attributes     = transformers.get("num_attributes")

    # Text
    if text_mode == "dense":
        X_text = csr_matrix(hdp.transform_denseTFIDF_SVD(
            df_input, transformers["tfidf_vectorizer"],
            transformers["svd"], transformers["normalizer"], text_attributes))
    else:
        X_text = hdp.transform_sparseTFIDF(
            df_input, transformers["tfidf_vectorizer"], text_attributes)

    # Numeric
    if num_mode == "raw":
        X_num = hdp.transform_rawNum(
            df_input, transformers["num_imputer"], num_attributes)
    else:
        X_num = hdp.transform_stdNum(
            df_input, transformers["num_imputer"],
            transformers["num_scaler"], num_attributes)

    # Boolean
    X_bool = hdp.transform_bool(
        df_input, transformers["bool_imputer"], boolean_attributes)

    # Categorical
    X_cat = hdp.transform_onehot_topk_test(
        df_input, transformers["cat_encoders"], transformers["cat_metadata"])

    return hstack([X_text, X_num, X_bool, X_cat], format="csr")
