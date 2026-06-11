"""
data_helpers.py
Low-level feature-engineering utilities for the EMSCAD fraud detection system.
Adapted from helper_data_parsing_v2.py (standalone — no package-relative imports).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MultiLabelBinarizer, Normalizer, OneHotEncoder, StandardScaler,
)

# ── Default column lists ──────────────────────────────────────────────────────
DEFAULT_TEXT_ATTRIBUTES    = ['description', 'requirements', 'benefits', 'company_profile']
DEFAULT_CAT_ATTRIBUTES     = ['title', 'location', 'department', 'employment_type',
                               'required_experience', 'required_education',
                               'industry', 'function']
DEFAULT_BOOLEAN_ATTRIBUTES = ['telecommuting', 'has_company_logo', 'has_questions',
                               'salary_range_missing']
DEFAULT_NUM_ATTRIBUTES     = ['salary_lower', 'salary_upper']


# ── Text features ─────────────────────────────────────────────────────────────

def make_denseTFIDF_SVD(df_input, text_attributes=None):
    if text_attributes is None:
        text_attributes = DEFAULT_TEXT_ATTRIBUTES
    combined = df_input[text_attributes].fillna("").apply(
        lambda r: " ".join(r.values.astype(str)), axis=1
    )
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=5000, min_df=5, max_df=0.90,
        ngram_range=(1, 1), lowercase=True, dtype=np.float32,
    )
    X_tfidf = vectorizer.fit_transform(combined)
    svd      = TruncatedSVD(n_components=300, random_state=42, algorithm="randomized")
    X_svd    = svd.fit_transform(X_tfidf).astype(np.float32)
    norm     = Normalizer(copy=False)
    X_svd    = norm.fit_transform(X_svd)
    return X_svd, vectorizer, svd, norm

def transform_denseTFIDF_SVD(df_input, vectorizer, svd, normalizer, text_attributes=None):
    if text_attributes is None:
        text_attributes = DEFAULT_TEXT_ATTRIBUTES
    combined = df_input[text_attributes].fillna("").apply(
        lambda r: " ".join(r.values.astype(str)), axis=1
    )
    return normalizer.transform(svd.transform(vectorizer.transform(combined)).astype(np.float32))

def make_sparseTFIDF(df_input, text_attributes=None):
    if text_attributes is None:
        text_attributes = DEFAULT_TEXT_ATTRIBUTES
    combined = df_input[text_attributes].fillna("").apply(
        lambda r: " ".join(r.values.astype(str)), axis=1
    )
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=5000, min_df=5, max_df=0.90,
        ngram_range=(1, 1), lowercase=True, dtype=np.float32,
    )
    return vectorizer.fit_transform(combined), vectorizer

def transform_sparseTFIDF(df_input, vectorizer, text_attributes=None):
    if text_attributes is None:
        text_attributes = DEFAULT_TEXT_ATTRIBUTES
    combined = df_input[text_attributes].fillna("").apply(
        lambda r: " ".join(r.values.astype(str)), axis=1
    )
    return vectorizer.transform(combined)


# ── Numeric features ──────────────────────────────────────────────────────────

def make_rawNum(df_input, num_attributes=None, strategy="median"):
    if num_attributes is None:
        num_attributes = DEFAULT_NUM_ATTRIBUTES
    X   = df_input[num_attributes].to_numpy(dtype=np.float32, copy=True)
    imp = SimpleImputer(strategy=strategy)
    return sp.csr_matrix(imp.fit_transform(X).astype(np.float32)), imp

def transform_rawNum(df_input, imputer, num_attributes=None):
    if num_attributes is None:
        num_attributes = DEFAULT_NUM_ATTRIBUTES
    X = df_input[num_attributes].to_numpy(dtype=np.float32, copy=True)
    return sp.csr_matrix(imputer.transform(X).astype(np.float32))

def make_stdNum(df_input, num_attributes=None, strategy="median"):
    if num_attributes is None:
        num_attributes = DEFAULT_NUM_ATTRIBUTES
    X   = df_input[num_attributes].to_numpy(dtype=np.float32, copy=True)
    imp = SimpleImputer(strategy=strategy)
    X   = imp.fit_transform(X)
    scl = StandardScaler()
    return sp.csr_matrix(scl.fit_transform(X).astype(np.float32)), imp, scl

def transform_stdNum(df_input, imputer, scaler, num_attributes=None):
    if num_attributes is None:
        num_attributes = DEFAULT_NUM_ATTRIBUTES
    X = df_input[num_attributes].to_numpy(dtype=np.float32, copy=True)
    return sp.csr_matrix(scaler.transform(imputer.transform(X)).astype(np.float32))


# ── Boolean features ──────────────────────────────────────────────────────────

def make_bool(df_input, boolean_attributes=None, strategy="most_frequent"):
    if boolean_attributes is None:
        boolean_attributes = DEFAULT_BOOLEAN_ATTRIBUTES
    X   = df_input[boolean_attributes].to_numpy(dtype=np.float32, copy=True)
    imp = SimpleImputer(strategy=strategy)
    return sp.csr_matrix(np.rint(imp.fit_transform(X)).astype(np.int32)), imp

def transform_bool(df_input, imputer, boolean_attributes=None):
    if boolean_attributes is None:
        boolean_attributes = DEFAULT_BOOLEAN_ATTRIBUTES
    X = df_input[boolean_attributes].to_numpy(dtype=np.float32, copy=True)
    return sp.csr_matrix(np.rint(imputer.transform(X)).astype(np.int32))


# ── Categorical features ──────────────────────────────────────────────────────

REQUIRED_EDU_MAP = {
    "Bachelor's Degree": "Bachelors",  "Master's Degree": "Masters",
    "High School or equivalent": "High School", "Associate Degree": "Associate",
    "Doctorate": "Doctorate",           "Certification": "Certification",
    "Professional": "Certification",   "Some College Coursework Completed": "Some College",
    "Some High School Coursework": "Some HS", "Unspecified": "Unspecified",
}

def _normalize_categories(df_input, cat_attributes=None):
    if cat_attributes is None:
        cat_attributes = DEFAULT_CAT_ATTRIBUTES
    df = df_input[cat_attributes].copy()
    df['required_education'] = df['required_education'].map(REQUIRED_EDU_MAP).fillna(
        df['required_education'])
    return df

def _is_multilabel_column(series, sample_size=100):
    non_null = series.dropna()
    return len(non_null) > 0 and non_null.head(sample_size).astype(str).str.contains(
        ',', na=False).any()

def _parse_multilabel_values(series, missing_token="__MISSING__"):
    result = []
    for val in series:
        if pd.isna(val) or val == "" or val is None:
            result.append([missing_token])
        else:
            labels = [l.strip() for l in str(val).split(",") if l.strip()]
            result.append(labels if labels else [missing_token])
    return result

def _learn_category_sets(df_cat, max_categories=10,
                          other_token="__OTHER__", missing_token="__MISSING__",
                          min_freq=2):
    keep_sets, multilabel_cols = {}, set()
    for col in df_cat.columns:
        if _is_multilabel_column(df_cat[col]):
            multilabel_cols.add(col)
            parsed  = _parse_multilabel_values(df_cat[col], missing_token)
            all_lbl = [l for row in parsed for l in row]
            counts  = pd.Series(all_lbl).value_counts()
            q = counts[counts >= min_freq]
            keep_sets[col] = (
                set(q.head(max_categories).index) | {missing_token}
            )
        else:
            vc = df_cat[col].fillna(missing_token).value_counts(dropna=False)
            keep_sets[col] = set(vc.head(max_categories).index.astype(str)) | {missing_token}
    return keep_sets, multilabel_cols

def _apply_category_buckets(df_cat, keep_sets, multilabel_cols,
                             other_token="__OTHER__", missing_token="__MISSING__"):
    regular_data, multilabel_data = {}, {}
    for col in df_cat.columns:
        keep = keep_sets[col]
        if col in multilabel_cols:
            parsed = _parse_multilabel_values(df_cat[col], missing_token)
            multilabel_data[col] = [
                ([l for l in row if l in keep] or [missing_token]) for row in parsed
            ]
        else:
            s = df_cat[col].fillna(missing_token).astype(str)
            regular_data[col] = s.where(s.isin(keep), other_token)
    df_reg = pd.DataFrame(regular_data, index=df_cat.index) if regular_data else None
    return df_reg, multilabel_data

def make_onehot_topk_train(df_input, cat_attributes=None, max_categories=10):
    if cat_attributes is None:
        cat_attributes = DEFAULT_CAT_ATTRIBUTES
    df_cat = _normalize_categories(df_input, cat_attributes).astype("object")
    keep_sets, multilabel_cols = _learn_category_sets(
        df_cat, max_categories=max_categories, min_freq=2)
    df_reg, ml_data = _apply_category_buckets(df_cat, keep_sets, multilabel_cols)
    matrices, reg_enc, ml_encs = [], None, {}
    if df_reg is not None and len(df_reg.columns) > 0:
        reg_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        matrices.append(reg_enc.fit_transform(df_reg))
    for col in multilabel_cols:
        mlb = MultiLabelBinarizer(sparse_output=True)
        matrices.append(mlb.fit_transform(ml_data[col]))
        ml_encs[col] = mlb
    X_cat = (sp.hstack(matrices, format="csr").astype(np.float32)
             if matrices else sp.csr_matrix((len(df_input), 0), dtype=np.float32))
    encoders = {'regular': reg_enc, 'multilabel': ml_encs}
    metadata = {'keep_sets': keep_sets, 'multilabel_cols': multilabel_cols,
                'cat_attributes': cat_attributes, 'max_categories': max_categories}
    return X_cat, encoders, metadata

def transform_onehot_topk_test(df_input, encoders, metadata):
    cat_attributes  = metadata.get('cat_attributes', DEFAULT_CAT_ATTRIBUTES)
    keep_sets       = metadata['keep_sets']
    multilabel_cols = metadata['multilabel_cols']
    df_cat = _normalize_categories(df_input, cat_attributes).astype("object")
    df_reg, ml_data = _apply_category_buckets(df_cat, keep_sets, multilabel_cols)
    matrices = []
    reg_enc  = encoders['regular']
    if reg_enc is not None and df_reg is not None and len(df_reg.columns) > 0:
        matrices.append(reg_enc.transform(df_reg))
    for col in multilabel_cols:
        matrices.append(encoders['multilabel'][col].transform(ml_data[col]))
    return (sp.hstack(matrices, format="csr").astype(np.float32)
            if matrices else sp.csr_matrix((len(df_input), 0), dtype=np.float32))
