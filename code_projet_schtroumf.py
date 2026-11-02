"""LELEC2870 2026-2027 - Smurf heart-failure risk - Part 1 (linear baseline).

Sources: 2026-2027 project guidelines; LELEC2870 Linear regression,
Feature selection, Model selection slides.

Dependencies: Python >=3.10; numpy, pandas, scikit-learn, reportlab, pypdf.

Run (with Moodle CSV files somewhere under DATA_ROOT):
    python partie1_linear.py --data-root . --out-root resultats

This script NEVER manufactures predictions if the original data is missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import sklearn
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
N_SPLITS = 5
MAX_ORIGINAL_FEATURES = 6
ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
ALL_FEATURES = (
    'age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin',
    'height', 'potassium', 'profession', 'sarsaparilla',
    'smurfberry liquor', 'smurfin donuts', 'vitamin d', 'weight',
)
CATEGORICAL = {'profession', 'sarsaparilla', 'smurfberry liquor', 'smurfin donuts'}


def rw_rmse(y_true, y_pred):
    """Exactly the risk-weighted RMSE requested by the project guidelines."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred have different lengths.')
    weights = 1.0 + y_true / 0.1
    if np.any(weights <= 0):
        raise ValueError('Risk-based weights must be positive.')
    return float(np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights)))


def normalize_name(value):
    value = str(value).lower().strip()
    value = re.sub(r'[_\-]+', ' ', value)
    return re.sub(r'\s+', ' ', value)


def find_csv(root: Path, desired: str) -> Path:
    """Locate X_train.csv etc., including inside unzipped nested folders."""
    desired_flat = re.sub(r'[^a-z0-9]', '', desired.lower())
    matches = []
    for p in root.rglob('*.csv'):
        if re.sub(r'[^a-z0-9]', '', p.name.lower()) == desired_flat:
            matches.append(p)
    if not matches:
        raise FileNotFoundError(f'Missing {desired}. Download it from Moodle and unzip into {root}.')
    if len(matches) > 1:
        raise FileExistsError(f'Several {desired} files found: {matches}. Keep only one version.')
    return matches[0]


def load_x(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python')
    if df.empty:
        raise ValueError(f'Empty X file: {path}')
    renamed = {name: normalize_name(name) for name in df.columns}
    df = df.rename(columns=renamed)
    if df.columns.duplicated().any():
        raise ValueError(f'Duplicate column names after normalization in {path}.')
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns in {path.name}: {missing}. Actual columns: {list(df.columns)}')
    # Ignore filenames of heart scans, IDs, and extra columns; part 1 is tabular only.
    data = df.loc[:, ALL_FEATURES].copy()
    for col in ALL_FEATURES:
        if col in CATEGORICAL:
            data[col] = data[col].astype('string').str.strip().str.lower()
            data[col] = data[col].str.replace(r'[_\-]+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
            data[col] = data[col].replace({'': pd.NA, 'nan': pd.NA, 'none': pd.NA, '?': pd.NA})
            data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
        else:
            data[col] = pd.to_numeric(
                data[col].astype('string').str.strip().str.replace(',', '.', regex=False),
                errors='coerce',
            )
    return data


def load_y(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 1:
        raise ValueError(f'{path.name} must have only one column; got {df.shape[1]}.')
    y = pd.to_numeric(df.iloc[:, 0].astype('string').str.replace(',', '.', regex=False), errors='coerce')
    if y.isna().any():
        raise ValueError(f'{path.name} contains missing, non-numeric values, or a header.')
    values = y.to_numpy(dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f'{path.name}: targets should be 10-year risks in [0,1], not percentages 0-100.')
    return values


def build_pipeline(features, alpha: float) -> Pipeline:
    numeric = [f for f in features if f not in CATEGORICAL]
    category = [f for f in features if f in CATEGORICAL]
    transformers = []
    if numeric:
        transformers.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric))
    if category:
        transformers.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), category))
    preprocess = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0)
    regressor = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return Pipeline([('preprocess', preprocess), ('model', regressor)])


def fit_model(X, y, features, alpha, weighted):
    pipeline = build_pipeline(features, alpha)
    params = {'model__sample_weight': 1 + y / 0.1} if weighted else {}
    pipeline.fit(X.loc[:, features], y, **params)
    return pipeline


def predict(model, X, features, clipping):
    pred = np.asarray(model.predict(X.loc[:, features]), dtype=float).ravel()
    return np.clip(pred, 0, 1) if clipping else pred


def prepare_cv_cache(X, y, splits):
    """Fit each per-column preprocessing transformer once per training fold.

    This is mathematically equivalent to doing it inside every subset pipeline,
    but much faster when evaluating hundreds of feature subsets. Each validation
    fold is transformed using its own training-fold statistics only (no leakage).
    """
    cache = []
    for idx_tr, idx_val in splits:
        xt, xv = X.iloc[idx_tr], X.iloc[idx_val]
        tr_blocks, val_blocks = {}, {}
        for name in ALL_FEATURES:
            if name in CATEGORICAL:
                prep = Pipeline([
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ])
            else:
                prep = Pipeline([
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler()),
                ])
            tr_blocks[name] = prep.fit_transform(xt[[name]])
            val_blocks[name] = prep.transform(xv[[name]])
        cache.append({'train_blocks': tr_blocks, 'val_blocks': val_blocks,
                      'y_train': y[idx_tr], 'y_val': y[idx_val]})
    return cache


def cv_scores(cache, features, alpha, weighted, clipping):
    scores = []
    for fold in cache:
        x_train = np.column_stack([fold['train_blocks'][f] for f in features])
        x_val = np.column_stack([fold['val_blocks'][f] for f in features])
        y_train, y_val = fold['y_train'], fold['y_val']
        model = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
        params = {'sample_weight': 1. + y_train / 0.1} if weighted else {}
        model.fit(x_train, y_train, **params)
        prediction = model.predict(x_val)
        if clipping:
            prediction = np.clip(prediction, 0., 1.)
        scores.append(rw_rmse(y_val, prediction))
    return np.asarray(scores)


def greedy_select(cache, features_all):
    """Wrapper: forward selection using ONLY X_train/y_train with weighted 5-fold CV.

    Scores for different subset sizes are computed with a fixed, regularized linear
    model (Ridge alpha=1). Alpha and sample weighting are optimized afterwards.
    The held-out test set is never used for feature selection.
    """
    chosen = []
    remaining = list(features_all)
    history = []
    for k in range(1, min(MAX_ORIGINAL_FEATURES, len(features_all)) + 1):
        candidates = []
        for name in remaining:
            subset = chosen + [name]
            folds = cv_scores(cache, subset, alpha=1., weighted=True, clipping=True)
            candidates.append((float(folds.mean()), name, float(folds.std(ddof=1))))
        candidates.sort(key=lambda entry: (entry[0], entry[1]))
        score, best_name, std = candidates[0]
        chosen.append(best_name)
        remaining.remove(best_name)
        history.append({'size': k, 'added': best_name, 'features': chosen.copy(),
                        'cv_rw_rmse': score, 'cv_std': std})
        print(f'  {k} variable(s): {chosen} | CV RW-RMSE = {score:.5f}')
    best_step = min(history, key=lambda h: (h['cv_rw_rmse'], h['size']))
    return best_step['features'], history


def select_model(cache, features):
    candidates = []
    for alpha in ALPHAS:
        for weighted in (True, False):
            for clipping in (True, False):
                folds = cv_scores(cache, features, alpha, weighted, clipping)
                candidates.append({
                    'alpha': alpha, 'weighted': weighted, 'clipping': clipping,
                    'cv_rw_rmse': float(folds.mean()),
                    'cv_std': float(folds.std(ddof=1)),
                })
    candidates.sort(key=lambda c: c['cv_rw_rmse'])
    return candidates[0], candidates


def make_report_pdf(destination: Path, result: dict):
    """Create an A4 one-page PDF with real computed numbers only."""
    page_w, page_h = A4
    doc = SimpleDocTemplate(str(destination), pagesize=A4,
                            topMargin=1.55*cm, bottomMargin=1.3*cm,
                            leftMargin=1.65*cm, rightMargin=1.65*cm)
    palette = {'navy': colors.HexColor('#142741'), 'teal': colors.HexColor('#15788c'),
               'muted': colors.HexColor('#526376'), 'line': colors.HexColor('#dfe5eb')}
    styles = {
        'eyebrow': ParagraphStyle('eyebrow', fontName='Helvetica-Bold', fontSize=8,
                                  leading=10, textColor=palette['teal'], spaceAfter=5),
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=16,
                                leading=18, textColor=palette['navy'], spaceAfter=5),
        'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=8,
                                   leading=11, textColor=palette['muted'], spaceAfter=12),
        'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=9.3,
                                  leading=12, textColor=palette['navy'], spaceBefore=9, spaceAfter=3),
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.55,
                               leading=12.3, spaceAfter=5, textColor=palette['navy']),
        'foot': ParagraphStyle('foot', fontName='Helvetica', fontSize=7.2,
                               leading=10, textColor=palette['muted'], spaceBefore=9),
        'metric': ParagraphStyle('metric', fontName='Helvetica', fontSize=8.0,
                                 leading=10.5, textColor=palette['navy']),
    }
    def p(t, style='body'):
        return Paragraph(t, styles[style])
    d = result
    features_text = escape(', '.join(d['selected_features']))
    model = ('Weighted least squares' if d['selected_model']['weighted'] else 'Ordinary least squares') if d['selected_model']['alpha'] == 0 else f"Ridge (alpha={d['selected_model']['alpha']:g})"
    training = 'risk-weighted fitting' if d['selected_model']['weighted'] else 'ordinary fitting'
    clip = ('clipped to [0, 1]' if d['selected_model']['clipping'] else 'without clipping')
    story = [p('LELEC2870  |  MACHINE LEARNING PROJECT  |  2026-2027', 'eyebrow'),
             p('Part 1 - Linear model (baseline)', 'title'),
             p('Smurf heart-failure risk  |  Interim report - 3 November 2026', 'subtitle'),
             p('Objective and dataset', 'heading'),
             p('We predict the 10-year heart-failure risk from tabular medical data. ' 
               'The labeled dataset contains 1,000 training and 500 held-out test patients. '
               'Images were deliberately excluded from this first linear baseline.'),
             p('Preprocessing', 'heading'),
             p('We removed the heart-scan filename/identifiers and retained only clinical and lifestyle variables. '
               'Missing numerical values were imputed by the training-fold median and then standardized. '
               'Categorical values (profession and consumption levels) were imputed with the most frequent '
               'training-fold category and one-hot encoded; unknown categories were ignored at prediction time. '
               'No missing values were found in the provided data; imputation was kept for robustness. '
               'All transformations were fitted inside each training fold to prevent leakage.'),
             p('Feature selection', 'heading'),
             p('Forward wrapper selection minimized mean five-fold cross-validated RW-RMSE on the 1,000 training cases, '
               'using risk-weighted Ridge regression as the selection model. The number of <i>original</i> variables '
               'was capped at six (one-hot columns of one variable count as one). '
               f'<b>Selected ({len(d["selected_features"])}): {features_text}.</b> '
               f'The CV RW-RMSE decreased from {d["selection_history"][0]["cv_rw_rmse"]:.4f} with one feature '
               f'to {d["selection_history"][-1]["cv_rw_rmse"]:.4f} with six.'),
             p('Model selection and assessment', 'heading'),
             p('We compared ordinary least squares with Ridge regression (alpha = 0.01 to 1,000), '
               'weighted versus unweighted training, and optional [0,1] output clipping, '
               'using the same five CV folds and the guideline-defined RW-RMSE. '
               f'The selected model was <b>{escape(model)}</b>, {training}, {clip}. '
               'The test set was evaluated only after these choices were fixed.'),
             p('Results', 'heading')]
    table_data = [
        [p('<b>Evaluation set</b>', 'metric'), p('<b>RW-RMSE</b>', 'metric'), p('<b>RMSE</b>', 'metric')],
        [p('5-fold validation (mean +/- SD)', 'metric'),
         p(f"{d['selected_model']['cv_rw_rmse']:.4f} +/- {d['selected_model']['cv_std']:.4f}", 'metric'),
         p('Not computed', 'metric')],
        [p('Training (1,000 patients)', 'metric'), p(f"{d['train_rw_rmse']:.4f}", 'metric'),
         p(f"{d['train_rmse']:.4f}", 'metric')],
        [p('Held-out test (500 patients)', 'metric'), p(f"<b>{d['test_rw_rmse']:.4f}</b>", 'metric'),
         p(f"{d['test_rmse']:.4f}", 'metric')],
    ]
    t = Table(table_data, colWidths=[8.8*cm, 4.05*cm, 3.9*cm], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f1f6f8')),
        ('LINEBELOW',(0,0),(-1,0),0.5,palette['teal']),
        ('LINEBELOW',(0,1),(-1,-2),0.35,palette['line']),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(t)
    story.extend([
        p('Interpretation and limitations', 'heading'),
        p(f"On the held-out test, RW-RMSE was {d['test_rw_rmse']:.4f}, compared with "
          f"{d['baseline_test_rw_rmse']:.4f} for a weighted constant-risk baseline "
          f"({100*(1-d['test_rw_rmse']/d['baseline_test_rw_rmse']):.1f}% lower). "
          'As feature selection and tuning both used the training CV folds, the held-out test is the more reliable '
          'generalization estimate. Selected features are predictive indicators, not proof of causality. '
          'For the 500 submitted predictions, the selected configuration was refitted on all 1,500 labeled cases.'),
        p('Sources: LELEC2870 course slides (Linear Regression; Feature Selection; Model Selection); '
          'LELEC2870 Project Guidelines 2026-2027. Software: Python, scikit-learn ' + escape(sklearn.__version__) + '.', 'foot'),
    ])
    doc.build(story)
    from pypdf import PdfReader
    page_count = len(PdfReader(str(destination)).pages)
    if page_count != 1:
        raise RuntimeError(f'Report was {page_count} pages, but the guideline limit is 1. Please adjust layout.')


def main():
    parser = argparse.ArgumentParser(description='Part 1: weighted linear baseline, evaluation and submission files.')
    parser.add_argument('--data-root', default='.', help='Folder that contains the extracted Moodle data somewhere inside.')
    parser.add_argument('--out-root', default='resultats', help='Folder for final output files.')
    args = parser.parse_args()
    root, out = Path(args.data_root).expanduser().resolve(), Path(args.out_root).expanduser().resolve()
    names = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv', 'X.csv']
    try:
        paths = {name: find_csv(root, name) for name in names}
    except (FileNotFoundError, FileExistsError) as exc:
        print(f'\nDATA NOT FOUND: {exc}\n', file=sys.stderr)
        print('Required: X_train.csv, y_train.csv, X_test.csv, y_test.csv, X.csv', file=sys.stderr)
        return 2

    print('Files found:')
    for name, path in paths.items():
        print(' ', name, '->', path)
    X_tr = load_x(paths['X_train.csv'])
    X_te = load_x(paths['X_test.csv'])
    X_un = load_x(paths['X.csv'])
    y_tr = load_y(paths['y_train.csv'])
    y_te = load_y(paths['y_test.csv'])
    if (len(X_tr), len(y_tr), len(X_te), len(y_te), len(X_un)) != (1000, 1000, 500, 500, 500):
        raise ValueError('Expected 1000 train, 500 test and 500 unlabeled rows. Found: '
                         f'{len(X_tr)}, {len(y_tr)}, {len(X_te)}, {len(y_te)}, {len(X_un)}.')
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_tr))
    print('\nPreparing leakage-free CV folds...')
    cache = prepare_cv_cache(X_tr, y_tr, splits)
    print('\nFeature selection (up to 6 original features):')
    selected, steps = greedy_select(cache, ALL_FEATURES)
    print('\nComparing linear models, weights, regularization and clipping...')
    best, candidates = select_model(cache, selected)
    print('Selected configuration:', best)

    # IMPORTANT: test is not used for either feature selection or hyperparameter tuning.
    model_train_only = fit_model(X_tr, y_tr, selected, best['alpha'], best['weighted'])
    pred_train = predict(model_train_only, X_tr, selected, best['clipping'])
    pred_test = predict(model_train_only, X_te, selected, best['clipping'])
    result = {
        'selected_features': selected, 'selection_history': steps,
        'selected_model': best, 'cv_candidates': candidates,
        'train_rw_rmse': rw_rmse(y_tr, pred_train),
        'train_rmse': float(np.sqrt(mean_squared_error(y_tr, pred_train))),
        'test_rw_rmse': rw_rmse(y_te, pred_test),
        'test_rmse': float(np.sqrt(mean_squared_error(y_te, pred_test))),
        'baseline_test_rw_rmse': rw_rmse(y_te, np.full(len(y_te), np.average(y_tr, weights=1+y_tr/0.1))),
        'n_training': int(len(y_tr)), 'n_test': int(len(y_te)),
        'n_unlabeled': int(len(X_un)), 'sklearn_version': sklearn.__version__,
    }
    print('\nTrain RW-RMSE: %.5f | Held-out test RW-RMSE: %.5f' %
          (result['train_rw_rmse'], result['test_rw_rmse']))

    # After finalising the model and evaluating it on untouched test labels,
    # refit on ALL 1500 labeled examples to maximize data available for predictions.
    X_all = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    y_all = np.concatenate([y_tr, y_te])
    model_full = fit_model(X_all, y_all, selected, best['alpha'], best['weighted'])
    pred_unlabeled = predict(model_full, X_un, selected, best['clipping'])
    if len(pred_unlabeled) != 500 or not np.isfinite(pred_unlabeled).all():
        raise RuntimeError('Invalid predictions: expected 500 finite numbers.')
    out.mkdir(parents=True, exist_ok=True)
    predpath = out / 'y_pred.csv'
    pd.Series(pred_unlabeled).to_csv(predpath, header=False, index=False, float_format='%.12g')
    with predpath.open('r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if len(lines) != 500 or any(not re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line) for line in lines):
        raise RuntimeError('CSV format validation failed (500 numeric rows, no header/index expected).')
    result['prediction_file'] = str(predpath)
    reportpath = out / 'rapport_partie1.pdf'
    make_report_pdf(reportpath, result)
    (out / 'resultats_detail.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nDONE - Submit these 2 files separately to Moodle:')
    print('  ', predpath)
    print('  ', reportpath)
    print('Technical results (for your own use):', out / 'resultats_detail.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
