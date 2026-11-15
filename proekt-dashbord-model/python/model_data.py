# Проект · дашборд «Можно ли доверять модели?» (датасет 2 – лог инференса) · подготовка данных (pandas)
# Лог MODEL_PREDICTIONS_target.parquet из комплекта практики 3.6 (BI-трек): модель-советчик wlarf-dec-jun-v3,
# 19 393 прогноза за 04.09–29.11.2025. Метрики – только по размеченным строкам (Flag_ValidLabel = 1):
# y = 1 – инцидент («Решён»), y = 0 – False Positive.
import json
import numpy as np
import pandas as pd

DEFAULT_PATH = 'data/MODEL_PREDICTIONS_target.parquet'
TAU0 = 0.18                     # порог приложения курса: 1 − 0,82 (порог модели в model_params задан для класса FP)
NONE = 'не указан'
SLICES = {'Вендор из SIEM': 'vendor', 'Пиктограмма IRP': 'irp', 'Правило корреляции SIEM': 'rule',
          'Критичность': 'sev', 'Конвейер SIEM': 'conv'}
BINS = [0, 0.05, 0.1, 0.18, 0.3, 0.5, 0.7, 1.0]
BIN_LAB = ['0–0,05', '0,05–0,1', '0,1–0,18', '0,18–0,3', '0,3–0,5', '0,5–0,7', '0,7–1']


def _first(s):
    """JSON-массив строк → первый элемент (правило хранится как ["..."])."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == '':
        return NONE
    try:
        a = json.loads(s)
        return a[0] if isinstance(a, list) and a else (a if isinstance(a, str) else NONE)
    except Exception:
        return str(s)


def load(path=DEFAULT_PATH):
    p = pd.read_parquet(path)
    p['day'] = pd.to_datetime(p['Дата'])
    p['week'] = p['day'].dt.to_period('W-SUN').dt.start_time
    p['labeled'] = p['Flag_ValidLabel'].eq(1)
    p['proba'] = p['y_pred_proba'].astype(float)
    p['vendor'] = p['Вендор из сием'].fillna(NONE)
    p['irp'] = p['Пиктограмма IRP'].fillna(NONE)
    p['rule'] = p['Правило корреляции SIEM'].map(_first)
    p['sev'] = p['Критичность'].fillna(NONE)
    p['conv'] = p['Конвейер SIEM'].fillna(NONE)
    return p


def confusion(v, tau):
    """v – размеченные строки; прогноз «инцидент» при оценке ≥ τ."""
    pr = v['proba'] >= tau
    y = v['y'] == 1
    TP, FN = int((pr & y).sum()), int((~pr & y).sum())
    FP, TN = int((pr & ~y).sum()), int((~pr & ~y).sum())
    n = TP + FN + FP + TN
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, n=n, inc=TP + FN,
                recall=TP / (TP + FN) if TP + FN else np.nan, miss=FN / (TP + FN) if TP + FN else np.nan,
                prec=TP / (TP + FP) if TP + FP else np.nan, agree=(TP + TN) / n if n else np.nan,
                flag=(TP + FP) / n if n else np.nan)


def weekly(v, tau):
    inc = v[v['y'] == 1]
    g = inc.assign(found=inc['proba'] >= tau).groupby('week')['found'].agg(found='sum', inc='size')
    g['missed'] = g['inc'] - g['found']
    g['miss'] = g['missed'] / g['inc']
    return g.reset_index()


def where_missed(v, tau, col, min_inc=5, top=8):
    inc = v[v['y'] == 1]
    g = inc.assign(m=inc['proba'] < tau).groupby(col)['m'].agg(missed='sum', inc='size')
    g = g[g['inc'] >= min_inc]
    g['miss'] = g['missed'] / g['inc']
    return g.sort_values(['missed', 'inc'], ascending=False).head(top).reset_index().rename(columns={col: 'group'})


def calibration(v):
    b = pd.cut(v['proba'], BINS, labels=BIN_LAB, include_lowest=True)
    g = v.groupby(b, observed=False).agg(n=('y', 'size'), fact=('y', 'mean'), model=('proba', 'mean'))
    return g.reset_index().rename(columns={'proba': 'bin'})


def threshold_curve(v, taus=np.round(np.arange(0.02, 0.61, 0.01), 2)):
    return pd.DataFrame([dict(tau=t, **{k: c[k] for k in ('recall', 'prec', 'flag', 'FN', 'FP')}) for t in taus for c in [confusion(v, t)]])


if __name__ == '__main__':
    p = load()
    v = p[p['labeled']]
    print(len(p), len(v), p['day'].min().date(), p['day'].max().date())
    print(confusion(v, TAU0))
    print(weekly(v, TAU0).to_string())
    for name, col in SLICES.items():
        print('==', name); print(where_missed(v, TAU0, col).to_string())
    print(calibration(v).round(3).to_string())
    for t in (0.02, 0.05, 0.1, TAU0):
        c = confusion(v, t); print(t, round(c['recall'], 3), round(c['flag'], 3), c['FN'])
    print('задержка: медиана', p['latency_s'].median(), '95 %', p['latency_s'].quantile(0.95))
