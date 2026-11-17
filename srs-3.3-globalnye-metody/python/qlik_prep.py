# СРС 3.3 · подготовка CSV для Qlik: десятичная запятая, без экспоненты, SHAP в длинном формате
# Запуск из корня папки: python python/qlik_prep.py  →  qlik/data/xai_*.csv
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from global_xai import rd, load

OUT = 'qlik/data'


def save(df, name):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.10f}'.rstrip('0').rstrip('.').replace('.', ','))
    df.to_csv(os.path.join(OUT, name), sep=';', index=False, encoding='utf-8')
    print(name, len(df))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    d = load()
    pdp, ale = rd('pdp_numeric.csv'), rd('ale_numeric.csv')
    pa = pdp.merge(ale, on=['feature', 'quantile'])
    save(pa.rename(columns={'feature': 'Признак', 'quantile': 'Квантиль', 'raw_value': 'Значение', 'mean_proba': 'PDP', 'ale': 'ALE'}),
         'xai_pdp_ale.csv')
    save(rd('ice_numeric.csv').rename(columns={'feature': 'Признак', 'line': 'Линия', 'quantile': 'ICE_Квантиль', 'proba': 'ICE_p'}),
         'xai_ice.csv')
    imp, ab = rd('importance.csv'), rd('ablation.csv')
    ia = imp[['rank', 'source_column', 'mean_abs_shap', 'share']].rename(
        columns={'rank': 'Ранг_SHAP', 'source_column': 'ПолеXAI', 'mean_abs_shap': 'meanAbsSHAP', 'share': 'Доля_SHAP'}).merge(
        ab[['rank', 'field', 'n_features', 'mean_abs_delta']].rename(
            columns={'rank': 'Ранг_abl', 'field': 'ПолеXAI', 'n_features': 'Признаков', 'mean_abs_delta': 'Дельта_abl'}), on='ПолеXAI')
    save(ia, 'xai_importance.csv')
    c, feats = d['cards'], d['shap_feats']
    save(c[['EntityId', 'team', 'rank_in_team', 'outcome', 'y_true', 'proba', 'base', 'z']].rename(
        columns={'EntityId': 'КартаXAI', 'team': 'Набор', 'rank_in_team': 'НомерВНаборе', 'outcome': 'Исход', 'y_true': 'y_факт',
                 'proba': 'p', 'base': 'base', 'z': 'z'}), 'xai_cards.csv')
    long = c[['EntityId'] + feats].melt(id_vars='EntityId', var_name='ПризнакSHAP', value_name='phi').rename(columns={'EntityId': 'КартаXAI'})
    save(long, 'xai_shap.csv')
