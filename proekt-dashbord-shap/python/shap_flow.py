# Проект, часть 3 · расчёты для дашборда «Можно ли верить объяснению модели?»
# Вход: data/shap_flow.parquet (готовит python/prep_shap_flow.py) – 20 584 карточки эксплуатации,
# вклады SHAP модели lgbm_gate1 по 73 полям, оценка score, метка y, значения ключевых полей.
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'shap_flow.parquet')
TAU = 0.019258739887821837
Z_TAU = float(np.log(TAU / (1 - TAU)))
OUTCOMES = ['FN', 'TP', 'FP', 'TN']
OUT_RU = {'TP': 'нашла инцидент', 'FN': 'пропустила инцидент', 'FP': 'ложная тревога', 'TN': 'верно сняла шум'}

GROUPS = {
    'время': ['Создан', 'event_duration_s', 'create_lag_s', 'event_age_days', 'is_business_hours', 'is_night'],
    'сеть': ['src', 'dst', 'Hostname Цели', 'Hostname атакующего', 'src_is_private', 'dst_is_private', 'ip_count_src',
             'ip_count_dst', 'dst_external_count', 'len_src', 'len_dst', 'host_depth', 'host_is_server'],
    'правило и источник': ['Наименование', 'freq_Наименование', 'categoryMain', 'categoryHigh', 'categoryLow',
                           'Пиктограмма IRP', 'Критичность', 'Критичность инцидента в источнике', 'Источник инцидента',
                           'Вендор из сием', 'freq_Вендор', 'Конвейер SIEM', 'Организация', 'object', 'n_rules'],
}
TEXT = 'текст алерта и команд'
GROUP_ORDER = ['сеть', 'правило и источник', TEXT, 'время']


def group_of(f):
    for g, cols in GROUPS.items():
        if f in cols:
            return g
    return TEXT


def load(path=DATA):
    d = pd.read_parquet(path).set_index('EntityId')
    phi = [c for c in d.columns if c.startswith('phi__')]
    d['z'] = d['base'] + d[phi].sum(axis=1)
    d['flag'] = d['score'] >= TAU
    lab = d['y'].notna()
    d['outcome'] = np.where(~lab, '', np.where(d['y'] == 1, np.where(d['flag'], 'TP', 'FN'),
                                               np.where(d['flag'], 'FP', 'TN')))
    t = pd.to_datetime(d['val__Создан'], utc=True).dt.tz_convert('Europe/Moscow')
    d['hour'] = t.dt.hour
    d['weekend'] = t.dt.dayofweek >= 5
    sip = d['val__src_is_private']
    d['src_kind'] = np.where(sip.isna(), 'не указан', np.where(sip == 0, 'внешний', 'внутренний'))
    d['grp_empty'] = np.where(d['val__len_Значение Инцидента (группировка)'] == 0, 'пустое', 'заполнено')
    d['type'] = d['card__Пиктограмма IRP'].replace('', 'не указан')
    for g in GROUP_ORDER:
        cols = ['phi__' + f for f in [c[5:] for c in phi] if group_of(f) == g]
        d['g__' + g] = d[cols].sum(axis=1)
    return d, phi


def filt(d, typ='все', out='все'):
    m = pd.Series(True, index=d.index)
    if typ != 'все':
        m &= d['type'] == typ
    if out != 'все':
        m &= d['outcome'] == out
    return d[m]


def kpi(d):
    lab = d[d['outcome'] != '']
    inc = int((lab['y'] == 1).sum())
    fn = int((lab['outcome'] == 'FN').sum())
    phi = [c for c in d.columns if c.startswith('phi__')]
    add = float((1 / (1 + np.exp(-(d['base'] + d[phi].sum(axis=1)))) - d['score']).abs().max()) if len(d) else np.nan
    return dict(n=len(d), lab=len(lab), inc=inc, fn=fn, miss=fn / inc if inc else np.nan, add=add)


def importance(d, phi, top=12):
    s = d[phi].abs().mean().sort_values(ascending=False).head(top)
    r = pd.DataFrame({'field': [c[5:] for c in s.index], 'imp': s.to_numpy()})
    r['group'] = r['field'].map(group_of)
    return r


def by_value(d, col, field, order):
    """Довод модели против реальности: средний вклад поля и доля инцидентов по значениям поля."""
    lab = d[d['outcome'] != '']
    g = lab.groupby(col).agg(n=('y', 'size'), inc=('y', 'mean'), phi=('phi__' + field, 'mean'))
    return g.reindex([o for o in order if o in g.index])


def by_hour(d):
    lab = d[d['outcome'] != '']
    return lab.groupby('hour').agg(n=('y', 'size'), inc=('y', 'mean'), phi=('phi__Создан', 'mean')).reindex(range(24))


def blind(d, top=8):
    """Пропуски по типу инцидента: сколько инцидентов и сколько из них модель пропустила."""
    lab = d[d['y'] == 1]
    g = lab.groupby('type').agg(inc=('y', 'size'), fn=('outcome', lambda s: int((s == 'FN').sum())))
    g = g[g['inc'] >= 15].sort_values('fn', ascending=False).head(top)
    g['miss'] = g['fn'] / g['inc']
    return g


def fn_vs_tp(d):
    lab = d[d['y'] == 1]
    cols = ['g__' + g for g in GROUP_ORDER]
    r = lab.groupby('outcome')[cols].mean().reindex(['FN', 'TP'])
    r.columns = GROUP_ORDER
    return r


def card_lov(d, n=200):
    """Карточки для разбора: сначала пропуски, внутри – ближе к порогу."""
    order = {'FN': 0, 'FP': 1, 'TP': 2, 'TN': 3, '': 4}
    s = d.assign(o=d['outcome'].map(order), gap=(d['z'] - Z_TAU).abs()).sort_values(['o', 'gap']).head(n)
    return [f'{e} · {r.outcome or "без вердикта"} · {r.type} · p = {r.score:.4f}'.replace('.', ',', 1)
            .replace('p = 0.', 'p = 0,') for e, r in s.iterrows()]


def explain(eid, d, phi, top=10):
    r = d.loc[eid]
    v = r[phi].astype(float)
    v.index = [c[5:] for c in v.index]
    o = v.abs().sort_values(ascending=False).index
    t = pd.DataFrame({'Поле': o[:top], 'Вклад': v[o[:top]].round(3).to_numpy()})
    t['Направление'] = np.where(t['Вклад'] > 0, 'к инциденту', 'к легитимному')
    t['Группа'] = t['Поле'].map(group_of)
    return dict(row=r, top=t, tail=float(v[o[top:]].sum()), n_tail=len(o) - top)


if __name__ == '__main__':
    d, phi = load()
    k = kpi(d)
    print(k)
    print(importance(d, phi).round(3).to_string())
    print(by_value(d, 'src_kind', 'src_is_private', ['внешний', 'внутренний', 'не указан']).round(3))
    print(by_value(d, 'grp_empty', 'len_Значение Инцидента (группировка)', ['пустое', 'заполнено']).round(3))
    h = by_hour(d); print('corr', round(np.corrcoef(h['phi'], h['inc'])[0, 1], 3))
    print(blind(d).round(3)); print(fn_vs_tp(d).round(2))
