# СРС 2.4 · общие расчёты для трёх гипотез (используют charts_matplotlib.py и hypotheses_taipy.py)
import sys
import pandas as pd

DEFAULT_PATH = 'data/iavd_incidents_raw_target.parquet'
COLS = ['EntityId', 'Создан', 'Критичность', 'Вердикт', 'Правило корреляции SIEM',
        'Дата взятия в работу', 'SLA по взятию в работу Атаки', 'SLA просрочен (взятие в работу)']
ORDER = ['Критическая', 'Высокая', 'Средняя', 'Низкая']

# Пороги гипотезы 1: «шумное» правило – не меньше 300 разобранных карточек и не меньше 95 % False Positive
H1_MIN_N, H1_MIN_FP = 300, 0.95
# Группы нагрузки для гипотезы 2: сколько карточек создано за тот же календарный час (МСК)
H2_BINS = [0, 5, 10, 20, 50, 100, float('inf')]
H2_LABELS = ['1–5', '6–10', '11–20', '21–50', '51–100', '> 100']

fmt = lambda v: f'{int(round(v)):,}'.replace(',', ' ')
pct = lambda v, d=0: f'{v * 100:.{d}f} %'.replace('.', ',')


def data_path():
    return next((a for a in sys.argv[1:] if not a.startswith('--')), DEFAULT_PATH)


def load(path=None):
    df = pd.read_parquet(path or data_path(), columns=COLS)
    df['msk'] = df['Создан'].dt.tz_convert('Europe/Moscow')
    return df


def h1_rules(df):
    """Гипотеза 1: доля False Positive по правилам корреляции SIEM (только карточки с вердиктом)."""
    v = df[(df['Вердикт'] != 'Не указан') & df['Правило корреляции SIEM'].notna()]
    g = v.groupby('Правило корреляции SIEM').agg(
        n=('EntityId', 'size'),
        fp=('Вердикт', lambda s: (s == 'False Positive').mean()),
        solved=('Вердикт', lambda s: (s == 'Решён').sum()))
    g['noisy'] = (g['n'] >= H1_MIN_N) & (g['fp'] >= H1_MIN_FP)
    noisy = g[g['noisy']]
    all_v = (df['Вердикт'] != 'Не указан').sum()
    stats = dict(rules=len(g), noisy_rules=len(noisy), noisy_cards=int(noisy['n'].sum()),
                 noisy_share=noisy['n'].sum() / all_v, noisy_solved=int(noisy['solved'].sum()),
                 fp_all=(df['Вердикт'] == 'False Positive').sum() / all_v, reviewed=int(all_v))
    return g.sort_values('n', ascending=False), stats


def h2_load(df):
    """Гипотеза 2: доля просрочки SLA взятия в работу в зависимости от нагрузки часа."""
    d = df.copy()
    d['slot'] = d['msk'].dt.floor('h')
    d['load'] = d.groupby('slot')['EntityId'].transform('size')
    k = d[d['SLA просрочен (взятие в работу)'].notna()].copy()
    k['over'] = k['SLA просрочен (взятие в работу)'].astype(str) == 'True'
    k['grp'] = pd.cut(k['load'], bins=H2_BINS, labels=H2_LABELS)
    g = k.groupby('grp', observed=False).agg(cards=('over', 'size'), over=('over', 'mean'), slots=('slot', 'nunique'))
    slots = k.groupby('slot').agg(load=('load', 'first'), over=('over', 'mean'), n=('over', 'size'))
    rho = slots[['load', 'over']].corr(method='spearman').iloc[0, 1]
    stats = dict(cards=len(k), over_all=k['over'].mean(), slots=len(slots), rho=rho)
    return g, slots, stats


def h3_take(df):
    """Гипотеза 3: время до взятия в работу по критичности против норматива SLA."""
    ok = df['Дата взятия в работу'].notna()
    t = (df['Дата взятия в работу'] - df['Создан']).dt.total_seconds() / 60
    sla = df['SLA по взятию в работу Атаки'] / 60000
    d = pd.DataFrame({'crit': df['Критичность'], 'take': t, 'sla': sla})[ok & (t >= 0)]
    d = d[d['crit'].isin(ORDER)]
    g = d.groupby('crit').agg(n=('take', 'size'), q1=('take', lambda s: s.quantile(.25)),
                              med=('take', 'median'), q3=('take', lambda s: s.quantile(.75)),
                              sla=('sla', lambda s: s.mode().iloc[0]))
    g['over'] = d.assign(o=d['take'] > d['sla']).groupby('crit')['o'].mean()
    g = g.reindex(ORDER)
    stats = dict(valid=len(d), no_take=int((~ok).sum()), negative=int((ok & (t < 0)).sum()),
                 crit_no_take=int(((df['Критичность'] == 'Критическая') & ~ok).sum()))
    return d, g, stats


if __name__ == '__main__':
    df = load()
    g1, s1 = h1_rules(df); print(s1); print(g1[g1.noisy].to_string())
    g2, _, s2 = h2_load(df); print(s2); print(g2.to_string())
    _, g3, s3 = h3_take(df); print(s3); print(g3.round(2).to_string())
