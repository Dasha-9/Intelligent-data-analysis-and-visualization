# СРС 3.4 · локальные объяснения: водопады SHAP по своему набору, аддитивность, вывод «данные / порог / признак»
# Запуск из корня папки (окружение курса: lightgbm 4.7, scikit-learn 1.7.2, pandas 2.3, pyarrow 18):
#   python python/local_shap.py <папка IAVD_3.7_ML_team> [N]
#   N – номер набора команды (практика 3.7), по умолчанию 37.
# Что делает:
#   1) прогоняет модель lgbm_gate1 v001 по 320 карточкам практики: вклады pred_contrib, base, raw_score, proba;
#   2) суммирует вклады по 73 исходным полям (как в приложении курса и в fact_shap) и сверяет с shap_cards.csv;
#   3) проверяет аддитивность base + Σφ = raw_score (допуск 1e-6) и σ(raw_score) = proba;
#   4) по карточкам набора N: топ-10, хвост «остальные 63», группы полей, расстояние до порога;
#   5) рисует водопады в images/ и сохраняет таблицы в results/.
import os, sys
KIT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'IAVD_3.7_ML_team')
TEAM = int(sys.argv[2]) if len(sys.argv) > 2 else 37
sys.path.insert(0, KIT)
import course_ml  # noqa: F401 – LightGBM загружается раньше pyarrow (см. model_card.md)
from course_ml import data as data_mod
from course_ml import train as train_mod
from course_ml.core import read_csv
from course_ml.inference import CourseModel
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRAY, LINE, ACC, RED = '#1F2328', '#6E7781', '#8C959F', '#D0D7DE', '#0969DA', '#CF222E'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': LINE, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
num = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',').replace('-', '−')
sgn = lambda v, d=2: ('+' if v > 0 else '') + num(v, d)
sig = lambda z: 1 / (1 + np.exp(-z))
TOL = 1e-6

# смысловые группы полей: коррелированные признаки читаем группой (лекция 3.4, слайд 33)
GROUPS = {
    'время': ['Создан', 'event_duration_s', 'create_lag_s', 'event_age_days', 'is_business_hours', 'is_night'],
    'сеть: адреса и хосты': ['src', 'dst', 'Hostname Цели', 'Hostname атакующего', 'src_is_private', 'dst_is_private',
                             'ip_count_src', 'ip_count_dst', 'dst_external_count', 'len_src', 'len_dst', 'host_depth',
                             'host_is_server'],
    'правило и источник': ['Наименование', 'freq_Наименование', 'categoryMain', 'categoryHigh', 'categoryLow',
                           'Пиктограмма IRP', 'Критичность', 'Критичность инцидента в источнике', 'Источник инцидента',
                           'Вендор из сием', 'freq_Вендор', 'Конвейер SIEM', 'Организация', 'object', 'n_rules'],
}


def group_of(col):
    for g, cols in GROUPS.items():
        if col in cols:
            return g
    return 'текст алерта, команд и объектов'


def run_model():
    model = CourseModel(os.path.join(KIT, 'models', 'lgbm_gate1__v001'))
    mapping = read_csv(model.dir / 'feature_mapping.csv')
    cards = read_csv(os.path.join(KIT, 'course_pack', 'cards_practice.csv'))
    cards['EntityId'] = cards['EntityId'].astype(str)
    raw = pd.read_parquet(os.path.join(KIT, 'course_pack', 'cards_raw.parquet'))
    raw['EntityId'] = raw['EntityId'].astype(str)
    raw = raw[raw['EntityId'].isin(cards['EntityId'])].drop_duplicates('EntityId').set_index('EntityId').loc[cards['EntityId']].reset_index()
    prep, _, _ = data_mod.prepare(raw, model.cfg)
    for column, source in data_mod.FREQ_FIELDS.items():
        prep[column] = prep[source].astype(str).map(model.freq_maps.get(source, {})).fillna(0.0).astype(float)
    X = np.asarray(model.preproc.transform(train_mod.feature_frame(prep, model.cfg)), dtype=np.float32)
    contrib = model.model.predict(X, pred_contrib=True)
    raw_score = model.model.predict(X, raw_score=True)
    proba = model.model.predict_proba(X)[:, 1]
    # 1 099 столбцов после one-hot → 73 исходных поля (как в приложении курса)
    phi = pd.DataFrame(contrib[:, :-1].T, index=mapping['source_column'].to_numpy()).groupby(level=0, sort=False).sum().T
    phi.index = raw['EntityId'].to_numpy()
    res = cards.set_index('EntityId')[['team', 'rank_in_team', 'y_true', 'outcome']].copy()
    res['proba_kit'] = cards.set_index('EntityId')['proba']
    res['base'] = contrib[:, -1]
    res['sum_phi'] = phi.sum(axis=1).to_numpy()
    res['raw_score'] = raw_score
    res['proba'] = proba
    res['add_err'] = (res['base'] + res['sum_phi'] - res['raw_score']).abs()
    res['sig_err'] = (sig(res['base'] + res['sum_phi']) - res['proba']).abs()
    return model, res, phi, raw.set_index('EntityId'), prep.set_index(raw['EntityId'].to_numpy())


def check_kit(phi):
    s = read_csv(os.path.join(KIT, 'course_pack', 'shap_cards.csv'))
    s['EntityId'] = s['EntityId'].astype(str)
    s = s.set_index('EntityId')[phi.columns]
    return float((s.loc[phi.index] - phi).abs().to_numpy().max())


def card_rows(eid, res, phi, prep, raw, top=10):
    """Топ-10 по модулю + хвост: одна строка на поле, со значением из карточки."""
    v = phi.loc[eid]
    order = v.abs().sort_values(ascending=False).index
    t = pd.DataFrame({'field': order[:top], 'shap': v[order[:top]].to_numpy()})
    t['group'] = t['field'].map(group_of)
    t['engineered'] = ~t['field'].isin(raw.columns)
    t['value'] = [value_of(f, eid, prep, raw) for f in t['field']]
    tail = v[order[top:]]
    return t, float(tail.sum()), int(len(tail))


def value_of(f, eid, prep, raw):
    if f == 'Создан':
        ts = pd.Timestamp(raw.loc[eid, 'Создан']).tz_convert('Europe/Moscow')
        return ts.strftime('%d.%m %H:%M МСК, ') + ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'][ts.dayofweek]
    x = prep.loc[eid, f] if f in prep.columns else raw.loc[eid, f]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'пусто'
    if isinstance(x, (float, np.floating)):
        return num(x, 3) if abs(x) < 10 else f'{x:,.0f}'.replace(',', ' ')
    s = str(x).replace('\\\\', '\\').replace('["', '').replace('"]', '').replace('\n', ' ')
    return s if len(s) <= 46 else s[:45] + '…'


def waterfall(eid, res, phi, prep, raw, title, fname, note=''):
    r = res.loc[eid]
    t, tail, n_tail = card_rows(eid, res, phi, prep, raw)
    z_tau = np.log(TAU / (1 - TAU))
    labels = ['база модели'] + [f'{f}  ·  {val}' + ('  (сконстр.)' if e else '') for f, val, e in zip(t['field'], t['value'], t['engineered'])] + \
             [f'остальные {n_tail} полей', 'итог z']
    steps = [r['base']] + list(t['shap']) + [tail]
    z = r['base'] + r['sum_phi']
    fig, ax = plt.subplots(figsize=(13, 7.6), dpi=150)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.83, bottom=0.09)
    fig.text(0.012, 0.975, title, fontsize=14, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.925, f'Вклады в логарифме шансов z · base {num(r["base"])} + топ-10 {sgn(t["shap"].sum())} + остальные {n_tail} {sgn(tail)} '
             f'= z {num(z)} → p = σ(z) = {num(r["proba"], 4)} · порог τ = 0,019259 ↔ z_τ = {num(z_tau, 2)}' + (f' · {note}' if note else ''),
             fontsize=10.2, color=MUTED, va='top')
    n = len(labels)
    y = np.arange(n)[::-1]
    cum = 0.0
    for i, (lab, st) in enumerate(zip(labels, steps + [None])):
        if i == 0:
            ax.barh(y[i], st, left=0, color='#C3C9D0', height=0.62); cum = st
            ax.text(st + 0.15, y[i], num(st), va='center', ha='left', color='white', fontsize=10, weight='bold')
        elif i == n - 1:
            ax.barh(y[i], z, left=0, color=RED if z >= z_tau else ACC, height=0.62)
            ax.text(z / 2, y[i], f'{num(z)}  (p = {num(r["proba"], 4)})', va='center', ha='center', color='white',
                    fontsize=10, weight='bold')
        else:
            col = RED if st > 0 else ACC
            if i == n - 2:
                col = '#E8A3A8' if st > 0 else '#9CC3EE'
            ax.barh(y[i], st, left=cum, color=col, height=0.62)
            end = cum + st
            ax.text(max(cum, end) + 0.12, y[i], sgn(st), va='center', ha='left', color=INK, fontsize=10)
            ax.plot([end, end], [y[i] - 0.31, y[i] - 0.69], color=GRAY, lw=0.7)
            cum = end
    ax.axvline(z_tau, color=ACC, ls=':', lw=1.4)
    ax.text(z_tau, n - 0.35, ' порог z_τ', color=ACC, fontsize=9.5, va='bottom')
    ax.axvline(0, color=LINE, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.8)
    for tl, i in zip(ax.get_yticklabels(), range(n)):
        if i in (0, n - 1): tl.set_weight('bold')
    lo = min(r['base'], 0) - 0.3
    hi = max(max(np.cumsum(steps)), z, 0) + 1.2
    ax.set_xlim(lo, hi)
    ax.set_xlabel('z · красный – к инциденту, синий – к легитимному, светлый – хвост остальных полей')
    ax.grid(axis='x', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    os.makedirs('images', exist_ok=True)
    fig.savefig(os.path.join('images', fname), facecolor='white'); plt.close(fig)
    return t, tail, n_tail


def overview(res, team, fname):
    """Восемь карточек набора: z относительно порога, база и доля топ-10."""
    z_tau = np.log(TAU / (1 - TAU))
    s = res[res['team'] == team].copy()
    s['z'] = s['base'] + s['sum_phi']
    s = s.sort_values('z')
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.14)
    fig.text(0.012, 0.965, f'Набор {team}: две «уверенные» ложные тревоги и два пропуска рядом с порогом', fontsize=14, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.895, 'Итог z = base + Σφ по каждой карточке · все стартуют с базы −8,85 · пунктир – порог z_τ = −3,93',
             fontsize=10.2, color=MUTED, va='top')
    y = np.arange(len(s))
    col = {'TP': RED, 'FP': RED, 'FN': ACC, 'TN': ACC}
    for yy, (eid, r) in zip(y, s.iterrows()):
        ax.plot([r['base'], r['z']], [yy, yy], color=LINE, lw=2.5)
        ax.scatter([r['base']], [yy], color=GRAY, s=18, zorder=3)
        wrong = r['outcome'] in ('FN', 'FP')
        ax.scatter([r['z']], [yy], color=col[r['outcome']], s=70 if wrong else 40, zorder=4,
                   edgecolor=INK if wrong else 'none', linewidth=1.2)
        ax.text(r['z'] + 0.3, yy, f'p = {num(r["proba"], 4)}', va='center', fontsize=9.5, color=INK,
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'), zorder=5)
    ax.axvline(z_tau, color=ACC, ls=':', lw=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{e} · {r.outcome} ({"инцидент" if r.y_true == 1 else "легитимно"})' for e, r in s.iterrows()], fontsize=10)
    ax.set_xlabel('логарифм шансов z · красная точка – модель сказала «инцидент», синяя – «легитимно», с обводкой – ошибка')
    ax.set_xlim(-9.5, 10.5)
    ax.grid(axis='x', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig(os.path.join('images', fname), facecolor='white'); plt.close(fig)


def groups_table(eids, phi):
    g = phi.loc[eids].T
    g['group'] = [group_of(c) for c in g.index]
    return g.groupby('group').sum().T


if __name__ == '__main__':
    model, res, phi, raw, prep = run_model()
    TAU = model.threshold
    z_tau = np.log(TAU / (1 - TAU))
    pd.set_option('display.width', 220)
    print(f'τ = {TAU}, z_τ = {z_tau:.4f}, base = {res["base"].iloc[0]:.6f}, карточек {len(res)}')
    print(f'сверка вкладов с shap_cards.csv: max |Δφ| = {check_kit(phi):.2e}')
    print(f'сверка proba с cards_practice.csv: max |Δp| = {(res["proba"] - res["proba_kit"]).abs().max():.2e}')
    print(f'аддитивность base + Σφ = raw_score: max {res["add_err"].max():.2e} (допуск {TOL:g}), '
          f'нарушений {(res["add_err"] > TOL).sum()} · σ(z) = proba: max {res["sig_err"].max():.2e}')

    s = res[res['team'] == TEAM].copy()
    s['z'] = s['base'] + s['sum_phi']
    s['gap'] = s['z'] - z_tau
    rows = []
    for eid in s.sort_values('rank_in_team').index:
        t, tail, n_tail = card_rows(eid, res, phi, prep, raw)
        rows.append(dict(EntityId=eid, outcome=s.loc[eid, 'outcome'], y=int(s.loc[eid, 'y_true']), proba=s.loc[eid, 'proba'],
                         base=s.loc[eid, 'base'], top10=t['shap'].sum(), tail=tail, z=s.loc[eid, 'z'], raw_score=s.loc[eid, 'raw_score'],
                         add_err=s.loc[eid, 'add_err'], gap=s.loc[eid, 'gap'],
                         top1=f'{t.field[0]} {sgn(t.shap[0])}', top10_share=t['shap'].sum() / s.loc[eid, 'sum_phi']))
    summary = pd.DataFrame(rows)
    print(f'\n=== Набор {TEAM}'); print(summary.round(6).to_string(index=False))
    print('\n=== Группы полей (сумма вкладов)'); print(groups_table(list(s.index), phi).round(3).to_string())
    for eid in s.sort_values('rank_in_team').index:
        t, tail, n_tail = card_rows(eid, res, phi, prep, raw)
        print(f'\n--- {eid} {s.loc[eid, "outcome"]}: топ-10 {t.shap.sum():+.4f}, остальные {n_tail} {tail:+.4f}')
        print(t.round(3).to_string(index=False))

    os.makedirs('results', exist_ok=True)
    summary.to_csv(f'results/set{TEAM}_cards.csv', sep=';', index=False, encoding='utf-8-sig')
    res.reset_index().rename(columns={'index': 'EntityId'})[['EntityId', 'team', 'outcome', 'base', 'sum_phi', 'raw_score', 'add_err', 'proba', 'sig_err']] \
        .to_csv('results/additivity_320.csv', sep=';', index=False, encoding='utf-8-sig')

    if TEAM == 37:
        overview(res, TEAM, 'w0_set37.png')
        waterfall('522033', res, phi, prep, raw, 'FN 522033: троян не удалён, вердикт «Решён», а модель – «легитимно»', 'w1_fn_522033.png')
        waterfall('519532', res, phi, prep, raw, 'FP 519532: попытка взлома через wget, аналитик закрыл как ложную, модель – p = 0,999',
                  'w2_fp_519532.png')
        waterfall('528183', res, phi, prep, raw, 'Непонятная карточка 528183: «вирус не вылечен» – это архив учебных скриптов RTA',
                  'w3_528183.png')
    print('ok')
