# СРС 3.3 · глобальные методы XAI: PDP / ICE / ALE, важность против ablation, p → z
# Запуск из корня папки: python python/global_xai.py [N]
#   N – номер набора команды (практика 3.7), по умолчанию 37.
# Вход (папка data/): файлы занятия 3.3 – importance.csv, ablation.csv, pdp_numeric.csv,
#   ice_numeric.csv, ale_numeric.csv, cards_practice.csv; из комплекта 3.7 – shap_cards.csv.
# Выход: числа в консоль + images/g1_pdp_ice_ale.png, g2_importance_ablation.png, g3_z.png, g4_card.png
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAU = 0.019258739887821837                 # порог модели lgbm_gate1 (spec.json)
Z_TAU = np.log(TAU / (1 - TAU))            # −3,9303
INK, MUTED, GRAY, LINE, ACC, RED = '#1F2328', '#6E7781', '#C3C9D0', '#D0D7DE', '#0969DA', '#CF222E'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': LINE, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
num = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',').replace('-', '−')
FEAT_RU = {'create_lag_s': 'create_lag_s\nзадержка создания карточки', 'event_age_days': 'event_age_days\nвозраст события',
           'alert_words': 'alert_words\nслов в алерте', 'cmd_flags': 'cmd_flags\nфлагов в командах'}
FEATS = list(FEAT_RU)
sig = lambda z: 1 / (1 + np.exp(-z))


def rd(name):
    return pd.read_csv(os.path.join('data', name), sep=';', encoding='utf-8-sig')


def load():
    d = {k: rd(f'{k}.csv') for k in ('importance', 'ablation', 'pdp_numeric', 'ice_numeric', 'ale_numeric', 'cards_practice')}
    s = rd('shap_cards.csv')
    feats = [c for c in s.columns if c not in ('EntityId', 'y_true', 'proba', 'outcome', 'base')]
    s['z'] = s['base'] + s[feats].sum(axis=1)
    c = d['cards_practice'].merge(s[['EntityId', 'base', 'z'] + feats], on='EntityId')
    with np.errstate(divide='ignore'):
        c['z_from_p'] = np.log(c['proba'] / (1 - c['proba']))
    d['cards'], d['shap_feats'] = c, feats
    return d


def task1(d):
    pdp, ice, ale = d['pdp_numeric'], d['ice_numeric'], d['ale_numeric']
    out = []
    for f in FEATS:
        p = pdp[pdp.feature == f].set_index('quantile')['mean_proba']
        a = ale[ale.feature == f].set_index('quantile')['ale']
        w = ice[ice.feature == f].pivot(index='line', columns='quantile', values='proba')
        rng = (w.max(1) - w.min(1)).sort_values(ascending=False)
        out.append(dict(feature=f, pdp_first=p.iloc[0], pdp_q09=p.loc[0.9], pdp_last=p.iloc[-1], pdp_range=p.max() - p.min(),
                        ale_first=a.iloc[0], ale_q09=a.loc[0.9], ale_last=a.iloc[-1], ale_range=a.max() - a.min(),
                        ice_lines=len(w), ice_moving=int((rng > 0.01).sum()), ice_top=list(rng.index[:2]),
                        ice_top_drop=[round(w.loc[i].iloc[0] - w.loc[i].iloc[-1], 3) for i in rng.index[:2]]))
    return pd.DataFrame(out)


def task2(d):
    imp, ab = d['importance'], d['ablation']
    m = imp[['rank', 'source_column', 'share', 'mean_abs_shap']].rename(columns={'rank': 'r_shap', 'source_column': 'field'}).merge(
        ab[['rank', 'field', 'n_features', 'mean_abs_delta']].rename(columns={'rank': 'r_abl'}), on='field')
    top_s, top_a = set(m.nsmallest(5, 'r_shap').field), set(m.nsmallest(5, 'r_abl').field)
    rho = m[['r_shap', 'r_abl']].corr(method='spearman').iloc[0, 1]
    return m, top_s & top_a, top_s - top_a, top_a - top_s, rho


def task4(d):
    c = d['cards']
    r = dict(n=len(c), base=c['base'].iloc[0], z_tau=Z_TAU, max_err=(sig(c['z']) - c['proba']).abs().max(),
             decision_ok=bool(((c['z'] >= Z_TAU) == c['outcome'].isin(['TP', 'FP'])).all()),
             p_zero=int((c['proba'] == 0).sum()))
    bands = pd.cut(c['proba'], [-1, 0, 1e-4, 0.01, 0.99, 1], labels=['p = 0', 'p < 0,0001', '0,0001–0,01', '0,01–0,99', '> 0,99'])
    dz = (c['z_from_p'] - c['z']).abs().replace(np.inf, np.nan)
    r['bands'] = pd.DataFrame({'band': bands, 'dz': dz}).groupby('band', observed=False)['dz'].agg(['size', 'max'])
    c['gap'] = c['z'] - Z_TAU
    r['nearest'] = c.loc[c['gap'].abs().nsmallest(5).index, ['team', 'EntityId', 'outcome', 'proba', 'z', 'gap']]
    return r


def prob_sum_demo(d, eid):
    """Почему вероятности нельзя складывать: сумма «вкладов в вероятность» против σ(base + Σφ)."""
    c, feats = d['cards'], d['shap_feats']
    row = c[c['EntityId'] == eid].iloc[0]
    ph = row[feats].astype(float)
    b = row['base']
    pos, neg = ph[ph > 0].sum(), ph[ph < 0].sum()
    naive = sig(b) + sum(sig(b + v) - sig(b) for v in ph)
    return dict(eid=eid, outcome=row['outcome'], team=int(row['team']), base=b, pos=pos, neg=neg, z=row['z'],
                p_base=sig(b), p_pos=sig(b + pos), p=sig(row['z']), naive=naive)


def task3(d, team):
    c = d['cards'][d['cards']['team'] == team].copy()
    c['gap'] = c['z'] - Z_TAU
    return c[['rank_in_team', 'EntityId', 'outcome', 'proba', 'z', 'gap']].sort_values('rank_in_team')


def figures(d, t1, t2, t4, demo):
    os.makedirs('images', exist_ok=True)
    pdp, ice, ale = d['pdp_numeric'], d['ice_numeric'], d['ale_numeric']
    # --- рис. 1: PDP и ALE (верх), ICE (низ) ---
    fig, axs = plt.subplots(2, 4, figsize=(15, 8.2), dpi=150, sharex=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.08, hspace=0.32, wspace=0.28)
    fig.text(0.008, 0.975, 'Самые большие задержки: PDP показывает падение −5 п. п., ALE – только −2; по ICE эффект есть у 2 карточек из 20',
             fontsize=14, weight='bold', color=INK, va='top')
    fig.text(0.008, 0.925, 'Ось X – квантиль признака на тестовой выборке (3 173 карточки) · верх: отклонение от средней оценки, п. п. · '
             'низ: ICE для 20 карточек, вероятность', fontsize=10.5, color=MUTED, va='top')
    for j, f in enumerate(FEATS):
        p = pdp[pdp.feature == f]; a = ale[ale.feature == f]
        ax = axs[0, j]
        ax.axhline(0, color=LINE, lw=1)
        ax.plot(p['quantile'], (p['mean_proba'] - p['mean_proba'].mean()) * 100, color=INK, lw=2.2, marker='o', ms=3.5, label='PDP')
        ax.plot(a['quantile'], (a['ale'] - a['ale'].mean()) * 100, color=ACC, lw=2.2, marker='o', ms=3.5, label='ALE')
        ax.set_title(FEAT_RU[f], fontsize=10.5, color=INK, loc='left')
        ax.set_ylim(-5.5, 3)
        ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel('п. п. от среднего')
            ax.legend(frameon=False, loc='lower left', fontsize=10)
        w = ice[ice.feature == f].pivot(index='line', columns='quantile', values='proba')
        rng = (w.max(1) - w.min(1))
        ax = axs[1, j]
        for i, row in w.iterrows():
            hi = rng.loc[i] > 0.01
            ax.plot(w.columns, row.values, color=RED if hi else GRAY, lw=2 if hi else 1, alpha=1 if hi else 0.8, zorder=3 if hi else 2)
        ax.axhline(TAU, color=ACC, ls=':', lw=1)
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel('квантиль')
        ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
        n_hi = int((rng > 0.01).sum())
        ax.text(0.02, 0.5, f'{20 - n_hi} линий лежат у нуля\n(p ≈ 0, эффекта не видно)' if n_hi else 'все 20 линий плоские',
                transform=ax.transAxes, color=MUTED, fontsize=9.5)
        if j == 0:
            ax.set_ylabel('вероятность')
    fig.savefig('images/g1_pdp_ice_ale.png', facecolor='white'); plt.close(fig)

    # --- рис. 2: ранги важности: mean|SHAP| против ablation ---
    m, both, only_s, only_a, rho = t2
    sel = m[(m.r_shap <= 8) | (m.r_abl <= 8) | (m.field == 'cmd_entropy')].copy()
    fig, ax = plt.subplots(figsize=(11, 7.4), dpi=150)
    fig.subplots_adjust(left=0.24, right=0.60, top=0.84, bottom=0.05)
    fig.text(0.012, 0.975, f'Топ-5 совпадает в трёх полях из пяти; ранги в целом согласованы (ρ Спирмена = {num(rho)})',
             fontsize=14, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.925, 'Слева – ранг по mean|SHAP| (логарифм шансов), справа – по ablation (Δ вероятности при выключении поля)',
             fontsize=10.5, color=MUTED, va='top')
    lim = sel[['r_shap', 'r_abl']].max().max()
    for _, r in sel.iterrows():
        col = ACC if r.field in both else (RED if abs(r.r_shap - r.r_abl) >= 4 else '#8C959F')
        ax.plot([0, 1], [r.r_shap, min(r.r_abl, 14.5)], color=col, lw=2.2 if col != '#8C959F' else 1.4, marker='o', ms=5)
        ax.text(-0.04, r.r_shap, f'{int(r.r_shap)}. {r.field}', ha='right', va='center', color=col if col != '#8C959F' else INK, fontsize=10)
        lab = f'{int(r.r_abl)}. {r.field}' + (f'  (полей в группе: {int(r.n_features)})' if r.n_features > 1 else '')
        ax.text(1.04, min(r.r_abl, 14.5), lab if r.r_abl <= 14 else f'{int(r.r_abl)}. {r.field} ↓', ha='left', va='center',
                color=col if col != '#8C959F' else INK, fontsize=10)
    ax.set_ylim(15, 0.3); ax.set_xlim(-0.02, 1.02); ax.axis('off')
    ax.text(0, 0, 'mean|SHAP|', ha='center', color=MUTED, fontsize=10.5); ax.text(1, 0, 'ablation', ha='center', color=MUTED, fontsize=10.5)
    fig.savefig('images/g2_importance_ablation.png', facecolor='white'); plt.close(fig)

    # --- рис. 3: z из вероятности против z из SHAP + пример сложения ---
    c = d['cards']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.11, wspace=0.28)
    fig.text(0.008, 0.965, 'base + Σφ = z сходится у всех 320 карточек, а сложение вероятностей даёт неверное решение',
             fontsize=13.5, weight='bold', color=INK, va='top')
    fig.text(0.008, 0.91, f'z = ln(p / (1 − p)) · порог τ = 0,019259 ↔ z_τ = {num(Z_TAU, 4)} · base = {num(c["base"].iloc[0], 4)} у всех карточек',
             fontsize=10.5, color=MUTED, va='top')
    fin = np.isfinite(c['z_from_p'])
    a1.plot([-16, 13], [-16, 13], color=LINE, lw=1)
    a1.scatter(c.loc[fin, 'z'], c.loc[fin, 'z_from_p'], s=14, color=INK, alpha=0.7, label='карточка')
    zero = c[~fin]
    a1.scatter(zero['z'], [-15.6] * len(zero), marker='v', s=50, color=RED, label=f'p = 0 в файле → z = −∞ ({len(zero)} шт.)')
    a1.axvline(Z_TAU, color=ACC, ls=':', lw=1.2); a1.axhline(Z_TAU, color=ACC, ls=':', lw=1.2)
    a1.text(Z_TAU + 0.3, 11.5, 'z_τ', color=ACC)
    a1.set_xlim(-16, 13); a1.set_ylim(-16, 13)
    a1.set_xlabel('z = base + Σφ (из SHAP)'); a1.set_ylabel('z из proba в cards_practice.csv')
    a1.legend(frameon=False, loc='upper left', fontsize=9.5)
    a1.set_title('Пересчёт p → z', loc='left', color=INK, fontsize=11.5)
    a1.grid(color='#EEF0F2', lw=0.8); a1.set_axisbelow(True)
    labels = ['база\nσ(base)', 'база + все «плюсы»', 'сумма вкладов\nв вероятность', 'итог\nσ(base + Σφ)']
    vals = [demo['p_base'], demo['p_pos'], demo['naive'], demo['p']]
    cols = [GRAY, '#8C959F', RED, ACC]
    y = np.arange(4)[::-1]
    a2.barh(y, vals, color=cols, height=0.55)
    a2.axvline(TAU, color=ACC, ls=':', lw=1.4)
    a2.text(TAU, -0.55, ' τ = 0,0193', color=ACC, fontsize=10)
    for yy, v in zip(y, vals):
        a2.text(v + 0.002, yy, num(v, 4), va='center', color=INK, fontsize=10.5)
    a2.set_yticks(y); a2.set_yticklabels(labels, fontsize=10); a2.set_ylim(-0.75, 3.5)
    a2.set_xlim(0, 0.13); a2.set_xlabel('вероятность')
    a2.set_title(f'Карточка {demo["eid"]} (набор {demo["team"]}, {demo["outcome"]}): '
                 f'{num(demo["base"])} + {num(demo["pos"])} − {num(-demo["neg"])} = {num(demo["z"], 3)} > z_τ', loc='left', color=INK, fontsize=10.5)
    a2.grid(axis='x', color='#EEF0F2', lw=0.8); a2.set_axisbelow(True)
    fig.savefig('images/g3_z.png', facecolor='white'); plt.close(fig)


CARD3 = 528183                              # задание 3: FN из набора 37


def card_figure(d, eid):
    c, feats = d['cards'], d['shap_feats']
    row = c[c['EntityId'] == eid].iloc[0]
    ph = row[feats].astype(float)
    top = ph.reindex(ph.abs().sort_values(ascending=False).index)[:10][::-1]
    rest = ph.sum() - top.sum()
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.10)
    fig.text(0.012, 0.965, f'Карточка {eid} (набор {int(row["team"])}, {row["outcome"]}): «вирус не вылечен», а модель дала p = {num(row["proba"], 4)} < τ',
             fontsize=14, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, f'Вклады в логарифме шансов · base {num(row["base"])} + плюсы {num(ph[ph > 0].sum())} − минусы {num(-ph[ph < 0].sum())} '
             f'= z {num(row["z"])}; до порога z_τ = {num(Z_TAU)} не хватило {num(Z_TAU - row["z"])}', fontsize=10.5, color=MUTED, va='top')
    y = np.arange(len(top))
    ax.barh(y, top.values, color=[RED if v > 0 else ACC for v in top.values], height=0.6)
    for yy, v in zip(y, top.values):
        ax.text(v + (0.03 if v > 0 else -0.03), yy, ('+' if v > 0 else '') + num(v), va='center', ha='left' if v > 0 else 'right',
                color=INK, fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=10.5)
    ax.axvline(0, color=LINE, lw=1)
    ax.set_xlim(min(top.min() * 1.7, -0.3), top.max() * 1.12)
    ax.set_xlabel(f'вклад в z · красный – «к инциденту», синий – «к легитимному» · остальные {len(ph) - 10} полей в сумме {num(rest)}')
    ax.grid(axis='x', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig('images/g4_card.png', facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    d = load()
    t1, t2, t4 = task1(d), task2(d), task4(d)
    demo = prob_sum_demo(d, 525452)
    pd.set_option('display.width', 200)
    print('=== Задание 1: PDP / ICE / ALE'); print(t1.round(4).to_string(index=False))
    m, both, only_s, only_a, rho = t2
    print('=== Задание 2: топ-5 – общие', both, '| только SHAP', only_s, '| только ablation', only_a, '| ρ', round(rho, 3))
    print(m.sort_values('r_shap').head(12).to_string(index=False))
    print('=== Задание 4'); print({k: v for k, v in t4.items() if k not in ('bands', 'nearest')})
    print(t4['bands']); print(t4['nearest'].to_string(index=False)); print(demo)
    team = int(sys.argv[1]) if len(sys.argv) > 1 else 37
    print(f'=== Задание 3: набор {team}'); print(task3(d, team).to_string(index=False))
    figures(d, t1, t2, t4, demo)
    card_figure(d, CARD3)
    print('ok')
