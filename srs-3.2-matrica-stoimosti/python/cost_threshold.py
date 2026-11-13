# СРС 3.2 · матрица стоимости и выбор порога по минимуму стоимости
# Запуск из корня папки: python python/cost_threshold.py [путь_к_parquet]
#   по умолчанию data/iavd_ml_dataset.parquet (карточки по 20.11.2025)
# Выход: числа в консоль + images/c1_cost.png, images/c2_fbeta.png, images/c3_matrix.png
#
# Оценка (score) – как в СРС 3.1: доля реальных инцидентов у правила корреляции SIEM в обучающем периоде.
# Обучение – карточки до 04.09.2025, проверка – 04.09–20.11.2025 (тот же период, что у лога инференса
# модели преподавателей). Класс 1 – реальный инцидент («Решён»/«Не решён»), класс 0 – False Positive.
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH = next((a for a in sys.argv[1:] if not a.startswith('--')), 'data/iavd_ml_dataset.parquet')
CUT = pd.Timestamp('2025-09-04', tz='Europe/Moscow')
SMOOTH = 20
K_BASE, K_SCEN = 20, [5, 20, 50]           # базовое соотношение цен и сценарии
INK, MUTED, GRAY, LINE, ACC, RED = '#1F2328', '#6E7781', '#C3C9D0', '#D0D7DE', '#0969DA', '#CF222E'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': LINE, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
fmt = lambda v: f'{int(round(v)):,}'.replace(',', ' ')
pct = lambda v, d=1: f'{v * 100:.{d}f} %'.replace('.', ',')
num = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',')


def load(path=PATH):
    d = pd.read_parquet(path, columns=['EntityId', 'Дата создания', 'Вердикт', 'Правило корреляции SIEM'])
    d = d[d['Вердикт'].isin(['False Positive', 'Решён', 'Не решён'])].copy()
    d['y'] = d['Вердикт'].isin(['Решён', 'Не решён']).astype(int)
    d['rule'] = d['Правило корреляции SIEM'].fillna('(нет правила)')
    tr, te = d[d['Дата создания'] < CUT], d[d['Дата создания'] >= CUT].copy()
    prior = tr['y'].mean()
    g = tr.groupby('rule')['y'].agg(['sum', 'count'])
    te['score'] = te['rule'].map((g['sum'] + SMOOTH * prior) / (g['count'] + SMOOTH)).fillna(prior)
    return te, dict(train=len(tr), test=len(te), pos=int(te['y'].sum()), neg=int((te['y'] == 0).sum()))


def sweep(te):
    """Все различимые пороги: оценка ≥ τ → «отдать аналитику» (класс 1)."""
    y = te['y'].values
    taus = np.r_[np.unique(te['score'].values), 1.0001]
    rows = []
    for t in taus:
        p = te['score'].values >= t
        TP, FP = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        FN, TN = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN)
        rows.append(dict(tau=t, TP=TP, FP=FP, FN=FN, TN=TN, P=P, R=R))
    return pd.DataFrame(rows)


def fbeta(P, R, b):
    return np.where(P + R > 0, (1 + b * b) * P * R / np.maximum(b * b * P + R, 1e-12), 0.0)


def best(sw, k):
    cost = k * sw['FN'] + sw['FP']
    i = int(cost.idxmin())
    fb = fbeta(sw['P'].values, sw['R'].values, np.sqrt(k))
    j = int(np.argmax(fb))
    return sw.loc[i], cost.loc[i], sw.loc[j], cost.loc[j], fb[j]


if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    te, info = load()
    sw = sweep(te)
    print(info)
    all_alarm = info['neg']                    # «тревога на всё»: FN = 0, FP = все легитимные
    rows = []
    for k in K_SCEN:
        rc, c, rf, cf, fbv = best(sw, k)
        rows.append(dict(k=k, beta=np.sqrt(k), tau_cost=rc.tau, FN=rc.FN, FP=rc.FP, cost=c, R=rc.R, P=rc.P,
                         tau_f=rf.tau, FN_f=rf.FN, FP_f=rf.FP, cost_f=cf, gain=1 - c / min(all_alarm, k * info['pos'])))
    res = pd.DataFrame(rows)
    print(res.round(4).to_string())
    print('тревога на всё:', all_alarm, ' молчание (k=20):', K_BASE * info['pos'])

    # --- рис. 1: стоимость по порогу для трёх k ---
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.78, bottom=0.11)
    r20 = res.set_index('k').loc[K_BASE]
    fig.text(0.012, 0.965, f'Стоимость ошибок Cost(τ) = k · FN + FP: при k = {K_BASE} минимум в τ* = {num(r20.tau_cost)}',
             fontsize=15, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, f'Проверочный период 04.09–20.11.2025 · {fmt(info["test"])} карточек с вердиктом ({fmt(info["pos"])} инцидентов)\n'
             'Стоимость в «разборах FP»: 1 = разбор одной ложной тревоги, пропуск инцидента = k', fontsize=10.5, color=MUTED, va='top')
    m = sw['tau'] <= 0.6
    for k, col, lw in [(5, GRAY, 1.6), (20, ACC, 2.6), (50, INK, 1.6)]:
        c = k * sw['FN'] + sw['FP']
        ax.plot(sw.loc[m, 'tau'], c[m], color=col, lw=lw, drawstyle='steps-post')
        r = res.set_index('k').loc[k]
        ax.plot(r.tau_cost, r.cost, 'o', color=col, ms=8, zorder=5)
        ax.text(0.605, c[m].iloc[-1], f'k = {k}', color=col, fontsize=10.5, va='center')
        off = {5: (12, -4), 20: (10, -18), 50: (-10, -22)}[k]
        ax.annotate(f'τ* = {num(r.tau_cost)} · {fmt(r.cost)}', (r.tau_cost, r.cost), xytext=off, ha='left' if k != 50 else 'right',
                    textcoords='offset points', color=col if k != 5 else '#8C959F', fontsize=9.5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.axhline(all_alarm, color=RED, lw=1, ls='--')
    ax.text(0.30, all_alarm * 1.03, f'«тревога на всё»: {fmt(all_alarm)}', color=RED, fontsize=9.5)
    ax.set_xlim(0, 0.6); ax.set_xlabel('Порог τ'); ax.set_ylabel('Стоимость, в разборах FP')
    ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig('images/c1_cost.png', facecolor='white'); plt.close(fig)

    # --- рис. 2: минимум стоимости против максимума F_β, β = √k ---
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.80, bottom=0.11)
    b = np.sqrt(K_BASE)
    fb = fbeta(sw['P'].values, sw['R'].values, b)
    c = K_BASE * sw['FN'] + sw['FP']
    cn = 1 - (c - c.min()) / (c.max() - c.min())
    fig.text(0.012, 0.965, f'k = {K_BASE}: минимум стоимости τ* = {num(r20.tau_cost)}, максимум F(β = {num(b, 1)}) – τ = {num(r20.tau_f)}',
             fontsize=15, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, 'Обе кривые нормированы к 0…1 · β = √k – эвристика из лекции 3.2', fontsize=10.5, color=MUTED, va='top')
    ax.plot(sw.loc[m, 'tau'], cn[m], color=ACC, lw=2.4, drawstyle='steps-post')
    ax.plot(sw.loc[m, 'tau'], (fb / fb.max())[m.values], color=INK, lw=1.8, ls='--', drawstyle='steps-post')
    ax.axvline(r20.tau_cost, color=ACC, ls=':', lw=1); ax.axvline(r20.tau_f, color=INK, ls=':', lw=1)
    ax.text(0.605, cn[m].iloc[-1] - 0.04, '1 − норм. стоимость', color=ACC, fontsize=10, va='center')
    ax.text(0.605, (fb / fb.max())[m.values][-1] + 0.04, f'F(β = {num(b, 1)}) норм.', color=INK, fontsize=10, va='center')
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.05); ax.set_xlabel('Порог τ'); ax.set_ylabel('чем выше, тем лучше')
    ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig('images/c2_fbeta.png', facecolor='white'); plt.close(fig)

    # --- рис. 3: матрица стоимости (тепловая карта) при τ* и k = 20 ---
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.10)
    fig.text(0.012, 0.965, f'Цена ошибок при τ* = {num(r20.tau_cost)}, k = {K_BASE}: {fmt(r20.cost)} «разборов FP»',
             fontsize=14, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, 'Число карточек × цена клетки · правильные решения стоят 0', fontsize=10.5, color=MUTED, va='top')
    TP = int(r20.R * info['pos'] + 0.5); FN = int(r20.FN); FP = int(r20.FP); TN = info['neg'] - FP
    cells = [[(TP, 0, 'TP'), (FN, K_BASE, 'FN')], [(FP, 1, 'FP'), (TN, 0, 'TN')]]
    vmax = max(FN * K_BASE, FP)
    for i in range(2):
        for j in range(2):
            n, w, code = cells[i][j]
            v = n * w
            col = RED if code == 'FN' else ('#8C959F' if code == 'FP' else '#F3F5F8')
            alpha = 0.15 + 0.85 * v / vmax if v else 1
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.98, 0.98, color=col, alpha=alpha))
            tc = 'white' if v > vmax * 0.5 else INK
            ax.text(j + 0.49, 1 - i + 0.62, f'{code}: {fmt(n)} карт.', ha='center', fontsize=13, weight='bold', color=tc)
            ax.text(j + 0.49, 1 - i + 0.36, f'× {w} = {fmt(v)}', ha='center', fontsize=12, color=tc)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis('off')
    ax.text(0.49, 2.04, 'Прогноз: аналитику', ha='center', color=MUTED); ax.text(1.49, 2.04, 'Прогноз: шум', ha='center', color=MUTED)
    ax.text(-0.04, 1.49, 'Факт: инцидент', ha='right', va='center', color=MUTED); ax.text(-0.04, 0.49, 'Факт: FP', ha='right', va='center', color=MUTED)
    fig.savefig('images/c3_matrix.png', facecolor='white'); plt.close(fig)
    res.to_csv('images/results.csv', index=False)
    print('ok')
