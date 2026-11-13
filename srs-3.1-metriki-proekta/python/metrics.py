# СРС 3.1 · метрики проекта: матрица ошибок, порог, ROC и PR, Fβ
# Запуск из корня папки: python python/metrics.py [путь_к_parquet]
#   по умолчанию data/iavd_ml_dataset.parquet (карточки по 20.11.2025)
# Выход: числа в консоль + images/m1_matrix.png, images/m2_threshold.png, images/m3_curves.png
#
# Класс 1 – реальный инцидент («Решён» / «Не решён»), класс 0 – False Positive.
# Оценка (score) – доля реальных инцидентов у правила корреляции SIEM в обучающем периоде,
# сглаженная к общей доле. Обучение – карточки до 04.09.2025, проверка – 04.09–20.11.2025
# (тот же период, что у лога инференса модели преподавателей). Тот же расчёт – в СРС 3.2.
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

PATH = next((a for a in sys.argv[1:] if not a.startswith('--')), 'data/iavd_ml_dataset.parquet')
CUT = pd.Timestamp('2025-09-04', tz='Europe/Moscow')
SMOOTH = 20
MIN_RECALL = 0.95          # требование: полнота не ниже 95 %
MID = 0.5                  # «середина» для сравнения
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
    return te, dict(train=len(tr), train_pos=prior, test=len(te), pos=int(te['y'].sum()), neg=int((te['y'] == 0).sum()))


def point(y, s, tau):
    """Матрица ошибок и метрики при пороге τ: оценка ≥ τ → «отдать аналитику»."""
    p = s >= tau
    TP, FP = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    FN, TN = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN)
    n = TP + FP + FN + TN
    return dict(tau=tau, TP=TP, FP=FP, FN=FN, TN=TN, P=P, R=R, F1=fb(P, R, 1), FPR=FP / (FP + TN),
                acc=(TP + TN) / n, flag=(TP + FP) / n)


def fb(P, R, b):
    return (1 + b * b) * P * R / (b * b * P + R) if P + R else 0.0


def sweep(y, s):
    return pd.DataFrame([point(y, s, t) for t in np.r_[np.unique(s), 1.0001]])


def work_tau(sw):
    """Рабочий порог: наибольшая точность среди порогов с полнотой ≥ MIN_RECALL."""
    ok = sw[sw['R'] >= MIN_RECALL]
    t = float(ok.loc[ok['P'].idxmax(), 'tau'])
    return np.floor(t * 100) / 100       # округляем вниз до сотых: полнота при этом не уменьшается


if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    te, info = load()
    y, s = te['y'].values, te['score'].values
    sw = sweep(y, s)
    t0 = work_tau(sw)
    a, m = point(y, s, t0), point(y, s, MID)
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    print(info)
    for name, r in (('рабочий', a), ('0,5', m)):
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
              {f'F{b}': round(fb(r['P'], r['R'], b), 4) for b in (0.5, 1, 2)})
    print('всегда FP: доля правильных', round(info['neg'] / info['test'], 4), ' AUC ROC', round(auc, 3), ' AP', round(ap, 3),
          ' значений оценки', len(np.unique(s)))

    # --- рис. 1: матрица ошибок в рабочей точке ---
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.76, bottom=0.08)
    fig.text(0.012, 0.965, f'Порог {num(t0)}: найдено {pct(a["R"])} инцидентов, точность {pct(a["P"])}',
             fontsize=15, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.895, f'Проверочный период 04.09–20.11.2025 · {fmt(info["test"])} карточек с вердиктом, '
             f'из них {fmt(info["pos"])} реальных инцидентов ({pct(info["pos"] / info["test"])})',
             fontsize=10.5, color=MUTED, va='top')
    cells = [[('TP', a['TP'], ACC), ('FN', a['FN'], RED)], [('FP', a['FP'], '#8C959F'), ('TN', a['TN'], '#F3F5F8')]]
    for i in range(2):
        for j in range(2):
            code, n, col = cells[i][j]
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.98, 0.98, color=col, alpha=1 if code != 'FP' else 0.55))
            tc = 'white' if code in ('TP', 'FN') else INK
            ax.text(j + 0.49, 1 - i + 0.58, f'{code} – {fmt(n)}', ha='center', fontsize=17, weight='bold', color=tc)
            sub = {'TP': 'инцидент отдан аналитику', 'FN': 'инцидент пропущен', 'FP': 'ложная тревога у аналитика',
                   'TN': 'шум снят с очереди'}[code]
            ax.text(j + 0.49, 1 - i + 0.34, sub, ha='center', fontsize=11, color=tc)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis('off')
    ax.text(0.49, 2.04, 'Прогноз: аналитику (оценка ≥ τ)', ha='center', color=MUTED)
    ax.text(1.49, 2.04, 'Прогноз: шум (оценка < τ)', ha='center', color=MUTED)
    ax.text(-0.04, 1.49, f'Факт: инцидент\n{fmt(info["pos"])}', ha='right', va='center', color=MUTED)
    ax.text(-0.04, 0.49, f'Факт: False Positive\n{fmt(info["neg"])}', ha='right', va='center', color=MUTED)
    fig.savefig('images/m1_matrix.png', facecolor='white'); plt.close(fig)

    # --- рис. 2: метрики по порогу ---
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.11)
    fig.text(0.012, 0.965, f'Требование «полнота ≥ 95 %» выполняется до τ = {num(t0)}; дальше полнота падает',
             fontsize=15, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, 'Рабочий порог – наибольшая точность среди порогов с полнотой ≥ 95 % · «середина» 0,5 – для сравнения',
             fontsize=10.5, color=MUTED, va='top')
    k = sw['tau'] <= 0.72
    f2 = [fb(p, r, 2) for p, r in zip(sw['P'], sw['R'])]
    ax.axhspan(MIN_RECALL, 1.02, color='#EEF4FB', zorder=0)
    series = [('R', ACC, 2.6, '-', 'полнота'), ('P', INK, 2, '-', 'точность'), ('F2', '#8C959F', 1.6, '--', 'F2'),
              ('flag', '#A8B0B8', 1.8, ':', 'доля карточек аналитику')]
    sw['F2'] = f2
    for col, c, lw, ls, lab in series:
        # τ между соседними значениями оценки отбирает те же карточки, что и следующее значение → steps-pre
        ax.plot(sw.loc[k, 'tau'], sw.loc[k, col], color=c, lw=lw, ls=ls, drawstyle='steps-pre', label=lab)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=10.5, handlelength=2.6)
    for t, c in ((t0, ACC), (MID, MUTED)):
        ax.axvline(t, color=c, ls=':', lw=1.2)
    ax.text(t0 + 0.008, 0.06, f'τ = {num(t0)}', color=ACC, fontsize=10)
    ax.text(MID + 0.008, 0.06, 'τ = 0,5', color=MUTED, fontsize=10)
    ax.text(0.30, 0.975, 'зона «полнота ≥ 95 %»', color=ACC, fontsize=9.5, va='center')
    ax.set_xlim(0, 0.7); ax.set_ylim(0, 1.02); ax.set_xlabel('Порог τ')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v * 100:.0f} %'))
    ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig('images/m2_threshold.png', facecolor='white'); plt.close(fig)

    # --- рис. 3: ROC и PR ---
    fpr, tpr, _ = roc_curve(y, s)
    pr, rc, _ = precision_recall_curve(y, s)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.11, wspace=0.25)
    fig.text(0.012, 0.965, f'ROC: площадь {num(auc, 3)} · точность и полнота: площадь {num(ap, 3)} при доле инцидентов {pct(info["pos"] / info["test"])}',
             fontsize=14.5, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, 'Точки – пороги 0,5 (серая) и рабочий (синяя) · пунктир – случайное угадывание',
             fontsize=10.5, color=MUTED, va='top')
    a1.plot(fpr, tpr, color=INK, lw=2); a1.plot([0, 1], [0, 1], color=GRAY, ls='--', lw=1)
    a2.plot(rc, pr, color=INK, lw=2, drawstyle='steps-post'); a2.axhline(info['pos'] / info['test'], color=GRAY, ls='--', lw=1)
    for r, c in ((m, MUTED), (a, ACC)):
        a1.plot(r['FPR'], r['R'], 'o', color=c, ms=9, zorder=5)
        a2.plot(r['R'], r['P'], 'o', color=c, ms=9, zorder=5)
    a1.annotate(f'τ = {num(t0)}', (a['FPR'], a['R']), xytext=(10, -16), textcoords='offset points', color=ACC)
    a1.annotate('τ = 0,5', (m['FPR'], m['R']), xytext=(10, -4), textcoords='offset points', color=MUTED)
    a2.annotate(f'τ = {num(t0)}', (a['R'], a['P']), xytext=(-14, 12), ha='right', textcoords='offset points', color=ACC)
    a2.annotate('τ = 0,5', (m['R'], m['P']), xytext=(10, 6), textcoords='offset points', color=MUTED)
    a1.set_xlabel('Доля ложных тревог (FPR)'); a1.set_ylabel('Полнота (TPR)'); a1.set_title('ROC-кривая', color=INK, loc='left')
    a2.set_xlabel('Полнота'); a2.set_ylabel('Точность'); a2.set_title('Кривая точности и полноты', color=INK, loc='left')
    for ax in (a1, a2):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig('images/m3_curves.png', facecolor='white'); plt.close(fig)
    print('ok')
