# СРС 2.4 · графики проверки трёх гипотез (matplotlib, статичные PNG для отчёта)
# Запуск из корня папки: python python/charts_matplotlib.py [путь_к_parquet]
# Результат: images/h1_rules.png, images/h2_load.png, images/h3_take.png
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eda_common import load, h1_rules, h2_load, h3_take, ORDER, H1_MIN_N, H1_MIN_FP, fmt, pct

INK, MUTED, GRAY, GRAY_D, LINE, ACC, RED = '#1F2328', '#6E7781', '#C3C9D0', '#57606A', '#D0D7DE', '#0969DA', '#CF222E'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': LINE, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
OUT = 'images'
os.makedirs(OUT, exist_ok=True)


def header(fig, title, sub):
    fig.text(0.012, 0.965, title, fontsize=15, weight='bold', color=INK, va='top')
    fig.text(0.012, 0.905, sub, fontsize=10.5, color=MUTED, va='top')


def chart_h1(df):
    g, s = h1_rules(df)
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.11)
    header(fig, f'Гипотеза 1. {s["noisy_rules"]} правил SIEM почти всегда дают False Positive – '
                f'{pct(s["noisy_share"])} разобранных карточек',
           f'Точка – правило корреляции SIEM ({s["rules"]} правил с вердиктом) · X – разобранных карточек, лог. шкала · '
           f'Y – доля False Positive · «Не указан» исключён')
    ax.axhspan(H1_MIN_FP * 100, 101, xmin=0, xmax=1, color='#DDF4FF', zorder=0)
    ax.axvline(H1_MIN_N, color=LINE, lw=1, ls='--', zorder=1)
    rest, noisy = g[~g.noisy], g[g.noisy]
    ax.scatter(rest.n, rest.fp * 100, s=22, color=GRAY, edgecolor='white', lw=0.4, zorder=2)
    ax.scatter(noisy.n, noisy.fp * 100, s=40, color=ACC, edgecolor='white', lw=0.5, zorder=3)
    ax.axhline(s['fp_all'] * 100, color=GRAY_D, lw=1, zorder=1)
    ax.text(1.05, s['fp_all'] * 100 - 1.2, f'в среднем по всем разобранным карточкам: {pct(s["fp_all"])} FP',
            color=GRAY_D, fontsize=9.5, va='top')
    ax.annotate(f'«шумные» правила: ≥ {H1_MIN_N} карточек и ≥ {int(H1_MIN_FP * 100)} % FP\n'
                f'{s["noisy_rules"]} правил · {fmt(s["noisy_cards"])} карточек ({pct(s["noisy_share"])} разобранных)\n'
                f'из них «Решён» – {s["noisy_solved"]}: отключать нельзя, только донастраивать',
                xy=(700, 94.5), xytext=(700, 18), ha='center', va='center', fontsize=10, color=ACC,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=ACC, lw=0.8),
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.2))
    big = g.sort_values('n', ascending=False).iloc[0]
    ax.annotate(f'{g.sort_values("n", ascending=False).index[0]}: {fmt(big.n)} карточек, {pct(big.fp)} FP',
                (big.n, big.fp * 100), xytext=(-10, -22), textcoords='offset points', ha='right', fontsize=8.5, color=INK)
    ax.set_xscale('log'); ax.set_xlim(0.8, 25000); ax.set_ylim(-2, 101)
    ax.set_xticks([1, 10, 100, 300, 1000, 10000]); ax.set_xticklabels(['1', '10', '100', '300', '1 000', '10 000'])
    ax.set_yticks(range(0, 101, 20)); ax.set_yticklabels([f'{v} %' for v in range(0, 101, 20)])
    ax.set_xlabel('Разобранных карточек по правилу (лог. шкала)'); ax.set_ylabel('Доля False Positive')
    ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig(f'{OUT}/h1_rules.png', facecolor='white'); plt.close(fig)
    return s


def chart_h2(df):
    g, slots, s = h2_load(df)
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.15)
    lo, hi = g['over'].iloc[0], g['over'].iloc[-1]
    header(fig, f'Гипотеза 2. Чем больше карточек за час, тем чаще просрочен SLA взятия: {pct(lo)} → {pct(hi)}',
           f'Доля карточек с «SLA просрочен (взятие в работу)» = True · группы по числу карточек, созданных в тот же час (МСК)\n'
           f'{fmt(s["cards"])} карточек с заполненным флагом, {fmt(s["slots"])} часов · Спирмен ρ = {s["rho"]:.2f}'.replace('.', ','))
    x = np.arange(len(g))
    cols = [ACC if v >= 0.35 else GRAY for v in g['over']]
    ax.bar(x, g['over'] * 100, color=cols, width=0.62, zorder=2)
    ax.axhline(s['over_all'] * 100, color=GRAY_D, lw=1, ls='--', zorder=3)
    ax.text(-0.45, s['over_all'] * 100 + 0.8, f'в среднем {pct(s["over_all"])}', color=GRAY_D, fontsize=9.5)
    for i, (idx, r) in enumerate(g.iterrows()):
        ax.text(i, r.over * 100 + 1.0, pct(r.over), ha='center', fontsize=12, weight='bold', color=ACC if r.over >= 0.35 else INK)
    ax.set_xticks(x); ax.set_xticklabels([f'{l}\n{fmt(r.cards)} карт. · {fmt(r.slots)} ч' for l, r in g.iterrows()], fontsize=10.5, color=INK)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.set_xlabel('Карточек создано за тот же час (МСК)', labelpad=6)
    ax.set_ylim(0, 55); ax.set_yticks(range(0, 51, 10)); ax.set_yticklabels([f'{v} %' for v in range(0, 51, 10)])
    ax.set_ylabel('Доля с просроченным SLA взятия')
    ax.grid(axis='y', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig(f'{OUT}/h2_load.png', facecolor='white'); plt.close(fig)
    return g, s


def chart_h3(df):
    d, g, s = h3_take(df)
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
    fig.subplots_adjust(left=0.13, right=0.83, top=0.78, bottom=0.11)
    c = g.loc['Критическая']
    header(fig, f'Гипотеза 3. Критические карточки ждут дольше всех: медиана {c.med:.0f} мин при SLA {c.sla:.0f} мин',
           f'Время от «Создан» до «Дата взятия в работу», мин, лог. шкала · ящик – Q1–Q3, линия – медиана, усы – 1,5 IQR\n'
           f'{fmt(s["valid"])} карточек (без пустых и отрицательных интервалов)')
    data = [np.log10(d.loc[d.crit == k, 'take'] + 0.1) for k in ORDER][::-1]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=GRAY_D), capprops=dict(color=GRAY_D))
    for patch, k in zip(bp['boxes'], ORDER[::-1]):
        patch.set_facecolor('#FFEBE9' if k == 'Критическая' else '#EEF0F2')
        patch.set_edgecolor(RED if k == 'Критическая' else GRAY_D)
    for i, k in enumerate(ORDER[::-1], start=1):
        r = g.loc[k]
        y = np.log10(r.sla + 0.1)
        ax.plot([y, y], [i - 0.36, i + 0.36], color=RED, lw=2.5, zorder=4)
        ax.text(y, i + 0.40, f'SLA {r.sla:.0f}', color=RED, fontsize=8.8, ha='center', va='bottom')
        ax.text(1.02, i, 
                 f'медиана {r.med:.1f} мин'.replace('.', ',') + f'\nсверх SLA: {pct(r.over)}\nn = {fmt(r.n)}', transform=ax.get_yaxis_transform(),
                 fontsize=10, va='center', color=RED if k == 'Критическая' else INK, weight='bold' if k == 'Критическая' else 'normal')
    ax.set_yticks(range(1, 5)); ax.set_yticklabels(ORDER[::-1], fontsize=11.5)
    for lbl in ax.get_yticklabels():
        lbl.set_color(RED if lbl.get_text() == 'Критическая' else INK)
    ticks = [1, 2, 5, 10, 30, 60, 240, 1440]
    ax.set_xticks(np.log10(np.array(ticks) + 0.1)); ax.set_xticklabels(['1', '2', '5', '10', '30', '1 ч', '4 ч', '24 ч'])
    ax.set_xlim(np.log10(0.1 + 0.1), np.log10(3000))
    ax.set_xlabel('Минут до взятия в работу (лог. шкала)')
    ax.grid(axis='x', color='#EEF0F2', lw=0.8); ax.set_axisbelow(True)
    fig.savefig(f'{OUT}/h3_take.png', facecolor='white'); plt.close(fig)
    return g, s


if __name__ == '__main__':
    df = load()
    print(chart_h1(df)); print(chart_h2(df)); print(chart_h3(df))
