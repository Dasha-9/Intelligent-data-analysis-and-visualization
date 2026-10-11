# СРС 2.2: график «до → после» и квартет Энскомба
# Запуск из корня репозитория: python python/charts_matplotlib.py [путь_к_parquet]
# Рисует images/matplotlib_do.png, images/matplotlib_posle.png и images/anscombe.png
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
plt.rcParams['font.family'] = 'DejaVu Sans'
PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/iavd_incidents_raw_target.parquet'
DAY = '2024-06-14'

df = pd.read_parquet(PATH, columns=['EntityId', 'Создан', 'Правило корреляции SIEM'])
df['msk'] = df['Создан'].dt.tz_convert('Europe/Moscow')
d = df[df['msk'].dt.date.astype(str) == DAY]
N = len(d)
vc = d['Правило корреляции SIEM'].dropna().value_counts()
top = vc.head(11)  # 10-е и 11-е места делят два правила с 36 карточками — берём оба
print(N, len(vc), top.sum(), d['Правило корреляции SIEM'].isna().sum())
print(top.to_string())

# ---------- ДО: типичное перегруженное оформление ----------
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
fig.patch.set_facecolor('#dfe7f2')
ax.set_facecolor('#eef2f8')
cols = plt.cm.rainbow(np.linspace(0, 1, len(top)))
xs = np.arange(len(top)); w = 0.6; dx, dy = 0.18, top.max() * 0.03
for i, (v, c) in enumerate(zip(top.values, cols)):
    ax.add_patch(Rectangle((i - w/2 + 0.12, 0), w, v - dy*0.4, fc='#555555', alpha=0.35, zorder=2))          # тень
    ax.add_patch(Rectangle((i - w/2, 0), w, v, fc=c, ec='black', lw=0.8, zorder=3))                          # фасад
    ax.add_patch(Polygon([(i-w/2, v), (i-w/2+dx, v+dy), (i+w/2+dx, v+dy), (i+w/2, v)], fc=c, ec='black', lw=0.8, alpha=0.75, zorder=3))  # верх
    ax.add_patch(Polygon([(i+w/2, 0), (i+w/2+dx, dy), (i+w/2+dx, v+dy), (i+w/2, v)], fc=c, ec='black', lw=0.8, alpha=0.55, zorder=3))  # бок
ax.set_xlim(-0.7, len(top) - 0.1); ax.set_ylim(0, top.max() * 1.12)
ax.set_xticks(xs); ax.set_xticklabels(top.index, rotation=45, ha='right', fontsize=8)
ax.minorticks_on()
ax.grid(which='major', color='#444444', lw=1.0, zorder=0)
ax.grid(which='minor', color='#777777', lw=0.5, ls=':', zorder=0)
for s in ax.spines.values(): s.set_linewidth(2.5); s.set_color('#1a2a4a')
ax.set_ylabel('Количество карточек, шт.', fontsize=10)
ax.set_xlabel('Правило корреляции SIEM', fontsize=10)
ax.set_title('Алерты по правилам SIEM за сутки 14.06.2024', fontsize=14, weight='bold')
h = Rectangle((0, 0), 1, 1, fc=cols[0], ec='black')
ax.legend([h], ['Количество карточек'], loc='upper right', frameon=True, shadow=True, fancybox=True)
fig.add_artist(Rectangle((0.005, 0.005), 0.99, 0.99, transform=fig.transFigure, fill=False, ec='#1a2a4a', lw=4))
fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
fig.savefig('images/matplotlib_do.png', facecolor=fig.get_facecolor())
plt.close(fig)

# ---------- ПОСЛЕ ----------
INK, MUTED, GRAY, ACC = '#1f2328', '#6e7781', '#c3c9d0', '#0969da'
t = top.iloc[::-1]
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
fig.patch.set_facecolor('white')
y = np.arange(len(t))
colors = [ACC if v == top.max() else GRAY for v in t.values]
ax.barh(y, t.values, color=colors, height=0.68, zorder=2)
ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=9.5, color=INK)
for yi, v in zip(y, t.values):
    lab = f'{v:,}'.replace(',', ' ') + (f' · {v/N:.0%} всех' if v == top.max() else '')
    if v == top.max():
        ax.text(v - top.max()*0.015, yi, lab, va='center', ha='right', fontsize=11, color='white', weight='bold')
    else:
        ax.text(v + top.max()*0.01, yi, lab, va='center', fontsize=10, color=INK)
ax.set_xlim(0, top.max() * 1.03)
for s in ['top', 'right', 'bottom']: ax.spines[s].set_visible(False)
ax.spines['left'].set_color(GRAY)
ax.tick_params(axis='y', length=0); ax.set_xticks([])
fig.text(0.015, 0.955, f'14.06.2024: {top.max()/N:.0%} карточек дало одно правило — сканирование портов',
         fontsize=15, weight='bold', color=INK)
NS = f'{N:,}'.replace(',', ' ')
fig.text(0.015, 0.915, f'Карточки по правилу корреляции SIEM, шт. · топ-11 из {len(vc)} правил · сутки по МСК, всего {NS} карточек',
         fontsize=10.5, color=MUTED)
fig.tight_layout(rect=(0.0, 0.0, 1, 0.89))
fig.savefig('images/matplotlib_posle.png', facecolor='white')
plt.close(fig)

# ---------- Квартет Энскомба ----------
x = [10,8,13,9,11,14,6,4,12,7,5]; x4 = [8,8,8,8,8,8,8,19,8,8,8]
Y = [[8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68],
     [9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74],
     [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73],
     [6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89]]
X = [x, x, x, x4]
names = ['I · линейное облако', 'II · криволинейная связь', 'III · один выброс тянет прямую', 'IV · связь задаёт одна точка']
fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), dpi=150, sharex=True, sharey=True)
xx = np.linspace(2, 20, 2)
for a, xi, yi, nm in zip(axs.flat, X, Y, names):
    a.plot(xx, 3 + 0.5*xx, color=GRAY, lw=1.5, zorder=1)
    a.scatter(xi, yi, s=36, color=INK, zorder=2)
    a.set_title(nm, fontsize=11, color=INK, loc='left')
    a.set_xlim(2, 20); a.set_ylim(2, 14)
    for s in ['top', 'right']: a.spines[s].set_visible(False)
    for s in ['left', 'bottom']: a.spines[s].set_color(GRAY)
    a.tick_params(colors=MUTED, labelsize=9)
fig.suptitle('Квартет Энскомба: одинаковые сводки, разная структура', fontsize=14, weight='bold', color=INK, x=0.02, ha='left')
fig.text(0.02, 0.915, 'Во всех наборах: x̄ = 9,0 · ȳ = 7,50 · s²x = 11,0 · s²y ≈ 4,12 · r ≈ 0,816 · прямая y = 3,0 + 0,5x (серая)',
         fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig('images/anscombe.png', facecolor='white')
