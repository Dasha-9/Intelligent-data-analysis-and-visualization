# СРС 2.4 · три EDA-гипотезы – интерактивная проверка в Taipy + Plotly (трек ML)
# Запуск из корня папки: pip install -r requirements.txt
#                        python python/hypotheses_taipy.py [путь_к_parquet]
# Откроется браузер (порт 5058): страницы /h1, /h2, /h3
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import plotly.graph_objects as go
from taipy.gui import Gui
from eda_common import load, h1_rules, h2_load, h3_take, ORDER, H1_MIN_N, H1_MIN_FP, fmt, pct

INK, MUTED, GRAY, GRAY_D, ACC, RED = '#1F2328', '#6E7781', '#C3C9D0', '#57606A', '#0969DA', '#CF222E'
BASE = dict(plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', color=INK, size=13),
            margin=dict(l=70, r=40, t=100, b=60), height=620, showlegend=False)


def title(t, sub):
    return dict(text=f'<b>{t}</b><br><span style="font-size:13px;color:{MUTED}">{sub}</span>', x=0.01, font=dict(size=19))


df = load()

# ---------- Гипотеза 1: правила SIEM × доля False Positive ----------
g1, s1 = h1_rules(df)
fig_h1 = go.Figure()
for mask, color, size, name in [(~g1.noisy, GRAY, 7, 'остальные'), (g1.noisy, ACC, 10, '«шумные»')]:
    d = g1[mask]
    fig_h1.add_scatter(x=d.n, y=d.fp * 100, mode='markers', name=name, text=d.index,
                       marker=dict(color=color, size=size, line=dict(color='white', width=0.5)),
                       customdata=d.solved,
                       hovertemplate='%{text}<br>%{x} карточек · %{y:.1f} % FP · «Решён»: %{customdata}<extra></extra>')
fig_h1.add_hrect(y0=H1_MIN_FP * 100, y1=101, fillcolor='#DDF4FF', line_width=0, layer='below')
fig_h1.add_vline(x=H1_MIN_N, line=dict(color='#D0D7DE', dash='dash'))
fig_h1.add_hline(y=s1['fp_all'] * 100, line=dict(color=GRAY_D, width=1),
                 annotation_text=f'в среднем {pct(s1["fp_all"])} FP', annotation_position='bottom left')
fig_h1.update_layout(**BASE, title=title('Правила SIEM и доля False Positive', 'Синий цвет · ≥ 300 карточек и ≥ 95 % False Positive'),
                     xaxis=dict(type='log', title='Разобранных карточек по правилу (лог. шкала)', showgrid=False, range=[-0.1, 4.4]),
                     yaxis=dict(title='Доля False Positive, %', gridcolor='#EEF0F2', range=[-2, 101]))

# ---------- Гипотеза 2: нагрузка часа × просрочка SLA ----------
g2, _, s2 = h2_load(df)
lbl = [f'{l}<br><span style="font-size:11px;color:{MUTED}">{fmt(r.cards)} карт. · {fmt(r.slots)} ч</span>' for l, r in g2.iterrows()]
fig_h2 = go.Figure(go.Bar(x=lbl, y=g2.over * 100, marker_color=[ACC if v >= 0.35 else GRAY for v in g2.over],
                          text=[pct(v) for v in g2.over], textposition='outside', textfont=dict(size=15),
                          hovertemplate='%{y:.1f} %<extra></extra>'))
fig_h2.add_hline(y=s2['over_all'] * 100, line=dict(color=GRAY_D, dash='dash'),
                 annotation_text=f'в среднем {pct(s2["over_all"])}', annotation_position='top left')
fig_h2.update_layout(**BASE, title=title('Почасовая нагрузка и просрочка SLA', 'Группировка по числу карточек за час (МСК)'),
                     xaxis=dict(title='Карточек создано за тот же час (МСК)'),
                     yaxis=dict(title='Доля с просроченным SLA взятия, %', gridcolor='#EEF0F2', range=[0, 55]), bargap=0.35)

# ---------- Гипотеза 3: время до взятия по критичности ----------
d3, g3, _ = h3_take(df)
fig_h3 = go.Figure()
for k in ORDER[::-1]:
    lx = np.log10(d3.loc[d3.crit == k, 'take'] + 0.1)          # усы считаем на лог. шкале, как в matplotlib-версии
    q1, med, q3 = lx.quantile([.25, .5, .75])
    lo, hi = lx[lx >= q1 - 1.5 * (q3 - q1)].min(), lx[lx <= q3 + 1.5 * (q3 - q1)].max()
    fig_h3.add_box(y=[k], q1=[10 ** q1], median=[10 ** med], q3=[10 ** q3], lowerfence=[10 ** lo], upperfence=[10 ** hi],
                   name=k, orientation='h', marker_color=RED if k == 'Критическая' else GRAY_D,
                   fillcolor='#FFEBE9' if k == 'Критическая' else '#EEF0F2', line=dict(width=1.5), hoverinfo='skip')
for i, k in enumerate(ORDER[::-1]):
    r = g3.loc[k]
    fig_h3.add_shape(type='line', x0=r.sla, x1=r.sla, y0=i - 0.4, y1=i + 0.4, yref='y', line=dict(color=RED, width=3))
    fig_h3.add_annotation(x=np.log10(r.sla), y=i + 0.45, text=f'SLA {r.sla:.0f}', showarrow=False, font=dict(color=RED, size=11))
fig_h3.update_layout(**BASE, title=title('Время до взятия в работу по критичности', 'Красная черта · SLA взятия в работу'),
                     xaxis=dict(type='log', title='Минут до взятия в работу (лог. шкала)', gridcolor='#EEF0F2',
                                tickvals=[1, 2, 5, 10, 30, 60, 240, 1440], ticktext=['1', '2', '5', '10', '30', '1 ч', '4 ч', '24 ч'],
                                range=[-0.7, 3.5]),
                     yaxis=dict(tickfont=dict(size=13), ticktext=[f'{k}<br><span style="font-size:11px;color:{MUTED}">медиана {g3.loc[k].med:.1f} мин · сверх SLA {pct(g3.loc[k].over)}</span>'.replace('.', ',', 1) for k in ORDER[::-1]], tickvals=ORDER[::-1]))

nav = [('/h1', 'Гипотеза 1 · правила'), ('/h2', 'Гипотеза 2 · нагрузка'), ('/h3', 'Гипотеза 3 · критичность')]
pages = {
    '/': '<|navbar|lov={nav}|>',
    'h1': '<|chart|figure={fig_h1}|height=640px|>',
    'h2': '<|chart|figure={fig_h2}|height=640px|>',
    'h3': '<|chart|figure={fig_h3}|height=640px|>',
}

if __name__ == '__main__':
    Gui(pages=pages).run(title='СРС 2.4 · EDA-гипотезы', port=5058, use_reloader=False, dark_mode=False,
        server_config={'socketio': {'manage_session': False}},
                         run_browser='--no-browser' not in sys.argv)
