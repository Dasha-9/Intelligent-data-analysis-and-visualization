# СРС 3.1 · «ручка чувствительности»: порог τ и β в Taipy + Plotly
# Запуск из корня папки: python python/threshold_taipy.py  →  http://127.0.0.1:5059
# Ползунок порога и переключатель β пересчитывают матрицу ошибок, полноту, точность, Fβ
# и долю карточек аналитику; строка статуса показывает, выполнено ли требование «полнота ≥ 95 %».
import os, sys
import numpy as np
import plotly.graph_objects as go
from taipy.gui import Gui

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load, point, fb, sweep, work_tau, MIN_RECALL

INK, MUTED, GRAY, ACC, RED = '#1F2328', '#6E7781', '#C3C9D0', '#0969DA', '#CF222E'
fmt = lambda v: f'{int(round(v)):,}'.replace(',', ' ')
pct = lambda v: f'{v * 100:.1f} %'.replace('.', ',')

te, info = load()
Y, S = te['y'].values, te['score'].values
SW = sweep(Y, S)
SW = SW[SW['tau'] <= 0.72]
T0 = work_tau(SW)

tau = T0
beta = '2'
beta_lov = ['0.5', '1', '2']
m_recall = m_prec = m_fb = m_flag = status = ''
fb_label = ''
fig_matrix = go.Figure()
fig_curve = go.Figure()


def matrix_fig(r):
    z = [[r['TP'], r['FN']], [r['FP'], r['TN']]]
    txt = [[f"TP<br><b>{fmt(r['TP'])}</b>", f"FN<br><b>{fmt(r['FN'])}</b>"],
           [f"FP<br><b>{fmt(r['FP'])}</b>", f"TN<br><b>{fmt(r['TN'])}</b>"]]
    colors = [[ACC, RED], ['#BFC5CC', '#F3F5F8']]
    f = go.Figure()
    for i in range(2):
        for j in range(2):
            f.add_shape(type='rect', x0=j, x1=j + 0.98, y0=1 - i, y1=1 - i + 0.98, fillcolor=colors[i][j], line_width=0)
            f.add_annotation(x=j + 0.49, y=1 - i + 0.49, text=txt[i][j], showarrow=False,
                             font=dict(size=20, color='white' if (i, j) in ((0, 0), (0, 1)) else INK))
    f.update_xaxes(range=[0, 2], tickvals=[0.49, 1.49], ticktext=['Прогноз: аналитику', 'Прогноз: шум'], side='top',
                   showgrid=False, zeroline=False)
    f.update_yaxes(range=[0, 2], tickvals=[1.49, 0.49], ticktext=['Факт: инцидент', 'Факт: FP'], showgrid=False, zeroline=False)
    f.update_layout(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white', height=330,
                    font=dict(color=MUTED, size=13))
    return f


def curve_fig(t, b):
    f = go.Figure()
    fbv = [fb(p, r, b) for p, r in zip(SW['P'], SW['R'])]
    f.add_hrect(y0=MIN_RECALL, y1=1.02, fillcolor='#EEF4FB', line_width=0)
    for y, c, w, d, name in ((SW['R'], ACC, 3, 'solid', 'полнота'), (SW['P'], INK, 2, 'solid', 'точность'),
                             (fbv, '#8C959F', 2, 'dash', f'F{b:g}'), (SW['flag'], '#A8B0B8', 2, 'dot', 'доля аналитику')):
        f.add_trace(go.Scatter(x=SW['tau'], y=y, name=name, line=dict(color=c, width=w, dash=d, shape='vh'),
                               hovertemplate='τ %{x:.2f} · %{y:.1%}<extra>' + name + '</extra>'))
    f.add_vline(x=t, line=dict(color=ACC, dash='dot', width=2))
    f.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor='white', paper_bgcolor='white',
                    legend=dict(orientation='h', y=1.12, x=0), font=dict(color=MUTED, size=12),
                    xaxis=dict(title='Порог τ', range=[0, 0.7], gridcolor='#EEF0F2'),
                    yaxis=dict(tickformat='.0%', range=[0, 1.02], gridcolor='#EEF0F2'))
    return f


def refresh(state):
    b = float(state.beta)
    r = point(Y, S, float(state.tau))
    state.m_recall, state.m_prec, state.m_flag = pct(r['R']), pct(r['P']), pct(r['flag'])
    state.fb_label = f'F{b:g}'.replace('.', ',')
    state.m_fb = pct(fb(r['P'], r['R'], b))
    ok = r['R'] >= MIN_RECALL
    state.status = (f"✓ Требование выполнено: полнота {pct(r['R'])} ≥ 95 %. Пропущено {fmt(r['FN'])} инцидентов из {fmt(info['pos'])}, "
                    f"с очереди снято {fmt(r['TN'] + r['FN'])} карточек." if ok else
                    f"✗ Полнота {pct(r['R'])} ниже 95 %: пропущено {fmt(r['FN'])} инцидентов из {fmt(info['pos'])}.")
    state.fig_matrix = matrix_fig(r)
    state.fig_curve = curve_fig(float(state.tau), b)


def on_change(state, var, val):
    if var in ('tau', 'beta'):
        refresh(state)


def on_init(state):
    refresh(state)


def to_work(state):
    state.tau = T0
    refresh(state)


page = f"""
# Порог триажа: полнота против точности

<|text|value=Проверочный период 04.09–20.11.2025 · {fmt(info['test'])} карточек с вердиктом, из них {fmt(info['pos'])} реальных инцидентов · оценка – доля инцидентов у правила SIEM|class_name=q|>

<|layout|columns=6 2 2|gap=20px|class_name=ctl|
<|part|
<|text|value=Порог τ: оценка ≥ τ → карточка уходит аналитику|class_name=kt|>
<|{{tau}}|slider|min=0|max=0.7|step=0.01|width=100%|labels=True|>
|>
<|part|
<|text|value=β в F-мере|class_name=kt|>
<|{{beta}}|toggle|lov={{beta_lov}}|>
|>
<|part|
<|Рабочий порог {str(T0).replace('.', ',')}|button|on_action=to_work|>
|>
|>

<|layout|columns=1 1 1 1|gap=12px|
<|part|class_name=kpi|
<|text|value=Полнота (Recall)|class_name=kt|>
<|{{m_recall}}|text|class_name=kv blue|>
|>
<|part|class_name=kpi|
<|text|value=Точность (Precision)|class_name=kt|>
<|{{m_prec}}|text|class_name=kv|>
|>
<|part|class_name=kpi|
<|{{fb_label}}|text|class_name=kt|>
<|{{m_fb}}|text|class_name=kv grey|>
|>
<|part|class_name=kpi|
<|text|value=Доля карточек аналитику|class_name=kt|>
<|{{m_flag}}|text|class_name=kv grey|>
|>
|>

<|{{status}}|text|class_name=status|>

<|layout|columns=5 7|gap=20px|
<|part|
### Матрица ошибок при выбранном пороге
<|chart|figure={{fig_matrix}}|plot_config={{{{'displayModeBar': False}}}}|>
|>
<|part|
### Метрики по всем порогам · синяя полоса – зона «полнота ≥ 95 %»
<|chart|figure={{fig_curve}}|plot_config={{{{'displayModeBar': False}}}}|>
|>
|>
"""

if __name__ == '__main__':
    Gui(page).run(   # стили – из threshold_taipy.css рядом со скриптом (Taipy подхватывает его сам)
        title='СРС 3.1 · порог', port=5059, dark_mode=False, use_reloader=False)
