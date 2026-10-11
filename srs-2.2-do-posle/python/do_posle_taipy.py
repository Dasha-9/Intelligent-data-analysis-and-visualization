# СРС 2.2 · «до → после» на Taipy + Plotly
# Запуск: pip install taipy plotly pandas pyarrow
#         python python/do_posle_taipy.py [путь_к_parquet]   (по умолчанию data/iavd_incidents_raw_target.parquet)
# Откроется браузер: страницы /do и /posle
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from taipy.gui import Gui

PATH = next((a for a in sys.argv[1:] if not a.startswith('--')), 'data/iavd_incidents_raw_target.parquet')
DAY, TOP = '2024-06-14', 11          # 10-е и 11-е места делят два правила (по 36) — берём оба

df = pd.read_parquet(PATH, columns=['EntityId', 'Создан', 'Правило корреляции SIEM'])
d = df[df['Создан'].dt.tz_convert('Europe/Moscow').dt.date.astype(str) == DAY]
N = len(d)
vc = d['Правило корреляции SIEM'].dropna().value_counts()
top = vc.head(TOP)
fmt = lambda v: f'{v:,}'.replace(',', ' ')

# ---------- ДО: перегруженное оформление ----------
fig_do = go.Figure()
palette = px.colors.qualitative.Alphabet
for i, (rule, v) in enumerate(top.items()):
    fig_do.add_bar(x=[rule], y=[v], name=rule, marker_color=palette[i % len(palette)],
                   marker_line_color='black', marker_line_width=1.5)
fig_do.update_layout(
    title=dict(text='Алерты по правилам SIEM за сутки', font=dict(size=20)),
    paper_bgcolor='#9fd3cf', plot_bgcolor='#c9ece9',
    showlegend=True, legend=dict(title='Правило корреляции SIEM', bordercolor='black', borderwidth=1),
    xaxis=dict(title='Правило корреляции SIEM', tickangle=-45, showline=True, linewidth=3, linecolor='black', mirror=True),
    yaxis=dict(title='Количество карточек', dtick=250, gridcolor='#4a4a4a', gridwidth=1.2,
               minor=dict(dtick=50, showgrid=True, gridcolor='#7a7a7a', griddash='dot'),
               showline=True, linewidth=3, linecolor='black', mirror=True),
    bargap=0.25, margin=dict(l=70, r=30, t=70, b=230), height=760, width=1500,
)

# ---------- ПОСЛЕ ----------
INK, MUTED, GRAY, ACC = '#1f2328', '#6e7781', '#c3c9d0', '#0969da'
t = top.iloc[::-1]
colors = [ACC if v == top.max() else GRAY for v in t.values]
labels = [f'{fmt(v)} · {v/N:.0%} всех' if v == top.max() else fmt(v) for v in t.values]
fig_posle = go.Figure(go.Bar(
    x=t.values, y=t.index, orientation='h', marker_color=colors,
    text=labels, textposition=['inside' if v == top.max() else 'outside' for v in t.values],
    insidetextanchor='end', textfont=dict(size=14), cliponaxis=False,
    hovertemplate='%{y}: %{x} карточек<extra></extra>'))
fig_posle.update_layout(
    title=dict(text=f'<b>14.06.2024: {top.max()/N:.0%} карточек дало одно правило — сканирование портов</b>'
                    f'<br><span style="font-size:14px;color:{MUTED}">Карточки по правилу корреляции SIEM, шт. · '
                    f'топ-{TOP} из {len(vc)} правил · сутки по МСК, всего {fmt(N)} карточек</span>',
               font=dict(size=20, color=INK), x=0.01),
    paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
    xaxis=dict(visible=False, range=[0, top.max() * 1.02]),
    yaxis=dict(showgrid=False, ticks='', tickfont=dict(size=13, color=INK), showline=True, linecolor=GRAY),
    margin=dict(l=20, r=40, t=100, b=20), height=760, width=1500, bargap=0.3,
)

nav = [("/do", "До"), ("/posle", "После")]
page_do = "## Версия «до»\n\n<|chart|figure={fig_do}|height=780px|>"
page_posle = "## Версия «после»\n\n<|chart|figure={fig_posle}|height=780px|>"

if __name__ == '__main__':
    print(N, len(vc)); print(top.to_string())
    Gui(pages={'/': '<|navbar|lov={nav}|>','do': page_do, 'posle': page_posle}).run(
        title='СРС 2.2 · до → после', run_browser='--no-browser' not in sys.argv, port=5055, use_reloader=False, dark_mode=False)
