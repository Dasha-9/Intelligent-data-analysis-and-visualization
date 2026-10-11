# СРС 2.1 · Обзор карточек алертов на Taipy + Plotly
# Запуск: pip install taipy plotly pandas pyarrow
#         python python/dashboard_taipy.py [путь_к_parquet]   (по умолчанию data/iavd_incidents_raw_target.parquet)
import sys, datetime as dt
import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Gui

PATH = next((a for a in sys.argv[1:] if not a.startswith('--')), 'data/iavd_incidents_raw_target.parquet')
INK, MUTED, GRAY_D, GRAY_L, RED = '#1F2328', '#6E7781', '#57606A', '#D0D7DE', '#CF222E'
ORDER = ['Критическая', 'Высокая', 'Средняя', 'Низкая']
SIGN = {'Не указан': '○ нет', 'False Positive': '● FP', 'Дубликат': '⊘ дубль', 'Решён': '✓ решён', 'Не решён': '⊖ не решён'}

df = pd.read_parquet(PATH, columns=['EntityId', 'Создан', 'Критичность', 'Вердикт', 'Количество инцидентов', 'Тип инцидента'])
df['msk'] = df['Создан'].dt.tz_convert('Europe/Moscow')
df['day'] = df['msk'].dt.date
df['Алертов'] = df['Количество инцидентов']
fmt = lambda v: f'{int(v):,}'.replace(',', ' ')

def compute(day):
    d = df[df['day'] == day]
    nv = d['Вердикт'] == 'Не указан'
    k = dict(k1=fmt(len(d)), k2=fmt(d['Алертов'].sum()),
             k3=f"{fmt(nv.sum())} ({nv.mean():.0%})" if len(d) else '0',
             k4=fmt((nv & (d['Критичность'] == 'Критическая')).sum()))
    q = d[nv].copy()
    q['r'] = q['Критичность'].map({c: i for i, c in enumerate(ORDER)}).fillna(9)
    q = q.sort_values(['r', 'Алертов'], ascending=[True, False], na_position='last')
    queue = pd.DataFrame({
        'ID': q['EntityId'].astype(str), 'Время': q['msk'].dt.strftime('%H:%M'),
        'Критичность': q['Критичность'], 'Тип инцидента': q['Тип инцидента'],
        'Алертов': q['Алертов'].map(lambda v: '—' if pd.isna(v) else fmt(v)),
        'Масштаб': q['Алертов'].map(lambda v: 'нет данных' if pd.isna(v) else '█' * min(int(-(-v // 60)), 10) + (' »' if v > 600 else '')),
        'Вердикт': q['Вердикт'].map(lambda v: SIGN.get(v, v))})
    # критичность: с вердиктом / без
    g = d.assign(nv=nv).groupby('Критичность')['nv'].agg(['sum', 'count']).reindex(ORDER).fillna(0)
    yv = list(reversed(ORDER))
    fig_c = go.Figure()
    fig_c.add_bar(y=yv, x=[g.loc[c, 'count'] - g.loc[c, 'sum'] for c in yv], orientation='h', name='С вердиктом', marker_color=GRAY_D)
    fig_c.add_bar(y=yv, x=[g.loc[c, 'sum'] for c in yv], orientation='h', name='Без вердикта', marker_color=GRAY_L,
                  text=[fmt(g.loc[c, 'count']) for c in yv], textposition='outside', cliponaxis=False)
    fig_c.update_layout(barmode='stack', title='Критичность: с вердиктом / без', plot_bgcolor='white', paper_bgcolor='white',
                        legend=dict(orientation="h", y=-0.12, traceorder="normal"), xaxis=dict(showgrid=False, visible=False),
                        yaxis=dict(tickfont=dict(color=INK)), margin=dict(l=10, r=40, t=50, b=10), height=330)
    # по часам
    h = d.groupby(d['msk'].dt.hour).size().reindex(range(24), fill_value=0)
    fig_h = go.Figure(go.Scatter(x=list(h.index), y=h.values, mode='lines+markers', line=dict(color=GRAY_D, width=2), marker=dict(size=6)))
    fig_h.update_layout(title=f'Динамика: карточки по часам суток (МСК)<br><span style="font-size:13px;color:{MUTED}">Пик: {h.max()} карточек в {h.idxmax()}:00</span>',
                        plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(dtick=1, showgrid=False, range=[-0.5, 23.5]),
                        yaxis=dict(gridcolor='#EEF0F2', rangemode='tozero'), margin=dict(l=40, r=20, t=70, b=30), height=300)
    vc = d['Вердикт'].value_counts()
    legend = '  \n'.join(f"{SIGN[v].split()[0]} {v}{' — нужен вердикт' if v == 'Не указан' else ''} ({vc.get(v, 0)})" for v in SIGN)
    return k, queue, fig_c, fig_h, legend

day = dt.date(2024, 6, 14)
_k, queue, fig_c, fig_h, legend = compute(day)
k1, k2, k3, k4 = _k['k1'], _k['k2'], _k['k3'], _k['k4']

def crit_style(state, value, index, row, column_name):
    return 'crit' if value == 'Критическая' else ''

def on_day(state):
    k, state.queue, state.fig_c, state.fig_h, state.legend = compute(pd.Timestamp(state.day).date())
    state.k1, state.k2, state.k3, state.k4 = k['k1'], k['k2'], k['k3'], k['k4']

page = """
<|layout|columns=3 1|
# Обзор карточек алертов
<|{day}|date|format=dd.MM.yyyy|on_change=on_day|label=Сутки (МСК)|>
|>

<|layout|columns=1 1 1 1|gap=12px|
<|part|class_name=kpi|
<|Карточек за сутки|text|class_name=kpi_title|>  
<|{k1}|text|class_name=kpi_val|>
|>
<|part|class_name=kpi|
<|Без вердикта|text|class_name=kpi_title|>  
<|{k3}|text|class_name=kpi_val|>
|>
<|part|class_name=kpi|
<|Критических без вердикта|text|class_name=kpi_title|>  
<|{k4}|text|class_name=kpi_red|>
|>
<|part|class_name=kpi|
<|Алертов в карточках|text|class_name=kpi_title|>  
<|{k2}|text|class_name=kpi_val|>
|>
|>

<|layout|columns=2 1|gap=16px|
<|part|
### Требует вердикта аналитика
<|{queue}|table|page_size=12|show_all=False|cell_class_name[Критичность]=crit_style|filter=False|sortable=False|width=100%|>
|>
<|part|
<|chart|figure={fig_c}|height=330px|>

**Кодировка Вердикта: форма + подпись**  
<|{legend}|text|mode=markdown|>
|>
|>

<|chart|figure={fig_h}|height=300px|>
"""

if __name__ == '__main__':
    import os
    Gui(page, css_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')).run(title='СРС 2.1 · обзор карточек', port=5056, use_reloader=False,
                                         run_browser='--no-browser' not in sys.argv, dark_mode=False)
