# Проект · дашборд «Можно ли доверять модели?» (датасет 2 – лог инференса, комплект практики 3.6) · Taipy + Plotly
# Зритель – руководство SOC, не специалисты по ML. Главная цифра – доля пропущенных реальных инцидентов.
# Запуск из корня папки: python python/app.py [путь_к_parquet]   → http://127.0.0.1:5081
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Gui
from model_data import load, confusion, weekly, where_missed, calibration, threshold_curve, SLICES, TAU0, DEFAULT_PATH

PATH = next((a for a in sys.argv[1:] if not a.startswith('--')), DEFAULT_PATH)
INK, MUTED, LINE, ACC, RED, GRAY, LIGHT = '#1F2328', '#6E7781', '#D0D7DE', '#0969DA', '#CF222E', '#8C959F', '#E6E9ED'
FONT = dict(family='Segoe UI, Arial, sans-serif', size=12, color=INK)
fmt = lambda v: f'{int(round(v)):,}'.replace(',', ' ')
pct = lambda v, d=1: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:.{d}f} %'.replace('.', ',')
num = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',')

P = load(PATH)
D0, D1 = P['day'].min().date(), P['day'].max().date()


def base(fig, h, l=10, r=20, t=8, b=30):
    fig.update_layout(height=h, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor='white', plot_bgcolor='white', font=FONT,
                      showlegend=False, hoverlabel=dict(bgcolor='white'), bargap=0.3)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def empty(h, txt='нет карточек с вердиктом аналитика за выбранный период'):
    fig = base(go.Figure(), h)
    fig.add_annotation(text=txt, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def fig_matrix(c):
    """Матрица согласия с аналитиком: строки – факт, столбцы – решение модели."""
    if not c['n']:
        return empty(300)
    fig = base(go.Figure(), 300, l=118, r=10, t=36, b=10)
    cells = [(0, 1, 'TP', 'нашла инцидент', ACC, 'white', c['TP'] / max(c['inc'], 1), 'инцидентов'),
             (1, 1, 'FN', 'пропустила инцидент', RED, 'white', c['FN'] / max(c['inc'], 1), 'инцидентов'),
             (0, 0, 'FP', 'ложная тревога', '#BFC5CC', INK, c['FP'] / max(c['n'] - c['inc'], 1), 'ложных'),
             (1, 0, 'TN', 'верно сняла шум', '#F3F5F8', INK, c['TN'] / max(c['n'] - c['inc'], 1), 'ложных')]
    for x, y, code, txt, col, tc, share, of in cells:
        fig.add_shape(type='rect', x0=x, x1=x + 0.97, y0=y, y1=y + 0.95, fillcolor=col, line_width=0)
        fig.add_annotation(x=x + 0.485, y=y + 0.64, text=f'<b>{fmt(c[code])}</b>', showarrow=False, font=dict(size=24, color=tc))
        fig.add_annotation(x=x + 0.485, y=y + 0.27, text=f'{txt}<br>{pct(share, 0)} {of}', showarrow=False, font=dict(size=12, color=tc))
    fig.update_xaxes(range=[0, 2], tickvals=[0.485, 1.485], ticktext=['Модель: «к аналитику»', 'Модель: «шум»'], side='top')
    fig.update_yaxes(range=[0, 2], tickvals=[1.48, 0.48], ticktext=['Аналитик:<br>инцидент', 'Аналитик:<br>ложное'])
    return fig


def fig_weekly(w):
    if w.empty:
        return empty(280)
    fig = base(go.Figure(), 280, r=10)
    x = w['week'].dt.strftime('%d.%m')
    fig.add_bar(x=x, y=w['found'], name='Найдено', marker_color=GRAY, hovertemplate='неделя с %{x}: найдено %{y}<extra></extra>')
    fig.add_bar(x=x, y=w['missed'], name='Пропущено', marker_color=RED, hovertemplate='неделя с %{x}: пропущено %{y}<extra></extra>')
    fig.add_scatter(x=x, y=w['inc'], mode='text', text=[f'<b>{pct(m, 0)}</b>' for m in w['miss']], textposition='top center',
                    textfont=dict(color=RED, size=12), hoverinfo='skip', showlegend=False)
    fig.update_layout(barmode='stack')
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=1.12, x=0))
    fig.update_yaxes(range=[0, w['inc'].max() * 1.18], showgrid=True, gridcolor='#EEF0F2')
    return fig


def fig_where(g):
    if g.empty:
        return empty(300)
    g = g.iloc[::-1]
    lab = [s if len(s) <= 36 else s[:35] + '…' for s in g['group']]
    fig = base(go.Figure(), 60 + 34 * len(g), l=10, r=10)
    fig.add_bar(y=lab, x=g['inc'], orientation='h', marker_color=LIGHT, customdata=g['group'],
                hovertemplate='%{customdata}: реальных инцидентов %{x}<extra></extra>')
    fig.add_bar(y=lab, x=g['missed'], orientation='h', marker_color=RED, customdata=g['group'],
                hovertemplate='%{customdata}: пропущено %{x}<extra></extra>')
    fig.add_scatter(y=lab, x=g['inc'], mode='text', textposition='middle right', hoverinfo='skip',
                    text=[f'  <b>{fmt(m)}</b> из {fmt(i)} · {pct(s, 0)}' for m, i, s in zip(g['missed'], g['inc'], g['miss'])],
                    textfont=dict(size=12, color=INK))
    fig.update_layout(barmode='overlay', bargap=0.35)
    fig.update_xaxes(visible=False, range=[0, g['inc'].max() * 1.55])
    fig.update_yaxes(ticksuffix='  ')
    return fig


def fig_calib(cb):
    fig = base(go.Figure(), 280, r=10)
    fig.add_bar(x=cb['bin'], y=cb['model'], marker_color=LIGHT, name='обещала модель',
                hovertemplate='оценка %{x}: модель в среднем %{y:.0%}<extra></extra>')
    fig.add_bar(x=cb['bin'], y=cb['fact'], marker_color=ACC, name='оказалось инцидентами',
                hovertemplate='оценка %{x}: инцидентов %{y:.0%}<extra></extra>')
    fig.add_scatter(x=cb['bin'], y=cb[['model', 'fact']].max(axis=1), mode='text', text=[f'{fmt(n)} карт.' for n in cb['n']],
                    textposition='top center', textfont=dict(size=10, color=MUTED), hoverinfo='skip', showlegend=False)
    fig.update_layout(barmode='group', showlegend=True, legend=dict(orientation='h', y=1.12, x=0, font=dict(size=11)))
    fig.update_yaxes(tickformat='.0%', range=[0, 1.05], showgrid=True, gridcolor='#EEF0F2')
    fig.update_xaxes(title=dict(text='оценка модели («вероятность инцидента»)', font=dict(size=11, color=MUTED)))
    return fig


def fig_tau(tc, tau):
    fig = base(go.Figure(), 280, r=10)
    fig.add_hrect(y0=0.95, y1=1.0, fillcolor='#EEF4FB', line_width=0)
    fig.add_scatter(x=tc['tau'], y=tc['recall'], line=dict(color=ACC, width=3), name='найдено инцидентов',
                    hovertemplate='τ %{x:.2f}: найдено %{y:.1%}<extra></extra>')
    fig.add_scatter(x=tc['tau'], y=tc['flag'], line=dict(color=GRAY, width=2, dash='dot'), name='доля карточек аналитику',
                    hovertemplate='τ %{x:.2f}: аналитику %{y:.1%}<extra></extra>')
    fig.add_vline(x=tau, line=dict(color=INK, width=1.5, dash='dot'))
    r0 = tc.iloc[0]
    fig.add_annotation(x=r0['tau'], y=r0['recall'], text=f'даже при τ = {num(r0["tau"])} найдено только {pct(r0["recall"], 0)}',
                       showarrow=True, arrowhead=0, ax=60, ay=40, xanchor='left', font=dict(size=11, color=INK))
    fig.add_annotation(x=0.6, y=0.975, text='цель «найдено ≥ 95 %»', showarrow=False, xanchor='right', font=dict(size=11, color=ACC))
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=1.12, x=0, font=dict(size=11)))
    fig.update_yaxes(tickformat='.0%', range=[0, 1.0], showgrid=True, gridcolor='#EEF0F2')
    fig.update_xaxes(title=dict(text='порог τ', font=dict(size=11, color=MUTED)), range=[0.02, 0.6])
    return fig


def build(s):
    d0, d1 = [pd.Timestamp(x) for x in s['dates']]
    period = P[(P['day'] >= d0) & (P['day'] <= d1)]
    p = period[period[SLICES[s['slice']]] == s['group']] if s['group'] else period
    v = p[p['labeled']]
    tau = float(s['tau'])
    c = confusion(v, tau)
    return dict(c=c, n_all=len(p), n_lab=len(v), lat=p['latency_s'].median() if len(p) else np.nan,
                f_matrix=fig_matrix(c), f_week=fig_weekly(weekly(v, tau)),
                f_where=fig_where(where_missed(period[period['labeled']], tau, SLICES[s['slice']])),
                f_calib=fig_calib(calibration(v)) if len(v) else empty(280),
                f_tau=fig_tau(threshold_curve(v), tau) if c['inc'] else empty(280))


# --- состояние страницы ---
dates = [D0, D1]
tau = TAU0
slice_lov = list(SLICES)
slice_ = 'Вендор из SIEM'
group = ''
pc = {'displayModeBar': False}


def texts(b, s):
    c = b['c']
    lab = f'Прогнозов: {fmt(b["n_all"])} · с вердиктом: {fmt(b["n_lab"])} · порог τ = {num(s["tau"])}'
    return dict(k_miss=f'{fmt(c["FN"])} из {fmt(c["inc"])}', k_miss_p=pct(c['miss']), k_rec=pct(c['recall']), k_prec=pct(c['prec']),
                k_flag=pct(c['flag']), k_agree=pct(c['agree']), k_lat='—' if np.isnan(b['lat']) else f'{num(b["lat"])} с', k_note=lab,
                chips=(f'Выбрано – {s["slice"]}: {s["group"]}' if s['group'] else
                       'Клик по группе применяет фильтр.'))


_s = dict(dates=dates, tau=tau, slice=slice_, group=group)
_b = build(_s)
_t = texts(_b, _s)
KEYS = ('k_miss', 'k_miss_p', 'k_rec', 'k_prec', 'k_flag', 'k_agree', 'k_lat', 'k_note', 'chips')
k_miss, k_miss_p, k_rec, k_prec, k_flag, k_agree, k_lat, k_note, chips = (_t[k] for k in KEYS)
FIGS = ('f_matrix', 'f_week', 'f_where', 'f_calib', 'f_tau')
f_matrix, f_week, f_where, f_calib, f_tau = (_b[k] for k in FIGS)


def cur(state):
    return dict(dates=state.dates, tau=state.tau, slice=state.slice_, group=state.group)


def refresh(state):
    s = cur(state)
    b = build(s)
    for k, v in texts(b, s).items():
        setattr(state, k, v)
    for k in FIGS:
        setattr(state, k, b[k])


def on_change(state, var, val):
    if var in ('dates', 'tau'):
        refresh(state)
    elif var == 'slice_':
        state.group = ''
        refresh(state)


def last(state, days):
    state.dates = [(pd.Timestamp(D1) - pd.Timedelta(days=days - 1)).date(), D1]
    refresh(state)


def last7(state):
    last(state, 7)


def last30(state):
    last(state, 30)


def all_period(state):
    state.dates = [D0, D1]
    refresh(state)


def click_where(state, id, p):
    y = p.get('y') or ''
    full = next((g for g in P[SLICES[state.slice_]].unique() if g == y or (y.endswith('…') and g.startswith(y[:-1]))), '')
    state.group = '' if state.group == full else full
    refresh(state)


def reset(state):
    state.dates, state.tau, state.group = [D0, D1], TAU0, ''
    refresh(state)


page = """
# Можно ли доверять модели? · лог инференса 04.09–29.11.2025

<|layout|columns=4 3 4 1|gap=14px|class_name=frow|
<|part|
<|{dates}|date_range|format=dd.MM.yyyy|label_start=Прогнозы с|label_end=по|>
|>
<|part|
<|Весь период|button|on_action=all_period|> <|30 дней|button|on_action=last30|> <|7 дней|button|on_action=last7|>
|>
<|part|
<|text|value=Порог τ · оценка ≥ τ → к аналитику|class_name=kt|>
<|{tau}|slider|min=0.02|max=0.6|step=0.01|width=100%|labels=True|>
|>
<|part|
<|Сбросить|button|on_action=reset|>
|>
|>

<|{chips}|text|class_name=chips|>

<|layout|columns=2 1 1 1 1 1|gap=12px|
<|part|class_name=kpi kpi_main|
<|text|value=Пропущено реальных инцидентов|class_name=kt|>
<|{k_miss}|text|class_name=kv red|> <|{k_miss_p}|text|class_name=kv2 red|>
|>
<|part|class_name=kpi|
<|text|value=Найдено инцидентов|class_name=kt|>
<|{k_rec}|text|class_name=kv blue|>
|>
<|part|class_name=kpi|
<|text|value=Тревог по делу|class_name=kt|>
<|{k_prec}|text|class_name=kv|>
|>
<|part|class_name=kpi|
<|text|value=Ушло аналитикам|class_name=kt|>
<|{k_flag}|text|class_name=kv grey|>
|>
<|part|class_name=kpi|
<|text|value=Совпало с аналитиком|class_name=kt|>
<|{k_agree}|text|class_name=kv grey|>
|>
<|part|class_name=kpi|
<|text|value=Ответ модели, медиана|class_name=kt|>
<|{k_lat}|text|class_name=kv grey|>
|>
|>
<|{k_note}|text|class_name=note|>

<|layout|columns=5 7|gap=20px|
<|part|
### Матрица ошибок
<|chart|figure={f_matrix}|plot_config={pc}|height=300px|>
|>
<|part|
### Пропуски по неделям
<|chart|figure={f_week}|plot_config={pc}|height=280px|>
|>
|>

<|layout|columns=7 5|gap=20px|
<|part|
### Пропуски по группам
<|{slice_}|toggle|lov={slice_lov}|>
<|chart|figure={f_where}|plot_config={pc}|on_click=click_where|height=340px|>
|>
<|part|
### Калибровка оценки модели
<|chart|figure={f_calib}|plot_config={pc}|height=290px|>
|>
|>

### Полнота и нагрузка при разных порогах
<|chart|figure={f_tau}|plot_config={pc}|height=280px|>
"""

if __name__ == '__main__':
    Gui(page).run(   # стили – app.css рядом со скриптом (Taipy подключает его автоматически)
        title='Можно ли доверять модели?', port=5081, use_reloader=False, dark_mode=False,
        server_config={'socketio': {'manage_session': False}},
        run_browser='--no-browser' not in sys.argv)
