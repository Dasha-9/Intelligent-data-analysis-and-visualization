# Проект · дашборд дежурной смены SOC (датасет 1 – карточки) · Taipy + Plotly
# Запуск из корня папки: pip install -r requirements.txt
#                        python python/app.py [путь_к_parquet]        → http://127.0.0.1:5080
import os, sys, datetime as dt
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Gui, navigate
from soc_data import load, filtered, kpis, queue_table, card, SEV, SEV_COLOR, AGE, DEFAULT_PATH

PATH = next((a for a in sys.argv[1:] if not a.startswith('--')), DEFAULT_PATH)
INK, MUTED, LINE, ACC, RED = '#1F2328', '#6E7781', '#D0D7DE', '#0969DA', '#CF222E'
FONT = dict(family='Segoe UI, Arial, sans-serif', size=12, color=INK)
GAP = dict(color='white', width=1.5)          # зазор между частями столбца
fmt = lambda v: f'{int(v):,}'.replace(',', ' ')

DF, SNAP = load(PATH)
SNAP_DAY = SNAP.normalize()
OLDEST_OPEN = DF.loc[DF['open'], 'day'].min()
TYPES = ['Все'] + DF['type'].value_counts().index.tolist()


def base(fig, h, left=10):
    fig.update_layout(height=h, margin=dict(l=left, r=60, t=8, b=28), paper_bgcolor='white', plot_bgcolor='white',
                      font=FONT, showlegend=False, bargap=0.25, barmode='stack', hoverlabel=dict(bgcolor='white'))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def empty(h, txt='нет карточек при выбранных фильтрах'):
    fig = base(go.Figure(), h)
    fig.add_annotation(text=txt, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def stacked_h(data, cat_col, cats, h, total_side=True, extra=None):
    """Горизонтальные столбцы по категориям, сложенные по критичности (красный – только критическая)."""
    fig = go.Figure()
    for s in SEV[::-1]:                       # красный сегмент – в конце столбца
        cnt = data[data['sev'] == s].groupby(cat_col).size().reindex(cats, fill_value=0)
        fig.add_bar(y=cats, x=cnt.values, orientation='h', name=s, marker=dict(color=SEV_COLOR[s], line=GAP),
                    hovertemplate=f'%{{y}} · {s}: %{{x}}<extra></extra>')
    tot = data.groupby(cat_col).size().reindex(cats, fill_value=0)
    if tot.sum() == 0:
        return empty(h)
    h = min(h, 50 + 40 * len(cats))
    lab = [f'<b>{fmt(v)}</b>' + (extra.get(c, '') if extra else '') for c, v in zip(cats, tot.values)]
    fig.add_scatter(y=cats, x=tot.values, mode='text', text=lab, textposition='middle right', hoverinfo='skip',
                    textfont=dict(size=12, color=INK))
    base(fig, h)
    fig.update_yaxes(autorange='reversed', ticksuffix='  ')
    fig.update_xaxes(visible=False, range=[0, max(tot.max(), 1) * (1.7 if extra else 1.2)])
    return fig


def build(s):
    """s – словарь фильтров; возвращает всё, что рисуется на странице."""
    start, end = [pd.Timestamp(d).tz_localize('Europe/Moscow') for d in s['dates']]
    f = filtered(DF, start, end, s['sev'], s['typ'], s['target'], s['age'], s['day'], s['only_sla'])
    o = f[f['open']]
    k = kpis(o)
    out = {'k': k}
    # вне периода: открытые карточки старше начала периода (с остальными фильтрами)
    f_all = filtered(DF, OLDEST_OPEN, end, s['sev'], s['typ'], s['target'], s['age'], s['day'], s['only_sla'])
    out['outside'] = int(f_all['open'].sum() - k['open'])
    out['queue'] = queue_table(o)
    # 1. возраст очереди
    out['f_age'] = stacked_h(o, 'age', AGE, 210)
    # 2. повторяющиеся цели – кандидаты на эскалацию
    t = o.explode('targets').dropna(subset=['targets'])
    top = t['targets'].value_counts()
    top = top[top > 1].head(8).index.tolist()
    out['f_tgt'] = stacked_h(t[t['targets'].isin(top)], 'targets', top or ['—'], 300)
    # 3. приток по дням периода (все карточки, не только открытые)
    days = pd.date_range(start, end, freq='D')
    fig = go.Figure()
    for sv in SEV[::-1]:
        cnt = f[f['sev'] == sv].groupby('day').size().reindex(days, fill_value=0)
        fig.add_bar(x=days, y=cnt.values, name=sv, marker=dict(color=SEV_COLOR[sv], line=GAP),
                    hovertemplate=f'%{{x|%d.%m}} · {sv}: %{{y}}<extra></extra>')
    base(fig, 250, left=40)
    fig.update_layout(bargap=0.15)
    fig.update_yaxes(showgrid=True, gridcolor='#EEF0F2', nticks=4, tickformat='d')
    fig.update_xaxes(tickformat='%d.%m', dtick=86400000 * (1 if len(days) <= 14 else 7) if len(days) <= 60 else 'M1')
    out['f_flow'] = fig
    last24 = f[f['msk'] > SNAP - pd.Timedelta('24h')]
    out['n24'] = len(last24)
    # 4. типы в работе: открытые + сколько из них без вердикта
    tt = o['type'].value_counts().head(7).index.tolist()
    nov = o[o['no_verdict']].groupby('type').size()
    out['f_types'] = stacked_h(o, 'type', tt or ['—'], 300, extra={c: f'  ○ {int(nov.get(c, 0))}' for c in tt})
    # 5. 24 ч к среднесуточному уровню (сортировка по приросту)
    n_prev = max(int((SNAP - pd.Timedelta(hours=24) - start) / pd.Timedelta(days=1)), 1)
    prev = f[f['msk'] <= SNAP - pd.Timedelta('24h')].groupby('type').size() / n_prev
    cur = last24.groupby('type').size()
    cmp_ = pd.DataFrame({'cur': cur, 'avg': prev}).fillna(0)
    cmp_['d'] = cmp_['cur'] - cmp_['avg']
    cmp_ = cmp_.sort_values('d', ascending=False).head(6)
    if not len(cmp_):
        out['f_24'] = empty(300)
    fig = go.Figure()
    fig.add_bar(y=cmp_.index, x=cmp_['avg'], orientation='h', name='среднее в сутки', marker_color='#D0D7DE',
                hovertemplate='%{y}: среднее %{x:.1f}<extra></extra>')
    fig.add_bar(y=cmp_.index, x=cmp_['cur'], orientation='h', name='последние 24 ч', marker_color=ACC,
                text=[f'{int(c)} · {"+" if d >= 0 else "−"}{abs(d):.0f}' for c, d in zip(cmp_['cur'], cmp_['d'])],
                textposition='outside', cliponaxis=False, hovertemplate='%{y}: %{x} за 24 ч<extra></extra>')
    base(fig, 300)
    fig.update_layout(barmode='group', bargap=0.35, bargroupgap=0.05)
    fig.update_yaxes(autorange='reversed', ticksuffix='  ')
    fig.update_xaxes(visible=False, range=[0, max(cmp_[['cur', 'avg']].max().max(), 1) * 1.3])
    if len(cmp_):
        fig.update_layout(height=min(300, 50 + 45 * len(cmp_)))
        out['f_24'] = fig
    # 6. вероятный шум в очереди (гипотеза 1)
    nz = o[o['noisy']].groupby('rule').size().sort_values(ascending=False).head(6)
    if not len(nz):
        out['f_noise'] = empty(300, 'в очереди нет карточек «шумных» правил')
    fig = go.Figure(go.Bar(y=[r if len(r) < 38 else r[:36] + '…' for r in nz.index] or ['—'], x=nz.values if len(nz) else [0],
                           orientation='h', marker_color='#8C959F', text=[fmt(v) for v in nz.values], textposition='outside',
                           cliponaxis=False, hovertemplate='%{y}: %{x}<extra></extra>'))
    base(fig, 300)
    fig.update_yaxes(autorange='reversed', ticksuffix='  ')
    fig.update_xaxes(visible=False, range=[0, max(nz.max() if len(nz) else 1, 1) * 1.3])
    if len(nz):
        fig.update_layout(height=min(300, 50 + 40 * len(nz)))
        out['f_noise'] = fig
    # 7. поток по часам суток и просрочка SLA (гипотеза 2)
    hh = f.groupby('hour').agg(n=('EntityId', 'size'), over=('sla_over', 'mean')).reindex(range(24))
    ndays = max(len(days), 1)
    avg_over = f['sla_over'].mean() if len(f) else 0
    col = [ACC if (v or 0) > avg_over * 1.2 else '#C3C9D0' for v in hh['over']]
    fig = go.Figure(go.Bar(x=hh.index, y=hh['n'].fillna(0) / ndays, marker_color=col,
                           customdata=(hh['over'].fillna(0) * 100).round(0),
                           hovertemplate='%{x}:00 · %{y:.1f} карточек в день · просрочено %{customdata:.0f} %<extra></extra>'))
    base(fig, 300, left=40)
    fig.update_xaxes(dtick=3, ticksuffix=':00')
    fig.update_yaxes(showgrid=True, gridcolor='#EEF0F2', nticks=4)
    out['f_hour'] = fig
    return out


# ---------- состояние ----------
def default_filters():
    return dict(dates=[(SNAP_DAY - pd.Timedelta(days=29)).date(), SNAP_DAY.date()], sev=[], typ='Все',
                target=None, age=None, day=None, only_sla=False)


F0 = default_filters()
dates, sev, typ = F0['dates'], F0['sev'], F0['typ']
target, age, day, only_sla = '', '', '', False
_b = build(F0)
k_open, k_nov, k_noisy = (fmt(_b['k'][x]) for x in ('open', 'nov', 'noisy'))
k_crit, k_high, k_old, k_sla = (fmt(_b['k'][x]) + ' ›' for x in ('crit', 'high', 'old', 'sla'))
outside_txt = (f'Открыто вне периода: {_b["outside"]} · «Вся очередь»' if _b['outside'] else '')
queue = _b['queue']
f_age, f_tgt, f_flow, f_types, f_24, f_noise, f_hour = (_b[x] for x in ('f_age', 'f_tgt', 'f_flow', 'f_types', 'f_24', 'f_noise', 'f_hour'))
n24 = fmt(_b['n24'])
chips = 'Фильтры не выбраны'
rb = False
pc = {'displayModeBar': False}
sev_lov, types_lov = SEV, TYPES
show_card = False
c_title = c_body = ''
c_link = ''


def cur(state):
    return dict(dates=state.dates, sev=list(state.sev), typ=state.typ, target=state.target or None,
                age=state.age or None, day=pd.Timestamp(state.day).tz_localize('Europe/Moscow') if state.day else None,
                only_sla=state.only_sla)


def refresh(state):
    s = cur(state)
    b = build(s)
    k = b['k']
    state.k_open, state.k_nov, state.k_noisy = fmt(k['open']), fmt(k['nov']), fmt(k['noisy'])
    state.k_crit, state.k_high, state.k_old, state.k_sla = (fmt(k[x]) + ' ›' for x in ('crit', 'high', 'old', 'sla'))
    state.outside_txt = (f'Открыто вне периода: {b["outside"]} · «Вся очередь»'
                         if b['outside'] else '')
    state.queue = b['queue']
    state.rb = not state.rb          # пересобрать таблицу, чтобы не осталась старая страница
    state.f_age, state.f_tgt, state.f_flow, state.f_types = b['f_age'], b['f_tgt'], b['f_flow'], b['f_types']
    state.f_24, state.f_noise, state.f_hour, state.n24 = b['f_24'], b['f_noise'], b['f_hour'], fmt(b['n24'])
    parts = []
    if s['sev']: parts.append('критичность: ' + ', '.join(s['sev']))
    if s['typ'] != 'Все': parts.append('тип: ' + s['typ'])
    if s['target']: parts.append('цель: ' + s['target'])
    if s['age']: parts.append('возраст: ' + s['age'])
    if s['day'] is not None: parts.append('день: ' + s['day'].strftime('%d.%m.%Y'))
    if s['only_sla']: parts.append('только просроченные по SLA')
    state.chips = 'Выбрано – ' + ' · '.join(parts) if parts else 'Фильтры не выбраны'


def on_change(state, var, val):
    if var in ('dates', 'sev', 'typ'):
        refresh(state)


# клики по KPI
def kpi_crit(state):
    state.sev = [] if list(state.sev) == ['Критическая'] else ['Критическая']; refresh(state)


def kpi_high(state):
    state.sev = [] if sorted(state.sev) == sorted(['Критическая', 'Высокая']) else ['Критическая', 'Высокая']; refresh(state)


def kpi_old(state):
    state.age = '' if state.age == '> 3 сут' else '> 3 сут'; refresh(state)


def kpi_sla(state):
    state.only_sla = not state.only_sla; refresh(state)


def all_queue(state):
    state.dates = [OLDEST_OPEN.date(), SNAP_DAY.date()]; refresh(state)


def reset(state):
    d = default_filters()
    state.dates, state.sev, state.typ = d['dates'], [], 'Все'
    state.target, state.age, state.day, state.only_sla = '', '', '', False
    refresh(state)


# клики по графикам – перекрёстная фильтрация (повторный клик снимает выбор)
def click_age(state, id, p):
    v = p.get('y'); state.age = '' if state.age == v else (v or ''); refresh(state)


def click_tgt(state, id, p):
    v = p.get('y'); state.target = '' if state.target == v or v == '—' else (v or ''); refresh(state)


def click_type(state, id, p):
    v = p.get('y'); state.typ = 'Все' if state.typ == v or v not in TYPES else v; refresh(state)


def click_day(state, id, p):
    v = str(p.get('x', ''))[:10]
    state.day = '' if state.day == v or not v else v; refresh(state)


# карточка инцидента
def open_card(state, var, p):
    eid = state.queue.iloc[p['index']]['№']
    c = card(DF, eid)
    state.c_title = f"Инцидент № {c['id']} · {c['sev']}"
    state.c_body = '\n\n'.join([
        f"### {state.c_title}",
        f"**Создан:** {c['created']} · **Фаза:** {c['phase']} · **SLA взятия:** {c['sla']}",
        f"**Тип:** {c['type']} · **Источник:** {c['source']}",
        f"**Правило:** {c['rule']}  \n{c['noisy']}",
        f"**Цели:** {c['targets']} · **Атакующий:** {c['attacker']}",
        f"**Ответственный:** {c['owner']} · **Вердикт:** {c['verdict']}",
        f"**Описание:** {c['descr']}",
        f"**Ссылка в источнике (NAD):** {c['link'] or 'в карточке не заполнена'}"])
    state.c_link = c['link']
    state.show_card = True


def card_action(state, id, p):
    if p.get('args') and p['args'][0] == 1 and state.c_link:
        navigate(state, state.c_link, tab='_blank')
    state.show_card = False


def sev_cell(state, value, index, row, column_name):
    return {'Критическая': 'crit', 'Высокая': 'high'}.get(value, '')


def sla_cell(state, value, index, row, column_name):
    return 'red' if value else ''


SNAP_TXT = SNAP.strftime('%d.%m.%Y %H:%M МСК')

page = f"""
# Дежурная смена SOC · снимок {SNAP_TXT}
<|part|class_name=legend|
<|text|value=■|class_name=c0|> Критическая &nbsp; <|text|value=■|class_name=c1|> Высокая &nbsp; <|text|value=■|class_name=c2|> Средняя &nbsp; <|text|value=■|class_name=c3|> Низкая &nbsp;·&nbsp; **○** – без вердикта &nbsp;·&nbsp; клик применяет фильтр · повторный клик снимает
|>

<|layout|columns=5 3 4 2 2|gap=10px|class_name=frow|
<|part|
<|{{dates}}|date_range|format=dd.MM.yyyy|label_start=Создано с|label_end=по|>
|>
<|part|
<|{{sev}}|selector|lov={{sev_lov}}|multiple|dropdown|label=Критичность|width=100%|>
|>
<|part|
<|{{typ}}|selector|lov={{types_lov}}|dropdown|filter|label=Тип инцидента|width=100%|>
|>
<|part|
<|Вся очередь|button|on_action=all_queue|>
|>
<|part|
<|Сбросить|button|on_action=reset|>
|>
|>

<|{{chips}}|text|class_name=chips|> <|{{outside_txt}}|text|class_name=note|>

<|layout|columns=1 1 1 1 1 1|gap=12px|
<|part|class_name=kpi|
<|text|value=Открыто в очереди|class_name=kt|>
<|{{k_open}}|text|class_name=kv|>
|>
<|part|class_name=kpi kpi_red|
<|text|value=Критических|class_name=kt|>
<|button|label={{k_crit}}|on_action=kpi_crit|class_name=kb red|>
|>
<|part|class_name=kpi|
<|text|value=Высокая и критическая|class_name=kt|>
<|button|label={{k_high}}|on_action=kpi_high|class_name=kb|>
|>
<|part|class_name=kpi|
<|text|value=Старше 3 суток|class_name=kt|>
<|button|label={{k_old}}|on_action=kpi_old|class_name=kb|>
|>
<|part|class_name=kpi kpi_red|
<|text|value=Просрочен SLA взятия|class_name=kt|>
<|button|label={{k_sla}}|on_action=kpi_sla|class_name=kb red|>
|>
<|part|class_name=kpi|
<|text|value=Вероятный шум (правила ≥ 95 % FP)|class_name=kt|>
<|{{k_noisy}}|text|class_name=kv grey|>
|>
|>

<|layout|columns=7 5|gap=20px|
<|part|
### Очередь на разбор
<|{{queue}}|table|page_size=8|height=330px|rebuild={{rb}}|on_action=open_card|cell_class_name[Критичность]=sev_cell|cell_class_name[SLA]=sla_cell|width[Тип]=170px|width[Цель]=170px|width[№]=70px|>
|>
<|part|
### Повторяющиеся цели
<|chart|figure={{f_tgt}}|plot_config={{pc}}|on_click=click_tgt|height=310px|>
|>
|>

<|layout|columns=5 10 7|gap=20px|
<|part|
### Возраст очереди
<|chart|figure={{f_age}}|plot_config={{pc}}|on_click=click_age|height=230px|>
|>
<|part|
### Приток карточек по дням · <|{{n24}}|text|> за последние 24 ч
<|chart|figure={{f_flow}}|plot_config={{pc}}|on_click=click_day|height=250px|>
|>
<|part|
### Типы в работе · ○ без вердикта
<|chart|figure={{f_types}}|plot_config={{pc}}|on_click=click_type|height=300px|>
|>
|>

<|layout|columns=1 1 1|gap=20px|
<|part|
### Типы: последние 24 ч к среднесуточному
<|chart|figure={{f_24}}|plot_config={{pc}}|on_click=click_type|height=300px|>
|>
<|part|
### Вероятный шум в очереди
<|chart|figure={{f_noise}}|plot_config={{pc}}|height=300px|>
|>
<|part|
### Поток по часам суток
<|chart|figure={{f_hour}}|plot_config={{pc}}|height=300px|>
|>
|>

<|{{show_card}}|dialog|labels=Закрыть;Открыть в источнике|on_action=card_action|width=760px|
<|{{c_body}}|text|mode=markdown|>
|>
"""

if __name__ == '__main__':
    Gui(page).run(   # стили – app.css рядом со скриптом (Taipy подключает его автоматически)
        title='Дежурная смена SOC', port=5080, use_reloader=False, dark_mode=False,
        server_config={'socketio': {'manage_session': False}},
        run_browser='--no-browser' not in sys.argv)
