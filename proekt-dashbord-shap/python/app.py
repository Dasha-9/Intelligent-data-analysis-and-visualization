# Проект, часть 3 · дашборд «Можно ли верить объяснению модели?» (SHAP-датасет) · Taipy + Plotly
# Зритель – аналитик SOC (L1/L2). Модель курса lgbm_gate1 решает «инцидент / легитимно», SHAP раскладывает решение
# по полям карточки. Дашборд проверяет доводы модели по реальным вердиктам на трёх уровнях: поток → группа → карточка.
# Запуск из корня папки: python python/app.py   → http://127.0.0.1:5083   (данные: python/prep_shap_flow.py)
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from taipy.gui import Gui
from shap_flow import (load, filt, kpi, importance, by_value, by_hour, blind, fn_vs_tp, card_lov, explain,
                       Z_TAU, TAU, GROUP_ORDER, OUT_RU)

INK, MUTED, LINE, ACC, RED, GRAY, LIGHT, DARK = '#1F2328', '#6E7781', '#D0D7DE', '#0969DA', '#CF222E', '#8C959F', '#E6E9ED', '#57606A'
GCOL = {'сеть': '#0969DA', 'правило и источник': '#57606A', 'текст алерта и команд': '#8C959F', 'время': '#C3C9D0'}
FONT = dict(family='Segoe UI, Arial, sans-serif', size=12, color=INK)
fmt = lambda v: f'{int(round(v)):,}'.replace(',', ' ')
pct = lambda v, d=1: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:.{d}f} %'.replace('.', ',')
num = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',').replace('-', '−')
sgn = lambda v, d=2: ('+' if v > 0 else '') + num(v, d)

D, PHI = load()
TYPES = ['все'] + D['type'].value_counts().index.tolist()
pc = {'displayModeBar': False}


def base(fig, h, l=10, r=20, t=8, b=30):
    fig.update_layout(height=h, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor='white', plot_bgcolor='white', font=FONT,
                      showlegend=False, hoverlabel=dict(bgcolor='white'), bargap=0.3)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def empty(h, txt='нет карточек в выборке'):
    fig = base(go.Figure(), h)
    fig.add_annotation(text=txt, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


# --- 1. Поток: на что модель смотрит ---------------------------------------------------------------
def fig_importance(r):
    if r.empty:
        return empty(330)
    r = r.iloc[::-1]
    lab = [f if len(f) <= 30 else f[:29] + '…' for f in r['field']]
    fig = base(go.Figure(), 330, r=60)
    fig.add_bar(y=lab, x=r['imp'], orientation='h', marker_color=[GCOL[g] for g in r['group']],
                text=[num(v) for v in r['imp']], textposition='outside', cliponaxis=False, customdata=r['group'],
                hovertemplate='%{y}: %{x:.3f} · группа «%{customdata}»<extra></extra>')
    fig.update_xaxes(title='Средний модуль SHAP')
    return fig


# --- 2–3. Довод модели против реальности -----------------------------------------------------------
def fig_value(g, title_phi='средний вклад поля, z', h=200):
    if g.empty:
        return empty(h)
    lab = [f'{i}<br><span style="color:{MUTED}">{fmt(n)} карт.</span>' for i, n in zip(g.index, g['n'])][::-1]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=(f'Что говорит модель: {title_phi}', 'Что в данных: доля инцидентов'))
    phi, inc = g['phi'][::-1], g['inc'][::-1]
    fig.add_bar(y=lab, x=phi, orientation='h', marker_color=[RED if v > 0 else ACC for v in phi], row=1, col=1,
                text=[sgn(v) for v in phi], textposition='outside', cliponaxis=False, hovertemplate='%{x:+.2f}<extra></extra>')
    fig.add_bar(y=lab, x=inc, orientation='h', marker_color=DARK, row=1, col=2,
                text=[pct(v) for v in inc], textposition='outside', cliponaxis=False, hovertemplate='%{x:.1%}<extra></extra>')
    base(fig, h, r=40, t=26)
    fig.update_annotations(font=dict(size=12, color=MUTED), x=None)
    for i, a in enumerate(fig.layout.annotations):
        a.update(x=[0.0, 0.53][i], xanchor='left')
    lo, hi = min(0, phi.min()), max(0, phi.max())
    fig.update_xaxes(visible=False, range=[lo - 0.5 * (hi - lo + 0.5), hi + 0.5 * (hi - lo + 0.5)], row=1, col=1)
    fig.update_xaxes(visible=False, range=[0, max(inc.max(), 0.01) * 1.45], row=1, col=2)
    fig.add_vline(x=0, line=dict(color=LINE, width=1), row=1, col=1)
    return fig


def fig_hour(h):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=('Что говорит модель: средний вклад поля «Создан», z',
                                        'Что в данных: доля инцидентов, пунктир – в среднем по выборке'))
    ok = h['n'].fillna(0) >= 30
    phi = h['phi'].where(ok); inc = h['inc'].where(ok)
    fig.add_bar(x=h.index, y=phi, marker_color=[RED if (v or 0) > 0 else ACC for v in phi.fillna(0)], row=1, col=1,
                hovertemplate='%{x}:00 · вклад %{y:+.2f}<extra></extra>')
    fig.add_bar(x=h.index, y=inc, marker_color=DARK, row=2, col=1, customdata=h['n'],
                hovertemplate='%{x}:00 · инцидентов %{y:.1%} из %{customdata} карточек<extra></extra>')
    avg = (h['inc'] * h['n']).sum() / h['n'].sum() if h['n'].sum() else 0
    fig.add_hline(y=avg, line=dict(color=GRAY, dash='dot', width=1.5), row=2, col=1)
    base(fig, 330, t=26, b=30)
    fig.update_annotations(font=dict(size=12, color=MUTED), x=0, xanchor='left')
    fig.update_xaxes(tickmode='array', tickvals=list(range(0, 24, 3)), ticktext=[f'{i}:00' for i in range(0, 24, 3)], row=2, col=1)
    fig.update_yaxes(tickformat='.0%', gridcolor='#EEF0F2', showgrid=True, row=2, col=1)
    fig.update_yaxes(gridcolor='#EEF0F2', showgrid=True, zeroline=True, zerolinecolor=LINE, row=1, col=1)
    return fig


# --- 4. Группа: где модель слепа ---------------------------------------------------------------------
def fig_blind(g, sel):
    if g.empty:
        return empty(300)
    g = g.iloc[::-1]
    lab = list(g.index)
    fig = base(go.Figure(), 60 + 34 * len(g), r=10)
    op = [1 if sel in ('все', t) else 0.35 for t in lab]
    fig.add_bar(y=lab, x=g['inc'], orientation='h', marker=dict(color=LIGHT, opacity=op),
                hovertemplate='%{y}: реальных инцидентов %{x}<extra></extra>')
    fig.add_bar(y=lab, x=g['fn'], orientation='h', marker=dict(color=RED, opacity=op),
                hovertemplate='%{y}: пропущено %{x}<extra></extra>')
    fig.add_scatter(y=lab, x=g['inc'], mode='text', textposition='middle right', hoverinfo='skip',
                    text=[f'  <b>{fmt(m)}</b> из {fmt(i)} · {pct(s, 0)}' for m, i, s in zip(g['fn'], g['inc'], g['miss'])],
                    textfont=dict(size=12, color=INK))
    fig.update_layout(barmode='overlay', bargap=0.35)
    fig.update_xaxes(visible=False, range=[0, g['inc'].max() * 1.6])
    fig.update_yaxes(ticksuffix='  ')
    return fig


def fig_fntp(r):
    if r.dropna(how='all').empty:
        return empty(260, 'нет реальных инцидентов в выборке')
    fig = base(go.Figure(), 260, r=40)
    groups = GROUP_ORDER[::-1]
    for out, col, name in (('TP', GRAY, 'найденные инциденты (TP)'), ('FN', RED, 'пропущенные инциденты (FN)')):
        if out in r.index and r.loc[out].notna().any():
            v = r.loc[out, groups]
            fig.add_bar(y=groups, x=v, orientation='h', name=name, marker_color=col, text=[sgn(x) for x in v],
                        textposition='outside', cliponaxis=False, hovertemplate=name + ': %{x:+.2f}<extra></extra>')
    fig.update_layout(barmode='group', showlegend=True, legend=dict(orientation='h', y=1.14, x=0, font=dict(size=11)),
                      margin=dict(t=30))
    fig.add_vline(x=0, line=dict(color=LINE, width=1))
    lo = float(np.nanmin(r.to_numpy())); hi = float(np.nanmax(r.to_numpy()))
    fig.update_xaxes(visible=False, range=[min(lo, 0) - 0.25 * (hi - min(lo, 0)) - 0.3, hi * 1.2 + 0.3])
    fig.update_yaxes(ticksuffix='    ')
    return fig


# --- 5. Карточка: почему так ----------------------------------------------------------------------
def fig_water(e):
    r, t = e['row'], e['top']
    labels = ['база модели'] + list(t['Поле']) + [f'остальные {e["n_tail"]} полей', 'итог z']
    vals = [float(r['base'])] + list(t['Вклад']) + [e['tail'], 0]
    meas = ['absolute'] + ['relative'] * (len(t) + 1) + ['total']
    fig = go.Figure(go.Waterfall(
        orientation='h', y=labels, x=vals, measure=meas,
        increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=ACC)), totals=dict(marker=dict(color=GRAY)),
        connector=dict(line=dict(color=LINE, width=1)),
        text=[(sgn(v) if m == 'relative' else num(float(r['z']) if m == 'total' else v)) for v, m in zip(vals, meas)],
        textposition=['inside' if m != 'relative' else 'outside' for m in meas], insidetextfont=dict(color='white'),
        cliponaxis=False, hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
    fig.add_vline(x=Z_TAU, line=dict(color=ACC, dash='dot', width=2))
    fig.add_annotation(x=Z_TAU, y=1.0, yref='paper', text='порог z_τ = −3,93', showarrow=False, xanchor='left',
                       yanchor='bottom', font=dict(color=ACC, size=12))
    base(fig, 120 + 30 * len(labels), r=60, t=24)
    fig.update_xaxes(title='логарифм шансов z · красный – к инциденту, синий – к легитимному', gridcolor='#EEF0F2',
                     showgrid=True, zeroline=True, zerolinecolor=LINE)
    fig.update_yaxes(autorange='reversed', automargin=True)
    return fig


# --- состояние ---------------------------------------------------------------------------------------
typ, out = 'все', 'все'
out_lov = ['все', 'FN', 'TP', 'FP', 'TN']
cards_lov = card_lov(D)
card = cards_lov[0]
chips = ''
k_n = k_inc = k_fn = k_fnp = k_add = ''
f_imp = f_src = f_grp = f_hour = f_blind = f_fntp = f_water = go.Figure()
card_title = card_text = ''
top_table = pd.DataFrame()


def refresh(state, keep_card=False):
    d_all = filt(D, state.typ, 'все')         # вопросы 2–4: исход не фильтруется, иначе «доля инцидентов» теряет смысл
    d = filt(D, state.typ, state.out)
    k = kpi(d)
    state.k_n = fmt(k['n'])
    state.k_inc = fmt(k['inc'])
    state.k_fn = f'{fmt(k["fn"])} из {fmt(k["inc"])}' if k['inc'] else 'нет инцидентов'
    state.k_fnp = pct(k['miss'])
    state.k_add = '—' if np.isnan(k['add']) else (f'{k["add"]:.7f}'.replace('.', ',') + (' ✓' if k['add'] <= 1e-6 else ' ✗'))
    parts = []
    if state.typ != 'все': parts.append(f'тип: {state.typ}')
    if state.out != 'все': parts.append(f'исход: {OUT_RU[state.out]} ({state.out})')
    state.chips = ('Выбрано – ' + ' · '.join(parts) + '') if parts else 'Весь поток эксплуатации'

    state.f_imp = fig_importance(importance(d, PHI))

    g = by_value(d_all, 'src_kind', 'src_is_private', ['внешний', 'внутренний', 'не указан'])
    state.f_src = fig_value(g, 'вклад src_is_private, z')

    g2 = by_value(d_all, 'grp_empty', 'len_Значение Инцидента (группировка)', ['пустое', 'заполнено'])
    state.f_grp = fig_value(g2, 'вклад длины поля, z', h=170)

    h = by_hour(d_all)
    state.f_hour = fig_hour(h)

    b = blind(D)
    state.f_blind = fig_blind(b, state.typ)

    r = fn_vs_tp(d_all)
    state.f_fntp = fig_fntp(r)

    state.cards_lov = card_lov(d)
    if not state.cards_lov:
        state.card = ''
    elif not keep_card or state.card not in state.cards_lov:
        state.card = state.cards_lov[0]
    refresh_card(state)


def refresh_card(state):
    if not state.card:
        state.f_water = empty(300, 'нет карточек в выборке'); state.card_title = state.card_text = state.card_sum = ''
        state.top_table = pd.DataFrame(); return
    eid = state.card.split(' · ')[0]
    e = explain(eid, D, PHI)
    r = e['row']
    state.f_water = fig_water(e)
    oc = r['outcome']
    state.card_title = (f'Карточка {eid} · {r["type"]} · p = {num(r["score"], 4)} · решение модели – '
                        f'{"инцидент" if r["flag"] else "легитимно"} · вердикт – {r["card__Вердикт"] or "нет"}'
                        + (f' · {OUT_RU[oc]} ({oc})' if oc else ''))
    f = lambda k: (r.get('card__' + k) or '—')
    desc = f('Описание инцидента')
    state.card_text = (f'Правило: {f("Наименование")[:90]}\n\nТип: {f("Тип инцидента")[:80]} · критичность: {f("Критичность")}\n\n'
                       f'Атакующий: {f("Hostname атакующего")[:60]} · цель: {f("Hostname Цели")[:60]}\n\n'
                       f'Описание: {desc[:260] + ("…" if len(desc) > 260 else "")}')
    t = e['top'].copy(); t['Вклад'] = [sgn(v) for v in t['Вклад']]
    state.top_table = t


def on_filter(state):
    refresh(state)


def click_blind(state, id, p):
    y = p.get('y') or ''
    state.typ = 'все' if state.typ == y else (y if y in TYPES else 'все')
    refresh(state)


def on_card(state):
    refresh_card(state)


def reset(state):
    state.typ, state.out = 'все', 'все'
    refresh(state)


def on_init(state):
    refresh(state)


page = """
# Можно ли верить объяснению модели? · поток эксплуатации, 20 584 карточки

<|layout|columns=3 4 1|gap=14px|class_name=frow|
<|part|
<|{typ}|selector|lov={TYPES}|dropdown|label=Тип инцидента (Пиктограмма IRP)|on_change=on_filter|width=100%|>
|>
<|part|
<|text|value=Исход · FN — пропуск · TP — найден · FP — ложная тревога · TN — верно снят шум|class_name=kt|>
<|{out}|toggle|lov={out_lov}|on_change=on_filter|>
|>
<|part|
<|Сбросить|button|on_action=reset|>
|>
|>
<|{chips}|text|class_name=chips|>

<|layout|columns=1 1 2 1|gap=12px|
<|part|class_name=kpi|
<|text|value=Карточек в выборке|class_name=kt|>
<|{k_n}|text|class_name=kv grey|>
|>
<|part|class_name=kpi|
<|text|value=Реальных инцидентов (по вердикту)|class_name=kt|>
<|{k_inc}|text|class_name=kv grey|>
|>
<|part|class_name=kpi kpi_main|
<|text|value=Пропущено инцидентов|class_name=kt|>
<|{k_fn}|text|class_name=kv red|> <|{k_fnp}|text|class_name=kv2 red|>
|>
<|part|class_name=kpi|
<|text|value=Макс. ошибка аддитивности SHAP|class_name=kt|>
<|{k_add}|text|class_name=kv blue|>
|>
|>

<|layout|columns=5 7|gap=20px|
<|part|
### Важность полей по SHAP
<|chart|figure={f_imp}|plot_config={pc}|height=330px|>
|>
<|part|
### Внешние и внутренние источники
<|chart|figure={f_src}|plot_config={pc}|height=200px|>
|>
|>

### Заполненность поля «Значение Инцидента (группировка)»
<|layout|columns=5 7|gap=20px|
<|part|
<|chart|figure={f_grp}|plot_config={pc}|height=170px|>
|>
<|part|
### Вклад времени создания и доля инцидентов по часам
<|chart|figure={f_hour}|plot_config={pc}|height=330px|>
|>
|>

<|layout|columns=7 5|gap=20px|
<|part|
### Пропуски по типу инцидента
<|chart|figure={f_blind}|plot_config={pc}|on_click=click_blind|height=300px|>
|>
<|part|
### Вклады групп полей для FN и TP
<|chart|figure={f_fntp}|plot_config={pc}|height=260px|>
|>
|>

### Водопад SHAP по карточке
<|layout|columns=3 2|gap=20px|
<|part|
<|{card}|selector|lov={cards_lov}|dropdown|label=Карточка (сначала пропуски, ближе к порогу)|on_change=on_card|width=100%|>
<|{card_title}|text|class_name=ct|>
<|chart|figure={f_water}|plot_config={pc}|height=480px|>
|>
<|part|
<|text|value=Что видит аналитик в карточке|class_name=kt|>
<|{card_text}|text|mode=pre|class_name=card|>
<|text|value=Крупнейшие вклады|class_name=kt|>
<|{top_table}|table|show_all|width=100%|page_size=10|>
|>
|>
"""

if __name__ == '__main__':
    Gui(page).run(title='Можно ли верить объяснению модели?', port=5083, use_reloader=False, dark_mode=False,
        server_config={'socketio': {'manage_session': False}},
                  run_browser='--no-browser' not in sys.argv)
