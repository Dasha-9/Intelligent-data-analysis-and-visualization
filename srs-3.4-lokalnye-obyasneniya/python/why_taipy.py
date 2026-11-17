# СРС 3.4 · вкладка «Почему так?» в Taipy + Plotly
# Запуск из корня папки: python python/why_taipy.py  →  http://127.0.0.1:5082
# Набор (1–40) и карточка выбираются вверху; водопад идёт от базы модели через 10 крупнейших вкладов
# и хвост «остальные поля» к итогу z; пунктир – порог. Ниже – проверка аддитивности, группы полей и поля карточки.
import os, sys
import plotly.graph_objects as go
from taipy.gui import Gui

sys.path.insert(0, os.path.dirname(__file__))
from shap_data import load, explain, NOTES, Z_TAU, TAU, TOL

INK, MUTED, GRAY, LINE, ACC, RED = '#1F2328', '#6E7781', '#C3C9D0', '#D0D7DE', '#0969DA', '#CF222E'
num = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',').replace('-', '−')
sgn = lambda v, d=2: ('+' if v > 0 else '') + num(v, d)
OUT = {'TP': 'нашла инцидент (TP)', 'FP': 'ложная тревога (FP)', 'FN': 'пропустила инцидент (FN)', 'TN': 'верно сняла шум (TN)'}

CARDS, SHAP, RAW, FEATS = load()
team_lov = [str(t) for t in sorted(CARDS['team'].unique())]


def card_lov(team):
    order = {'FN': 0, 'FP': 1, 'TN': 2, 'TP': 3}
    part = CARDS[CARDS['team'] == int(team)].assign(o=lambda d: d['outcome'].map(order)).sort_values(['o', 'z'])
    return [f'{e} · {r.outcome} · p = {num(r.proba, 4)}' for e, r in part.iterrows()]


team = '37'
title = 'Почему так? · набор 37'
cards_lov = card_lov(team)
card = cards_lov[0]
top_n = '10'
fig_water = go.Figure()
fig_groups = go.Figure()
k_p = k_dec = k_ver = k_out = k_gap = ''
add_line = note_kind = note_text = sum_line = ''
top_table = None
fields_table = None


def water_fig(e, n):
    c = e['card']
    t = e['top']
    labels = ['база модели'] + [f'{f}  ·  {v}' if len(str(v)) < 34 else f'{f}  ·  {str(v)[:33]}…'
                                for f, v in zip(t['Поле'], t['Значение в карточке'])] + [f'остальные {e["n_tail"]} полей', 'итог z']
    vals = [float(c['base'])] + list(t['Вклад']) + [e['tail'], 0]
    meas = ['absolute'] + ['relative'] * (len(t) + 1) + ['total']
    f = go.Figure(go.Waterfall(
        orientation='h', y=labels, x=vals, measure=meas,
        increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=ACC)),
        totals=dict(marker=dict(color='#8C959F')),
        connector=dict(line=dict(color=LINE, width=1)),
        text=[(sgn(v) if m == 'relative' else num(e['z'] if m == 'total' else v)) for v, m in zip(vals, meas)],
        textposition=['inside' if m != 'relative' else 'outside' for m in meas], insidetextfont=dict(color='white'), cliponaxis=False,
        hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
    f.add_vline(x=Z_TAU, line=dict(color=ACC, dash='dot', width=2))
    f.add_annotation(x=Z_TAU, y=1.0, yref='paper', text='порог z_τ = −3,93', showarrow=False, xanchor='left', yanchor='bottom',
                     font=dict(color=ACC, size=12))
    f.update_layout(height=120 + 34 * len(labels), margin=dict(l=10, r=70, t=30, b=40), plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(color=INK, size=12), showlegend=False,
                    xaxis=dict(title='логарифм шансов z · красный – к инциденту, синий – к легитимному', gridcolor='#EEF0F2',
                               zeroline=True, zerolinecolor=LINE, color=MUTED),
                    yaxis=dict(automargin=True, autorange='reversed'))
    return f


def groups_fig(e):
    g = e['groups']
    f = go.Figure(go.Bar(x=g.values, y=g.index, orientation='h', marker_color=[RED if v > 0 else ACC for v in g.values],
                         text=[sgn(v) for v in g.values], textposition='inside', insidetextanchor='middle', textfont=dict(color='white'),
                         hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
    f.update_layout(height=230, margin=dict(l=10, r=60, t=10, b=40), plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(color=INK, size=12), xaxis=dict(title='сумма вкладов группы, z', gridcolor='#EEF0F2',
                                                               zeroline=True, zerolinecolor=LINE, color=MUTED),
                    yaxis=dict(automargin=True))
    return f


def refresh(state):
    state.title = f'Почему так? · набор {state.team}'
    eid = state.card.split(' · ')[0]
    e = explain(eid, CARDS, SHAP, RAW, top=int(state.top_n))
    c = e['card']
    state.k_p = num(c['proba'], 6)
    state.k_dec = 'Инцидент' if c['proba'] >= TAU else 'Легитимно'
    state.k_ver = str(c['verdict'])
    state.k_out = OUT[c['outcome']]
    state.k_gap = sgn(e['gap'])
    state.sum_line = (f'base {num(c["base"], 4)} + топ-{state.top_n} {sgn(e["top"]["Вклад"].sum(), 3)} + остальные {e["n_tail"]} '
                      f'{sgn(e["tail"], 3)} = z {num(e["z"], 4)} → p = σ(z) = {num(e["p_sig"], 6)}. '
                      f'Топ-{state.top_n} объясняет {e["top"]["Вклад"].sum() / (e["z"] - c["base"]) * 100:.0f} % отклонения от базы.')
    ok = e['err'] <= TOL
    state.add_line = (f'Проверка аддитивности: |σ(base + Σφ) − p| = {e["err"]:.1e} ' + ('≤ 5·10⁻⁷ – объяснение сходится с оценкой ✓' if ok
                      else '> 5·10⁻⁷ – объяснение НЕ сходится с оценкой ✗'))
    kind, text = NOTES.get(eid, ('', ''))
    if not text:
        t0 = e['top'].iloc[0]
        kind = 'шаблон 3.7'
        text = (f'Наибольший вклад дало поле {t0["Поле"]}: {t0["Направление"]} на {sgn(t0["Вклад"])} в логарифме шансов; '
                f'база {num(c["base"])}. Решение – {state.k_dec.lower()}; что это – данные, порог или признак, решает разбор карточки.')
    state.note_kind = kind
    state.note_text = text
    state.top_table = e['top'][['Поле', 'Значение в карточке', 'Вклад', 'Направление', 'Тип']]
    from shap_data import KEY_FIELDS, short
    state.fields_table = __import__('pandas').DataFrame({'Поле карточки': KEY_FIELDS,
                                                         'Значение': [short(RAW.loc[eid].get(k), 110) for k in KEY_FIELDS]})
    state.fig_water = water_fig(e, int(state.top_n))
    state.fig_groups = groups_fig(e)


def on_team(state):
    state.cards_lov = card_lov(state.team)
    state.card = state.cards_lov[0]
    refresh(state)


def on_init(state):
    refresh(state)


page = """
<|{title}|text|class_name=h1|>
<|text|value=Модель lgbm_gate1 v001 · порог τ = 0,019259 (z_τ = −3,9303) · вклады SHAP в логарифме шансов, режим структуры деревьев (pred_contrib)|class_name=q|>

<|layout|columns=1 4 2|gap=20px|class_name=ctl|
<|part|
<|{team}|selector|lov={team_lov}|dropdown|label=Набор|on_change=on_team|>
|>
<|part|
<|{card}|selector|lov={cards_lov}|dropdown|label=Карточка|on_change=refresh|width=100%|>
|>
<|part|
<|text|value=Сколько вкладов показать|class_name=kt|>
<|{top_n}|toggle|lov=5;10;20|on_change=refresh|>
|>
|>

<|layout|columns=1 1 1 1 1|gap=12px|
<|part|class_name=kpi|
<|text|value=Оценка модели p|class_name=kt|>
<|{k_p}|text|class_name=kv|>
|>
<|part|class_name=kpi|
<|text|value=Решение модели|class_name=kt|>
<|{k_dec}|text|class_name=kv|>
|>
<|part|class_name=kpi|
<|text|value=Вердикт аналитика|class_name=kt|>
<|{k_ver}|text|class_name=kv grey|>
|>
<|part|class_name=kpi|
<|text|value=Исход|class_name=kt|>
<|{k_out}|text|class_name=kv small|>
|>
<|part|class_name=kpi|
<|text|value=До порога в z (z − z_τ)|class_name=kt|>
<|{k_gap}|text|class_name=kv blue|>
|>
|>

<|layout|columns=3 2|gap=24px|
<|part|
### Водопад: от базы модели к итогу
<|text|value=Почему водопад: показывает, откуда взялся итог – старт с базы, каждый шаг – вклад поля, отдельной строкой хвост остальных полей, чтобы топ-10 не выдавался за всё объяснение|class_name=why|>
<|chart|figure={fig_water}|>
<|{sum_line}|text|class_name=sum|>
<|{add_line}|text|class_name=add|>
|>
<|part|
### Вывод: данные, порог или признак?
<|{note_kind}|text|class_name=tag|>
<|{note_text}|text|class_name=note|>
### Группы связанных полей
<|text|value=Почему группы: вклад одного поля размазан по связанным колонкам (адреса, хосты, IP), поэтому выводы формулируем про группу – «сработала сеть», а не «виноват хост»|class_name=why|>
<|chart|figure={fig_groups}|>
|>
|>

### Крупнейшие вклады и значения полей
<|text|value=Почему таблица: аналитику нужно сверить вклад с тем, что написано в карточке; у сконструированных признаков (len_, freq_, cmd_) такого поля в карточке SOAR нет|class_name=why|>
<|{top_table}|table|show_all|width=100%|>

### Поля карточки
<|{fields_table}|table|show_all|width=100%|>
"""

if __name__ == '__main__':
    Gui(page).run(   # стили – из why_taipy.css рядом со скриптом (Taipy подхватывает его сам)
        title='Почему так? · СРС 3.4', port=5082, dark_mode=False, run_browser=False, use_reloader=False)
