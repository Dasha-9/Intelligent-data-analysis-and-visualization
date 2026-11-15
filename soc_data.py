# Проект · дашборд дежурной смены SOC · подготовка данных (pandas, без Taipy)
# Все времена – МСК. «Снимок» – последнее время создания карточки в выгрузке (03.09.2025 15:21 МСК).
import re
import pandas as pd

DEFAULT_PATH = 'data/iavd_incidents_raw_target.parquet'
COLS = ['EntityId', 'Создан', 'Критичность', 'Вердикт', 'Тип инцидента', 'Фаза реагирования', 'Hostname Цели',
        'Hostname атакующего', 'Правило корреляции SIEM', 'SLA просрочен (взятие в работу)', 'Дата взятия в работу',
        'Ответственный', 'Описание инцидента', 'Ссылка в NAD (actual)', 'Источник инцидента']
SEV = ['Критическая', 'Высокая', 'Средняя', 'Низкая']
SEV_COLOR = {'Критическая': '#CF222E', 'Высокая': '#424A53', 'Средняя': '#8C959F', 'Низкая': '#D0D7DE'}
AGE = ['< 1 ч', '1–4 ч', '4–24 ч', '1–3 сут', '> 3 сут']
NOISY_MIN_N, NOISY_MIN_FP = 300, 0.95      # гипотеза 1 СРС 2.4: «шумное» правило
OTHER = 'Не указан'


def _targets(s):
    """'ip\\fqdn\\short|\\fqdn\\short' → уникальные цели карточки (FQDN, иначе IP; 0.0.0.0 отбрасываем)."""
    out = []
    for part in str(s or '').split('|'):
        toks = [t for t in re.split(r'\\+', part) if t]
        fq = [t for t in toks if '.' in t and re.search('[a-zA-Z]', t)]
        ip = [t for t in toks if re.fullmatch(r'\d+\.\d+\.\d+\.\d+', t) and t != '0.0.0.0']
        name = (fq or ip or [None])[0]
        if name and name not in out:
            out.append(name)
    return out


def load(path=DEFAULT_PATH):
    df = pd.read_parquet(path, columns=COLS)
    df['msk'] = df['Создан'].dt.tz_convert('Europe/Moscow')
    df['day'] = df['msk'].dt.normalize()
    df['hour'] = df['msk'].dt.hour
    snap = df['msk'].max()
    df['open'] = ~df['Фаза реагирования'].fillna('').str.contains('Закрыт')
    df['age_h'] = (snap - df['msk']).dt.total_seconds() / 3600
    df['age'] = pd.cut(df['age_h'], [-1, 1, 4, 24, 72, 1e9], labels=AGE).astype(str)
    df['sev'] = df['Критичность'].where(df['Критичность'].isin(SEV), 'Низкая')
    df['sev_rank'] = df['sev'].map({s: i for i, s in enumerate(SEV)})
    df['type'] = df['Тип инцидента'].fillna(OTHER)
    df['rule'] = df['Правило корреляции SIEM'].fillna(OTHER)
    df['sla_over'] = df['SLA просрочен (взятие в работу)'].astype(str) == 'True'
    df['no_verdict'] = df['Вердикт'].eq('Не указан')
    df['targets'] = df['Hostname Цели'].map(_targets)
    # гипотеза 1: история вердиктов по правилу → «вероятный шум»
    v = df[~df['no_verdict']]
    g = v.groupby('rule')['Вердикт'].agg(n='size', fp=lambda s: (s == 'False Positive').mean())
    noisy = g[(g.n >= NOISY_MIN_N) & (g.fp >= NOISY_MIN_FP)]
    df['noisy'] = df['rule'].isin(noisy.index)
    df['rule_fp'] = df['rule'].map(g['fp'])
    return df, snap


def filtered(df, start, end, sev, typ, target, age, day, only_sla):
    """Общие фильтры: период по дате создания, критичность, тип, цель, корзина возраста, день, просрочка SLA."""
    m = (df['day'] >= start) & (df['day'] <= end)
    if sev:
        m &= df['sev'].isin(sev)
    if typ and typ != 'Все':
        m &= df['type'].eq(typ)
    if target:
        m &= df['targets'].map(lambda t: target in t)
    if age:
        m &= df['age'].eq(age)
    if day is not None:
        m &= df['day'].eq(day)
    if only_sla:
        m &= df['sla_over']
    return df[m]


def kpis(o):
    """o – открытые карточки после фильтров."""
    return dict(open=len(o), crit=int((o['sev'] == 'Критическая').sum()),
                high=int(o['sev'].isin(['Критическая', 'Высокая']).sum()),
                nov=int(o['no_verdict'].sum()), old=int((o['age_h'] > 72).sum()),
                sla=int(o['sla_over'].sum()), noisy=int(o['noisy'].sum()))


def queue_table(o):
    q = o.sort_values(['sev_rank', 'sla_over', 'age_h'], ascending=[True, False, False])

    def age_txt(h):
        return f'{h:.1f} ч'.replace('.', ',') if h < 24 else f'{h / 24:.0f} сут'
    return pd.DataFrame({
        '№': q['EntityId'].astype(str),
        'Критичность': q['sev'],
        'Тип': q['type'],
        'Возраст': q['age_h'].map(age_txt),
        'SLA': q['sla_over'].map(lambda b: 'просрочен' if b else ''),
        'Вердикт': q['no_verdict'].map(lambda b: 'нет' if b else 'есть'),
        'Пометка': q['noisy'].map(lambda b: 'вероятный шум' if b else ''),
        'Цель': q['targets'].map(lambda t: ', '.join(t[:2]) if t else '—'),
    }).reset_index(drop=True)


def card(df, eid):
    r = df[df['EntityId'] == int(eid)].iloc[0]
    link = str(r['Ссылка в NAD (actual)'] or '')
    m = re.search(r'https?://[^\s"<>]+', link)
    return dict(
        id=str(r['EntityId']), created=r['msk'].strftime('%d.%m.%Y %H:%M МСК'), sev=r['sev'], type=r['type'],
        rule=r['rule'], phase=str(r['Фаза реагирования']), owner=str(r['Ответственный'] or '—'),
        targets=', '.join(r['targets']) or '—', attacker=str(r['Hostname атакующего'] or '—').replace('~~||~~', ', ').strip('\\') or '—',
        source=str(r['Источник инцидента'] or '—'), descr=str(r['Описание инцидента'] or '—').split('~~||~~')[0][:600],
        sla='просрочен' if r['sla_over'] else 'в норме', verdict=str(r['Вердикт']),
        noisy=(f"правило «шумное»: {r['rule_fp']:.0%} вердиктов – False Positive" if r['noisy'] else
               (f"доля FP у правила: {r['rule_fp']:.0%}" if pd.notna(r['rule_fp']) and r['rule'] != OTHER else 'правило не указано или нет истории вердиктов')),
        link=m.group(0) if m else '')


if __name__ == '__main__':
    df, snap = load()
    print(snap, len(df))
    o = df[df['open']]
    print(kpis(o))
    print(o['age'].value_counts())
    print(o.explode('targets')['targets'].value_counts().head(8))
    print(queue_table(o).head(8).to_string())
