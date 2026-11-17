# СРС 3.4 · данные для вкладки «Почему так?»: готовые вклады SHAP из комплекта практики 3.7 (без запуска модели)
# Вход (папка data/): cards_practice.csv, shap_cards.csv, cards_raw.parquet из IAVD_3.7_ML_team/course_pack/
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
TAU = 0.019258739887821837                      # порог lgbm_gate1 v001
Z_TAU = float(np.log(TAU / (1 - TAU)))          # −3,9303
TOL = 5e-7                                      # допуск σ(base + Σφ) = proba (proba в файле округлена до 1e-6)
sig = lambda z: 1 / (1 + np.exp(-z))

GROUPS = {
    'время': ['Создан', 'event_duration_s', 'create_lag_s', 'event_age_days', 'is_business_hours', 'is_night'],
    'сеть: адреса и хосты': ['src', 'dst', 'Hostname Цели', 'Hostname атакующего', 'src_is_private', 'dst_is_private',
                             'ip_count_src', 'ip_count_dst', 'dst_external_count', 'len_src', 'len_dst', 'host_depth',
                             'host_is_server'],
    'правило и источник': ['Наименование', 'freq_Наименование', 'categoryMain', 'categoryHigh', 'categoryLow',
                           'Пиктограмма IRP', 'Критичность', 'Критичность инцидента в источнике', 'Источник инцидента',
                           'Вендор из сием', 'freq_Вендор', 'Конвейер SIEM', 'Организация', 'object', 'n_rules'],
}
TEXT_GROUP = 'текст алерта, команд и объектов'
KEY_FIELDS = ['Наименование', 'Тип инцидента', 'Критичность', 'Пиктограмма IRP', 'Вендор из сием', 'Организация',
              'Hostname атакующего', 'Hostname Цели', 'src', 'dst', 'Алерт', 'Команда', 'Решение']

# Выводы по набору 37 (разбор в README и отчёте): что это – данные, порог или признак
NOTES = {
    '522033': ('порог и данные',
               'Троян HEUR:Trojan.PowerShell.Generic в кэше браузера Edge, файлы удалены, аналитик всё же поставил «Решён». '
               'Карточка выглядит как типичный «шум» правила TTS_10.1: до порога не хватило 0,83 в z (шансы ×2,3). '
               'Поймать её можно порогом ≤ 0,0085, но на потоке эксплуатации это +1 027 ложных тревог ради +44 найденных инцидентов. '
               'Сначала проверить разметку: такие очищенные срабатывания в кэше в других карточках закрывают как FP.'),
    '528183': ('данные (метка) и признак',
               'Касперский нашёл архив RTA-master.zip – набор скриптов, имитирующих атаки (сотрудник писал сценарии для киберполигона). '
               'Карточку эскалировали на L2 и закрыли как «Решён», то есть по политике это инцидент, хотя атаки не было. '
               'Модель сказала «легитимно» почти без доводов: главный «плюс» – время (вечер воскресенья, +1,71), '
               'а «плюс» от поля Команда (+0,61) дан за то, что поле пустое. Признака «чем опасен файл и где лежит» в модели нет.'),
    '519532': ('данные',
               'Та же сигнатура PT «Possible webshell: WGET», что у двух инцидентов набора (519755, 522274), – модель ошиблась «по-честному». '
               'Аналитик закрыл карточку как ложную, потому что атака не удалась (сервер ответил 404) и адрес уже был заблокирован; '
               'этого в полях карточки на момент создания нет. Порог не поможет (p = 0,999), '
               'а около +4,7 в z дали пустые поля: карточка из сети другой организации (DataLearn) пришла без обогащения.'),
    '519576': ('данные',
               'Внешний адрес 195.96.129.68 – тот же, что у 519532, – и сигнатура эксплойта checkpoint. '
               'Почти весь «плюс» дала сетевая группа (+8,2). Аналитик: «спамящее срабатывание, требует доработки» – чинить правило, а не модель.'),
    '519755': ('решение верное', 'Сетевая группа (+10,0): внешний атакующий, цель – adfs.iaivd.ru. Аргумент для отчёта – сетевые поля, а не один хост.'),
    '522274': ('решение верное', 'Сетевая группа (+10,1): внешний атакующий 196.251.83.20, попытка RCE, адрес заблокирован.'),
    '520257': ('решение верное', 'Вклады мелкие (сумма +3,08 при базе −8,85): модель «не нашла» признаков инцидента – норма для шумного правила TTS_10.2.'),
    '528152': ('решение верное', 'Внутренний адрес источника (−1,22 и −1,20) перевесил время (+1,45): IDS-сводка без атаки.'),
}


def group_of(col):
    for g, cols in GROUPS.items():
        if col in cols:
            return g
    return TEXT_GROUP


def rd(name):
    return pd.read_csv(os.path.join(DATA, name), sep=';', encoding='utf-8-sig')


def load():
    cards = rd('cards_practice.csv')
    shap = rd('shap_cards.csv')
    feats = [c for c in shap.columns if c not in ('EntityId', 'y_true', 'proba', 'outcome', 'base')]
    raw = pd.read_parquet(os.path.join(DATA, 'cards_raw.parquet'))
    for t in (cards, shap, raw):
        t['EntityId'] = t['EntityId'].astype(str)
    raw = raw.drop_duplicates('EntityId').set_index('EntityId')
    shap = shap.set_index('EntityId')
    cards = cards.set_index('EntityId')
    cards['base'] = shap['base']
    cards['z'] = shap['base'] + shap[feats].sum(axis=1)
    cards['err'] = (sig(cards['z']) - cards['proba']).abs()
    cards['verdict'] = raw['Вердикт'].reindex(cards.index)
    return cards, shap[feats], raw, feats


def short(v, n=60):
    if isinstance(v, pd.Timestamp) or (isinstance(v, str) and len(v) >= 19 and v[4] == '-' and v[10] == ' ' and v[13] == ':'):
        try:
            ts = pd.Timestamp(v)
            ts = (ts.tz_localize('UTC') if ts.tzinfo is None else ts).tz_convert('Europe/Moscow')
            return ts.strftime('%d.%m.%Y %H:%M МСК, ') + ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'][ts.dayofweek]
        except Exception:
            pass
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'пусто'
    s = str(v).replace('\\\\', '\\').replace('["', '').replace('"]', '').replace('", "', ' ; ').replace('\n', ' ')
    return s if len(s) <= n else s[:n - 1] + '…'


def explain(eid, cards, shap, raw, top=10):
    """Топ-N по модулю, хвост остальных полей, группы, проверка аддитивности."""
    c = cards.loc[eid]
    v = shap.loc[eid].astype(float)
    order = v.abs().sort_values(ascending=False).index
    t = pd.DataFrame({'Поле': order[:top], 'Вклад': v[order[:top]].round(3).to_numpy()})
    t['Направление'] = np.where(t['Вклад'] > 0, 'к инциденту', 'к легитимному')
    t['Тип'] = np.where(t['Поле'].isin(raw.columns), 'поле карточки', 'сконструированный')
    t['Значение в карточке'] = [short(raw.loc[eid, f]) if f in raw.columns else '— (считается из полей)' for f in t['Поле']]
    tail = float(v[order[top:]].sum())
    groups = v.groupby([group_of(f) for f in v.index]).sum().sort_values()
    z = float(c['base'] + v.sum())
    return dict(card=c, top=t, tail=tail, n_tail=len(order) - top, groups=groups, z=z, p_sig=float(sig(z)),
                err=abs(float(sig(z)) - float(c['proba'])), gap=z - Z_TAU)


if __name__ == '__main__':
    cards, shap, raw, feats = load()
    print('карточек', len(cards), 'признаков', len(feats), 'max |σ(z) − proba|', f'{cards["err"].max():.1e}',
          'решение по z совпадает с исходом:', bool(((cards['z'] >= Z_TAU) == cards['outcome'].isin(['TP', 'FP'])).all()))
    for eid in cards[cards['team'] == 37].index:
        e = explain(eid, cards, shap, raw)
        print(eid, e['card']['outcome'], f"z={e['z']:.4f} top10={e['top']['Вклад'].sum():+.3f} tail={e['tail']:+.3f} err={e['err']:.1e}")
