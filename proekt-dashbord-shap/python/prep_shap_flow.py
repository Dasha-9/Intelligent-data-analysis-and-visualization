# Проект, часть 3 · подготовка данных: поток эксплуатации с вкладами SHAP
# Запуск из корня папки (окружение курса: lightgbm 4.7, scikit-learn 1.7.2, pandas 2.3, pyarrow 18):
#   python python/prep_shap_flow.py <папка IAVD_3.7_ML_team> <папка IAVD_3.7_BI_team>
# Вход (BI-комплект практики 3.7, data/5/Parquet/Gen и data/2_5/Parquet/Gen):
#   fact_shap.parquet   – вклады 73 признаков, срез productive_20584 (20 584 карточки эксплуатации)
#   fact_score.parquet  – оценка модели score
#   dim_cards.parquet   – метка y и вердикт
#   iavd_ml_dataset.parquet – исходные поля карточек
# Значения сконструированных признаков (src_is_private, len_*, время) считаются тем же кодом,
# что и при обучении модели (course_ml из ML-комплекта), – чтобы сравнивать вклад с реальным значением поля.
# Выход: data/shap_flow.parquet – одна строка на карточку.
import os, sys
KIT = os.path.abspath(sys.argv[1]); BI = os.path.abspath(sys.argv[2])
sys.path.insert(0, KIT)
import course_ml  # noqa: F401 – LightGBM раньше pyarrow
from course_ml import data as data_mod
from course_ml.inference import CourseModel
import pandas as pd

GEN = os.path.join(BI, 'data', '5', 'Parquet', 'Gen')
SRC = os.path.join(BI, 'data', '2_5', 'Parquet', 'Gen', 'iavd_ml_dataset.parquet')
VALS = ['src_is_private', 'len_Значение Инцидента (группировка)', 'len_Значение Атаки (группировка)', 'Создан']
CARD = ['Наименование', 'Тип инцидента', 'Пиктограмма IRP', 'Критичность', 'Вендор из сием', 'Hostname атакующего',
        'Hostname Цели', 'src', 'dst', 'Описание инцидента', 'Вердикт', 'Решение']

model = CourseModel(os.path.join(KIT, 'models', 'lgbm_gate1__v001'))
sh = pd.read_parquet(os.path.join(GEN, 'fact_shap.parquet'), columns=['entity_id', 'split', 'source_column', 'shap', 'base'])
sh = sh[sh['split'] == 'productive_20584']
phi = sh.pivot_table(index='entity_id', columns='source_column', values='shap', observed=True)
phi.columns = ['phi__' + str(c) for c in phi.columns]
base = sh.groupby('entity_id')['base'].first().rename('base')

sc = pd.read_parquet(os.path.join(GEN, 'fact_score.parquet'))
sc = sc[sc['split'] == 'productive'].set_index('entity_id')['score']
cards = pd.read_parquet(os.path.join(GEN, 'dim_cards.parquet')).set_index('EntityId')[['y']]

raw = pd.read_parquet(SRC)
raw['EntityId'] = raw['EntityId'].astype(str)
raw = raw[raw['EntityId'].isin(phi.index)].drop_duplicates('EntityId')
prep, _, _ = data_mod.prepare(raw, model.cfg)
prep.index = raw['EntityId'].to_numpy()
vals = prep[VALS].add_prefix('val__')
info = raw.set_index('EntityId')[CARD].astype(str).replace({'None': '', 'nan': ''})
info.columns = ['card__' + c for c in CARD]

df = phi.join(base).join(sc).join(cards).join(vals).join(info)
df.index.name = 'EntityId'
os.makedirs('data', exist_ok=True)
df.reset_index().to_parquet(os.path.join('data', 'shap_flow.parquet'), index=False)
print(f'карточек {len(df)}, с вердиктом {int(df["y"].notna().sum())}, столбцов {df.shape[1]} → data/shap_flow.parquet')
