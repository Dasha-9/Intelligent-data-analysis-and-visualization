# Данные

Файл `shap_flow.parquet` в репозиторий не загружается. Его готовит `python/prep_shap_flow.py` из двух комплектов практики 3.7:

- `IAVD_3.7_BI_team/data/5/Parquet/Gen/` – `fact_shap.parquet` (срез `productive_20584`), `fact_score.parquet`, `dim_cards.parquet`;
- `IAVD_3.7_BI_team/data/2_5/Parquet/Gen/iavd_ml_dataset.parquet` – исходные поля карточек;
- `IAVD_3.7_ML_team/` – модель `lgbm_gate1__v001` и код `course_ml`: им считаются значения сконструированных полей.

Одна строка – одна карточка: 73 вклада `phi__*`, `base`, `score`, `y`, значения полей `val__*` и поля карточки `card__*`.
