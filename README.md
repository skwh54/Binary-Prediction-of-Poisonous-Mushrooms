# Binary Prediction of Poisonous Mushrooms

This project tackles the binary classification task of determining whether a mushroom is poisonous or edible, using **AutoGluon** on Kaggle's [Playground Series - S4E8](https://www.kaggle.com/competitions/playground-series-s4e8) dataset.

 **Kaggle Ranking**: 13th / 2,422 (Top 0.5%)  
 **Framework**: AutoGluon Tabular (with GPU & stacking)  
 **Model Objective**: Binary classification (`edible` vs. `poisonous`) using MCC as the main evaluation metric.

---

##  Overview

This solution leverages **AutoGluon TabularPredictor** with advanced settings (`best_quality`, `auto_stack`, `refit_full`) to create an ensemble of models optimized for Matthews Correlation Coefficient (MCC). It also enriches the training dataset by merging external mushroom data.

---

##  Data

- **train.csv**: Primary training data from Kaggle (with `id` column dropped)
- **secondary_data.csv**: External mushroom dataset merged with train data
- **test.csv**: Test data for final prediction (with `id` column dropped)

---

##  Model Setup

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label='class',
    eval_metric='mcc',
    problem_type='binary'
).fit(
    train_data,
    presets='best_quality',
    time_limit=36000,  
    ag_args_fit={'num_gpus': 2},
    auto_stack=True,
    refit_full=True
)
