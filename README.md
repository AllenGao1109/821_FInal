# 🏥 EHR Data Analysis Toolkit

A flexible and modular Python toolkit for Electronic Health Record (EHR) data preprocessing, filtering, PCA analysis, and machine learning (LightGBM & XGBoost) modeling.

---

## 📦 Features

- ✅ Categorical variable support (prefix `cat__`)
- ✅ NaN handling (remove or impute)
- ✅ Flexible filtering and selection
- ✅ Range-based subgroup selection
- ✅ PCA analysis and visualization
- ✅ LightGBM and XGBoost modeling
- ✅ Hyperparameter tuning (grid or Bayesian)
- ✅ One-hot encoding for categorical variables
- ✅ Output visualization

---

## 📁 Standardized Input Format

- Categorical variables must be **prefixed with `cat__`**, e.g., `cat__gender`.
- NaNs are recognized as either `NaN` or `"?"`.

---

## 🔧 1. Data Preprocessing

```python
EHR_preprocessing(dataset, remove = True, impute = False)
```

**Parameters**  
- `remove`: (bool) Remove rows with missing data  
- `impute`: (str | bool) Imputation strategy:  
  - `"mean"`: Fill with column mean  
  - `"largest"`: Fill with column max  
  - `"smallest"`: Fill with column min  
  - `"knn5"`: Fill using KNN (k=5)  

**Example**  
```python
cleaned_df = HER_preprocessing(raw_df, remove=False, impute="mean")
```


## 🔍 2. Filtering and Selection

### Filter and select specific variables
```python
Demo_filter(variablelist)
Demo_select(variablelist)
```

**Example**  
```python
filtered_df = Demo_filter(['age','race','diabetes'])
selected_df = Demo_select(['age','race','diabetes'])
```


### Filter by range
```python
Demo_range(variable, range)
```

**Example**
```
# continuous variables
filtered_df = Demo_range("age", "40,60")    # get those age range from 40 - 60, 60 is not included

# categorical variables
filtered_df = Demo_range("race","white hispano") # get those race are white or hispano
```
- `numeric values`: use , to split the interval
- `categories`: provide space-separated string of desired levels



## 🔬 3. PCA Analysis
```python
EHR_PCA(dataset, num = 20, plot = True) -> list
```
**Parameters**  
- `num`: number of principal components to retain
- `plot`: Generate visualizations if True

**Returns**

List of most important variables contributing to PCA

**Example**  
```python
principal_components = EHR_PCA(df, 20, 0)
```


## 🌲 4. LightGBM Modeling
```python
EHR_LGBM(dataset, y_variable, plot = True, hyperparameters = None) -> dict
```

**Behavior**

- hyperparameters=None: Performs grid search/Bayesian optimization with CV

- hyperparameters provided: Uses specified parameters directly

- Automatic one-hot encoding for categorical variables

- Generates performance visualization when plot=True


**Returns**

A dictionary that stores model data, sorted important lists, and plot data

```python
return {'model': trained_model, 
        'feature_importance":feature_importance,
        "plot': plot_figures}
```


## 🚀 5. XGBoost Modeling

```python
EHR_XGB(dataset, y_variable, hyperparameters=None, plot=True)
```

**Same** interface and behavior as EHR_LGBM(), but uses XGBoost backed

