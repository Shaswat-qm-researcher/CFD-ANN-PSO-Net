# Artificial Neural Network Training Pipeline for Complex System Optimization

The code provides a comprehensive Python framework designed for building, training, and evaluating Artifical Neural Networks (ANNs) with automated hyperparameter optimization using GridSearchCV and MLPRegressor comparison. The pipline is specifically engineered for complex engineering applications requiring sophisticated nonlinear regression capabilities.

## Overview

This framework addresses the computational challenges inherent in modeling complex engineering systems where traditional analytical approaches prove inadequate. The pipeline implements a dual-model architecture combining scikit-learn's MLPRegressor baseline with custom TensorFlow/Keras Deep Neural Networks (DNNs), enabling comprehensive performance comparison and robust model selection for optimization tasks. 

### Core Engineering Applications

The framework is particularly suited for complex systems characterization and optimization, including:

**Electronic Cooling Systems:**
- Heat sink thermal performance modeling
- Multi-parameter optimization of cooling geometries
- Thermal resistance prediction under varying operating conditions
- Fan-fin interaction modeling for electronic component cooling

**Multi-Physics Engineering Problems:**
- Coupled heat and mass transfer systems
- Fluid-structure interaction modeling
- Multi-objective design optimization scenarios
- Parameter sensitivity analysis in complex geometries

**System-Level Optimization:**
- Design space exploration with limited experimental data
- Surrogate model development for computationally expensive simulations
- Real-time control system parameter tuning
- Inverse design problems requiring high-dimensional mapping

### Advanced Computational Features

The framework integrates state-of-the-art machine learning methodologies with engineering-specific requirements to provide an end-to-end solution for neural network regression tasks, featuring:


#### Data Management and Preprocessing
- **Intelligent Data Interface:** Multi-format compatibility (CSV, Excel, TXT) with automated missing value detection and interactive column selection
- **Adaptive Normalization:** StandardScaler implementation ensuring optimal neural network convergence across diverse engineering parameter ranges
- **Dynamic Test-Set Allocation:** Sample-size dependent train-test splitting optimized for small-to-medium engineering datasets

#### Neural Architecture Optimization
- **Dual-Model Framework:** Comparative analysis between scikit-learn MLPRegressor baseline and custom TensorFlow/Keras implementations
- **Topology-Aware Architecture Selection:** Dynamic network depth and width optimization based on input dimensionality and dataset complexity
- **Parameter Space Stratification:** Hierarchical hyperparameter grids organized by network complexity (shallow, medium, deep, very deep architectures)

#### Advanced Training and Optimization
- **Multi-Modal Hyperparameter Search:** Both manual specification and automated GridSearchCV optimization with intelligent parallel processing
- **Robust Convergence Protocols:** Multi-criteria termination including validation loss thresholds, wall-clock time limits, early stopping, and maximum iteration bounds
- **Overfitting Prevention:** Statistical gap analysis with automated model filtering based on train-test performance differentials

#### Validation and Reliability Assessment
- **Comprehensive Cross-Validation:** 5-fold stratified validation with multi-metric evaluation (MAE, RMSE, R²) and confidence interval estimation
- **Model Reliability Quantification:** Statistical significance testing and performance stability analysis across validation folds
- **Comparative Model Assessment:** Systematic evaluation between competing architectures with statistical hypothesis testing

#### Production Deployment Infrastructure
- **Complete Model Serialization:** TensorFlow/Keras model persistence with associated preprocessing scalers and comprehensive metadata tracking
- **Engineering Documentation:** Automated generation of model specifications, performance metrics, and deployment parameters
- **Industrial Integration Support:** JSON metadata export and Excel-based reporting for seamless integration into engineering workflows




## Dataset Example

The pipeline was tested with `IJMS_HS_data.xlsx` containing heat sink optimization data:

**Original Columns:**
- `Fin_shape`: Fin geometry type
- `Fan_speed`: Fan speed (RPM)
- `Fin_material`: Material type
- `NL`: Number of longitudinal fins
- `NT`: Number of transverse fins
- `T GPU`: GPU temperature (°C) - Target variable
- `Total Cost`: Manufacturing cost

**Used Configuration:**
- **Input Features (X):** Fin_shape, Fan_speed, Fin_material, NL, NT
- **Target Variable (y):** T GPU (°C)
- **Dataset Size:** 30 samples (24 LHS + 8 extrapolated samples)
- **Test Split:** 25% (based on dataset size)

## Key Features

### 1. Interactive Data Loading
- Supports multiple file formats (CSV, Excel, TXT)
- Automatic missing value detection
- Interactive column selection and dropping
- User-friendly file path specification

### 2. Automated Hyperparameter Optimization
- **GridSearchCV** with MLPRegressor for initial optimization
- Multiple parameter grids based on network complexity:
  - **Shallow Networks:** Single/double layer architectures
  - **Medium Networks:** Moderate complexity
  - **Deep Networks:** Multi-layer architectures
  - **Very Deep Networks:** High complexity architectures
- 5-fold cross-validation with multiple scoring metrics (MAE, RMSE, R²)
- Overfitting detection and filtering

### 3. Advanced DNN Implementation
- **Architecture:** Sequential model with PReLU activation
- **Regularization:** L2 regularization (0.0001)
- **Optimizer:** AdamW with weight decay
- **Multiple Termination Conditions:**
  - Maximum epochs (1500)
  - Validation loss threshold (1e-6)
  - Wall-clock time limit (2 hours)
  - Early stopping on validation plateau

### 4. Comprehensive Evaluation
- Training and test performance metrics
- Actual vs. Predicted visualizations
- Loss curve plotting
- 5-fold cross-validation analysis
- Model comparison (MLP vs. DNN)

## Installation Requirements

```bash
pip install --user numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 joblib==1.2.0 tensorflow==2.15.0 tqdm==4.65.0 scikit-learn==1.5.1 openpyxl==3.0.10
```

## Usage

### 1. Prepare Your Data
Ensure your dataset is in Excel (.xlsx), CSV (.csv), or TXT format with:
- No missing values
- Proper column headers
- Numeric data for regression

### 2. Run the Pipeline
```python
python ann_pipeline.py
```

### 3. Interactive Configuration
The script will guide you through:
- File location and name specification
- Column selection (inputs and target)
- Target variable units
- Save location preferences
- Hyperparameter optimization choice (manual/auto)

### 4. Example Session
```
Do you want to load the file from your home directory? (y/n): y
Enter the file name: IJMS_HS_data.xlsx
Enter input (X) column names: Fin_shape, Fan_speed, Fin_material, NL, NT
Enter the output (y) column name: T GPU
Enter the unit for the (Y) column: C
Would you like to manually enter hyperparameters or use automatic tuning? auto
```

## Output Files

The pipeline generates comprehensive outputs in the specified directory:

### Model Files
- `{savename}_trained_model.keras` - Trained DNN model
- `{savename}_scaler_X.joblib` - Input feature scaler
- `{savename}_scaler_y.joblib` - Target variable scaler
- `{savename}_metadata.json` - Model metadata

### Results and Visualizations
- `{savename}_GridSearch_results.xlsx` - Complete GridSearch results
- `{savename}_GridSearch_Filtered_NoOverfit.xlsx` - Filtered results
- `{savename}_MLP_Actual_vs_Predicted_Data.jpg` - MLP performance plot
- `{savename}_DNN_Actual_vs_Predicted_Data.jpg` - DNN performance plot
- `{savename}_DNN_loss_vs_epoch.jpg` - Training loss curves
- `{savename}_Best5Fold_BarPlot_MLP.jpg` - Cross-validation results

### Prediction Files
- `{savename}_DNN_predicted_train_data_{target}.xlsx` - Training predictions
- `{savename}_DNN_predicted_test_data_{target}.xlsx` - Test predictions
- `{savename}_model_info.xlsx` - Model metadata summary

## Algorithm and Mathematical Framework

### Data Preprocessing

#### Feature Scaling
The pipeline employs StandardScaler for both input and output normalization:

**Input Features (X):**
```math
X_{scaled} = \frac{(X - μ_X)}{σ_X}
```
where $μ_X$ is the mean and $σ_X$ is the standard deviation of each feature.

**Target Variable (y):**
```math
y_{scaled} = \frac{(y - μ_y)}{σ_y}
```

This standardization ensures zero mean and unit variance, facilitating neural network convergence.

#### Train-Test Split
Data is partitioned using stratified sampling with adaptive test sizes:
- > < 20 samples: 15% test
- > 20-30 samples: 20% test  
- > 30-40 samples: 25% test
- > \> 40 samples: 30% test

### GridSearchCV Optimization

#### Parameter Space
The hyperparameter grid is stratified by network complexity:

**Architecture Groups:**
- **Shallow:** [(32,), (64,), (128,), (32,16), (64,32), (32,16,8)]
- **Medium:** [(512,), (1024,), (128,64), (256,128)]
- **Deep:** [(512,256), (128,64,32), (256,128,64,32)]
- **Very Deep:** [(512,256,128,64,32,16,8)]

- **Learning Rates:** log-spaced from 10⁻¹ to 10⁻³
- **Batch Sizes:** {2, 4, 8, 16, 32}
- **Epochs:** {100, 200, 300, 400, 500, 600, 800, 1000, 1500}

#### Cross-Validation Scoring
Multi-objective optimization using:

**Mean Absolute Error:**
```math
MAE = \frac{1}{n} \Sigma|y_i - ŷ_i|
```

**Root Mean Square Error:**
```math
RMSE = \sqrt{\left[\frac{1}{n} \Sigma{(y_i - ŷ_i)^2} \right]}
```

**Coefficient of Determination:**
```math
R² = 1 - \left(\frac{SS_{res}}{SS_{tot}}\right)
```
where $SS_{res} = Σ(y_i - ŷ_i)²$ and $SS_{tot} = Σ(y_i - ȳ)²$


#### Overfitting Detection
Models are filtered using train-test gap analysis:
```
Overfitting_Gap = R²_train - R²_test
```
Models with gap > 0.07 are excluded from final selection.

### Multi-Layer Perceptron (MLP) Baseline

#### Architecture
The MLP uses scikit-learn's MLPRegressor with:
- **Activation:** ReLU (Rectified Linear Unit)
- **Solver:** adam (Adaptive Moment Estimation)
- **Weight Initialization:** Xavier/Glorot uniform

#### Forward Propagation
For layer l with input x:
```math
z^{(l)} = W^{(l)}x^{(l-1)} + b^{(l)}
```
```math
a^{(l)} = ReLU(z^{(l)}) = max(0, z^{(l)})
```
#### Loss Function
Mean Squared Error for regression:
```math
L = \frac{1}{2n} Σ(y_i - ŷ_i)²
```

### Deep Neural Network (DNN) Implementation

#### Network Architecture
Sequential model with adaptive depth:
- **Input Layer:** Dense(```n_features```)
- **Hidden Layers:** Variable depth with decreasing neuron counts
- **Output Layer:** Dense(1) for regression

#### Activation Function: PReLU
Parametric ReLU with learnable parameter $α$:
```
PReLU(x) = {
  x,        if x > 0
  αx,       if x ≤ 0
}
```
where $α$ is learned during training for each neuron.

#### Weight Initialization
Glorot/Xavier uniform initialization:
```math
W \sim U\left(-\sqrt{\frac{6}{(n_in + n_out)}}\right), \left(\sqrt{\frac{6}{(n_in + n_out)}}\right) 
```

#### Regularization
L2 weight penalty:
```math
L_{reg} = \lambda \Sigma{W^2}
```
with $\lambda$ = 0.0001

#### Optimizer: AdamW
Adaptive learning rate with weight decay:

**Momentum updates:**
```math
m_t = \beta_1 m_{t-1} + (1-\beta_1)∇f(\theta_t)
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2)[∇f(\theta_t)]^2
```

**Bias correction:**
```math
\hat{m}_t = \frac{m_t}{(1-\beta_1^t)}
```
```math
\hat{v}_t = \frac{v_t}{(1-\beta_1^t)}
```

**Parameter update:**
```math
\theta_{t+1} = \theta_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{(\hat{v}_t + \epsilon)}}\right) - \lambda \theta_t
```

where:
- $\alpha$: learning rate
- $\beta_1$ = 0.9, β₂ = 0.999: momentum parameters
- $\epsilon$ = 1e-8: numerical stability
- $\lambda$ = 1e-5: weight decay coefficient

#### Training Termination Conditions

**1. Maximum Epochs:** Hard limit at 1500 iterations

**2. Validation Loss Threshold:**
```
if L_val < 1e-6 for patience_epochs: stop_training = True
```

**3. Wall-Clock Time:** Maximum 2-hour training duration

**4. Early Stopping:**
```
if L_val(t) ≥ L_val(t-patience): stop_training = True
```
with patience = 40 epochs

#### Loss Function
Mean Squared Error with gradient:
```math
L(\theta) = \frac{1}{2n} \Sigma (y_i -\hat{y}_i)^2
```
```math
\nabla L = -\frac{1}{n} \Sigma (y_i - \hat{y}_i)\nabla \hat{y}_i
```

### Model Evaluation Metrics

#### Performance Metrics
**Training and Test Evaluation:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)  
- Coefficient of Determination (R²)

#### Cross-Validation
5-fold stratified cross-validation with performance aggregation:
```
CV_score = (1/k) Σ score_i
CV_std = √[(1/k) Σ(score_i - CV_score)²]
```

### Statistical Analysis

#### Confidence Intervals
95% confidence intervals for cross-validation scores:
```
CI = CV_score ± 1.96 × (CV_std/√k)
```

#### Model Comparison
Statistical significance testing between MLP and DNN using paired t-test on cross-validation folds:
```
t = (μ_diff)/(s_diff/√n)
```

### Computational Complexity

#### Training Complexity
For a network with L layers and maximum width W:
- **Forward Pass:** O(LW²)
- **Backward Pass:** O(LW²)  
- **Total per epoch:** O(nLW²) for n samples

#### GridSearchCV Complexity
Total evaluations: |Hyperparameter_space| × k-folds
Parallel speedup: Linear with available CPU cores

### Convergence Analysis

#### Learning Rate Adaptation
AdamW provides adaptive per-parameter learning rates:
```
effective_lr_i = α × m̂_t,i / √(v̂_t,i + ε)
```

#### Convergence Criteria
Multiple stopping conditions ensure robust convergence:
1. Gradient norm < threshold
2. Loss improvement < minimum delta  
3. Resource constraints (time/epochs)
4. Validation performance plateau

## Model Architecture

### MLP Baseline
- Scikit-learn MLPRegressor
- GridSearchCV optimization
- Multiple architecture configurations
- 5-fold cross-validation

### Custom DNN
- TensorFlow/Keras Sequential model
- PReLU activation functions
- L2 regularization
- AdamW optimizer
- Advanced early stopping

## Advanced Features

### Overfitting Detection
- Automatic filtering of overfit models
- Train-test gap threshold (0.07)
- Best model selection from non-overfit candidates

### Robust Training
- Multiple termination conditions prevent infinite training
- Wall-clock time limits for resource management
- Validation threshold stopping for convergence
- Comprehensive logging and progress tracking

### Scalability
- Dynamic plot sizing based on dataset size
- CPU core utilization for parallel processing
- Memory-efficient data handling

## Customization Options

### Manual Hyperparameter Entry
- Interactive parameter specification
- Guided input validation
- Custom architecture design
- Fine-grained control over training parameters

### Automated Optimization
- GridSearchCV with comprehensive parameter space
- Multiple complexity levels
- Parallel processing support
- Overfitting prevention

## Troubleshooting

### Common Issues
1. **Missing Values:** Ensure dataset has no NaN values
2. **File Path Errors:** Use forward slashes or double backslashes
3. **Memory Issues:** Reduce batch size or network complexity
4. **Long Training:** Enable early stopping or reduce epochs

### Performance Optimization
- Use `-1` for maximum CPU cores in GridSearchCV
- Consider dataset size when selecting test split
- Monitor memory usage for large networks
- Use GPU acceleration if available

