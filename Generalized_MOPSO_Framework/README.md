# Generalized Multi-Objective Particle Swarm Optimization (MOPSO) Framework

A comprehensive Python framework for multi-objective optimization using Particle Swarm Optimization (PSO) with Artificial Neural Network (ANN) surrogate models. This framework is designed to optimize complex engineering systems by leveraging pre-trained neural network models as objective function surrogates.

## Overview

The Generalized MOPSO framework addresses complex multi-objective optimization problems where traditional optimization methods are inadequate due to:
- **Computationally expensive objective functions**
- **Multiple competing objectives**
- **Complex constraint handling**
- **Mixed discrete-continuous parameter spaces**
- **Nonlinear, non-convex design spaces**

The framework uses pre-trained DNN models (generated from the ANN training pipeline) as surrogate models to enable efficient multi-objective optimization with sophisticated constraint handling and early termination capabilities.

## Key Features

### 1. Multi-Objective Optimization
- **Flexible Objective Configuration**: Support for any number of objectives (minimize or target-specific values)
- **Weighted Objective Combination**: Intelligent weight calculation based on target preferences
- **Surrogate Model Integration**: Uses pre-trained DNNs for fast objective evaluation
- **Performance Tracking**: Comprehensive history tracking for all objectives

### 2. Advanced Constraint Handling
- **Flexible Constraint Definition**: Min/max constraints for each objective
- **Penalty-Based Enforcement**: Customizable penalty weights for constraint violations
- **Feasibility Tracking**: Real-time monitoring of solution feasibility
- **Violation Reporting**: Detailed constraint violation analysis

### 3. Intelligent Parameter Handling
- **Mixed Parameter Types**: Support for both continuous and discrete parameters
- **Automatic Mapping**: Seamless conversion between optimization variables and actual parameter values
- **Bound Enforcement**: Automatic parameter bound handling during optimization
- **Smart Initialization**: Parameter space exploration with proper bounds

### 4. Early Termination System
- **Convergence Detection**: Automatic termination when optimization stagnates
- **Resource Management**: Time-based and iteration-based stopping criteria
- **Efficiency Optimization**: Prevents unnecessary computation while ensuring convergence
- **Adaptive Thresholds**: Configurable convergence criteria for different problem types

### 5. Comprehensive Results Analysis
- **Multi-Format Output**: Excel, JSON, and visualization outputs
- **Convergence Visualization**: Real-time and post-optimization plotting
- **Constraint Analysis**: Detailed feasibility and violation reporting
- **Performance Statistics**: Comprehensive optimization metrics and efficiency analysis

## Prerequisites

### Software Requirements
```bash
pip install numpy pandas pyswarm scikit-learn tensorflow joblib matplotlib tqdm openpyxl
```

### Input Files Required
The optimizer requires pre-trained models from the ANN training pipeline to be stored in a single folder:

**For each objective (e.g., "T_GPU", "C_total"):**
- `{objective_name}_trained_model.keras` - Trained TensorFlow/Keras model
- `{objective_name}_scaler_X.joblib` - Input feature scaler
- `{objective_name}_scaler_y.joblib` - Target variable scaler
- `{objective_name}_metadata.json` - Model metadata (optional to give manual inputs)

**File Organization:**
```
model_directory/
├── T_GPU_trained_model.keras
├── T_GPU_scaler_X.joblib
├── T_GPU_scaler_y.joblib
├── C_total_trained_model.keras
├── C_total_scaler_X.joblib
└── C_total_scaler_y.joblib
```

## Algorithm Framework

### Particle Swarm Optimization (PSO) Core

#### Particle Dynamics
Each particle i maintains:
- **Position**: $x_i(t) ∈ ℝ^n$ (current solution)
- **Velocity**: $v_i(t) ∈ ℝ^n$ (search direction and speed)
- **Personal Best**: $p_i$ (best position found by particle i)
- **Global Best**: $g$ (best position found by entire swarm)

#### Velocity Update Equation
```math
v_i(t+1) = w·v_i(t) + c₁·r₁·(p_i - x_i(t)) + c₂·r₂·(g - x_i(t))
```
where:
- w = 0.5: inertia weight (controls exploration vs exploitation)
- c₁ = c₂ = 2.0: acceleration coefficients
- r₁, r₂ ~ U(0,1): random numbers

#### Position Update
```math
x_i(t+1) = x_i(t) + v_i(t+1)
```

### Multi-Objective Function Formulation

#### Objective Evaluation
For each particle position x, the framework evaluates:

1. **Parameter Conversion**: x → actual parameter values
2. **Model Prediction**: Use pre-trained DNNs to predict objective values
3. **Constraint Evaluation**: Check constraint violations
4. **Weight Calculation**: Compute objective-specific weights

#### Combined Objective Function
```math
f_{combined}(x) = Σᵢ wᵢ·f̂ᵢ(x) + Σⱼ Pⱼ(x)
```
where:
- wᵢ: weight for objective i
- f̂ᵢ(x): normalized prediction from DNN i  
- Pⱼ(x): penalty for constraint j violation

#### Weight Calculation

**For Minimization Objectives:**
```math
w_i = 1 - min(1, f̂ᵢ(x)/f_{i,max})
```

**For Target Objectives:**
```math
w_i = 1 - min(1, |f̂ᵢ(x) - f_{i,target}|/f_{i,max})
```

#### Constraint Penalty Function
```math
P_j(x) = λⱼ · (violation_amount/constraint_value)²
```
where λⱼ is the user-defined penalty weight.

### Early Termination Algorithm

#### Convergence Detection
The framework tracks a convergence window of recent best values:
```python
convergence_window = [f_best(t-49), ..., f_best(t)]  # Last 50 iterations
```

#### Stagnation Criteria
Optimization terminates early when:
```math
max(convergence_window) - min(convergence_window) ≤ ε_{threshold}
```
for `stagnation_limit` consecutive iterations.

**Default Parameters:**
- $ε_{threshold}$ = 1×10⁻⁵
- stagnation_limit = 50 iterations

## Usage Guide

### 1. Environment Setup
```python
optimizer = GeneralizedMOPSO()
optimizer.setup_environment()
```

The setup process involves:
1. **Directory Path**: Specify location of pre-trained models
2. **Objectives Definition**: Number and names of optimization objectives
3. **Parameters Setup**: Define input parameters (continuous/discrete)
4. **Model Loading**: Automatic loading of DNNs and scalers
5. **Targets & Constraints**: Configure optimization goals and constraints

### 2. Example Configuration Session

```
=== GENERALIZED MULTI-OBJECTIVE PSO OPTIMIZER ===

Enter the directory path for loading model and scalers: C:/Users/models/DNN

--- OPTIMIZATION OBJECTIVES SETUP ---
Enter number of objectives to optimize: 2
Enter name for objective 1: T_GPU
Enter name for objective 2: C_total

--- INPUT PARAMETERS SETUP ---
Enter number of input parameters: 5

Parameter 1: Fin_shape (discrete: 1, 2, 3) [1 = Conical, 2 = Cylindrical and 3 = Square]
Parameter 2: Fan_speed (continuous: 500-4200)
Parameter 3: Fin_material (discrete: 1, 2, 3) [1 = Aluminium, 2 = Silicon Carbide and 3 = Copper]  
Parameter 4: NL (continuous: 10-70)
Parameter 5: NT (continuous: 10-45)

--- OPTIMIZATION TARGETS & CONSTRAINTS SETUP ---
T_GPU: minimize with max constraint ≤ 90°C
C_total: minimize (no constraints)
```

### 3. Run Optimization
```python
best_solution, best_value = optimizer.run_optimization(
    swarmsize=20,
    maxiter=3000
)
```

### 4. Results Analysis
```python
# Save comprehensive results
optimizer.save_results()

# Generate visualizations
optimizer.plot_convergence()
optimizer.plot_actual_objective_histories()

# Print detailed results
optimizer.print_final_results(best_solution, best_value)
```

## Input Parameters Configuration

### Continuous Parameters
```
Parameter: Fan_speed
Type: Continuous
Bounds: [500, 4200]
```

### Discrete Parameters
```
Parameter: Fin_shape  
Type: Discrete
Values: [1, 2, 3]
Optimization bounds: [0, 2] (mapped internally)
```

## Optimization Targets & Constraints

### Target Types

#### Minimization Objective
```python
objective_config = {
    'type': 'minimize',
    'max_value': 111.064  # For normalization
}
```

#### Target-Specific Objective
```python
objective_config = {
    'type': 'target',
    'target_value': 75.0,
    'max_value': 111.064
}
```

### Constraint Configuration

#### Constraint Setup
```python
constraint_config = {
    'has_constraints': True,
    'min_constraint': None,      # No minimum limit
    'max_constraint': 90.0,      # Maximum temperature
    'penalty_weight': 10000.0    # High penalty for violations
}
```

#### Constraint Penalty Calculation
```math
penalty = \lambda \times \left(\frac{\mathrm{violation}}{\mathrm{constraint\ value}} \right)^2
```

**Example**: For T_GPU > 90°C with actual value = 95°C:
```math
penalty = 10000 \times \frac{(95-90)}{90^2} = 10000 \times \frac{5}{90^2} = 30.86
```

## Early Termination System

### Convergence Monitoring
The framework maintains a sliding window of recent objective values and monitors for stagnation:

```python
def _check_convergence(self, current_value):
    self.convergence_window.append(current_value)
    
    if len(self.convergence_window) > self.stagnation_limit:
        self.convergence_window.pop(0)
    
    if len(self.convergence_window) >= self.stagnation_limit:
        variation = max(self.convergence_window) - min(self.convergence_window)
        return variation <= self.convergence_threshold
    
    return False
```

### Termination Benefits
- **Computational Efficiency**: Prevents unnecessary iterations
- **Resource Management**: Saves time and computational resources  
- **Automatic Convergence**: No manual monitoring required
- **Flexible Thresholds**: Adjustable for different problem complexities

### Configuration Options
```python
# Default settings
optimizer.convergence_threshold = 1e-5
optimizer.stagnation_limit = 50

# Custom settings
optimizer.convergence_threshold = 1e-6  # More strict
optimizer.stagnation_limit = 100       # More patient
```

## Output Files and Results

### Comprehensive Results File
`MOPSO_Results_{objectives}.xlsx` contains:

**Optimization_Results Sheet:**
- Iteration number
- Parameter values at each iteration
- Actual objective values (unscaled)
- Normalized weighted objective values
- Constraint violation status and amounts
- Combined objective value
- Feasibility status

**Termination_Info Sheet:**
- Termination reason and status
- Total iterations completed
- Convergence parameters used
- Final best value achieved

### Visualization Outputs

#### 1. Convergence Plot (`convergence.jpg`)
- **Combined objective value** progression over iterations
- **Constraint penalty** tracking
- **Early termination indicator** if applicable
- **Feasibility region** highlighting

#### 2. Objective Histories (`actual_objective_histories.jpg`)
- **Individual objective values** (unscaled, actual values)
- **Separate subplot** for each objective
- **Early termination markers**
- **Trend analysis** for each objective

### Performance Metrics

#### Final Results Summary
```
FINAL OPTIMIZATION RESULTS
==========================================================

Termination Information:
  Status: Early Terminated / Completed All Iterations
  Reason: Early termination: No significant improvement for 50 consecutive iterations
  Total iterations: 847 (out of 3000 max)
  Convergence threshold: 1e-05
  Stagnation limit: 50 iterations

Best Parameter Values:
  Fin_shape: 3
  Fan_speed: 2847.3
  Fin_material: 2
  NL: 45.7
  NT: 32.1

Best Combined Objective Value: 0.234567

Final Objective Values:
  T_GPU: 78.45°C
  C_total: 2341.2

Constraint Status:
  Total Penalty: 0.0000
  Solution is FEASIBLE

Optimization Statistics:
  Total iterations: 847
  Feasible solutions: 623 (73.6%)
  Infeasible solutions: 224 (26.4%)
  Early termination efficiency: 71.8% iterations saved
```

## Advanced Features

### 1. Mixed Parameter Space Handling
The framework seamlessly handles optimization problems with both continuous and discrete parameters:

```python
def _convert_parameters(self, x):
    """Convert optimization variables to actual parameter values"""
    param_values = {}
    
    for i, param_name in enumerate(self.parameter_info.keys()):
        if self.parameter_info[param_name]['type'] == 'continuous':
            param_values[param_name] = x[i]
        else:  # discrete
            idx = int(np.round(np.clip(x[i], 0, len(discrete_values)-1)))
            param_values[param_name] = discrete_values[idx]
    
    return param_values
```

### 2. Robust Model Integration
Automatic loading and validation of pre-trained models:

```python
def _load_models_and_scalers(self):
    """Load pre-trained models and their scalers from files"""
    for obj_name in self.objectives:
        # Load input/output scalers
        self.scalers_X[obj_name] = joblib.load(f"{obj_name}_scaler_X.joblib")
        self.scalers_y[obj_name] = joblib.load(f"{obj_name}_scaler_y.joblib")
        
        # Load trained model
        self.models[obj_name] = tf.keras.models.load_model(f"{obj_name}_trained_model.keras")
```

### 3. Intelligent Constraint Handling
Flexible constraint system with customizable penalties:

```python
def calculate_constraint_penalties(self, objectives):
    """Calculate constraint penalties for each objective"""
    penalties = []
    total_penalty = 0.0
    
    for i, obj_name in enumerate(self.objectives):
        constraint_config = self.objective_constraints[obj_name]
        actual_value = objectives[i]
        penalty = 0.0
        
        if constraint_config['has_constraints']:
            # Check minimum constraint
            if constraint_config['min_constraint'] is not None:
                if actual_value < constraint_config['min_constraint']:
                    violation = constraint_config['min_constraint'] - actual_value
                    penalty += constraint_config['penalty_weight'] * (violation/constraint_config['min_constraint'])**2
            
            # Check maximum constraint  
            if constraint_config['max_constraint'] is not None:
                if actual_value > constraint_config['max_constraint']:
                    violation = actual_value - constraint_config['max_constraint']
                    penalty += constraint_config['penalty_weight'] * (violation/constraint_config['max_constraint'])**2
        
        penalties.append(penalty)
        total_penalty += penalty
        
    return penalties, total_penalty
```

## Performance and Scalability

### Computational Complexity
- **Per Iteration**: O(S × N × M) where S=swarm size, N=parameters, M=objectives
- **Model Evaluation**: O(1) using pre-trained DNNs (vs. expensive simulations)
- **Memory Usage**: Linear with swarm size and iteration count
- **Parallel Potential**: Each particle evaluation is independent

### Efficiency Optimizations
1. **Early Termination**: Prevents unnecessary iterations (up to 70%+ savings)
2. **Surrogate Models**: Fast DNN evaluation vs. expensive simulations
3. **Vectorized Operations**: NumPy-based calculations for performance
4. **Memory Management**: Efficient storage of optimization history

### Scalability Considerations
- **Parameter Count**: Scales well with increasing problem dimensions
- **Objective Count**: Linear scaling with number of objectives
- **Constraint Complexity**: Flexible constraint handling without performance penalty
- **Swarm Size**: Linear computational scaling

## Troubleshooting and Best Practices

### Common Issues

#### 1. File Loading Problems
```
Error: No file found containing 'T_GPU_trained_model' in directory
```
**Solution**: Ensure all required files are present and follow naming convention:
- `{objective}_trained_model.keras`
- `{objective}_scaler_X.joblib`  
- `{objective}_scaler_y.joblib`

#### 2. Parameter Name Mismatches
```
Error: Parameter 'fan_speed' not found in model input features
```
**Solution**: Use exact parameter names from ANN training (check JSON metadata)

#### 3. Constraint Violations
```
Warning: No feasible solutions found during optimization
```
**Solutions**:
- Relax constraint bounds
- Reduce penalty weights  
- Expand parameter search space
- Increase maximum iterations

#### 4. Slow Convergence
```
Optimization running for extended time without convergence
```
**Solutions**:
- Increase convergence threshold (e.g., 1e-4 instead of 1e-5)
- Reduce stagnation limit for earlier termination
- Check objective function scaling
- Verify model prediction quality

### Best Practices

#### 1. Model Quality Assurance
- Verify DNN model performance on test data before optimization
- Check model metadata for feature scaling information
- Validate prediction ranges against expected objective bounds

#### 2. Parameter Space Design
- Set realistic bounds based on physical constraints
- Use appropriate discretization for categorical parameters
- Consider parameter interactions when setting bounds

#### 3. Constraint Configuration
- Start with relaxed constraints and gradually tighten
- Use moderate penalty weights (1000-10000) initially
- Monitor feasibility rates during optimization

#### 4. Convergence Settings
- Use default settings initially (threshold=1e-5, stagnation=50)
- Adjust based on problem complexity and time constraints
- Monitor convergence plots for appropriate threshold selection

#### 5. Results Validation
- Verify optimal solutions using original models/simulations
- Check constraint satisfaction in final results
- Analyze objective trade-offs in multi-objective scenarios

## Integration with ANN Training Pipeline

### Workflow Integration
```
1. Data Collection → 2. ANN Training → 3. Model Generation → 4. MOPSO Optimization
```

### File Dependencies
The MOPSO optimizer directly uses outputs from the ANN training pipeline:

**ANN Pipeline Outputs:**
```
model_directory/
├── {objective}_trained_model.keras     # ← Required by MOPSO
├── {objective}_scaler_X.joblib         # ← Required by MOPSO  
├── {objective}_scaler_y.joblib         # ← Required by MOPSO
├── {objective}_metadata.json           # ← Optional for MOPSO
└── training_results.xlsx               # ← Not used by MOPSO
```

### Parameter Consistency
Ensure parameter names match between ANN training and MOPSO:

**ANN Training Input Features:**
```python
input_features = ['Fin_shape', 'Fan_speed', 'Fin_material', 'NL', 'NT']
```

**MOPSO Parameter Names:**
```python
parameter_names = ['Fin_shape', 'Fan_speed', 'Fin_material', 'NL', 'NT']  # Must match exactly
```

### Multi-Objective Model Requirements
For each optimization objective, ensure you have:
1. Separate ANN models trained for each target variable
2. Individual scalers for inputs and outputs  
3. Consistent parameter names across all models
4. Compatible parameter bounds and types

## Example Applications

### 1. Electronic Cooling System Optimization
**Objectives:**
- Minimize GPU temperature (T_GPU)
- Minimize total cost (C_total)

**Parameters:**
- Fin_shape: Discrete (conical=1, cylindrical=2, square=3)
- Fan_speed: Continuous (500-4200 RPM)
- Fin_material: Discrete (aluminum=1, SiC xtal=2, copper=3)
- Number of longitudinal fins (NL): Continuous (10-70)
- Number of transverse fins (NT): Continuous (10-45)

**Constraints:**
- T_GPU ≤ 90°C (thermal limit for GPUs)
- C_total: unconstrained minimization

### 2. Multi-Physics System Design
**Objectives:**
- Minimize energy consumption
- Maximize performance metric
- Target specific temperature

**Parameters:**
- Geometric dimensions (continuous)
- Material selections (discrete)
- Operating conditions (continuous)
- Control parameters (mixed)

## Theoretical Background

### Multi-Objective Optimization Theory

#### Pareto Optimality
A solution $x^*$ is Pareto optimal if there exists no other solution x such that:
- $f_i(x) ≤ f_i(x^*)$ for all objectives i
- $f_j(x) < f_j(x^*)$ for at least one objective j

The MOPSO framework approximates Pareto optimal solutions through weighted objective combination.

#### Weighted Sum Approach
```math
minimize: f_{combined}(x) = Σᵢ wᵢ · f_i(x)\
subject\ to: g_j(x) ≤ 0, j = 1,...,m
           h_k(x) = 0, k = 1,...,p
```

### Particle Swarm Optimization Theory

#### Swarm Intelligence Principles
1. **Social Learning**: Particles learn from global best solution
2. **Cognitive Learning**: Particles remember personal best experiences  
3. **Velocity Persistence**: Inertia maintains search momentum
4. **Stochastic Elements**: Random factors prevent premature convergence

#### Convergence Properties
- **Global Convergence**: Under certain conditions, PSO converges to global optimum
- **Exploration vs Exploitation**: Balanced through inertia weight and acceleration coefficients
- **Population Diversity**: Maintained through swarm-based search

### Surrogate Model Integration

#### Model-Based Optimization
Using pre-trained DNNs as surrogate models provides:
- **Computational Efficiency**: Fast evaluation vs. expensive simulations
- **Smooth Approximation**: Continuous, differentiable objective functions
- **Noise Reduction**: Filtered, consistent objective evaluations
- **Scalability**: Handles high-dimensional parameter spaces

#### Metamodel Accuracy Considerations
- **Model Fidelity**: Surrogate accuracy affects optimization quality
- **Training Data Coverage**: Model valid within training parameter ranges
- **Interpolation vs Extrapolation**: Best performance within training bounds
- **Model Uncertainty**: Consider prediction confidence in optimization

## Future Extensions and Enhancements

### Potential Improvements
1. **Multi-Swarm Implementation**: Parallel swarm populations for better exploration
2. **Adaptive Parameters**: Dynamic inertia weight and acceleration coefficient adjustment
3. **Pareto Frontier Approximation**: Explicit multi-objective Pareto set generation
4. **Uncertainty Quantification**: Model prediction uncertainty integration
5. **Hybrid Optimization**: Combination with gradient-based local search
6. **Real-Time Optimization**: Online model updating and optimization
7. **Multi-Fidelity Models**: Integration of models with different accuracy levels

### Advanced Constraint Handling
1. **Constraint Aggregation**: Multiple constraint combination strategies
2. **Adaptive Penalties**: Dynamic penalty weight adjustment
3. **Constraint Relaxation**: Automatic constraint bound adjustment
4. **Feasible Region Mapping**: Explicit feasible space characterization

