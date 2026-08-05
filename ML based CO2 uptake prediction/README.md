
<h1>  CO₂ Uptake Prediction in Materials Porous  using Machine Learning</h1>

 <div class="section">
      <h2>✅ 1. Aim</h2>
      <p>
        To develop a <b>robust machine learning framework</b> for predicting <b>CO₂ uptake (wt%)</b> in Metal–Organic Frameworks (MOFs)
        using experimental data and physicochemical descriptors. The model will accelerate MOF screening and reduce reliance on expensive laboratory experiments.
      </p>
    </div>

<div class="section">
      <h2> 2. Objectives</h2>
      <ul>
        <li>Predict CO₂ uptake (wt%) from MOF properties and experimental conditions.</li>
        <li>Handle dataset challenges such as <b>limited size</b>, <b>skewed target distribution</b>, and <b>multiple measurements per MOF</b>.</li>
        <li>Compare multiple ML algorithms and identify the best-performing model.</li>
        <li>Perform <b>hyperparameter tuning</b> to optimize model performance.</li>
        <li>Analyze feature importance and interpret the model results.</li>
      </ul>
    </div>

 <div class="section">
      <h2> 3. Dataset</h2>
      <h3> Source</h3>
      <p>
        Experimental MOF adsorption data from an ACS journal dataset (589 measurements). Each row represents a unique adsorption experiment under different conditions.
      </p>
  <h3> Features</h3>
      <table>
        <thead>
          <tr>
            <th>Column</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Name</td><td>MOF name</td></tr>
          <tr><td>largest electronegativity diff</td><td>Max electronegativity difference</td></tr>
          <tr><td>T / K</td><td>Temperature in Kelvin</td></tr>
          <tr><td>P / bar</td><td>Pressure in bar</td></tr>
          <tr><td>Gas Mr</td><td>Gas molecular weight</td></tr>
          <tr><td>log(wt%)</td><td>Log-transformed uptake</td></tr>
          <tr><td>uptake Wt</td><td>CO₂ uptake (target variable)</td></tr>
        </tbody>
      </table>

  <h3> Key Facts</h3>
      <ul>
        <li><b>589 samples</b>, <b>304 unique MOFs</b></li>
        <li>Most measurements at <b>298 K</b> and <b>1 bar</b></li>
        <li>Dataset contains <b>commas instead of dots</b>, <b>punctuation</b>, and <b>missing values</b></li>
      </ul>
    </div>

 <div class="section">
      <h2>4. Data Preprocessing (Full Pipeline)</h2>

 <h3> 🔹Step 1 — Replace commas with dots (global)</h3>
      <div class="code">
df = df.applymap(lambda x: str(x).replace(',', '.'))
      </div>

<h3> 🔹Step 2 — Remove punctuation and invalid characters</h3>
      <div class="code">
import re

def clean_text(x):
    x = str(x)
    x = re.sub(r'[^\d\.\-]', '', x)  # keep only numbers, dot, minus
    return x

df = df.applymap(clean_text)
      </div>

      <h3> 🔹Step 3 — Convert all columns to numeric</h3>
      <div class="code">
df = df.apply(pd.to_numeric, errors='coerce')
      </div>

      <h3>🔹 Step 4 — Round numeric values</h3>
      <div class="code">
df = df.round(2)
      </div>
  <h3>🔹 Step 5 — Format numbers with commas (for display only)</h3>
      <div class="code">
df_display = df.copy()
for col in df_display.select_dtypes(include=['float', 'int']).columns:
    df_display[col] = df_display[col].map('{:,.2f}'.format)
      </div>
  <h3>🔹 Step 6 — Handle Missing Values</h3>
      <h4>Numerical columns — mean imputation</h4>
      <div class="code">
from sklearn.impute import SimpleImputer

num_cols = df.select_dtypes(include=['float', 'int']).columns
imputer_num = SimpleImputer(strategy='mean')
df[num_cols] = imputer_num.fit_transform(df[num_cols])
      </div>

  <h4>Categorical columns — most frequent</h4>
      <div class="code">
cat_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
      </div>

 <h3> Check Missing Values</h3>
      <div class="code">
print("Missing Values After Cleaning:\n", df.isnull().sum())
      </div>
    </div>

<div class="section">
      <h2> 5. Machine Learning Workflow</h2>
    <h3>🔹 Train-Test Split</h3>
      <ul>
        <li>Train: 70%</li>
        <li>Test: 30%</li>
      </ul>

  <h3>🔹 Models Evaluated</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Purpose</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Decision Tree</td><td>Baseline</td></tr>
          <tr><td>Random Forest</td><td>Strong tree ensemble</td></tr>
          <tr><td>Gradient Boosting</td><td>Best tree-based model</td></tr>
          <tr><td>XGBoost</td><td>Efficient boosting</td></tr>
          <tr><td>SVR</td><td>Kernel regression</td></tr>
          <tr><td>MLP</td><td>Neural network</td></tr>
        </tbody>
      </table>
    </div>

 <div class="section">
      <h2> 6. Feature Engineering</h2>
      <ul>
        <li>Added physics-based features: <b>log(P)</b>, <b>1/T</b>, <b>P × T</b></li>
        <li>Removed highly correlated features (optional)</li>
        <li>Standardized features for SVR and MLP</li>
      </ul>
    </div>

 <div class="section">
      <h2> 7. Hyperparameter Tuning</h2>
      <p>Used <b>GridSearchCV</b> and <b>RandomizedSearchCV</b> to optimize RMSE.</p>

<div class="code">
n_estimators = range(50, 501, 50)
depths = range(1, 11)
learning_rates = np.logspace(-3, 0, 10)
subsamples = np.linspace(0.5, 1.0, 6)
      </div>
    </div>

 <div class="section">
      <h2> 8. Evaluation Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Meaning</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>RMSE</td><td>Average prediction error</td></tr>
          <tr><td>MAE</td><td>Absolute error</td></tr>
          <tr><td>R²</td><td>Explained variance</td></tr>
        </tbody>
      </table>
    </div>

<div class="section">
      <h2> 9. Best Hyperparameters & Results</h2>

<table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Best Parameters</th>
            <th>RMSE</th>
            <th>R²</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Gradient Boosting</td>
            <td>learning_rate=0.1, max_depth=3, n_estimators=150, subsample=0.8</td>
            <td><b>0.2071</b></td>
            <td><b>0.9997</b></td>
          </tr>
          <tr>
            <td>Random Forest</td>
            <td>n_estimators=100, max_depth=None</td>
            <td>0.2090</td>
            <td>0.9997</td>
          </tr>
          <tr>
            <td>Decision Tree</td>
            <td>max_depth=None</td>
            <td>0.2354</td>
            <td>0.9996</td>
          </tr>
          <tr>
            <td>MLP Regressor</td>
            <td>hidden_layer_sizes=(100,), learning_rate_init=0.01</td>
            <td>2.4076</td>
            <td>0.9569</td>
          </tr>
        </tbody>
      </table>
    </div>

<div class="section">
      <h2> 10. Feature Importance</h2>
      <p>Top features influencing CO₂ uptake:</p>
      <ul>
        <li>T / K</li>
        <li>P / bar</li>
        <li>log(P)</li>
        <li>1/T</li>
        <li>largest electronegativity diff</li>
      </ul>
    </div>

<div class="section">
      <h2> 11. Final Conclusion</h2>
      <p>
        Gradient Boosting achieved the best performance with the lowest RMSE and highest R².
        The model is robust despite limited data size and heterogeneity.
        Feature importance confirms that temperature and pressure dominate adsorption,
        aligning with physical principles.
        The framework can be used as a screening tool for MOF carbon capture.
      </p>
    </div>

 <div class="section">
      <h2> 12. Future Improvements</h2>
      <ul>
        <li>Add more MOF descriptors (surface area, pore volume, etc.)</li>
        <li>Use GroupKFold based on MOF name to prevent leakage</li>
        <li>Use Bayesian optimization for faster tuning</li>
        <li>Apply SHAP for deeper interpretability</li>
      </ul>
    </div>


