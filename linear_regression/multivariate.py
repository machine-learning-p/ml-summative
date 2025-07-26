# WAEMU Banking Risk Assessment - Linear Regression Model
# Mission: Predict Bank Financial Health (Zscore) in West African Economic and Monetary Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🏦 WAEMU Banking Risk Assessment System")
print("=" * 50)

# Load the dataset
df = pd.read_csv('WAEMU_Banking.csv', encoding='ISO-8859-1')

# Clean column names and handle encoding issues
df = df.drop(df.columns[0], axis=1)  # Remove unnamed first column
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Fix encoding issues in Countries column
df['Countries'] = df['Countries'].str.replace('B�nin', 'Benin')
df['Countries'] = df['Countries'].str.replace('C�te', 'Cote')

print("\n📊 Dataset Overview:")
print(df.head())

print("\n🔍 Data Quality Assessment:")
print(df.info())
print("\n📈 Statistical Summary:")
print(df.describe())

# Check for missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

# VISUALIZATION 1: Correlation Heatmap
plt.figure(figsize=(15, 12))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Create correlation heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('📊 Banking Metrics Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()

# VISUALIZATION 2: Distribution of Target Variable (Zscore)
plt.subplot(2, 2, 2)
plt.hist(df['Zscore'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(df['Zscore'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["Zscore"].mean():.2f}')
plt.xlabel('Z-Score (Financial Health)')
plt.ylabel('Frequency')
plt.title('📈 Distribution of Bank Z-Score (Target Variable)')
plt.legend()
plt.grid(True, alpha=0.3)

# VISUALIZATION 3: Feature relationships with target
plt.subplot(2, 2, 3)
# Select top correlated features with Zscore
zscore_corr = correlation_matrix['Zscore'].abs().sort_values(ascending=False)[1:6]
top_features = zscore_corr.index.tolist()

for i, feature in enumerate(top_features[:3]):
    plt.scatter(df[feature], df['Zscore'], alpha=0.6, label=feature, s=20)
plt.xlabel('Feature Values')
plt.ylabel('Z-Score')
plt.title('🎯 Top Features vs Z-Score')
plt.legend()
plt.grid(True, alpha=0.3)

# VISUALIZATION 4: Banking performance by country
plt.subplot(2, 2, 4)
country_zscore = df.groupby('Countries')['Zscore'].mean().sort_values(ascending=False)
plt.bar(range(len(country_zscore)), country_zscore.values, color='lightcoral')
plt.xticks(range(len(country_zscore)), country_zscore.index, rotation=45, ha='right')
plt.ylabel('Average Z-Score')
plt.title('🌍 Average Banking Health by Country')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('banking_analysis_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎯 Target Variable Analysis:")
print(f"Z-Score Range: {df['Zscore'].min():.3f} to {df['Zscore'].max():.3f}")
print(f"Z-Score Mean: {df['Zscore'].mean():.3f}")
print(f"Z-Score Std: {df['Zscore'].std():.3f}")

# Feature Engineering
print("\n🔧 Feature Engineering:")

# Encode categorical variables
le_countries = LabelEncoder()
le_banks = LabelEncoder()

df['Countries_Encoded'] = le_countries.fit_transform(df['Countries'])
df['Banks_Encoded'] = le_banks.fit_transform(df['Banks'])

# Create additional features
df['Risk_Debt_Ratio'] = df['RIR'] * df['DEBT']
df['Stability_Size_Ratio'] = df['SFS'] / (df['SIZE'] + 1)  # Add 1 to avoid division by zero
df['Governance_Performance'] = (df['GE'] + df['PS']) / 2

# Select features for modeling (excluding target and non-numeric)
feature_cols = ['Countries_Num', 'Year', 'RIR', 'SFS', 'INF', 'ERA', 'INL', 
                'DEBT', 'SIZE', 'CC', 'GE', 'PS', 'RQ', 'RL', 'VA',
                'Countries_Encoded', 'Banks_Encoded', 'Risk_Debt_Ratio', 
                'Stability_Size_Ratio', 'Governance_Performance']

X = df[feature_cols]
y = df['Zscore']

print(f"Features selected: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=None)

print(f"\n📊 Data Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Data preprocessing completed!")

# MODEL 1: Linear Regression with Gradient Descent Implementation
print("\n🤖 Model 1: Linear Regression")

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.train_losses = []
        self.test_losses = []
        
    def fit(self, X, y, X_test=None, y_test=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate cost
            train_cost = np.mean((y_pred - y) ** 2)
            self.train_losses.append(train_cost)
            
            # Calculate test cost if provided
            if X_test is not None and y_test is not None:
                y_test_pred = np.dot(X_test, self.weights) + self.bias
                test_cost = np.mean((y_test_pred - y_test) ** 2)
                self.test_losses.append(test_cost)
            
            # Calculate gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train custom gradient descent model
gd_model = GradientDescentLinearRegression(learning_rate=0.01, n_iterations=1000)
gd_model.fit(X_train_scaled, y_train, X_test_scaled, y_test)

gd_train_pred = gd_model.predict(X_train_scaled)
gd_test_pred = gd_model.predict(X_test_scaled)

gd_train_mse = mean_squared_error(y_train, gd_train_pred)
gd_test_mse = mean_squared_error(y_test, gd_test_pred)
gd_r2 = r2_score(y_test, gd_test_pred)

print(f"Gradient Descent Linear Regression:")
print(f"  Train MSE: {gd_train_mse:.4f}")
print(f"  Test MSE: {gd_test_mse:.4f}")
print(f"  R² Score: {gd_r2:.4f}")

# Also train with sklearn for comparison
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"Scikit-learn Linear Regression:")
print(f"  Test MSE: {lr_mse:.4f}")
print(f"  R² Score: {lr_r2:.4f}")

# MODEL 2: Random Forest
print(f"\n🌲 Model 2: Random Forest")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest:")
print(f"  Test MSE: {rf_mse:.4f}")
print(f"  R² Score: {rf_r2:.4f}")

# MODEL 3: Decision Tree
print(f"\n🌳 Model 3: Decision Tree")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

print(f"Decision Tree:")
print(f"  Test MSE: {dt_mse:.4f}")
print(f"  R² Score: {dt_r2:.4f}")

# Compare models and select best
models_performance = {
    'Gradient Descent LR': {'mse': gd_test_mse, 'r2': gd_r2, 'model': gd_model},
    'Scikit-learn LR': {'mse': lr_mse, 'r2': lr_r2, 'model': lr_model},
    'Random Forest': {'mse': rf_mse, 'r2': rf_r2, 'model': rf_model},
    'Decision Tree': {'mse': dt_mse, 'r2': dt_r2, 'model': dt_model}
}

best_model_name = min(models_performance.keys(), 
                     key=lambda x: models_performance[x]['mse'])
best_model = models_performance[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best MSE: {models_performance[best_model_name]['mse']:.4f}")
print(f"Best R²: {models_performance[best_model_name]['r2']:.4f}")

# Plot Loss Curves
plt.figure(figsize=(15, 10))

# Loss curve for gradient descent
plt.subplot(2, 3, 1)
plt.plot(gd_model.train_losses, label='Training Loss', color='blue')
plt.plot(gd_model.test_losses, label='Test Loss', color='red')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('📉 Gradient Descent Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Scatter plot: Actual vs Predicted (Linear Regression)
plt.subplot(2, 3, 2)
plt.scatter(y_test, lr_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Z-Score')
plt.ylabel('Predicted Z-Score')
plt.title('🎯 Linear Regression: Actual vs Predicted')
plt.grid(True, alpha=0.3)

# Feature importance for Random Forest
plt.subplot(2, 3, 3)
if hasattr(rf_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    plt.barh(range(len(importance_df)), importance_df['importance'], color='lightgreen')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance')
    plt.title('🌲 Random Forest Feature Importance')
    plt.grid(True, alpha=0.3)

# Model comparison
plt.subplot(2, 3, 4)
model_names = list(models_performance.keys())
mse_values = [models_performance[name]['mse'] for name in model_names]
plt.bar(model_names, mse_values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.ylabel('Mean Squared Error')
plt.title('📊 Model Performance Comparison (MSE)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# R² comparison
plt.subplot(2, 3, 5)
r2_values = [models_performance[name]['r2'] for name in model_names]
plt.bar(model_names, r2_values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.ylabel('R² Score')
plt.title('📊 Model Performance Comparison (R²)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Residual plot for best model
plt.subplot(2, 3, 6)
if best_model_name == 'Random Forest':
    best_pred = rf_pred
elif best_model_name == 'Decision Tree':
    best_pred = dt_pred
else:
    best_pred = lr_pred

residuals = y_test - best_pred
plt.scatter(best_pred, residuals, alpha=0.6, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'📈 Residual Plot - {best_model_name}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the best model and preprocessing objects
print(f"\n💾 Saving the best model and preprocessing objects...")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save label encoders
joblib.dump(le_countries, 'le_countries.pkl')
joblib.dump(le_banks, 'le_banks.pkl')

# Save the best model
if best_model_name in ['Gradient Descent LR', 'Scikit-learn LR']:
    joblib.dump(lr_model, 'best_model.pkl')  # Use sklearn version for API
    print("Saved scikit-learn Linear Regression model")
else:
    joblib.dump(best_model, 'best_model.pkl')
    print(f"Saved {best_model_name} model")

# Save feature names
joblib.dump(feature_cols, 'feature_names.pkl')

print("✅ All models and preprocessing objects saved!")

# Create prediction function for API
def predict_bank_health(countries_num, year, rir, sfs, inf, era, inl, debt, size, 
                       cc, ge, ps, rq, rl, va, countries, banks):
    """
    Predict bank financial health (Z-score) based on input features
    """
    # Load saved objects
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_countries = joblib.load('le_countries.pkl')
    le_banks = joblib.load('le_banks.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Encode categorical variables
    try:
        countries_encoded = le_countries.transform([countries])[0]
    except:
        countries_encoded = 0  # Default value for unknown countries
    
    try:
        banks_encoded = le_banks.transform([banks])[0]
    except:
        banks_encoded = 0  # Default value for unknown banks
    
    # Create engineered features
    risk_debt_ratio = rir * debt
    stability_size_ratio = sfs / (size + 1)
    governance_performance = (ge + ps) / 2
    
    # Create feature array
    features = np.array([[countries_num, year, rir, sfs, inf, era, inl, debt, size,
                         cc, ge, ps, rq, rl, va, countries_encoded, banks_encoded,
                         risk_debt_ratio, stability_size_ratio, governance_performance]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return float(prediction[0])

# Test the prediction function with a sample from test set
print(f"\n🧪 Testing prediction function:")
test_idx = 0
sample_data = X_test.iloc[test_idx]

predicted_zscore = predict_bank_health(
    countries_num=sample_data['Countries_Num'],
    year=sample_data['Year'],
    rir=sample_data['RIR'],
    sfs=sample_data['SFS'],
    inf=sample_data['INF'],
    era=sample_data['ERA'],
    inl=sample_data['INL'],
    debt=sample_data['DEBT'],
    size=sample_data['SIZE'],
    cc=sample_data['CC'],
    ge=sample_data['GE'],
    ps=sample_data['PS'],
    rq=sample_data['RQ'],
    rl=sample_data['RL'],
    va=sample_data['VA'],
    countries='Benin',  # Default country
    banks='Default Bank'  # Default bank
)

actual_zscore = y_test.iloc[test_idx]
print(f"Predicted Z-Score: {predicted_zscore:.4f}")
print(f"Actual Z-Score: {actual_zscore:.4f}")
print(f"Prediction Error: {abs(predicted_zscore - actual_zscore):.4f}")

print(f"\n🎉 WAEMU Banking Risk Assessment Model Complete!")
print(f"📁 Files saved: best_model.pkl, scaler.pkl, feature_names.pkl")
print(f"🔮 Ready for API deployment!")

# Final model summary
print(f"\n📋 FINAL MODEL SUMMARY:")
print(f"{'='*50}")
print(f"🎯 Mission: Predict Bank Financial Health in WAEMU")
print(f"📊 Dataset: 742 banks across 8 countries (2013-2020)")
print(f"🏆 Best Model: {best_model_name}")
print(f"📈 Performance: MSE = {models_performance[best_model_name]['mse']:.4f}, R² = {models_performance[best_model_name]['r2']:.4f}")
print(f"🔧 Features: {len(feature_cols)} engineered features")
print(f"✅ Ready for production deployment!")