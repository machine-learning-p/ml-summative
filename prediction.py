"""
Prediction API for WAEMU Banking Risk Assessment
"""


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import uvicorn
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🏦 WAEMU Banking Risk Assessment API",
    description="Predict bank financial health (Z-score) in West African Economic and Monetary Union",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing objects
model = None
scaler = None
le_countries = None
le_banks = None
feature_names = None

@app.on_event("startup")
async def load_models():
    """Load models and preprocessing objects at startup"""
    global model, scaler, le_countries, le_banks, feature_names
    
    try:
        # Try to load from models directory first (Docker), then current directory
        model_paths = [
            ('/app/models/', 'Docker path'),
            ('./models/', 'Local models directory'),
            ('./', 'Current directory')
        ]
        
        for path, description in model_paths:
            try:
                if os.path.exists(f"{path}best_model.pkl"):
                    logger.info(f"Loading models from {description}: {path}")
                    model = joblib.load(f'{path}best_model.pkl')
                    scaler = joblib.load(f'{path}scaler.pkl')
                    le_countries = joblib.load(f'{path}le_countries.pkl')
                    le_banks = joblib.load(f'{path}le_banks.pkl')
                    feature_names = joblib.load(f'{path}feature_names.pkl')
                    logger.info("✅ All models loaded successfully!")
                    return
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {e}")
                continue
        
        logger.error("❌ Could not load models from any location")
        
    except Exception as e:
        logger.error(f"❌ Error during model loading: {e}")

# Pydantic models for request validation
class BankingData(BaseModel):
    """Input data model for banking risk prediction"""
    
    countries_num: int = Field(
        ..., 
        ge=1, 
        le=8, 
        description="Country numeric code (1-8 for WAEMU countries)"
    )
    
    year: int = Field(
        ..., 
        ge=2010, 
        le=2025, 
        description="Year of data collection"
    )
    
    rir: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Risk Index Rating (0-10)"
    )
    
    sfs: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Solvency and Financial Stability (0-100)"
    )
    
    inf: float = Field(
        ..., 
        ge=-5.0, 
        le=20.0, 
        description="Inflation Rate (-5% to 20%)"
    )
    
    era: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Economic Risk Assessment (0-10)"
    )
    
    inl: float = Field(
        ..., 
        ge=0.0, 
        le=50.0, 
        description="Internationalization Level (0-50)"
    )
    
    debt: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Debt Level (0-100)"
    )
    
    size: float = Field(
        ..., 
        ge=0.0, 
        le=30.0, 
        description="Bank Size indicator (0-30)"
    )
    
    cc: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Capital Adequacy (0-100)"
    )
    
    ge: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Governance and Ethics (0-100)"
    )
    
    ps: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Profitability and Sustainability (0-100)"
    )
    
    rq: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Regulatory Compliance (0-100)"
    )
    
    rl: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Liquidity Risk (0-100)"
    )
    
    va: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Value Added (0-100)"
    )
    
    countries: str = Field(
        default="Benin",
        description="Country name (Benin, Burkina Faso, Cote d'Ivoire, Guinea-Bissau, Mali, Niger, Senegal, Togo)"
    )
    
    banks: str = Field(
        default="Default Bank",
        description="Bank name"
    )

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    predicted_zscore: float = Field(..., description="Predicted Z-Score (financial health)")
    risk_level: str = Field(..., description="Risk assessment level")
    interpretation: str = Field(..., description="Human-readable interpretation")
    model_confidence: str = Field(..., description="Model confidence level")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🏦 WAEMU Banking Risk Assessment API",
        "version": "1.0.0",
        "description": "Predict bank financial health in West African Economic and Monetary Union",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
            "model_info": "/model-info (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0",
        "environment": os.getenv("RENDER_SERVICE_NAME", "local")
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_bank_health(data: BankingData):
    """
    Predict bank financial health (Z-score) based on banking metrics
    
    The Z-score indicates the likelihood of bank bankruptcy:
    - Higher Z-score = Better financial health
    - Lower Z-score = Higher bankruptcy risk
    """
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server configuration."
        )
    
    try:
        logger.info(f"Making prediction for country: {data.countries}, bank: {data.banks}")
        
        # Encode categorical variables
        try:
            countries_encoded = le_countries.transform([data.countries])[0]
        except:
            countries_encoded = 0  # Default for unknown countries
            logger.warning(f"Unknown country: {data.countries}, using default encoding")
        
        try:
            banks_encoded = le_banks.transform([data.banks])[0]
        except:
            banks_encoded = 0  # Default for unknown banks
            logger.warning(f"Unknown bank: {data.banks}, using default encoding")
        
        # Create engineered features
        risk_debt_ratio = data.rir * data.debt
        stability_size_ratio = data.sfs / (data.size + 1)
        governance_performance = (data.ge + data.ps) / 2
        
        # Create feature array in the correct order
        features = np.array([[
            data.countries_num, data.year, data.rir, data.sfs, data.inf, 
            data.era, data.inl, data.debt, data.size, data.cc, data.ge, 
            data.ps, data.rq, data.rl, data.va, countries_encoded, 
            banks_encoded, risk_debt_ratio, stability_size_ratio, 
            governance_performance
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        logger.info(f"Prediction made: {prediction}")
        
        # Determine risk level and interpretation
        if prediction >= 2.0:
            risk_level = "Low Risk"
            interpretation = "Excellent financial health. Very low bankruptcy risk."
            confidence = "High"
        elif prediction >= 1.5:
            risk_level = "Moderate Risk"
            interpretation = "Good financial health. Moderate bankruptcy risk."
            confidence = "High"
        elif prediction >= 1.0:
            risk_level = "High Risk"
            interpretation = "Concerning financial health. High bankruptcy risk."
            confidence = "Medium"
        else:
            risk_level = "Very High Risk"
            interpretation = "Poor financial health. Very high bankruptcy risk."
            confidence = "Medium"
        
        return PredictionResponse(
            predicted_zscore=round(float(prediction), 4),
            risk_level=risk_level,
            interpretation=interpretation,
            model_confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": str(type(model).__name__),
        "features_count": len(feature_names) if feature_names is not None else "Unknown",
        "supported_countries": [
            "Benin", "Burkina Faso", "Cote d'Ivoire", "Guinea-Bissau", 
            "Mali", "Niger", "Senegal", "Togo"
        ],
        "target_variable": "Z-Score (Financial Health)",
        "prediction_range": "Continuous values (higher = better health)",
        "deployment_environment": os.getenv("RENDER_SERVICE_NAME", "local")
    }

# For testing the API locally
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "prediction:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info"
    )

# Pydantic models for request validation
class BankingData(BaseModel):
    """Input data model for banking risk prediction"""
    
    countries_num: int = Field(
        ..., 
        ge=1, 
        le=8, 
        description="Country numeric code (1-8 for WAEMU countries)"
    )
    
    year: int = Field(
        ..., 
        ge=2010, 
        le=2025, 
        description="Year of data collection"
    )
    
    rir: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Risk Index Rating (0-10)"
    )
    
    sfs: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Solvency and Financial Stability (0-100)"
    )
    
    inf: float = Field(
        ..., 
        ge=-5.0, 
        le=20.0, 
        description="Inflation Rate (-5% to 20%)"
    )
    
    era: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Economic Risk Assessment (0-10)"
    )
    
    inl: float = Field(
        ..., 
        ge=0.0, 
        le=50.0, 
        description="Internationalization Level (0-50)"
    )
    
    debt: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Debt Level (0-100)"
    )
    
    size: float = Field(
        ..., 
        ge=0.0, 
        le=30.0, 
        description="Bank Size indicator (0-30)"
    )
    
    cc: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Capital Adequacy (0-100)"
    )
    
    ge: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Governance and Ethics (0-100)"
    )
    
    ps: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Profitability and Sustainability (0-100)"
    )
    
    rq: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Regulatory Compliance (0-100)"
    )
    
    rl: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Liquidity Risk (0-100)"
    )
    
    va: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Value Added (0-100)"
    )
    
    countries: str = Field(
        default="Benin",
        description="Country name (Benin, Burkina Faso, Cote d'Ivoire, Guinea-Bissau, Mali, Niger, Senegal, Togo)"
    )
    
    banks: str = Field(
        default="Default Bank",
        description="Bank name"
    )

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    predicted_zscore: float = Field(..., description="Predicted Z-Score (financial health)")
    risk_level: str = Field(..., description="Risk assessment level")
    interpretation: str = Field(..., description="Human-readable interpretation")
    model_confidence: str = Field(..., description="Model confidence level")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🏦 WAEMU Banking Risk Assessment API",
        "version": "1.0.0",
        "description": "Predict bank financial health in West African Economic and Monetary Union",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_bank_health(data: BankingData):
    """
    Predict bank financial health (Z-score) based on banking metrics
    
    The Z-score indicates the likelihood of bank bankruptcy:
    - Higher Z-score = Better financial health
    - Lower Z-score = Higher bankruptcy risk
    """
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server configuration."
        )
    
    try:
        # Encode categorical variables
        try:
            countries_encoded = le_countries.transform([data.countries])[0]
        except:
            countries_encoded = 0  # Default for unknown countries
        
        try:
            banks_encoded = le_banks.transform([data.banks])[0]
        except:
            banks_encoded = 0  # Default for unknown banks
        
        # Create engineered features
        risk_debt_ratio = data.rir * data.debt
        stability_size_ratio = data.sfs / (data.size + 1)
        governance_performance = (data.ge + data.ps) / 2
        
        # Create feature array in the correct order
        features = np.array([[
            data.countries_num, data.year, data.rir, data.sfs, data.inf, 
            data.era, data.inl, data.debt, data.size, data.cc, data.ge, 
            data.ps, data.rq, data.rl, data.va, countries_encoded, 
            banks_encoded, risk_debt_ratio, stability_size_ratio, 
            governance_performance
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Determine risk level and interpretation
        if prediction >= 2.0:
            risk_level = "Low Risk"
            interpretation = "Excellent financial health. Very low bankruptcy risk."
            confidence = "High"
        elif prediction >= 1.5:
            risk_level = "Moderate Risk"
            interpretation = "Good financial health. Moderate bankruptcy risk."
            confidence = "High"
        elif prediction >= 1.0:
            risk_level = "High Risk"
            interpretation = "Concerning financial health. High bankruptcy risk."
            confidence = "Medium"
        else:
            risk_level = "Very High Risk"
            interpretation = "Poor financial health. Very high bankruptcy risk."
            confidence = "Medium"
        
        return PredictionResponse(
            predicted_zscore=round(float(prediction), 4),
            risk_level=risk_level,
            interpretation=interpretation,
            model_confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": str(type(model).__name__),
        "features_count": len(feature_names) if 'feature_names' in globals() else "Unknown",
        "supported_countries": [
            "Benin", "Burkina Faso", "Cote d'Ivoire", "Guinea-Bissau", 
            "Mali", "Niger", "Senegal", "Togo"
        ],
        "target_variable": "Z-Score (Financial Health)",
        "prediction_range": "Continuous values (higher = better health)"
    }

# For testing the API locally
if __name__ == "__main__":
    uvicorn.run(
        "prediction:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )