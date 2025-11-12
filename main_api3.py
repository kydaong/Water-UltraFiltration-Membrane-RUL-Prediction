"""
Membrane Filter RUL Prediction API
FastAPI service for predicting remaining useful life of water treatment membrane filters
Updated to include membrane_id and timestamp tracking
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
import joblib
from pathlib import Path
import traceback
import math
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('membrane_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Membrane RUL Prediction API",
    description="Predicts remaining useful life of membrane filters in water treatment systems",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler_X = None
scaler_y = None
device = None
SEQUENCE_LENGTH = 60

# Feature names (24 features as per your training)
FEATURE_NAMES = [
    'feed_pressure_bar', 'temperature_c', 'feed_flow_m3h', 'tmp_bar',
    'turbidity_ntu', 'toc_mgl', 'ph', 'conductivity_uscm', 'tds_mgl',
    'permeate_flux_lm2h', 'permeate_flow_m3h', 'recovery_rate_pct',
    'specific_flux_lm2hbar', 'normalized_flux', 'flux_decline_pct',
    'flux_decline_rate', 'pressure_normalized_flux', 'flux_7day_avg',
    'flux_30day_avg', 'flux_std_7day', 'age_days', 'cumulative_operating_hours',
    'days_since_cleaning', 'cycles_completed'
]


# Model Architecture Classes
class PositionalEncoding(nn.Module):
    """Adds positional information to sequence embeddings"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerRULPredictor(nn.Module):
    """Transformer Encoder architecture for RUL prediction"""
    
    def __init__(
        self,
        input_dim: int = 24,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose: (batch, seq, features) -> (seq, batch, features)
        x = x.transpose(0, 1)
        
        # Project and encode
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=0)
        
        # Regression
        output = self.regression_head(x)
        
        return output


# Pydantic Models
class MembraneFeatures(BaseModel):
    """Single timestep of membrane operating parameters - accepts multiple data types"""
    
    feed_pressure_bar: Union[float, int, str, None] = Field(default=0.0, description="Feed pressure (bar)")
    temperature_c: Union[float, int, str, None] = Field(default=0.0, description="Temperature (°C)")
    feed_flow_m3h: Union[float, int, str, None] = Field(default=0.0, description="Feed flow (m³/h)")
    tmp_bar: Union[float, int, str, None] = Field(default=0.0, description="Transmembrane pressure (bar)")
    turbidity_ntu: Union[float, int, str, None] = Field(default=0.0, description="Turbidity (NTU)")
    toc_mgl: Union[float, int, str, None] = Field(default=0.0, description="TDS (mg/L)")
    ph: Union[float, int, str, None] = Field(default=0.0, description="pH")
    conductivity_uscm: Union[float, int, str, None] = Field(default=0.0, description="Conductivity (µS/cm)")
    tds_mgl: Union[float, int, str, None] = Field(default=0.0, description="TDS (mg/L)")
    permeate_flux_lm2h: Union[float, int, str, None] = Field(default=0.0, description="Permeate flux (L/m²/h)")
    permeate_flow_m3h: Union[float, int, str, None] = Field(default=0.0, description="Permeate flow (m³/h)")
    recovery_rate_pct: Union[float, int, str, None] = Field(default=0.0, description="Recovery rate (%)")
    specific_flux_lm2hbar: Union[float, int, str, None] = Field(default=0.0, description="Specific flux (L/m²/h/bar)")
    normalized_flux: Union[float, int, str, None] = Field(default=0.0, description="Normalized flux")
    flux_decline_pct: Union[float, int, str, None] = Field(default=0.0, description="Flux decline (%)")
    flux_decline_rate: Union[float, int, str, None] = Field(default=0.0, description="Flux decline rate")
    pressure_normalized_flux: Union[float, int, str, None] = Field(default=0.0, description="Pressure normalized flux")
    flux_7day_avg: Union[float, int, str, None] = Field(default=0.0, description="7-day avg flux")
    flux_30day_avg: Union[float, int, str, None] = Field(default=0.0, description="30-day avg flux")
    flux_std_7day: Union[float, int, str, None] = Field(default=0.0, description="7-day flux std dev")
    age_days: Union[float, int, str, None] = Field(default=0.0, description="Membrane age (days)")
    cumulative_operating_hours: Union[float, int, str, None] = Field(default=0.0, description="Operating hours")
    days_since_cleaning: Union[float, int, str, None] = Field(default=0.0, description="Days since cleaning")
    cycles_completed: Union[int, float, str, None] = Field(default=0, description="Cleaning cycles")


class SequencePredictionRequest(BaseModel):
    """Request for RUL prediction using time-series sequence"""
    membrane_id: Optional[str] = Field(None, description="Unique membrane identifier")
    timestamp: Optional[str] = Field(None, description="Timestamp of measurement (ISO format)")
    sequence: List[MembraneFeatures] = Field(
        ..., 
        min_items=60, 
        max_items=60,
        description="60 consecutive membrane measurements"
    )


class SinglePredictionRequest(BaseModel):
    """Request for single-point prediction"""
    membrane_id: Optional[str] = Field(None, description="Unique membrane identifier")
    timestamp: Optional[str] = Field(None, description="Timestamp of measurement (ISO format)")
    features: MembraneFeatures


class PredictionResponse(BaseModel):
    """Prediction response"""
    membrane_id: Optional[str] = Field(None, description="Membrane identifier from request")
    timestamp: Optional[str] = Field(None, description="Measurement timestamp from request")
    days_until_failure: float = Field(..., description="Predicted days until failure")
    health_status: str = Field(..., description="Health status classification")
    recommendation: str = Field(..., description="Maintenance recommendation")
    prediction_timestamp: str = Field(..., description="When prediction was made")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    scalers_loaded: bool
    device: str
    timestamp: str


def load_model_and_scalers():
    """Load the trained model and scalers"""
    global model, scaler_X, scaler_y, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load scalers
        scaler_X_path = Path("C:/Users/adminuser/Projects/water_filter_membrane_pred/scaler_X.pkl")
        scaler_y_path = Path("C:/Users/adminuser/Projects/water_filter_membrane_pred/scaler_y.pkl")
        
        if not scaler_X_path.exists() or not scaler_y_path.exists():
            raise FileNotFoundError("Scaler files not found")
        
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        logger.info("Scalers loaded")
        
        # Load model
        model_path = Path("C:/Users/adminuser/Projects/water_filter_membrane_pred/best_transformer_rul_model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = TransformerRULPredictor(
            input_dim=24,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint contains training metadata
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Checkpoint is just the model state dict
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def get_health_status(rul_days: float) -> tuple:
    """Get health status and recommendation"""
    if rul_days > 30: 
        return "HEALTHY", "Continue normal operation"
    elif rul_days > 14:
        return "WARNING", "Schedule cleaning within 2 weeks"
    elif rul_days > 7:
        return "CRITICAL", "Schedule immediate cleaning"
    else:
        return "URGENT", "Immediate action required"


def prepare_features(features_dict: Dict) -> np.ndarray:
    """Convert feature dict to array"""
    return np.array([features_dict[feat] for feat in FEATURE_NAMES])


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting API...")
    success = load_model_and_scalers()
    if success:
        logger.info("Ready to serve predictions")
    else:
        logger.error("Failed to load model")

def safe_float(value: any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or value == "null":
            return default
        return float(value)
    except:
        return default

def safe_int(value: any, default: int = 0) -> int:
    try:
        if value is None or value == "" or value == "null":
            return default
        return int(float(value))
    except:
        return default

# Add XMPro request model
class XMProPredictionRequest(BaseModel):
    """XMPro flat structure - 24 numeric fields only"""
    feed_pressure_bar: Union[float, int, str, None] = Field(default=0.0)
    temperature_c: Union[float, int, str, None] = Field(default=0.0)
    feed_flow_m3h: Union[float, int, str, None] = Field(default=0.0)
    tmp_bar: Union[float, int, str, None] = Field(default=0.0)
    turbidity_ntu: Union[float, int, str, None] = Field(default=0.0)
    toc_mgl: Union[float, int, str, None] = Field(default=0.0)
    ph: Union[float, int, str, None] = Field(default=0.0)
    conductivity_uscm: Union[float, int, str, None] = Field(default=0.0)
    tds_mgl: Union[float, int, str, None] = Field(default=0.0)
    permeate_flux_lm2h: Union[float, int, str, None] = Field(default=0.0)
    permeate_flow_m3h: Union[float, int, str, None] = Field(default=0.0)
    recovery_rate_pct: Union[float, int, str, None] = Field(default=0.0)
    specific_flux_lm2hbar: Union[float, int, str, None] = Field(default=0.0)
    normalized_flux: Union[float, int, str, None] = Field(default=0.0)
    flux_decline_pct: Union[float, int, str, None] = Field(default=0.0)
    flux_decline_rate: Union[float, int, str, None] = Field(default=0.0)
    pressure_normalized_flux: Union[float, int, str, None] = Field(default=0.0)
    flux_7day_avg: Union[float, int, str, None] = Field(default=0.0)
    flux_30day_avg: Union[float, int, str, None] = Field(default=0.0)
    flux_std_7day: Union[float, int, str, None] = Field(default=0.0)
    age_days: Union[float, int, str, None] = Field(default=0.0)
    cumulative_operating_hours: Union[float, int, str, None] = Field(default=0.0)
    days_since_cleaning: Union[float, int, str, None] = Field(default=0.0)
    cycles_completed: Union[int, float, str, None] = Field(default=0)

# Add XMPro response model
class XMProPredictionResponse(BaseModel):
    """Simplified response"""
    days_until_failure: float
    health_status: str
    recommendation: str
    prediction_timestamp: str




@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Membrane RUL Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_sequence": "/predict/sequence",
            "predict_single": "/predict/single",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy" if (model and scaler_X and scaler_y) else "unhealthy",
        model_loaded=model is not None,
        scalers_loaded=(scaler_X is not None and scaler_y is not None),
        device=str(device) if device else "unknown",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict/sequence", response_model=PredictionResponse)
async def predict_from_sequence(request: SequencePredictionRequest):
    """
    Predict RUL from 60-measurement sequence (RECOMMENDED)
    Now includes membrane_id and timestamp tracking
    """
    try:
        if not model or not scaler_X or not scaler_y:
            raise HTTPException(503, "Model not loaded")
        
        # Convert sequence to array
        sequence_data = []
        for measurement in request.sequence:
            features = prepare_features(measurement.dict())
            sequence_data.append(features)
        
        sequence_array = np.array(sequence_data)
        
        # Normalize
        sequence_normalized = scaler_X.transform(sequence_array)
        
        # Predict
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_norm = model(sequence_tensor).cpu().numpy()
        
        # Inverse transform
        rul_days = float(max(0, scaler_y.inverse_transform(pred_norm)[0, 0]))
        
        health_status, recommendation = get_health_status(rul_days)
        
        # Log with membrane_id if provided
        log_msg = f"Prediction: {rul_days:.2f} days, {health_status}"
        if request.membrane_id:
            log_msg = f"Membrane {request.membrane_id} - {log_msg}"
        logger.info(log_msg)
        
        return PredictionResponse(
            membrane_id=request.membrane_id,
            timestamp=request.timestamp,
            days_until_failure=round(rul_days, 2),
            health_status=health_status,
            recommendation=recommendation,
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/predict/single", response_model=PredictionResponse)
async def predict_from_single(request: SinglePredictionRequest):
    """
    Predict from single measurement (repeats to create sequence)
    Less accurate than /predict/sequence
    Now includes membrane_id and timestamp tracking
    """
    try:
        if not model or not scaler_X or not scaler_y:
            raise HTTPException(503, "Model not loaded")
        
        features = prepare_features(request.features.dict())
        sequence_array = np.tile(features, (SEQUENCE_LENGTH, 1))
        
        # Normalize and predict
        sequence_normalized = scaler_X.transform(sequence_array)
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_norm = model(sequence_tensor).cpu().numpy()
        
        rul_days = float(max(0, scaler_y.inverse_transform(pred_norm)[0, 0]))
        
        health_status, recommendation = get_health_status(rul_days)
        
        # Log with membrane_id if provided
        log_msg = f"Single prediction: {rul_days:.2f} days"
        if request.membrane_id:
            log_msg = f"Membrane {request.membrane_id} - {log_msg}"
        logger.info(log_msg)
        
        return PredictionResponse(
            membrane_id=request.membrane_id,
            timestamp=request.timestamp,
            days_until_failure=round(rul_days, 2),
            health_status=health_status,
            recommendation=recommendation,
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/predict/xmpro", response_model=XMProPredictionResponse)
async def predict_from_xmpro(request: XMProPredictionRequest):
    """XMPro endpoint - flat structure"""
    try:
        if not model or not scaler_X or not scaler_y:
            raise HTTPException(503, "Model not loaded")
        
        # Convert all to float safely
        features_dict = {
            'feed_pressure_bar': safe_float(request.feed_pressure_bar, 4.5),
            'temperature_c': safe_float(request.temperature_c, 20.0),
            'feed_flow_m3h': safe_float(request.feed_flow_m3h, 12.0),
            'tmp_bar': safe_float(request.tmp_bar, 0.8),
            'turbidity_ntu': safe_float(request.turbidity_ntu, 0.2),
            'toc_mgl': safe_float(request.toc_mgl, 2.5),
            'ph': safe_float(request.ph, 7.0),
            'conductivity_uscm': safe_float(request.conductivity_uscm, 500.0),
            'tds_mgl': safe_float(request.tds_mgl, 300.0),
            'permeate_flux_lm2h': safe_float(request.permeate_flux_lm2h, 90.0),
            'permeate_flow_m3h': safe_float(request.permeate_flow_m3h, 7.0),
            'recovery_rate_pct': safe_float(request.recovery_rate_pct, 70.0),
            'specific_flux_lm2hbar': safe_float(request.specific_flux_lm2hbar, 20.0),
            'normalized_flux': safe_float(request.normalized_flux, 0.9),
            'flux_decline_pct': safe_float(request.flux_decline_pct, 10.0),
            'flux_decline_rate': safe_float(request.flux_decline_rate, 0.1),
            'pressure_normalized_flux': safe_float(request.pressure_normalized_flux, 25.0),
            'flux_7day_avg': safe_float(request.flux_7day_avg, 95.0),
            'flux_30day_avg': safe_float(request.flux_30day_avg, 95.0),
            'flux_std_7day': safe_float(request.flux_std_7day, 2.0),
            'age_days': safe_float(request.age_days, 100.0),
            'cumulative_operating_hours': safe_float(request.cumulative_operating_hours, 2400.0),
            'days_since_cleaning': safe_float(request.days_since_cleaning, 30.0),
            'cycles_completed': safe_int(request.cycles_completed, 1)
        }
        
        features = prepare_features(features_dict)
        sequence_array = np.tile(features, (SEQUENCE_LENGTH, 1))
        sequence_normalized = scaler_X.transform(sequence_array)
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_norm = model(sequence_tensor).cpu().numpy()
        
        rul_days = float(max(0, scaler_y.inverse_transform(pred_norm)[0, 0]))
        health_status, recommendation = get_health_status(rul_days)
        
        logger.info(f"XMPro prediction: {rul_days:.2f} days, {health_status}")
        
        return XMProPredictionResponse(
            days_until_failure=round(rul_days, 2),
            health_status=health_status,
            recommendation=recommendation,
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"XMPro error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)