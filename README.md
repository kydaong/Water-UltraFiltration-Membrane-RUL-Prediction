## Membrane Filter RUL Prediction API
A production-ready FastAPI service that predicts remaining useful life (RUL) of water treatment membrane filters using a Transformer-based neural network. Processes 60-measurement temporal sequences of operational parameters to forecast membrane degradation.
## Overview
This API monitors water treatment membrane health by analyzing flux decline patterns, pressure dynamics, and water quality metrics. It predicts time-to-failure in days and provides health status classification and maintenance recommendations.

**Key Features:**

Transformer encoder architecture for temporal pattern recognition
24 operational and derived features from membrane sensors
Three inference endpoints for different data formats
XMPro integration support for enterprise data pipelines
Comprehensive logging and error handling
CORS enabled for cross-platform deployment

**Architecture:**

The model stack consists of:
Input Layer - Accepts 60 sequential measurements with 24 features each (1,440-dimensional input)
Transformer Encoder - 4 stacked encoder layers with 8 attention heads and 512-dimensional feed-forward networks. Positional encoding preserves temporal ordering within sequences.
Output Head - Multi-layer perceptron (256 → 128 → 64 → 1) that maps encoder representation to RUL prediction (days)
Normalization - Standard scalers fitted on training data normalize input features and predictions separately
Installation

Requirements

Python 3.8+
PyTorch 1.9+
FastAPI 0.95+
numpy, joblib, scikit-learn

## API Endpoints
**Endpoint: POST /predict/sequence**

Payload: 60 consecutive measurements with 24 operational parameters each

```python
{
  "membrane_id": "MEM001",
  "timestamp": "2025-11-12T10:30:00Z",
  "sequence": [
    {
      "feed_pressure_bar": 4.5,
      "temperature_c": 22.0,
      "feed_flow_m3h": 12.0,
      "tmp_bar": 0.8,
      "turbidity_ntu": 0.2,
      "toc_mgl": 2.5,
      "ph": 7.0,
      "conductivity_uscm": 500,
      "tds_mgl": 300,
      "permeate_flux_lm2h": 90.0,
      "permeate_flow_m3h": 7.0,
      "recovery_rate_pct": 70.0,
      "specific_flux_lm2hbar": 20.0,
      "normalized_flux": 0.9,
      "flux_decline_pct": 10.0,
      "flux_decline_rate": 0.1,
      "pressure_normalized_flux": 25.0,
      "flux_7day_avg": 95.0,
      "flux_30day_avg": 95.0,
      "flux_std_7day": 2.0,
      "age_days": 100,
      "cumulative_operating_hours": 2400,
      "days_since_cleaning": 30,
      "cycles_completed": 1
    },
    ...59 more measurements
  ]
}
```
Response:
```python 
{
  "membrane_id": "MEM001",
  "timestamp": "2025-11-12T10:30:00Z",
  "days_until_failure": 20.7,
  "health_status": "degrading",
  "recommendation": "Schedule preventive cleaning within 7 days",
  "prediction_timestamp": "2025-11-12T10:30:45Z"
}
```


Single Measurement Prediction

**Endpoint: POST /predict/single**

Takes one measurement and replicates it 60 times for inference. Useful for real-time updates or when full sequence unavailable.

```python
{
  "membrane_id": "MEM001",
  "timestamp": "2025-11-12T10:30:00Z",
  "features": {
    "feed_pressure_bar": 4.5,
    "temperature_c": 22.0,
    "feed_flow_m3h": 12.0,
    ...all 24 features
  }
}
```
**Endpoint: GET/health**
```python
{
  "status": "healthy",
  "model_loaded": true,
  "scalers_loaded": true,
  "device": "cuda",
  "timestamp": "2025-11-12T10:30:45Z"
}
```
## Features

**Operating Parameters (12 direct measurements):**
- `feed_pressure_bar` - System inlet pressure
- `temperature_c` - Membrane operating temperature
- `feed_flow_m3h` - Volumetric inlet flow
- `tmp_bar` - Transmembrane pressure (key degradation driver)
- `turbidity_ntu` - Feed water turbidity
- `toc_mgl` - Total organic carbon
- `ph` - Water pH
- `conductivity_uscm` - Water conductivity
- `tds_mgl` - Total dissolved solids
- `permeate_flux_lm2h` - Permeate flux (primary degradation metric)
- `permeate_flow_m3h` - Permeate volumetric flow
- `recovery_rate_pct` - % of feed water recovered as permeate

**Derived Metrics (12 calculated features):**
- `specific_flux_lm2hbar` - Flux normalized to pressure
- `normalized_flux` - Ratio of current to initial flux
- `flux_decline_pct` - Percentage flux loss from initial
- `flux_decline_rate` - Rate of flux degradation
- `pressure_normalized_flux` - Flux adjusted for operating pressure
- `flux_7day_avg` - 7-day moving average flux
- `flux_30day_avg` - 30-day moving average flux
- `flux_std_7day` - 7-day flux volatility
- `age_days` - Days since membrane installation
- `cumulative_operating_hours` - Total operation hours
- `days_since_cleaning` - Days since last CIP/cleaning
- `cycles_completed` - Membrane cleaning cycles


## Health Status Classification

Predictions are categorized based on remaining days:

| Status | RUL Threshold | Action |
|--------|---------------|--------|
| healthy | > 30 days | Continue normal operation |
| caution | 15-30 days | Plan maintenance schedule |
| degrading | 7-15 days | Schedule preventive cleaning |
| critical | < 7 days | Replace/clean immediately |


## Integration Example 

An excerpt from the script for FASTapi deployment at client site. Confidential information are masked for client data protection. 
```python
import requests

url = "http://localhost:5050/predict/sequence"

payload = {
    "membrane_id": "MEM001",
    "timestamp": "2025-11-12T10:30:00Z",
    "sequence": sequence_list  # 60 measurements
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Days until failure: {result['days_until_failure']}")
print(f"Status: {result['health_status']}")
print(f"Action: {result['recommendation']}")
```


**Performance Notes**

Single prediction latency: ~50ms (GPU), ~200ms (CPU)

Sequence validation: ~10ms

Throughput: 50-100 predictions/sec on GPU