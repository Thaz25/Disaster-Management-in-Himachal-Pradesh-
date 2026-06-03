# AI-Based Multi-Modal Disaster Prediction & Relief Optimization System

**Himachal Pradesh — Flood, Landslide & Extreme Weather Prediction**

A Python-based system using 4 collaborating ML models to predict disasters and optimize relief operations.

## Architecture

```
Data Collection → Image Processing → [CNN + LSTM + RF + XGBoost] → Ensemble → Risk Score → Relief Strategy → Map
```

### Collaborating Models:
1. **CNN (PyTorch)** — Classifies satellite images as Flood / Landslide / Normal
2. **LSTM (PyTorch)** — Forecasts 7-day rainfall from 30-day history
3. **Random Forest** — Scores terrain/elevation features
4. **XGBoost** — Scores historical pattern features
5. **Ensemble Predictor** — Weighted fusion of all 4 models

### Risk Score Formula:
```
Risk = 0.4×Rainfall + 0.3×Slope + 0.2×WaterExpansion + 0.1×ConstructionDensity
```

### Relief Decision Logic:
- **Low risk + roads accessible** → Road transport relief
- **Medium risk + hilly terrain** → Air support (supply drops)
- **Critical risk + flooding** → Air evacuation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate synthetic training data
python main.py generate-data

# Step 2: Train all models
python main.py train

# Step 3: Run prediction
python main.py predict --region Kullu      # Single region
python main.py predict --all               # All 8 HP regions
```

## Project Structure

```
disaster_prediction_hp/
├── config/config.yaml              # API keys, model params, thresholds
├── src/
│   ├── data_collection/            # Synthetic data + API fetchers
│   ├── image_processing/           # NDWI, NDVI, DEM, change detection
│   ├── models/                     # CNN, LSTM, Ensemble, Risk Scorer
│   ├── relief/                     # Route optimization, relief strategy
│   ├── visualization/              # Folium maps, HTML reports
│   └── pipeline/                   # Training & prediction orchestration
├── main.py                         # CLI entry point
├── models/saved/                   # Trained model weights
├── data/synthetic/                 # Generated training data
└── outputs/                        # Maps, reports
```

## Regions Covered

Kullu, Shimla, Mandi, Kangra, Manali, Chamba, Bilaspur, Solan

## Output

- **Console**: Colored risk reports with model contribution tables
- **outputs/disaster_map.html**: Interactive Folium map with risk zones
- **outputs/report.html**: Full HTML report with metrics

## Tech Stack

Python, PyTorch, scikit-learn, XGBoost, NetworkX, Folium, NumPy, pandas, OpenCV
