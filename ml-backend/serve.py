# serve.py
# --------------------------------------------------
# Minimal REST API that returns a card risk score
# --------------------------------------------------
# Start:
#   pip install fastapi uvicorn joblib scikit-learn
#   python serve.py --model_dir artifacts --host 0.0.0.0 --port 8000
#
# POST /score { "card_number": "...", "limit": 8000 }
# ➜ { "riskScore": 27.3 }
# --------------------------------------------------
import argparse
import json
import math
import pathlib
from typing import List
import re
import os
import logging
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
feature_names = None
model = None

# ---- Luhn helpers -----------------------------------------------------------

def _luhn_checksum(num: str) -> int:
    digits = [int(d) for d in num]
    odd = sum(digits[-1::-2])
    even = sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
    return (odd + even) % 10


def passes_luhn(num: str) -> bool:
    return _luhn_checksum(num) == 0

# ---- Feature extraction -----------------------------------------------------

def extract_features(name: str, card_number: str, limit: float) -> pd.DataFrame:
    """Extract features for prediction."""
    global feature_names
    features = pd.DataFrame(columns=feature_names)
    
    # Name-based features
    features['name_length'] = [len(name)]
    features['name_word_count'] = [len(name.split())]
    features['name_has_numbers'] = [int(any(c.isdigit() for c in name))]
    features['name_has_special_chars'] = [int(not all(c.isalpha() or c.isspace() for c in name))]
    
    # Card number features
    features['card_number_length'] = [len(card_number)]
    
    # Limit features
    features['limit'] = [limit]
    features['limit_is_round'] = [int(limit % 1000 == 0)]
    features['limit_is_high'] = [int(limit > 50000)]
    features['limit_is_low'] = [int(limit < 1000)]
    
    return features

# ---- Pydantic models ---------------------------------------------------------

class CreditCardRequest(BaseModel):
    card_number: str = Field(
        ...,
        min_length=13,
        max_length=19,
        pattern=r'^\d+$',
        description="Credit card number (13-19 digits)",
        example="4532015112830366"
    )
    limit: float = Field(
        ...,
        gt=0,
        description="Credit limit amount",
        example=5000.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "card_number": "4532015112830366",
                "limit": 5000.0
            }
        }
    }

class ScoreResponse(BaseModel):
    riskScore: float = Field(
        ...,
        description="Risk score between 0 and 100",
        example=27.3,
        ge=0.0,
        le=100.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "riskScore": 27.3
            }
        }
    }

# ---- App factory ------------------------------------------------------------

def create_app(model_dir: pathlib.Path = pathlib.Path("artifacts")) -> FastAPI:
    global feature_names, model
    
    model_path = os.path.join(model_dir, 'risk_model.joblib')
    feature_path = os.path.join(model_dir, 'feature_names.txt')

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        raise RuntimeError("Model files not found. Please run train.py first.")

    model = joblib.load(model_path)
    with open(feature_path, 'r') as f:
        feature_names = f.read().splitlines()

    app = FastAPI(
        title="Credit Card Risk Scoring API",
        description="API for scoring credit card applications based on name and limit patterns",
        version="1.0.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/score", response_model=ScoreResponse)
    async def score_card(request: CreditCardRequest):
        """
        Score a credit card application based on the card number and limit.
        
        Parameters:
        - card_number: 13-19 digit card number
        - limit: Credit limit amount
        
        Returns:
        - riskScore: Risk score between 0 and 100
        """
        try:
            # Extract features
            features = extract_features("", request.card_number, request.limit)
            
            # Make prediction
            risk_score = float(model.predict(features)[0])
            
            return ScoreResponse(riskScore=round(risk_score, 2))
        except Exception as e:
            logger.error(f"Error scoring card: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app

# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path("artifacts"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8111)
    args = ap.parse_args()

    app = create_app(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
