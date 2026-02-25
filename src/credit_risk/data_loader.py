import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def load_credit_data(file_path: str) -> pd.DataFrame:
    """Loads credit risk data from a CSV file."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file {file_path} not found.")
    
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic validation
    required_columns = [
        "income", "total_debt_outstanding", "fico_score", 
        "credit_lines_outstanding", "loan_amt_outstanding", 
        "years_employed", "default"
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns in CSV: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"Successfully loaded {len(df)} records.")
    return df
