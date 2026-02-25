from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATA_PATH: str = Field("data/Task 3 and 4_Loan_Data.csv", env="DATA_PATH")
    MODEL_RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Risk Assessment Thresholds
    LOW_RISK_THRESHOLD: float = 0.2
    MEDIUM_RISK_THRESHOLD: float = 0.5
    
    # Financial constants
    DEFAULT_RECOVERY_RATE: float = 0.10

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()
