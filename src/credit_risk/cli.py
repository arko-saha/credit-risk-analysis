import sys
import logging
from .data_loader import load_credit_data
from .preprocessor import CreditPreprocessor
from .model import CreditRiskModel
from .risk_calculator import RiskCalculator
from .config import settings
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_user_profile():
    """Prompts user for customer profile data."""
    print("\n--- Credit Risk Assessment Profile ---")
    try:
        profile = {
            "income": float(input("Annual Income ($): ")),
            "total_debt_outstanding": float(input("Total Debt Outstanding ($): ")),
            "fico_score": int(input("FICO Score (300-850): ")),
            "credit_lines_outstanding": int(input("Number of Credit Lines: ")),
            "loan_amt_outstanding": float(input("Current Loan Amount ($): ")),
            "years_employed": float(input("Years of Employment: ")),
        }
        return profile
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return None

def run_assessment():
    """E2E workflow for training and interactive assessment."""
    try:
        # 1. Load Data
        df = load_credit_data(settings.DATA_PATH)
        
        # 2. Preprocessing
        preprocessor = CreditPreprocessor()
        processed_df = preprocessor.transform_features(df)
        
        X = processed_df.drop(columns=['default'])
        y = processed_df['default']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, random_state=settings.MODEL_RANDOM_STATE
        )
        
        X_train_res, y_train_res = preprocessor.fit_resample(X_train, y_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # 3. Model Training
        model = CreditRiskModel(random_state=settings.MODEL_RANDOM_STATE)
        model.train(X_train_res, y_train_res)
        
        # 4. Evaluation
        acc, report = model.evaluate(X_test_scaled, y_test)
        logger.info(f"Model Evaluation Summary:\n{report}")
        
        # 5. Interactive Assessment
        calculator = RiskCalculator(model, preprocessor)
        
        while True:
            profile = get_user_profile()
            if not profile:
                continue
                
            results = calculator.calculate_risk_profile(profile)
            
            print("\n" + "="*40)
            print("         RISK ASSESSMENT REPORT         ")
            print("="*40)
            print(f"Risk Classification:    {results['risk_level'].upper()}")
            print(f"Prob. of Default:       {results['probability_of_default']:.2%}")
            print(f"Expected Loss (EL):     ${results['expected_loss']:,.2f}")
            print("="*40)
            
            cont = input("\nAssess another customer? (y/n): ")
            if cont.lower() != 'y':
                break
                
    except Exception as e:
        logger.exception("An error occurred during assessment:")
        sys.exit(1)

if __name__ == "__main__":
    run_assessment()
