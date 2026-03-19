# SME Credit Scoring & Financial Health Prediction

## Project Overview
This project involved the development and deployment of an end-to-end machine learning system designed to predict the financial health index of Small and Medium Enterprises (SMEs) in emerging markets. The solution transitioned from a raw tabular data challenge to a production-ready API and interactive dashboard.

## Technical Architecture & Modeling
The core inference engine utilizes a weighted ensemble of Gradient Boosting Decision Trees (GBDTs), specifically LightGBM, CatBoost, and XGBoost. The multi-model approach was designed to minimize variance and improve generalization across diverse macroeconomic regimes.

- **Data Engineering:** Developed a custom preprocessing pipeline handling high-cardinality categorical variables through target encoding and frequency-based mapping.
- **Model Performance:** Achieved a Top 10% ranking in the Zindi Financial Health Prediction Challenge (Public Score: 0.8863, Private Score: 0.8718).
- **Optimization:** Hyperparameters were tuned using Bayesian Optimization (Optuna) to maximize Macro-F1 scores, prioritizing the detection of high-risk "Low" financial health profiles.

## Engineering & Solutions Architect Approach
A significant challenge in the project was the "Sparsity Bias" encountered during the transition from batch training to real-time inference. Since the training data utilized over 90 survey features while the production UI captures only 7 key metrics, the model initially defaulted to high-risk classifications due to the high volume of missing data.

- **Baseline Inversion Strategy:** Implemented a non-trivial data engineering fix by extracting the statistical mode of high-risk profiles into a JSON-based baseline template. Incoming API requests overlay user data onto this baseline, ensuring the model operates within its trained distribution while maintaining strict risk-aversion.
- **Microservices Deployment:** Architected a FastAPI backend served through a non-root Docker container, ensuring enterprise-grade security and cross-platform portability.
- **User Interface:** Developed a high-performance Vanilla JavaScript frontend inspired by modern fintech design systems (e.g., Stripe, Brex) to visualize probability distributions and architectural insights.

## Deployment & Live Demo
The system is deployed as a live Docker Space on Hugging Face, utilizing an optimized CI/CD workflow that excludes heavy training datasets in favor of lean inference-only dependencies.

- **Live URL:** [https://huggingface.co/spaces/okechobonyo/sme-credit-scoring](https://huggingface.co/spaces/okechobonyo/sme-credit-scoring)
- **Deployment Platform:** Hugging Face Spaces (Docker SDK)
- **API Framework:** FastAPI (Python 3.10)

## Technical Stack
- **Languages:** Python, JavaScript, HTML, CSS.
- **Libraries:** LightGBM, CatBoost, XGBoost, Scikit-Learn, Pandas, Joblib.
- **Tools:** Docker, Hugging Face CLI, Uvicorn, Pydantic.

## Key Developer Documentation
- To run locally: `uvicorn app:app --reload`
- To train models: `python Financial Health Data/export_api_models.py`
- To verify API: `python test.py`
