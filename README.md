# Customer Churn Prediction Web App

A containerized, cloud-ready web application that predicts customer churn using a **Random Forest** machine learning model. This project demonstrates a full-stack ML deployment workflow with **FastAPI** backend, **Streamlit** frontend, and **Docker & Kubernetes** for scalable, production-ready deployment.

---

## Features

- **Churn Prediction**: Predict whether a customer is likely to churn based on **43 customer attributes**.  
- **Adaptive Input UI**: Users choose which features to input; missing features are automatically filled with the **most common (mode) values**.  
- **Categorical Grouping**: Related features (e.g., Partner Yes/No) are grouped for cleaner input.  
- **Interactive Frontend**: Streamlit interface provides intuitive input forms and instant prediction results with probability scores.  
- **Containerized Deployment**: Uses Docker to package the app with all dependencies.  
- **Kubernetes-Ready**: Includes Deployment and Service YAML files for scalable orchestration.

---

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn  
- **Frontend**: Streamlit  
- **Machine Learning**: scikit-learn, XGBoost, joblib  
- **Containerization**: Docker  
- **Orchestration**: Kubernetes  

---
