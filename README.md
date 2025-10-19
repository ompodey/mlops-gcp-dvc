#  IRIS Classification MLOps Pipeline

This repository demonstrates a complete **MLOps pipeline** built around the **Iris classification problem**.  
It integrates **Data Version Control (DVC)**, **Continuous Integration (CI)** using **GitHub Actions**, and **Continuous Model Evaluation** using **Iterative CML**.

---

## Project Overview

This project automates the end-to-end ML workflow:

1. **Data Versioning:**  
   DVC manages datasets and trained models.
2. **Model Training:**  
   A training script trains and stores model artifacts and metrics.
3. **Testing (PyTest):**  
   Automated testing ensures the data and model quality.
4. **Continuous Integration (CI):**  
   GitHub Actions automatically runs tests and validations on every push or PR.
5. **Continuous Machine Learning (CML):**  
   A CML job posts evaluation results (accuracy, metrics table, etc.) directly to your pull request.

---

##  Tech Stack

| Component | Technology Used |
|------------|----------------|
| Language | Python 3.10 |
| ML Library | scikit-learn |
| Data Versioning | DVC with Google Drive Remote |
| CI/CD | GitHub Actions |
| Model Evaluation | PyTest |
| Reporting | Iterative CML |
| Cloud | Google Cloud Storage (optional for remote storage) |

---

## Learning Outcomes

- Automated testing and validation for ML models

- Reproducible pipelines using DVC

- Git-based experiment tracking

- CI/CD with real-time reporting via GitHub Actions and CML

