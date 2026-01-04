

---
## Padmaja ML OPS Implementation

This repository demonstrates the implementation of MLOps practices for a Flower Prediction Model. It covers steps from basic setup to DevOps integration using CI/CD pipelines.

---

### 1. Add .gitignore File

Create a `.gitignore` file in the root directory to ignore files and folders that should not be tracked by Git (e.g., Python cache, environment files, logs, etc.).

---

### 2. Add Flower Prediction Model (Without DevOps)

Start by adding your machine learning model code and related files to the repository.

---

### 3. Create CI GitHub Actions Workflow

Created a CI workflow YAML file to automate:
- Checkout job (retrieves code from repository)
- Build job (installs dependencies and trains model)
- Upload artifacts for each Python version using matrix pip strategy

This uses GitHub Actions matrix to run jobs for multiple Python versions and saves model artifacts separately.


### 4. Test Model API Locally with curl

After starting your Flask app, you can test the /predict endpoint locally using the following PowerShell-friendly curl command:

```powershell
curl -X POST "http://127.0.0.1:5001/predict" -H "Content-Type: application/json" -d '{ "features": [5.1, 3.5, 1.4, 0.2] }'
```

This sends a sample request to your model API and returns the prediction result.

---

### 5. DevOps Integration (MLOps Stages: CI & CD)

** Dockerizing your Model **

Create a Dockerfile to containerize your application for consistent deployment.

**a. Build Docker Image:**

```powershell
docker build -t padmaja-ml-ops:latest .
```

**b. Run Docker Container:**

```powershell
docker run -p 5001:5001 padmaja-ml-ops:latest
```

**c. Test Model API in Docker:**

```powershell
curl -X POST "http://localhost:5001/predict" -H "Content-Type: application/json" -d '{ "features": [5.1, 3.5, 1.4, 0.2] }'
```
