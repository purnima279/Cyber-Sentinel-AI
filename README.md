
# 🛡️ CyberSentinel
## NLP-Based Cybersecurity Incident Intelligence and Analytics Platform

CyberSentinel is a full-stack web application that analyzes cybersecurity incident descriptions using **Natural Language Processing (NLP)**. It provides incident management, automated text analysis, database storage, and interactive analytics.

## 🎯 Project Objectives
- Automate cybersecurity incident text analysis.
- Apply NLP preprocessing, feature extraction, classification, and sentiment analysis.
- Integrate REST APIs, relational databases, and interactive dashboards.
- Generate insights using SQL, Excel, and data visualization.

## ✨ Key Features

**Incident Management**
- Create, view, update, delete, search, and filter incidents.
- Filter by category, severity, and status.

**NLP Analysis**
- Text cleaning, tokenization, stopword removal, and lemmatization.
- Named Entity Recognition (NER) and regex-based extraction.
- TF-IDF and n-gram feature extraction.
- Incident category prediction, sentiment analysis, word count, and keyword count.

**Analytics Dashboard**
- Total incidents and category distribution.
- Severity, status, and sentiment analysis.
- Critical/high-severity incident monitoring.
- Interactive charts using Chart.js, SQL analytics, and Excel reports.

## 🛠️ Technology Stack

- **Frontend:** React, Vite, Bootstrap, Axios, React Router, Chart.js
- **Backend:** Node.js, Express.js, MySQL2, CORS, dotenv
- **NLP Service:** Python, FastAPI, Uvicorn, NLTK, spaCy, scikit-learn, Pandas, NumPy, Joblib
- **Database:** MySQL 8.4
- **Analytics:** SQL, Microsoft Excel, Chart.js

## 🏗️ System Architecture

```text
React Frontend (React + Chart.js)
              |
              v
Node.js + Express REST API
         /         \
        v           v
   MySQL DB     Python FastAPI
                    |
                    v
              NLP Processing
                    |
                    v
       Features, Classification,
       Sentiment & Text Statistics
                    |
                    v
          MySQL Storage & Analytics
```

## ⚙️ Processing Workflow

1. User submits or updates an incident.
2. React sends the request to the Express REST API.
3. The backend sends incident text to the Python NLP service.
4. NLP performs preprocessing, feature extraction, classification, and sentiment analysis.
5. Analysis results are stored in MySQL.
6. The frontend displays incident details and updated analytics.

## 📊 Project Outcomes

- Centralized cybersecurity incident management.
- Automated NLP-based text analysis.
- Database-backed incident records and analytics.
- Interactive visualizations for identifying incident trends.

## 🔮 Future Improvements

- Train and integrate advanced machine learning classifiers.
- Add real-time alerts and severity prediction.
- Implement authentication, role-based access, and audit logs.
- Deploy the platform to the cloud.

---
**License:** MIT
