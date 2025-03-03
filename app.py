import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
import plotly.figure_factory as ff

# Set up the app with a modern theme
st.set_page_config(page_title="Gaurav Bhoi - AI Portfolio", layout="wide")

# Custom CSS to remove white bar issue
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(-45deg, #1f1c2c, #928dab, #1f4037, #99f2c8);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .block-container { padding-top: 0px; padding-bottom: 0px; }
        .main-title { text-align: center; font-size: 50px; color: #fff; font-weight: bold; }
        .subtitle { text-align: center; font-size: 24px; color: #ddd; }
        .profile-card, .card {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
            text-align: center;
            margin-bottom: 20px;
            color: white;
        }
        .profile-img {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar Navigation (Added Power BI Dashboard)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Projects", "🤖 ML Demos", "🛠 Skills", "📊 Power BI Dashboard", "📜 Resume", "📞 Contact"])

# 📌 Home Page
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>Gaurav Bhoi</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle'>🚀 Aspiring Data Scientist & Machine Learning Engineer</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("D:/portfolio_2025/profile.jpg", caption="Gaurav Bhoi", use_container_width=True, output_format="JPEG")
    with col2:
        st.markdown("<div class='profile-card'>", unsafe_allow_html=True)
        st.write("Passionate about AI, predictive modeling, and data-driven insights. I build AI-powered applications, ML models, and data solutions.")
        st.write("[🔗 LinkedIn](https://www.linkedin.com/in/gauravbhoi) | [📂 GitHub](https://github.com/GauravN03)")
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Projects", "10+", "🆕 Updated")
    col2.metric("📚 Certifications", "5", "📜 IBM Certified")
    col3.metric("👥 LinkedIn Network", "300+", "🚀 Growing")
if page == "🛠 Skills":
    st.title("🛠 My Technical Skills")

    skills = {
        "💻 Programming & Databases": [
            "Python (Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch)",
            "SQL (MySQL, PostgreSQL)",
            "MongoDB (NoSQL)"
        ],
        "☁️ Big Data & Cloud": [
            "Apache Spark, Hadoop",
            "Google Cloud",
            "AWS (EC2, Lambda, S3)",
            "Azure"
        ],
        "🛠 DevOps & Tools": [
            "Docker, Kubernetes",
            "Git, GitHub",
            "Jupyter Notebook",
            "VS Code, PyCharm"
        ],
        "🤖 Machine Learning & AI": [
            "Deep Learning, NLP",
            "Random Forest, KNN, SVM, PCA",
            "XGBoost, Reinforcement Learning"
        ],
        "📊 Data Visualization": [
            "Power BI, Tableau",
            "Matplotlib, Seaborn",
            "Excel (Pivot tables, Charts)"
        ]
    }

    for category, skill_list in skills.items():
        st.subheader(category)
        for skill in skill_list:
            st.write(f"✅ {skill}")

    st.success("🚀 Skills Section Updated! Let me know if you need modifications.")

# 📊 Projects Page
elif page == "📊 Projects":
    st.title("📂 Data Science & AI Projects")

    projects = {
        "🛩 UAV Propeller Performance Analysis": {
            "description": "Optimized UAV propeller thrust using Computational Fluid Dynamics (CFD).",
            "tech": "Python, TensorFlow, Power BI, CFD Simulations",
            "image": "D:/portfolio_2025/uav.jpg"
        },
        "🏥 Healthcare Insurance Fraud Detection": {
            "description": "Identified fraudulent claims saving $500K using ML models.",
            "tech": "Python, SQL, Random Forest, Logistic Regression",
            "image": "D:/portfolio_2025/health.jpg"
        },
        "🍷 Wine Quality Prediction": {
            "description": "Developed ML models to predict wine quality with 87% accuracy.",
            "tech": "Python, Pandas, Matplotlib, Scikit-Learn",
            "image": "D:/portfolio_2025/wine.jpg"
        }
    }

    for project, details in projects.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(details["image"], caption=project, use_container_width=True)
        with col2:
            st.subheader(project)
            st.write(f"📌 **Description:** {details['description']}")
            st.write(f"🛠 **Technologies:** {details['tech']}")
        st.markdown("---")

# 🤖 ML Demo Page (Now with Confusion Matrix & Feature Importance)
elif page == "🤖 ML Demos":
    st.title("🤖 Machine Learning Demos")
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    model_choice = st.selectbox("Choose an ML Model", ["Random Forest", "Logistic Regression", "SVM"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "SVM":
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown(f"### ✅ Model Accuracy: {accuracy:.2f}")

    if model_choice == "Random Forest":
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=[f"Feature {i}" for i in range(len(feature_importance))], ax=ax, palette="viridis")
        ax.set_title("🔍 Feature Importance in Random Forest")
        st.pyplot(fig)

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"], colorscale="blues")
    st.plotly_chart(fig)

# 📊 Power BI Dashboard
elif page == "📊 Power BI Dashboard":
    st.title("📊 Interactive Power BI Dashboard")
    st.write("Explore my real-time data visualizations below:")
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiNGM5MGE3ZTktZDQwOC00YWNiLTg2ZTYtYTZiZmM0N2Q5ZTBmIiwidCI6IjMwMDJhODgwLWYxMTItNGUwZi05Nzk2LTA4MGExOGYxMWRkNSJ9"
    st.markdown(f'<iframe width="1000" height="600" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)


# 📜 Resume Page
elif page == "📜 Resume":
    st.title("📜 My Resume")
    resume_path = "D:/portfolio_2025/GauravBhoi_Resume_DataScience_AI.pdf"
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as f:
            st.download_button("📥 Download Resume", f, file_name="GauravBhoi_Resume.pdf")

# 📞 Contact Page
elif page == "📞 Contact":
    st.title("📞 Contact Me")
    st.write("📧 Email: gauravbhoi2003@gmail.com")
    st.write("📞 Mobile: **+91 8668417061**")
    st.write("🔗 LinkedIn: [Gaurav Bhoi](https://www.linkedin.com/in/gauravbhoi)")
    st.write("📂 GitHub: [GauravN03](https://github.com/GauravN03)")
