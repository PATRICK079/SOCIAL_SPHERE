import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch






# Load Models and Scalers
@st.cache_resource
def load_classification_model():
    model = joblib.load("xgboost_classifier_mod.pkl")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

@st.cache_resource
def load_regression_model():
    model = joblib.load("random_regressor_model.pkl")
    scaler = joblib.load("random_forest_scaler.pkl")
    return model, scaler

# One-hot encoded column names expected
classification_cols = [
    'age', 'gender', 'academic_level', 'avg_daily_usage_hours',
    'affects_academic_performance', 'sleep_hours_per_night',
    'mental_health_score', 'addicted_score',
    'most_used_platform_Instagram', 'most_used_platform_KakaoTalk',
    'most_used_platform_LINE', 'most_used_platform_LinkedIn',
    'most_used_platform_Snapchat', 'most_used_platform_TikTok',
    'most_used_platform_Twitter', 'most_used_platform_VKontakte',
    'most_used_platform_WeChat', 'most_used_platform_WhatsApp',
    'most_used_platform_YouTube', 'relationship_status_In Relationship',
    'relationship_status_Single', 'continent_America', 'continent_Asia',
    'continent_Europe', 'continent_Oceania'
]

regression_cols = [
    'age', 'gender', 'academic_level', 'avg_daily_usage_hours',
    'affects_academic_performance', 'sleep_hours_per_night',
    'mental_health_score', 'conflicts_over_social_media',
    'most_used_platform_Instagram', 'most_used_platform_KakaoTalk',
    'most_used_platform_LINE', 'most_used_platform_LinkedIn',
    'most_used_platform_Snapchat', 'most_used_platform_TikTok',
    'most_used_platform_Twitter', 'most_used_platform_VKontakte',
    'most_used_platform_WeChat', 'most_used_platform_WhatsApp',
    'most_used_platform_YouTube', 'relationship_status_In Relationship',
    'relationship_status_Single', 'continent_America', 'continent_Asia',
    'continent_Europe', 'continent_Oceania'
]

ordinal_maps = {
    'academic_level': {'High School': 0, 'Undergraduate': 1, 'Graduate': 2},
    'gender': {'Female': 0, 'Male': 1},
    'affects_academic_performance': {'No': 0, 'Yes': 1}
}
st.sidebar.title("📌 Main Menu")
menu = st.sidebar.radio("Go to", [
    "About App",
    "Classification Task",
    "Regression Task",
    "Clustering Task",
    "Disclaimer",
    "Dataset Citation"
])

# --------------------------------------------------------
# 🔶 CLASSIFICATION SECTION
# --------------------------------------------------------
if menu == "Classification Task":
    st.markdown("## Social Media Conflict level Prediction")

    model, scaler = load_classification_model()

    # Inputs
    age = st.number_input("Age", 15, 35, 20)
    gender = st.selectbox("Gender", ["Female", "Male"])
    academic = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
    affects = st.selectbox("Affects Academic Performance?", ["No", "Yes"])
    usage = st.number_input("Avg Daily Usage (hrs)", 0.0, 24.0, 4.0)
    sleep = st.number_input("Sleep Hours", 0.0, 12.0, 6.0)
    mental = st.slider("Mental Health Score (1–10)", 1, 10, 5)
    addiction = st.slider("Addicted Score (1–10)", 1, 10, 6)
    platform = st.selectbox("Most Used Platform", [
        "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat", "TikTok", "Twitter",
        "VKontakte", "WeChat", "WhatsApp", "YouTube"
    ])
    status = st.selectbox("Relationship Status", ["Single", "Complicated", "In Relationship"])
    continent = st.selectbox("Continent", ["America", "Asia", "Europe", "Oceania","Africa"])

    if st.button("Predict Conflict Level"):
        try:
            data = {
                "age": age,
                "gender": ordinal_maps["gender"][gender],
                "academic_level": ordinal_maps["academic_level"][academic],
                "avg_daily_usage_hours": usage,
                "affects_academic_performance": ordinal_maps["affects_academic_performance"][affects],
                "sleep_hours_per_night": sleep,
                "mental_health_score": mental,
                "addicted_score": addiction,
                f"most_used_platform_{platform}": 1,
                f"relationship_status_{status}": 1,
                f"continent_{continent}": 1
            }

            for col in classification_cols:
                if col not in data:
                    data[col] = 0

            df = pd.DataFrame([data])[classification_cols]
            scaled = scaler.transform(df)
            pred = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]

            result = " High Conflict" if pred == 1 else " Low Conflict"
            st.success(f"Prediction: This student will likely have  {result} in their relationship over social media usage")
            #st.info(f"Confidence → Low: {proba[0]:.2f}, High: {proba[1]:.2f}")
            # Plot pie chart
            fig, ax = plt.subplots()
            if pred == 1:
              colors = ['lightgray', 'red']
            else:
              colors = ['green', 'lightgray']

            labels = ['Low Conflict', 'High Conflict']
            ax.pie(proba, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error during classification: {e}")


if menu == "About App":
    st.title("Welcome to the Social Sphere App")
    st.image("Screenshot 2025-07-09 at 21.59.03.png")#use_container_width=True)


    st.markdown("""
    **Social Sphere** is an interactive machine learning-powered platform designed to explore how social media habits impact students’ well-being.  
    It analyzes anonymized data from students aged **16 to 25** across multiple countries, offering insights into how digital behaviors correlate with:

    - 🎓 Academic performance  
    - 🧠 Mental health and sleep patterns  
    - 💬 Relationship dynamics and social conflicts  

    This project serves as a valuable tool for **researchers**, **educators**, and **mental health advocates**, providing a data-driven lens into student digital behavior.
    """)

    st.markdown("---")
    st.subheader("🔍 What This App Does")
    st.markdown("""
    Using a combination of supervised and unsupervised machine learning models, Social Sphere performs **three core tasks**:

    **1. 📊 Classification Task**  
    Predicts the likelihood of **relationship conflict** over social media usage by analyzing student behavior and lifestyle patterns.  
    ✅ Helps identify students at risk of interpersonal tension related to digital habits.

    **2. 📈 Regression Task**  
    Estimates the student’s **self-reported addiction score** based on demographic and usage patterns.  
    ✅ Useful for gauging levels of digital dependency for support or intervention.

    **3. 🧠 Clustering Task**  
    Groups students into **behavioral clusters** using unsupervised learning to uncover hidden patterns.  
    ✅ Helps spotlight vulnerable students who may benefit from targeted guidance or help.
    """)
    st.markdown("USE THE SIDE BAR TO CHOOSE A TASK ")
    st.markdown("---")
    st.subheader("🌐 Why It Matters")
    st.markdown("""
    With students increasingly immersed in online spaces, **Social Sphere** provides actionable insight into how these digital behaviors affect real-life outcomes.  
    By combining **predictive analytics** with a user-friendly interface built in **Streamlit**, this app empowers stakeholders to:

    - Explore data interactively  
    - Make informed decisions  
    - Drive meaningful intervention programs  
    """)


elif menu == "Clustering Task":
    run_clustering(df)

elif menu == "Disclaimer":
    st.title("⚠️ Disclaimer")
    st.write("""
        This application is for research and educational purposes only. 
        Predictions made by machine learning models are probabilistic 
        and should not replace expert advice or judgment.
    """)




elif menu == "Dataset Citation":
    st.title("📚 Dataset Citation")
    st.markdown("""
        The dataset used in this project is publicly available on Kaggle:

    **Title:** Social Media Addiction vs Relationships  
    **Author:** Adil Shamim  
    **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)

    **License:** This dataset is made available for academic and educational purposes. Please ensure proper citation if reused.

    ---
    *We acknowledge the efforts of the data creator and contributors for making this resource available to the public.*
    
    """)


# --------------------------------------------------------
# 🔷 REGRESSION SECTION
# --------------------------------------------------------
elif menu == "Regression Task":
    st.markdown("## Social Media Addiction Score ")

    model, scaler = load_regression_model()

    # Inputs
    age = st.number_input("Age", 15, 35, 20)
    gender = st.selectbox("Gender", ["Female", "Male"])
    academic = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
    affects = st.selectbox("Affects Academic Performance?", ["No", "Yes"])
    usage = st.number_input("Avg Daily Usage (hrs)", 0.0, 24.0, 4.0)
    sleep = st.number_input("Sleep Hours", 0.0, 12.0, 6.0)
    mental = st.slider("Mental Health Score (1–10)", 1, 10, 5)
    conflict = st.slider("Conflicts Over Social Media (1–10)", 1, 10, 5)
    platform = st.selectbox("Most Used Platform", [
        "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat", "TikTok", "Twitter",
        "VKontakte", "WeChat", "WhatsApp", "YouTube"
    ])
    status = st.selectbox("Relationship Status", ["Single", "In Relationship"])
    continent = st.selectbox("Continent", ["America", "Asia", "Europe", "Oceania"])

    if st.button("Predict Addicted Score"):
        try:
            data = {
                "age": age,
                "gender": ordinal_maps["gender"][gender],
                "academic_level": ordinal_maps["academic_level"][academic],
                "avg_daily_usage_hours": usage,
                "affects_academic_performance": ordinal_maps["affects_academic_performance"][affects],
                "sleep_hours_per_night": sleep,
                "mental_health_score": mental,
                "conflicts_over_social_media": conflict,
                f"most_used_platform_{platform}": 1,
                f"relationship_status_{status}": 1,
                f"continent_{continent}": 1
            }

            for col in regression_cols:
                if col not in data:
                    data[col] = 0

            df = pd.DataFrame([data])[regression_cols]
            scaled = scaler.transform(df)
            pred = model.predict(scaled)[0]
            st.success(f"Prediction: This student social media addiction score is  {pred:.2f} out of 10")
            # Pie chart
            fig, ax = plt.subplots()
            values = [pred, max(0, 10 - pred)]
            colors = ['green', 'lightgray']
            labels = [f"Score: {pred:.2f}", ""]

            ax.pie(values, labels=labels, startangle=90, colors=colors,
            counterclock=False, wedgeprops={"width": 0.4}, autopct='%1.1f%%')
            ax.axis("equal")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during regression: {e}")
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Model Tracking & Experimentation")
    st.markdown("""
    All machine learning models in this app were **logged using MLflow** and tracked via **DagsHub**.

    For the **classification task**, performance was monitored using:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1 Score**

    For the **regression task**, key metrics included:
    - **MAE** (Mean Absolute Error)
    - **RMSE** (Root Mean Squared Error)
    - **R²** and **Adjusted R²**
    - **Cross-Validation R²**

    🔗 [**View Experiments on DagsHub**](https://dagshub.com/PATRICK079/SDS-CP029-social-sphere/experiments)
    """)
