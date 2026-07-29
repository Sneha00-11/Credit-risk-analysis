import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Load model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv("Loan Prediction Dataset.csv")

st.set_page_config(page_title="Credit Risk Analysis System", layout="wide")
st.title("🏦 Credit Risk Analysis System")

# --- Sidebar: applicant input form ---
st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 240, 60])
credit_history = st.sidebar.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# --- Encode inputs to match training format ---
input_data = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],
    'Married': [1 if married == "Yes" else 0],
    'Dependents': [dependents],
    'Education': [1 if education == "Graduate" else 0],
    'Self_Employed': [1 if self_employed == "Yes" else 0],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [1 if "Good" in credit_history else 0],
    'Property_Area_Semiurban': [1 if property_area == "Semiurban" else 0],
    'Property_Area_Urban': [1 if property_area == "Urban" else 0]
})

# --- Scale and predict ---
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

# --- Risk category ---
if probability >= 0.8:
    risk_category = "Low Risk"
    recommendation = "Approve Loan"
elif probability >= 0.4:
    risk_category = "Medium Risk"
    recommendation = "Review Manually"
else:
    risk_category = "High Risk"
    recommendation = "Reject Application"

# --- Display results ---
st.subheader("Prediction Result")
col1, col2, col3 = st.columns(3)
col1.metric("Approval Probability", f"{probability*100:.1f}%")
col2.metric("Risk Category", risk_category)
col3.metric("Recommendation", recommendation)

st.divider()

# --- Professional Gauge Chart ---
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=probability * 100,

    number={
        'suffix': "%",
        'font': {'size': 36, 'color': "#1f2937"}
    },

    title={
        'text': "Approval Probability",
        'font': {'size': 16, 'color': "#6b7280"}
    },

    gauge={
        'axis': {
            'range': [0, 100],
            'tickwidth': 1,
            'tickcolor': "#9ca3af"
        },

        'bar': {
            'color': "#3b82f6",
            'thickness': 0.3
        },

        'bgcolor': "white",

        'borderwidth': 2,
        'bordercolor': "#e5e7eb",

        'steps': [
            {'range': [0, 40], 'color': "#fee2e2"},   # soft red
            {'range': [40, 80], 'color': "#fef3c7"},  # soft yellow
            {'range': [80, 100], 'color': "#dcfce7"}  # soft green
        ],

        'threshold': {
            'line': {'color': "#111827", 'width': 4},
            'thickness': 0.75,
            'value': probability * 100
        }
    }
))

fig_gauge.update_layout(
    height=320,
    margin=dict(l=20, r=20, t=60, b=20),

    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    font={'family': "Arial"}
)

st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================
# 👤 APPLICANT ANALYSIS (ALL KPI STYLE)
# ============================================
st.subheader("👤 This Applicant's Analysis")

avg_income = df['ApplicantIncome'].mean()
avg_loan = df['LoanAmount'].mean()
total_income = applicant_income + coapplicant_income

income_diff = ((applicant_income - avg_income) / avg_income) * 100
loan_diff = ((loan_amount - avg_loan) / avg_loan) * 100
income_percentile = (df['ApplicantIncome'] < applicant_income).mean() * 100

ratio = total_income / loan_amount if loan_amount > 0 else 0

# ============================================
# 🔹 ROW 1 (MAIN KPIs)
# ============================================
col1, col2, col3 = st.columns(3)

col1.metric("Applicant Income", f"₹{applicant_income:,}", f"{income_diff:+.1f}% vs avg")
col2.metric("Loan Amount", f"₹{loan_amount:,}", f"{loan_diff:+.1f}% vs avg", delta_color="inverse")
col3.metric("Income Rank", f"{income_percentile:.0f}th percentile")

# ============================================
# 🔹 ROW 2 (FINANCIAL STRENGTH)
# ============================================
col1, col2, col3 = st.columns(3)

col1.metric("Total Income", f"₹{total_income:,}")
col2.metric("Income-Loan Ratio", f"{ratio:.2f}x")

if ratio >= 6:
    col3.metric("Repayment Capacity", "Strong", "Good")
elif ratio >= 3:
    col3.metric("Repayment Capacity", "Moderate", "Average")
else:
    col3.metric("Repayment Capacity", "Weak", "Risky")

# ============================================
# 🔹 ROW 3 (POSITIONING + CREDIT)
# ============================================
col1, col2, col3 = st.columns(3)

col1.metric(
    "Credit History",
    "Good" if "Good" in credit_history else "Poor"
)

# --- Similar applicants ---
st.write("**How Similar Applicants Fared Historically**")
similar = df[
    (df['Credit_History'] == (1 if "Good" in credit_history else 0)) &
    (df['Property_Area'] == property_area)
]
if len(similar) > 0:
    similar_approval_rate = (similar['Loan_Status'] == 'Y').mean() * 100
    st.write(
        f"Among **{len(similar)}** applicants sharing this applicant's Credit History and Property Area, "
        f"**{similar_approval_rate:.0f}%** were historically approved."
    )
    fig_similar = go.Figure(go.Bar(
        x=['Similar Applicants Approval Rate', "This Applicant's Predicted Probability"],
        y=[similar_approval_rate, probability * 100],
        marker_color=['#4C72B0', '#DD8452']
    ))
    fig_similar.update_layout(yaxis_title="Percentage (%)", height=300)
    st.plotly_chart(fig_similar, use_container_width=False)
else:
    st.write("No closely similar applicants found in the dataset for comparison.")

# --- Feature importance ---
st.write("**Key Factors Influencing This Prediction**")
importance = pd.Series(model.feature_importances_, index=input_data.columns).sort_values(ascending=True)
fig_importance = go.Figure(go.Bar(
    x=importance.values,
    y=importance.index,
    orientation='h',
    marker_color='teal'
))
fig_importance.update_layout(height=400, xaxis_title="Importance Score")
st.plotly_chart(fig_importance, use_container_width=True)

st.divider()

# --- Dataset insights section ---
st.subheader("Dataset Insights")

tab1, tab2, tab3 = st.tabs(["Income Patterns", "Credit History Impact", "Property Area Trends"])

with tab1:
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.histplot(df['ApplicantIncome'], bins=40, kde=True, ax=ax)
    ax.set_title("Applicant Income Distribution", fontsize=10)
    ax.set_xlabel('Application Income', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.18,
    top=0.90
)
    st.pyplot(fig, use_container_width = False)

with tab2:
    fig, ax = plt.subplots(figsize=(4.5, 1.8))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=ax)
    ax.set_title("Loan Approval by Credit History", fontsize=10)
    plt.xlabel('Credit History (1=Good, 0=Bad)', fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'], fontsize=5)
    total = len(df)
    for container in ax.containers:
      labels = [f'{(v.get_height()/total)*100:.1f}%' for v in container]
      ax.bar_label(container, labels=labels,label_type='edge', padding=-1, fontsize=5)
    st.pyplot(fig, use_container_width = False)

with tab3:
    fig, ax = plt.subplots(figsize=(4, 2.2))
    pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', stacked=True, colormap = 'Set2', ax=ax)
    ax.set_title("Risk Distribution by Property Area", fontsize=10)
    plt.xlabel('Property Area', fontsize=8)
    plt.tick_params(labelsize=6)
    plt.xticks(rotation=0)
    plt.ylabel('Count', fontsize=8)
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'], loc="upper left",
    bbox_to_anchor=(1.02, 1),   # outside the plot
    borderaxespad=0,
    fontsize=5,
    title_fontsize=5)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f',label_type='center', color='white', fontsize=6)
    plt.tight_layout()
    st.pyplot(fig, use_container_width = False)

st.divider()

# --- All Customers Overview ---
st.subheader("All Customers Overview")


df_analysis = df.copy()
df_analysis['Credit_History'] = df_analysis['Credit_History'].fillna(df_analysis['Credit_History'].mode()[0])

st.write("**Full Customer Data**")
st.dataframe(df_analysis)
