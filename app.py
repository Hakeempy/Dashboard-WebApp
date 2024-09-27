import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(page_title="A-Z of AI Data Dashboard", layout="wide", page_icon="🤖")

# Custom CSS (same as before, so I'm omitting it for brevity)
# ...

# Top Bar (same as before, so I'm omitting it for brevity)
# ...

# User count
st.markdown('<p class="big-font">600 & counting Users</p>', unsafe_allow_html=True)


# Load data from CSV
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/hakeemwikireh/Downloads/user_data.csv")
    return df


df = load_data()

# Filters
st.subheader("Filters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    country = st.multiselect("Country", ["All"] + sorted(df['country'].unique().tolist()))

with col2:
    gender = st.radio("Gender", ["All", "Male", "Female"])

with col3:
    education = st.multiselect("Education", ["All"] + sorted(df['education'].unique().tolist()))

with col4:
    schools = st.multiselect("Schools", ["All"] + sorted(df['school'].unique().tolist()))


# Apply filters function
def apply_filters(df, country, gender, education, schools):
    filtered_df = df.copy()

    if "All" not in country and country:
        filtered_df = filtered_df[filtered_df['country'].isin(country)]

    if gender != "All":
        filtered_df = filtered_df[filtered_df['gender'] == gender]

    if "All" not in education and education:
        filtered_df = filtered_df[filtered_df['education'].isin(education)]

    if "All" not in schools and schools:
        filtered_df = filtered_df[filtered_df['school'].isin(schools)]

    return filtered_df


# Initialize session state
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = df

# Button actions
col1, col2 = st.columns(2)
with col1:
    if st.button("GENERATE REPORT"):
        st.session_state.filtered_df = apply_filters(df, country, gender, education, schools)
with col2:
    if st.button("RESET VALUES"):
        st.session_state.filtered_df = df

# Main content
st.header("Data Insights")

# Age distribution
st.subheader("Age Distribution")
age_fig = px.histogram(st.session_state.filtered_df, x='age', nbins=20, title='Age Distribution')
st.plotly_chart(age_fig, use_container_width=True)

# Education distribution
st.subheader("Education Distribution")
education_counts = st.session_state.filtered_df['education'].value_counts()
education_fig = px.pie(values=education_counts.values, names=education_counts.index, title='Education Distribution')
st.plotly_chart(education_fig, use_container_width=True)

# Country distribution
st.subheader("Country Distribution")
country_counts = st.session_state.filtered_df['country'].value_counts()
country_fig = px.bar(x=country_counts.index, y=country_counts.values, title='Country Distribution')
st.plotly_chart(country_fig, use_container_width=True)

# Gender distribution
st.subheader("Gender Distribution")
gender_counts = st.session_state.filtered_df['gender'].value_counts()
gender_fig = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution')
st.plotly_chart(gender_fig, use_container_width=True)

# School distribution
st.subheader("Top 10 Schools")
school_counts = st.session_state.filtered_df['school'].value_counts().head(10)
school_fig = px.bar(x=school_counts.index, y=school_counts.values, title='Top 10 Schools')
st.plotly_chart(school_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("FIXXIES • A-Z of AI 2024")