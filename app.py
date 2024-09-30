import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import streamlit.components.v1 as components

# from scipy.special import style

# components.iframe("https://flutter.dev/term", height=500)

#Set page config
st.set_page_config(page_title="A-Z of AI Data Dashboard", layout="wide", page_icon="🤖")


#Load external CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


load_css('style.css')

# st.button("login")
# st.button("log out")
#
# st.markdown("""
# <div class="div-container">
#     <div>
#         <button class="stButton">Login</button>
#     </div>
#     <div>
#         <button class="stButton">Log Out</button>
#     </div>
# </div>
# """, unsafe_allow_html=True)
#
# col1, col2 = st.columns(2)
#
# with col1:
#     st.title("Hakeem")
# with col2

#Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Surveys", "Quick Quiz", "Users"])

# Top Bar
# Header div
st.markdown('<div class="stHeader">', unsafe_allow_html=True)

# Columns layout
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.markdown('<div class="stLogo">', unsafe_allow_html=True)
    st.image('logo.png', width=50)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stTitle">', unsafe_allow_html=True)
    st.markdown("### Data Dashboard - Ghana")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stButtons">', unsafe_allow_html=True)
    # st.button("login")
    st.button("refresh")
    st.markdown('</div>', unsafe_allow_html=True)

# Close header div
st.markdown('</div>', unsafe_allow_html=True)

if page == "Surveys":
    st.title("Surveys")
    st.write("This is the Surveys page. You can add survey-related content here.")

    # Placeholder content for Surveys
    st.subheader("Available Surveys")
    surveys = ["Customer Satisfaction", "Product Feedback", "User Experience"]
    for survey in surveys:
        if st.button(f"Take {survey} Survey"):
            st.write(f"You've selected the {survey} survey. (Placeholder for survey content)")

elif page == "Quick Quiz":
    st.title("Quick Quiz")
    st.write("This is the Quick Quiz page. You can add quiz-related content here.")

    # Placeholder content for Quick Quiz
    st.subheader("Today's Quiz")
    question = "What is the capital of Ghana?"
    options = ["Accra", "Kumasi", "Tamale", "Cape Coast"]
    answer = st.radio("Select your answer:", options)
    if st.button("Submit Answer"):
        if answer == "Accra":
            st.success("Correct! Accra is the capital of Ghana.")
        else:
            st.error(f"Sorry, {answer} is not correct. The capital of Ghana is Accra.")

elif page == "Users":
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
    green = ['#00A674']

    # Age distribution
    st.subheader("Age Distribution")
    age_fig = px.histogram(st.session_state.filtered_df, x='age', nbins=20, title='Age Distribution',
                       color_discrete_sequence=['#00A674'])  # Set bar color to green
    st.plotly_chart(age_fig, use_container_width=True)

# Education distribution
    st.subheader("Education Distribution")
    education_counts = st.session_state.filtered_df['education'].value_counts()
    education_fig = px.pie(values=education_counts.values, names=education_counts.index, title='Education Distribution',
                       color_discrete_sequence=green)  # Shades of green for pie chart
    st.plotly_chart(education_fig, use_container_width=True)

# Country distribution
    st.subheader("Country Distribution")
    country_counts = st.session_state.filtered_df['country'].value_counts()
    country_fig = px.bar(x=country_counts.index, y=country_counts.values, title='Country Distribution',
                     color_discrete_sequence=['#00A674'])  # Set bar color to green
    st.plotly_chart(country_fig, use_container_width=True)

# Gender distribution
    st.subheader("Gender Distribution")
    gender_counts = st.session_state.filtered_df['gender'].value_counts()
    gender_fig = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution',
                    color_discrete_sequence=green)  # Shades of green for pie chart
    st.plotly_chart(gender_fig, use_container_width=True)

# School distribution
    st.subheader("Top 10 Schools")
    school_counts = st.session_state.filtered_df['school'].value_counts().head(10)
    school_fig = px.bar(x=school_counts.index, y=school_counts.values, title='Top 10 Schools',
                    color_discrete_sequence=['#00A674'])  # Set bar color to green
    st.plotly_chart(school_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("FIXXIES • A-Z of AI 2024")