import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Title
st.title("Advanced Machine Learning & Data Analysis Web App")

# Upload Dataset
st.sidebar.subheader("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Data Info
    st.write("### Dataset Information")
    st.write(df.describe())

    # Display Data Types and Allow Conversions
    st.write("### Data Types")
    st.write(df.dtypes)

    # Handle Date Columns
    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower()]
    if date_columns:
        st.write("### Date Conversion")
        for date_col in date_columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Extract year, month, and day to avoid mixing datetime and numeric types
            df[f"{date_col}_year"] = df[date_col].dt.year.astype('Int64')  # Convert to Int64 for consistency
            df[f"{date_col}_month"] = df[date_col].dt.month.astype('Int64')  # Convert to Int64
            df[f"{date_col}_day"] = df[date_col].dt.day.astype('Int64')  # Convert to Int64
        # After extraction, drop the original datetime column if not needed
        df.drop(columns=date_columns, inplace=True)
        st.write(f"Converted {date_columns} into year, month, and day columns and removed original datetime columns.")

    # Convert Categorical Columns Using Label Encoding or OneHot Encoding
    st.write("### Categorical Encoding")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        encoder_choice = st.radio("Choose encoding method for categorical data", ["Label Encoding", "One-Hot Encoding"])
        if encoder_choice == "Label Encoding":
            label_encoders = {}
            for col in categorical_columns:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))
            st.write(f"Applied Label Encoding to columns: {categorical_columns}")
        else:
            df = pd.get_dummies(df, columns=categorical_columns)
            st.write(f"Applied One-Hot Encoding to columns: {categorical_columns}")

    # Handle Missing Values
    st.write("### Handle Missing Values")
    missing_value_option = st.radio("How do you want to handle missing values?",
                                    ["Drop rows", "Fill with mean/median/mode"])
    if missing_value_option == "Drop rows":
        df.dropna(inplace=True)
        st.write("Dropped rows with missing values.")
    else:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == np.number:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        st.write("Filled missing values with mean for numerical columns and mode for categorical columns.")

    # Feature Scaling for numeric columns
    st.write("### Feature Scaling")
    scale_columns = st.multiselect("Select numeric columns to scale",
                                   df.select_dtypes(include=[np.number]).columns.tolist())
    if scale_columns:
        scaler = StandardScaler()
        df[scale_columns] = scaler.fit_transform(df[scale_columns])
        st.write(f"Scaled columns: {scale_columns}")

    # Data Visualization Section
    st.write("### Suggested Visuals")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Interactive Visuals with Larger Figures
    st.write("#### Customize your visual")
    visual_option = st.selectbox("Choose a chart type", ["Heatmap", "Scatter Plot", "Bar Plot", "Histogram", "Box Plot", "Pair Plot", "Line Plot", "None"])

    if visual_option == "Heatmap" and len(numeric_columns) > 1:
        st.write("#### Heatmap of Correlations")
        plt.figure(figsize=(15, 10))  # Increased figure size
        fig, ax = plt.subplots(figsize=(15, 10))  # Added larger size for better readability
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif visual_option == "Scatter Plot":
        st.write("#### Scatter Plot")
        x_axis = st.selectbox("X-Axis", numeric_columns)
        y_axis = st.selectbox("Y-Axis", numeric_columns)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
        st.pyplot(fig)

    elif visual_option == "Bar Plot":
        st.write("#### Bar Plot")
        col_to_plot = st.selectbox("Select column for bar plot", df.columns)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=col_to_plot, data=df, ax=ax)
        st.pyplot(fig)

    elif visual_option == "Histogram":
        st.write("#### Histogram")
        col_to_plot = st.selectbox("Select column for histogram", numeric_columns)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[col_to_plot], kde=True, ax=ax)
        st.pyplot(fig)

    elif visual_option == "Box Plot":
        st.write("#### Box Plot")
        col_to_plot = st.selectbox("Select column for box plot", numeric_columns)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=col_to_plot, data=df, ax=ax)
        st.pyplot(fig)

    elif visual_option == "Pair Plot":
        st.write("#### Pair Plot")
        sns.pairplot(df[numeric_columns])
        st.pyplot()

    elif visual_option == "Line Plot":
        st.write("#### Line Plot")
        col_to_plot = st.selectbox("Select column for line plot", numeric_columns)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df[col_to_plot], ax=ax)
        st.pyplot(fig)

    # Machine Learning Preprocessing
    st.write("### Machine Learning Preprocessing")

    target_column = st.selectbox("Select Target Variable", df.columns)
    task_type = st.radio("Choose task type", ["Classification", "Regression"])

    if st.checkbox("Perform Machine Learning Analysis"):

        # Data Preparation
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        if task_type == "Classification":
            df[target_column] = LabelEncoder().fit_transform(df[target_column])
            st.write("Categorical Target variable encoded")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model Selection
        if task_type == "Classification":
            st.write("### Select Classification Models to Evaluate")
            model_options = st.multiselect("Classification Models",
                                           ["Random Forest", "Logistic Regression", "SVM", "k-NN", "Decision Tree"])
        else:
            st.write("### Select Regression Models to Evaluate")
            model_options = st.multiselect("Regression Models",
                                           ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor",
                                            "Ridge", "Lasso"])

        # Model Performance Evaluation
        # Check if at least one model is selected
        if len(model_options) == 0:
            st.warning("Please select at least one model to evaluate.")
        else:
            model_performance = {}

            if task_type == "Classification":
                if "Random Forest" in model_options:
                    model = RandomForestClassifier(n_estimators=100)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_performance["Random Forest"] = accuracy

                if "Logistic Regression" in model_options:
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_performance["Logistic Regression"] = accuracy

                if "SVM" in model_options:
                    model = SVC()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_performance["SVM"] = accuracy

                if "k-NN" in model_options:
                    model = KNeighborsClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_performance["k-NN"] = accuracy

                if "Decision Tree" in model_options:
                    model = DecisionTreeClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_performance["Decision Tree"] = accuracy

            else:
                if "Linear Regression" in model_options:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_performance["Linear Regression"] = r2

                if "Random Forest Regressor" in model_options:
                    model = RandomForestRegressor(n_estimators=100)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_performance["Random Forest Regressor"] = r2

                if "Decision Tree Regressor" in model_options:
                    model = DecisionTreeRegressor()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_performance["Decision Tree Regressor"] = r2

                if "Ridge" in model_options:
                    model = Ridge()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_performance["Ridge"] = r2

                if "Lasso" in model_options:
                    model = Lasso()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_performance["Lasso"] = r2

            # Display Results
            st.write("### Model Performance")
            st.write(model_performance)

        #0245116942
