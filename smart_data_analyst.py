import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe

# Initialize the LLM model
model = LocalLLM(api_base="http://localhost:11434/v1", model="codegemma")

# Streamlit app title
st.set_page_config(page_title="Data Analysis and Preprocessing", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis with Your Data"])

# Initialize session state for storing data and history
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def save_processed_data():
    """Save the current processed data permanently in the session state."""
    st.session_state.processed_data = st.session_state.processed_data.copy()

def ensure_arrow_compatibility(df):
    """Ensure the DataFrame is Arrow-compatible by fixing column types and replacing incompatible values."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == np.float64 or df[col].dtype == np.int64:
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        elif df[col].dtype == object:
            df[col] = df[col].astype(str)
    return df

if page == "Home":
    st.title("Data Analysis and Preprocessing with PandasAI")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        st.session_state.original_data = pd.read_csv(uploaded_file)
        st.session_state.processed_data = st.session_state.original_data.copy()
        st.write("Original Data:")
        st.write(st.session_state.processed_data.head(6))

        # Sidebar for preprocessing options
        st.sidebar.subheader("Data Preprocessing Options")

        # Option to handle missing values
        handle_missing = st.sidebar.checkbox("Handle Missing Values", key="handle_missing")
        remove_duplicates = st.sidebar.checkbox("Remove Duplicates", key="remove_duplicates")

        # Perform preprocessing based on user selection
        if handle_missing:
            for column in st.session_state.processed_data.columns:
                if st.session_state.processed_data[column].dtype == "object":
                    mode_value = st.session_state.processed_data[column].mode()[0]
                    st.session_state.processed_data[column].fillna(mode_value, inplace=True)
                else:
                    mean_value = st.session_state.processed_data[column].mean()
                    st.session_state.processed_data[column].fillna(mean_value, inplace=True)
            save_processed_data()  # Save processed data permanently
            st.write("Missing values handled and saved.")
            st.write(st.session_state.processed_data.head(6))

        if remove_duplicates:
            st.session_state.processed_data.drop_duplicates(inplace=True)
            save_processed_data()  # Save processed data permanently
            st.write("Duplicates removed and saved.")
            st.write(st.session_state.processed_data.head(6))

elif page == "Data Analysis with Your Data":
    st.title("Data Analysis with Your Data")

    # Display the processed data
    if st.session_state.processed_data is not None:
        st.write("Processed Data:")
        st.write(st.session_state.processed_data.head(6))

        # Handle missing values
        st.subheader("Handle Missing Values")
        if st.button("Handle Missing Values", key="handle_missing_values"):
            for column in st.session_state.processed_data.columns:
                if st.session_state.processed_data[column].dtype == "object":
                    mode_value = st.session_state.processed_data[column].mode()[0]
                    st.session_state.processed_data[column].fillna(mode_value, inplace=True)
                else:
                    mean_value = st.session_state.processed_data[column].mean()
                    st.session_state.processed_data[column].fillna(mean_value, inplace=True)
            save_processed_data()  # Save processed data permanently
            st.write("Missing values handled and saved.")
            st.write(st.session_state.processed_data.head(6))

        # Show duplicate values
        st.subheader("Duplicate Values")
        duplicates = st.session_state.processed_data.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicates}")

        if st.button("Remove Duplicates", key="remove_duplicates_button"):
            st.session_state.processed_data.drop_duplicates(inplace=True)
            save_processed_data()  # Save processed data permanently
            st.write("Duplicates removed and saved.")
            st.write(st.session_state.processed_data.head(6))

        # Check and fix data types
        st.subheader("Check and Fix Data Types")
        st.write("Current Data Types:")
        st.write(st.session_state.processed_data.dtypes)

        columns_to_fix = st.multiselect(
            "Select columns to change data types:",
            options=st.session_state.processed_data.columns,
            default=[], key="columns_to_fix"
        )

        if st.button("Fix Data Types", key="fix_data_types"):
            for column in columns_to_fix:
                correct_type = st.selectbox(f"Select correct data type for {column}:", options=["int", "float", "object", "bool"], key=f"correct_type_{column}")
                if correct_type == "int":
                    st.session_state.processed_data[column] = st.session_state.processed_data[column].astype(int)
                elif correct_type == "float":
                    st.session_state.processed_data[column] = st.session_state.processed_data[column].astype(float)
                elif correct_type == "object":
                    st.session_state.processed_data[column] = st.session_state.processed_data[column].astype(str)
                elif correct_type == "bool":
                    st.session_state.processed_data[column] = st.session_state.processed_data[column].astype(bool)
            save_processed_data()  # Save processed data permanently
            st.write("Data types corrected and saved.")
            st.write(st.session_state.processed_data.dtypes)

        # Data Encoding Section
        st.subheader("Data Encoding")
        encoding_method = st.selectbox("Select Encoding Method:", ["Label Encoding", "Ordinal Encoding", "One-Hot Encoding"], key="encoding_method")
        string_columns = st.multiselect("Select columns to encode:", options=st.session_state.processed_data.select_dtypes(include=['object']).columns, key="string_columns")

        if st.button("Apply Encoding", key="apply_encoding"):
            if encoding_method == "Label Encoding":
                encoder = LabelEncoder()
                for col in string_columns:
                    st.session_state.processed_data[col] = encoder.fit_transform(st.session_state.processed_data[col])
            elif encoding_method == "Ordinal Encoding":
                for col in string_columns:
                    unique_values = list(st.session_state.processed_data[col].unique())
                    st.session_state.processed_data[col] = st.session_state.processed_data[col].apply(lambda x: unique_values.index(x))
            elif encoding_method == "One-Hot Encoding":
                st.session_state.processed_data = pd.get_dummies(st.session_state.processed_data, columns=string_columns)
            save_processed_data()  # Save processed data permanently
            st.write(f"{encoding_method} applied to selected columns.")
            st.write(st.session_state.processed_data.head(6))

        # Outlier Detection and Handling
        st.subheader("Outlier Detection and Handling")
        int_columns = st.session_state.processed_data.select_dtypes(include=['int64', 'float64']).columns

        if st.button("Show Box Plot of Numeric Columns", key="show_boxplot"):
            for column in int_columns:
                st.write(f"Box plot for {column}:")
                plt.figure(figsize=(10, 4))
                sns.boxplot(x=st.session_state.processed_data[column])
                st.pyplot(plt)

        outlier_column = st.selectbox("Select a column to handle outliers:", options=int_columns, key="outlier_column")

        if st.button("Handle Outliers using IQR Method", key="handle_outliers_iqr"):
            Q1 = st.session_state.processed_data[outlier_column].quantile(0.25)
            Q3 = st.session_state.processed_data[outlier_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            st.session_state.processed_data = st.session_state.processed_data[(st.session_state.processed_data[outlier_column] >= lower_bound) & (st.session_state.processed_data[outlier_column] <= upper_bound)]
            save_processed_data()  # Save processed data permanently
            st.write(f"Outliers handled for {outlier_column}. Updated box plot:")
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=st.session_state.processed_data[outlier_column])
            st.pyplot(plt)

        # **Capping Method for Outliers**
        if st.button("Cap Outliers", key="cap_outliers"):
            Q1 = st.session_state.processed_data[outlier_column].quantile(0.25)
            Q3 = st.session_state.processed_data[outlier_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            st.session_state.processed_data[outlier_column] = np.where(
                st.session_state.processed_data[outlier_column] > upper_limit, upper_limit,
                np.where(st.session_state.processed_data[outlier_column] < lower_bound, lower_bound, st.session_state.processed_data[outlier_column])
            )
            save_processed_data()  # Save processed data permanently
            st.write(f"Outliers capped for {outlier_column}. Updated box plot:")
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=st.session_state.processed_data[outlier_column])
            st.pyplot(plt)

        # **Feature Scaling**
        st.subheader("Feature Scaling")
        scaling_method = st.selectbox("Select Scaling Method:", ["Normalization", "Standardization"], key="scaling_method")
        columns_to_scale = st.multiselect("Select columns to scale:", options=int_columns, key="columns_to_scale")

        if st.button("Apply Scaling", key="apply_scaling"):
            if scaling_method == "Normalization":
                scaler = MinMaxScaler()
                st.session_state.processed_data[columns_to_scale] = scaler.fit_transform(st.session_state.processed_data[columns_to_scale])
            elif scaling_method == "Standardization":
                scaler = StandardScaler()
                st.session_state.processed_data[columns_to_scale] = scaler.fit_transform(st.session_state.processed_data[columns_to_scale])
            save_processed_data()  # Save processed data permanently
            st.write(f"{scaling_method} applied to selected columns.")
            st.write(st.session_state.processed_data.head(6))

        # Exploratory Data Analysis
        st.subheader("Exploratory Data Analysis")
        eda_chart = st.selectbox("Select chart type for EDA:", ["Bar Chart", "Box Plot", "Heat Map", "Scatter Plot", "Pie Chart", "Histogram", "Distplot"], key="eda_chart")

        if eda_chart == "Bar Chart":
            eda_column = st.selectbox("Select column for Bar Chart:", options=st.session_state.processed_data.columns, key="bar_chart_column")
            plt.figure(figsize=(10, 4))
            sns.countplot(x=st.session_state.processed_data[eda_column])
            st.pyplot(plt)

        elif eda_chart == "Box Plot":
            eda_column = st.selectbox("Select column for Box Plot:", options=int_columns, key="box_plot_column")
            st.write(f"Box Plot for {eda_column}:")
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=st.session_state.processed_data[eda_column])
            st.pyplot(plt)

        elif eda_chart == "Heat Map":
            st.write("Heat Map of correlations:")
            plt.figure(figsize=(10, 6))
            sns.heatmap(st.session_state.processed_data.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

        elif eda_chart == "Scatter Plot":
            x_axis = st.selectbox("Select X-axis column for Scatter Plot:", options=int_columns, key="scatter_plot_x")
            y_axis = st.selectbox("Select Y-axis column for Scatter Plot:", options=int_columns, key="scatter_plot_y")
            hue_column = st.selectbox("Select column for color encoding (optional):", options=[None] + list(st.session_state.processed_data.columns), index=0, key="scatter_plot_hue")

            st.write(f"Scatter Plot between {x_axis} and {y_axis}:")
            plt.figure(figsize=(10, 6))
            if hue_column == 'None' or hue_column is None:
                sns.scatterplot(x=st.session_state.processed_data[x_axis], y=st.session_state.processed_data[y_axis])
            else:
                sns.scatterplot(x=st.session_state.processed_data[x_axis], y=st.session_state.processed_data[y_axis], hue=st.session_state.processed_data[hue_column])
            st.pyplot(plt)

        elif eda_chart == "Pie Chart":
            pie_column = st.selectbox("Select column for Pie Chart:", options=st.session_state.processed_data.columns, key="pie_chart_column")
            pie_data = st.session_state.processed_data[pie_column].value_counts()
            st.write(f"Pie Chart for {pie_column}:")
            plt.figure(figsize=(8, 8))
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            st.pyplot(plt)

        elif eda_chart == "Histogram":
            hist_column = st.selectbox("Select column for Histogram:", options=int_columns, key="histogram_column")
            st.write(f"Histogram for {hist_column}:")
            plt.figure(figsize=(10, 4))
            sns.histplot(st.session_state.processed_data[hist_column], kde=False, bins=30)
            st.pyplot(plt)

        elif eda_chart == "Distplot":
            dist_column = st.selectbox("Select column for Distplot:", options=int_columns, key="distplot_column")
            st.write(f"Distribution Plot (Distplot) for {dist_column}:")
            plt.figure(figsize=(10, 4))
            sns.histplot(st.session_state.processed_data[dist_column], kde=True)
            st.pyplot(plt)

        # Model Application Section
        st.subheader("Model Application")
        model_type = st.selectbox("Select Model Type:", ["Regression", "Classification"], key="model_type")

        target_column = st.selectbox("Select Target Column:", options=st.session_state.processed_data.columns, key="target_column")

        if st.button("Apply Model", key="apply_model"):
            if model_type == "Regression":
                from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
                from sklearn.svm import SVR
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                X = st.session_state.processed_data.drop(columns=[target_column])
                y = st.session_state.processed_data[target_column]

                models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso Regression": Lasso(),
                    "Ridge Regression": Ridge(),
                    "ElasticNet Regression": ElasticNet(),
                    "Support Vector Regressor": SVR(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor()
                }

                results = {}
                for name, model in models.items():
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    results[name] = {
                        "R2 Score": r2_score(y, y_pred),
                        "Mean Squared Error": mean_squared_error(y, y_pred),
                        "Mean Absolute Error": mean_absolute_error(y, y_pred),
                        "Root Mean Squared Error": np.sqrt(mean_squared_error(y, y_pred))
                    }

                st.write("Model Performance:")
                for model_name, metrics in results.items():
                    st.write(f"**{model_name}:**")
                    st.write(f"R2 Score: {metrics['R2 Score']:.4f}")
                    st.write(f"Mean Squared Error: {metrics['Mean Squared Error']:.4f}")
                    st.write(f"Mean Absolute Error: {metrics['Mean Absolute Error']:.4f}")
                    st.write(f"Root Mean Squared Error: {metrics['Root Mean Squared Error']:.4f}")
                    st.write("---")

            elif model_type == "Classification":
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.naive_bayes import GaussianNB
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                X = st.session_state.processed_data.drop(columns=[target_column])
                y = st.session_state.processed_data[target_column]

                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Support Vector Classifier": SVC(),
                    "Naive Bayes": GaussianNB(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "K-Nearest Neighbors": KNeighborsClassifier()
                }

                results = {}
                for name, model in models.items():
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    results[name] = {
                        "Accuracy": accuracy_score(y, y_pred),
                        "Precision": precision_score(y, y_pred, average='weighted'),
                        "Recall": recall_score(y, y_pred, average='weighted'),
                        "F1 Score": f1_score(y, y_pred, average='weighted')
                    }

                st.write("Model Performance:")
                for model_name, metrics in results.items():
                    st.write(f"**{model_name}:**")
                    st.write(f"Accuracy: {metrics['Accuracy']:.4f}")
                    st.write(f"Precision: {metrics['Precision']:.4f}")
                    st.write(f"Recall: {metrics['Recall']:.4f}")
                    st.write(f"F1 Score: {metrics['F1 Score']:.4f}")
                    st.write("---")

# Option to download the cleaned and analyzed data after applying the models
st.sidebar.subheader("Download Analyzed Data")
analyzed_download_format = st.sidebar.selectbox("Select format", ["CSV", "Excel"], key="analyzed_download_format")

if st.sidebar.button("Download Analyzed Data", key="download_analyzed_data"):
    # Ensure DataFrame is compatible with Arrow
    compatible_df = ensure_arrow_compatibility(st.session_state.processed_data)

    if analyzed_download_format == "CSV":
        analyzed_csv = compatible_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=analyzed_csv,
            file_name="analyzed_data.csv",
            mime="text/csv"
        )
    elif analyzed_download_format == "Excel":
        # Save to Excel in-memory and provide a download button
        import io
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        compatible_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        output.seek(0)
        st.download_button(
            label="Download Excel",
            data=output,
            file_name="analyzed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
