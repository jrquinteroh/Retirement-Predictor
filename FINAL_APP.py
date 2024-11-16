import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from PIL import Image
import base64

# -------------------------
# Set Streamlit Page Configuration
# -------------------------
st.set_page_config(
    page_title="Bank Customer Retirement Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Custom CSS for Dark Theme Styling
# -------------------------
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Define CSS Variables for Theme Colors */
        :root {
            --primary-color: #28a745; /* Money-Like Green */
            --primary-hover-color: #218838; /* Darker Shade for Hover */
            --background-color: #2c2c2c;
            --header-bg-color: #3a3a3a;
            --text-color: #ffffff;
            --border-radius: 8px;
            --font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }

        /* Apply background and text color to the main container */
        .reportview-container {
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 2rem;
            font-family: var(--font-family);
        }

        /* Center header images */
        .header-img {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        /* Style for emojis */
        .emoji {
            font-size: 1.5em;
            vertical-align: middle;
        }

        /* Adjust DataFrame styles for dark theme */
        .stDataFrame table {
            color: var(--text-color);
            background-color: var(--background-color);
            border-collapse: collapse;
            width: 100%;
        }
        .stDataFrame th {
            background-color: var(--header-bg-color);
            color: var(--text-color);
            padding: 8px;
            text-align: left;
        }
        .stDataFrame td {
            padding: 8px;
            border-bottom: 1px solid #444;
        }

        /* Customize button appearance to match money-like green */
        .stButton>button {
            background-color: var(--primary-color);
            color: var(--text-color);
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: var(--border-radius);
            transition: background-color 0.3s ease;
            font-family: var(--font-family);
        }
        .stButton>button:hover {
            background-color: var(--primary-hover-color);
        }

        /* Optional: Adjust margins and padding for better layout */
        .block-container {
            padding: 1rem 2rem;
        }

        /* Ensure consistency for other interactive elements */
        .stTextInput>div>div>input {
            background-color: #3a3a3a;
            color: var(--text-color);
            border: 1px solid #555;
            border-radius: var(--border-radius);
            padding: 8px;
            font-size: 16px;
            font-family: var(--font-family);
        }

        .stSelectbox>div>div>div>div {
            background-color: #3a3a3a;
            color: var(--text-color);
            border: 1px solid #555;
            border-radius: var(--border-radius);
            padding: 8px;
            font-size: 16px;
            font-family: var(--font-family);
        }

        /* Customize other components as needed to match the theme */

        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# -------------------------
# Function to Load Data
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('Bank_Customer_retirement.csv')
    if 'Customer ID' in data.columns:
        data = data.drop(['Customer ID'], axis=1)
    return data

# -------------------------
# Function to Load the Trained Model
# -------------------------
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()


# -------------------------
# Load Data and Model
# -------------------------
data = load_data()
model = load_model()

# -------------------------
# Sidebar Content
# -------------------------

# Sidebar title
st.sidebar.title("Navigation")

st.sidebar.write('From here, navegate to wherever you want! Just click anywhere in the select box and choose which page to go into!')

page = st.sidebar.radio(
    "Navigate to:",
    options=[
        "Home",
        "Get Your Predictions",
        "About the Dataset",
        "About the Model",
    ],
)


st.sidebar.markdown("---")

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("Home")

    st.markdown(
        """
        Welcome to the **Bank Customer Retirement Analysis** app! 

        This app is designed to help you explore and analyze retirement-related data, and predict whether a customer is likely to retire (or be retired) based on their age and 401K savings.
        """
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            """
            ## Features

            - **Get Your Predictions:** Input your age and 401K savings to predict your retirement status.
            - **About the Dataset:** Dive deep into the dataset with interactive visualizations and discover meaningful insights.
            - **About the Model:** Learn about the machine learning model used in this app in order to make the predictions.
            - **Fun Interactive Elements:** Enjoy quizzes, fun facts, and interactive charts to make your data exploration enjoyable.
            """
        )
    with col2:
        st.markdown(
            """
            ## Technologies Used

            - **Streamlit:** For building the interactive web app.
            - **Plotly:** For creating dynamic and interactive visualizations.
            - **Pandas & NumPy:** For data manipulation and analysis.
            - **Scikit-learn:** For machine learning modeling.
            - **GenIA:** For troubleshooting help, error bug fixing, brainstorming ideas.
            """
        )

    
    csv_data = data.to_csv(index=False).encode('utf-8')
    b64_csv = base64.b64encode(csv_data).decode('utf-8')

    st.markdown(
        """
        <div style='text-align: center;'>
            <p><strong>Data Source:</strong></p>
            <a href="data:file/csv;base64,{b64}" download="Bank_Customer_retirement.csv">
                <button style="background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer;">
                    📥 Download Bank Customer Retirement Dataset
                </button>
            </a>
        </div>
        """.format(b64=b64_csv),
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='text-align: center;'>
        Developed By: Juan Quintero

        Developed For: IE University: MBD - Machine Learning 2 

            📧 Contact

            Have questions or feedback? Reach out at jrquinteroh@student.ie.edu 📬
        </div>
        """,
        unsafe_allow_html=True
    )

    image = Image.open("streamlit.jpg")
    resized_img = image.resize((500, 400)) 

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(resized_img, caption="Credits: DALL·E")


# -------------------------
# Get Your Predictions Page
# -------------------------
elif page == "Get Your Predictions":
    st.title("Retirement Prediction")

    image = Image.open("prediction.jpg")
    st.image(image, caption="Credits: DALL·E", use_column_width=True)

    st.markdown("---")

    st.subheader("📝 Input Features")

    Age = st.number_input(
        label="Age",
        min_value=int(data['Age'].min()),
        max_value=int(data['Age'].max()),
        value=int(data['Age'].mean()),
        step=1
    )

    savings_401k = st.number_input(
        label="401K Savings ($)",
        min_value=float(data['401K Savings'].min()),
        max_value=float(data['401K Savings'].max()),
        value=float(data['401K Savings'].mean()),
        step=1000.0
    )

    input_data = pd.DataFrame({
        'Age': [Age],
        '401K Savings': [savings_401k]
    })

    st.write("### 📄 Input Data")
    st.write(input_data.round({'Age': 0, '401K Savings': 2}))

    if st.button("🔍 Predict"):
        with st.spinner('Predicting...'):
            try:
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                class_mapping = {0: 'Not Retired', 1: 'Retired'}
                predicted_class_label = class_mapping.get(prediction, 'Unknown')

                prob = prediction_proba[prediction] * 100

                st.markdown(f"**Explanation:** The model is **{prob:.2f}%** confident that this customer is **{predicted_class_label.lower()}** basing the analysis on their 401K savings and their age.")

                proba_df = pd.DataFrame({
                    'Class': [class_mapping.get(cls, 'Unknown') for cls in model.named_steps['classifier'].classes_],
                    'Probability': prediction_proba
                })
                proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{x*100:.2f}%")

                st.table(proba_df)

                if predicted_class_label == 'Retired':
                    st.success("🎉 **Retired!** 🎉")
                    st.balloons()
                elif predicted_class_label == 'Not Retired':
                    st.error("😞 **Not Retired :(**")
                else:
                    st.warning("⚠️ **Prediction Uncertain.**")

            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")

    st.subheader("📊 Feature Importance")
    try:
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = ['Age', '401K Savings']  

        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        fig_importance = px.bar(
            feature_importances,
            x='Importance',
            y='Feature',
            orientation='h',
            title='🔍 Feature Importances from Random Forest',
            text='Importance',
            color='Feature',
            color_discrete_sequence=px.colors.qualitative.Set2,
            template='plotly_dark'  
        )
        fig_importance.update_traces(texttemplate='%{text:.4f}', textposition='inside')
        fig_importance.update_layout(
            yaxis={'categoryorder':'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error in displaying feature importances: {e}")

    st.subheader("🎯 Quick Quiz: Test Your Retirement Knowledge!")

    quiz_question = "At what age do you think the average person retires in the USA?"
    options = ["60", "62", "65", "68"]

    user_answer = st.radio(quiz_question, options)

    if st.button("Submit Answer"):
        correct_answer = "65"
        if user_answer == correct_answer:
            st.success("✅ Correct! The average American retires at age 65.")
        else:
            st.error(f"❌ Incorrect. The correct answer is {correct_answer}.")

# -------------------------
# About the Dataset Page
# -------------------------
elif page == "About the Dataset":
    st.title("About the Dataset: Bank Customer Retirement Insights")

    st.markdown(
        """
        <div class="header-img">
            <img src="https://media2.giphy.com/media/67ThRZlYBvibtdF9JH/giphy.gif?cid=6c09b952jg7ns0jme6spv2anset5msh0453sru59xuwalaep&ep=v1_gifs_search&rid=giphy.gif&ct=g" alt="Dataset GIF" width="700">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("""
    Welcome to the **Bank Customer Retirement Dataset**! This dataset offers a glimpse into the financial planning of 500 customers, focusing on their journey towards retirement. Let's break it down!
    """)

    st.subheader("Dataset Overview")
    st.markdown("""
    - **Customer ID**: Unique identifier for each customer.
    - **Age**: Age of the customer, measured in years.
    - **401K Savings**: Total savings in the customer's 401K retirement account.
    - **Retire**: Binary indicator:
    - `1`: Retired.
    - `0`: Not retired.
    """)

    st.subheader("Key Features")
    st.write("""
    With **500 records** and **no missing data** (yay! 🎉), this dataset is ***"perfect"*** for:
    - Predictive modeling.
    - Financial behavior analysis.
    - Insights into retirement trends.
    """)

    st.markdown("Let's dive into the data and uncover some interesting patterns! 🔍")


    st.markdown("---")

    st.header("Dataset Overview")
    st.write("**Shape of the Dataset:**", data.shape)
    st.write("**Columns:**", data.columns.tolist())

    st.subheader("Data Types and Missing Values")
    data_types = data.dtypes.reset_index()
    data_types.columns = ['Column', 'Data Type']
    missing_values = data.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Values']
    buffer_df = pd.merge(data_types, missing_values, on='Column')
    st.dataframe(buffer_df.style.set_properties(**{'text-align': 'center'}))


    st.subheader("Statistical Summary")
    st.dataframe(
        data.describe().round(2).style.format("{:.2f}").set_properties(**{'text-align': 'center', 'background-color': '#2c2c2c'}),
        use_container_width=True
    )

    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig_corr = px.imshow(
        corr,
        text_auto='.2f',  
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Correlation Heatmap',
        labels={"color": "Correlation"},
        template='plotly_dark'  
    )
    fig_corr.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Distribution of Numerical Features")
    numerical_features = ['Age', '401K Savings']
    selected_features = st.multiselect("Select Features for Distribution Plots", numerical_features, default=numerical_features)
    bin_size = st.slider("Select Number of Bins", min_value=10, max_value=50, value=30, step=5)

    for feature in selected_features:
        fig_dist = px.histogram(
            data,
            x=feature,
            nbins=bin_size,
            title=f'Distribution of {feature}',
            marginal='box',  
            opacity=0.75,
            color_discrete_sequence=['#636EFA'],
            histnorm='percent',  
            template='plotly_dark'  
        )
        fig_dist.update_layout(
            bargap=0.1,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Box Plots of Numerical Features")
    selected_box = st.multiselect("Select Features for Box Plots", numerical_features, default=numerical_features)
    group_by = st.selectbox("Group by (optional)", [None] + data.columns.tolist(), index=0)

    for feature in selected_box:
        fig_box = px.box(
            data,
            y=feature,
            title=f'Box Plot of {feature}' + (f' Grouped by {group_by}' if group_by else ''),
            points='all',
            color=group_by if group_by else None,
            color_discrete_sequence=px.colors.qualitative.Set2,
            template='plotly_dark'  
        )
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_box, use_container_width=True)

    
    st.subheader("Count Plot for Target Variable")
    if 'Retire' in data.columns:
        count_data = data['Retire'].value_counts().reset_index()
        count_data.columns = ['Retire', 'Count']
        count_data['Percentage'] = (count_data['Count'] / count_data['Count'].sum()) * 100

        fig_count = px.bar(
            count_data,
            x='Retire',
            y='Count',
            title='Distribution of Target Variable: Retire',
            text='Percentage',
            color='Retire',
            color_discrete_sequence=['#636EFA', '#EF553B'],
            template='plotly_dark' 
        )
        fig_count.update_traces(texttemplate='%{text:.2f}%', textposition='auto')
        fig_count.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_count, use_container_width=True)
    else:
        st.write("Target variable 'Retire' not found in the dataset.")

    st.subheader("Savings Distribution by Age Group")

    age_bins = st.slider("Define Number of Age Groups", min_value=3, max_value=10, value=5)
    data['Age Group'] = pd.cut(data['Age'], bins=age_bins).astype(str)

    fig_age_savings = px.box(
        data,
        x='Age Group',
        y='401K Savings',
        title='401K Savings by Age Group',
        color='Age Group',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='plotly_dark'  
    )
    fig_age_savings.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_age_savings, use_container_width=True)

    st.subheader("👥 Age vs. Savings Correlation by Retirement Status")
    if 'Retire' in data.columns:
        fig_scatter = px.scatter(
            data,
            x='Age',
            y='401K Savings',
            color='Retire',
            title='Age vs. 401K Savings by Retirement Status',
            trendline='ols',  
            color_discrete_sequence=['#636EFA', '#EF553B'],
            template='plotly_dark'  
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Savings Goal Achievement")
    savings_goal = st.number_input("Set a Savings Goal ($)", value=100000, step=50000)
    goal_achievers = data[data['401K Savings'] >= savings_goal]
    goal_percentage = len(goal_achievers) / len(data) * 100

    st.metric("Percentage of Individuals Meeting Savings Goal", f"{goal_percentage:.2f}%")
    fig_goal = px.histogram(
        data,
        x='401K Savings',
        nbins=30,
        title=f'Distribution of Savings with Goal Line at ${savings_goal}',
        color_discrete_sequence=['#636EFA'],
        template='plotly_dark'  
    )
    fig_goal.add_vline(x=savings_goal, line_width=3, line_dash="dash", line_color="red", annotation_text="Goal")
    fig_goal.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_goal, use_container_width=True)

   
    st.subheader("Retirement Likelihood by Age and Savings")
    if 'Retire' in data.columns:
        fig_density = px.density_heatmap(
            data,
            x='Age',
            y='401K Savings',
            z='Retire',
            histfunc='avg',
            title='Heatmap of Retirement Likelihood by Age and Savings',
            color_continuous_scale='Viridis',
            template='plotly_dark'  
        )
        fig_density.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_density, use_container_width=True)

    st.subheader("Fun Facts About Retirement in the USA 🇺🇸")
    fun_facts = [
        "💡 **Fact 1:** The average American retires at age 65.",
        "💡 **Fact 2:** Approximately 50% of retirees feel financially unprepared.",
        "💡 **Fact 3:** 401K accounts are a popular retirement savings option in the USA.",
        "💡 **Fact 4:** Regular savings can significantly impact your retirement lifestyle.",
        "💡 **Fact 5:** Investing early in your 401K can lead to substantial growth over time."
    ]
    for fact in fun_facts:
        st.write(fact)

# -------------------------
# About the Model Page
# -------------------------
elif page == "About the Model":
    st.title("About the Model")

    st.markdown(
        """
        ## Random Forest Classifier Model Used

        ### Why Random Forest?

        The **Random Forest Classifier** is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) of the individual trees.

        - **Robustness to Overfitting:** Random Forest reduces overfitting by averaging multiple trees, resulting in a model with less variance than a single decision tree.
        - **Handles Non-linear Data Well:** It can capture complex patterns and interactions between features.
        - **Feature Importance:** Provides estimates of feature importance, helping us understand which features contribute most to the prediction.

        ### Model Explanation

        The model predicts whether a customer is likely to retire based on their age and 401K savings.

        - **Features Used:**
            - **Age**
            - **401K Savings**

        - **Target Variable:**
            - **Retire** (1 for Retired, 0 for Not Retired)""")



    st.subheader("Model Building and Evaluation: Random Forest Classifier")

    st.write("""
    In this section, we showcase how the model was optimized using **RandomizedSearchCV**, followed by an evaluation of its performance on test data. Let's dive in! 🚀
    """)

    st.subheader("Randomized Search for Random Forest")

    st.write("""
    Let's outline how the model was built using **RandomizedSearchCV** to find the optimal hyperparameters for a Random Forest Classifier. Let's walk through the process! 🚀
    """)

    st.subheader("Parameter Grid")
    st.code("""
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 400, 500],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': [2, 5, 10, 15],
        'classifier__min_samples_leaf': [1, 2, 4, 6],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__bootstrap': [True, False]
    }
    """, language='python')

    st.write("""
    The **parameter grid** above explores different combinations of hyperparameters:
    - **n_estimators**: Number of trees in the forest.
    - **max_depth**: Maximum depth of each tree.
    - **min_samples_split**: Minimum number of samples required to split an internal node .
    - **min_samples_leaf**: Minimum number of samples required to be at a leaf node.
    - **max_features**: Number of features considered for splitting at each node.
    - **bootstrap**: Whether to use bootstrap samples when building trees.
    """)

    st.subheader("🔧 RandomizedSearchCV Setup")
    st.code("""
    rf_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings sampled
        cv=5,  # Number of cross-validation folds
        verbose=2,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        scoring='accuracy'  # Use 'roc_auc' for classification, 'neg_mean_squared_error' for regression
    )
    """, language='python')

    st.write("""
    Here's what each parameter in **RandomizedSearchCV** means:
    - **n_iter=10**: Randomly sample 10 combinations of hyperparameters.
    - **cv=5**: Perform 5-fold cross-validation for each combination.
    - **verbose=2**: Show real-time updates on the fitting process.
    - **random_state=42**: Ensures reproducibility of results.
    - **n_jobs=-1**: Utilizes all available CPU cores for parallel computation.
    - **scoring='accuracy'**: Evaluates model performance based on accuracy. 
    """)

    st.markdown("With this setup, we efficiently search for the best hyperparameters to optimize the model's performance. 🚀")


    st.subheader("🏆 Best Hyperparameters")
    st.write("""
    After running the randomized search, these were the best hyperparameters identified:
    """)
    best_params = {
        'classifier__n_estimators': 400,
        'classifier__min_samples_split': 15,
        'classifier__min_samples_leaf': 2,
        'classifier__max_features': 'log2',
        'classifier__max_depth': None,
        'classifier__bootstrap': False
    }
    st.json(best_params)

    st.subheader("📊 Classification Report")
    st.write("""
    Here’s the performance of the optimized model on the test dataset:
    """)
    classification_report_data = {
        "Class": [0, 1, "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": [0.98, 0.92, None, 0.95, 0.95],
        "Recall": [0.92, 0.98, None, 0.95, 0.95],
        "F1-Score": [0.95, 0.95, 0.95, 0.95, 0.95],
        "Support": [50, 50, 100, 100, 100]
    }
    df_report = pd.DataFrame(classification_report_data).set_index("Class")
    st.table(df_report)

    st.markdown("""
    - **Precision**: Proportion of true positives out of predicted positives.
    - **Recall**: Proportion of true positives out of actual positives.
    - **F1-Score**: Harmonic mean of precision and recall.
    - **Support**: Number of true instances for each class.
    """)

    st.write("Overall, the model achieves an impressive **accuracy of 95%** on the test set! 🎉 (Taking into account that the test set was of 100 data entries).")
    
    
    st.markdown(
        """
        ### Limitations

        - **Limited Features:** The model only uses Age and 401K Savings, which may not capture all factors influencing retirement.
            - The dataset is only 500 data entry points. Not enough to capture capture the reality and to preform a good prediction.
        - **Data Bias:** If the dataset is not representative of the broader population, the model's predictions may not generalize well.

        ### Future Improvements

        - **Include More Features:** Incorporate additional relevant features such as income, employment status, health indicators, etc.
        - If possible, get more data into the model to further analyse and get better and more accurate predictions. 

        ## 📚 Learn More

        - **About Retirement Planning:** [USA Gov. - Retirement Planning](https://www.usa.gov/retirement-planning-tools)
        - **401K Plans:** [IRS - 401(k) Resource Guide](https://www.irs.gov/retirement-plans/plan-participant-employee/401k-resource-guide-plan-participants-summary-plan-description)

        """
    )


    image = Image.open("rfc.jpg")
    st.image(image, caption="Credits: DALL·E", use_column_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    ---
    <div style="text-align: center; color: grey;">
        &copy; 2024 Bank Customer Retirement Analysis App | Developed with GenAI by Juan Quintero (https://github.com/jrquinteroh)
    </div>
    """,
    unsafe_allow_html=True
)
