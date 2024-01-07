import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import  SimpleImputer
import requests
from io import BytesIO

@st.cache_resource(show_spinner=False)
def load_model_from_pickle(model_pickle_url):
    # Send a GET request to GitHub
    file_like_object = BytesIO(requests.get(model_pickle_url).content)

    # Load the logistic regression model
    model = sm.load(file_like_object)
    return model

# Data preparation
def process_data(data):

    # Check for correct column names
    columns_list = ['enrollee_id', 'city', 'city_development_index', 'gender',
                    'relevent_experience', 'enrolled_university', 'education_level',
                    'major_discipline', 'experience', 'company_size', 'company_type',
                    'last_new_job', 'training_hours']
    if not set(data.columns) == set(columns_list): # regardless of the order
        missing_columns = list(set(columns_list) - set(data.columns))
        if missing_columns == []:
            additional_columns = list(set(data.columns) - set(columns_list))
            st.error(f'{additional_columns} columns shall not be in the csv file.', icon='❌')
            st.write(f'The csv must contain the following columns: {columns_list}')
        # Check for the target variable
        if additional_columns == ['target']:
            st.error('You are passing a file that already contains the target column.', icon='❌')
            st.write('Would you like to drop it?')
            if st.button('Drop target column'):
                data.drop('target', axis=1, inplace=True)
            else:
                st.stop()
        else: 
            st.error(f'{missing_columns} columns are missing.', icon='❌')
            st.write(f'The csv must contain the following columns: {columns_list}')
            st.stop()

    # Drop the city
    data.drop('city', axis=1, inplace=True)

    # Correct the typo
    data.rename({'relevent_experience': 'relevant_experience'}, axis=1, inplace=True)
    data.relevant_experience = data.relevant_experience.str.replace('relevent', 'relevant')

    # Ordinal encoding for education_level
    education_mapping = {
        'Primary School': 1,
        'High School': 2,
        'Graduate': 3,
        'Masters': 4,
        'Phd': 5
    }
    data.education_level = data.education_level.map(education_mapping)

    # Interval encoding for experience
    # Replace limit values and cast the column to float type
    experience_replacements = {
        '>20': 20.5,
        '<1': 0.5
    }
    data.experience = data.experience.replace(experience_replacements).astype(float)
    # Divide the column into intervals of experience years
    data.experience = pd.cut(x=data.experience, bins=[0, 1, 5, 10, 20, 21], right=False, labels=['<1', '1-5', '6-10', '11-20', '>20'])

    # Data format fixing for company_size
    data.company_size.replace({'10/49': '10-49', '10000+': '>10000'}, inplace=True)

    # Drop the rows with NaNs in enrolled_university, education_level, experience, last_new_job (see NaN handling)
    data = data[(data.enrolled_university.isna() == False) & (data.education_level.isna() == False) & (data.experience.isna() == False) & (data.last_new_job.isna() == False)].reset_index(drop=True)

    # Reproduce the One-Hot-Encoded dataframe leaving out company_size and company_type
    processed_data = pd.get_dummies(data, columns=['gender', 'relevant_experience', 'enrolled_university', 'major_discipline', 'experience', 'last_new_job'], dtype=int, dummy_na=True)

    # Drop gender and major_discipline
    processed_data.drop(['gender_Female','gender_Male','gender_Other','gender_nan','major_discipline_Arts', 'major_discipline_Business Degree',
           'major_discipline_Humanities', 'major_discipline_No Major',
           'major_discipline_Other', 'major_discipline_STEM',
           'major_discipline_nan'], axis=1, inplace=True)
    
    # Initate the imputer class
    simple_imputer = SimpleImputer(missing_values=pd.NA, strategy='most_frequent')
    # Imputwe the NaNs with the most frequent value
    processed_data = pd.DataFrame(simple_imputer.fit_transform(processed_data), columns=processed_data.columns)
    # Get the dataframe without NaNs and all the dummies
    processed_data = pd.get_dummies(processed_data, columns=['company_size', 'company_type'], dtype=int, dummy_na=False)
    # Drop the NaNs dummies
    processed_data = processed_data.drop(['relevant_experience_nan', 'enrolled_university_nan', 'experience_nan','last_new_job_nan'], axis=1)
    # Drop one dummy per category to avoid the dummy trap. Leave in the enrollee_id column
    processed_data = processed_data.drop(['relevant_experience_Has relevant experience', 'enrolled_university_Part time course', 'experience_<1','last_new_job_never','company_size_<10', 'company_type_Other'], axis=1)

    return processed_data

# Function to make predictions
def make_predictions(model, exog_data, enrollee_ids):
    enrollee_ids = enrollee_ids.astype('str')
    predictions = pd.Series(model.predict(exog=exog_data), name='churn_probability')
    churn_probabilities = pd.concat((enrollee_ids, predictions), axis=1)
    return churn_probabilities

# Streamlit app
def main():
    st.title("Churn Prediction")

    # Explanation of logistic model
    st.header("Model Explanation")
    st.markdown('''
                The model used to perform the predictions is a Logistic regression trained on the actual company data.
                
                The variables weights are represented below.
                Each prediction is made by multiplying the weights and the variables values and then passing the result into a logistic function:
                ''')
    st.latex(r'''$$\[ f(x) = \frac{1}{1 + e^{-x}} \]$$''')
    st.image("https://github.com/LeonardoAcquaroli/xtream-ai-assignment/blob/main/assignment-LeonardoAcquaroli/weights_plot.png?raw=true")

    # Upload CSV file
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        position = uploaded_file.tell() # current position of the file pointer
        # Load data
        churn = pd.read_csv(uploaded_file)
        if ';' in churn.columns[0]: # robust to ';' separator
            # Reset the file pointer to the original position
            uploaded_file.seek(position)
            churn = pd.read_csv(uploaded_file, sep=';')

        # Data preparation
        processed_data = process_data(churn)
        X = processed_data.drop('enrollee_id', axis=1).astype(float)
        X_with_intercept = sm.add_constant(X)

        # Make predictions
        logreg = load_model_from_pickle(model_pickle_url = "https://github.com/LeonardoAcquaroli/xtream-ai-assignment/raw/main/assignment-LeonardoAcquaroli/logreg_fit.pickle")
        predictions = make_predictions(model=logreg, exog_data=X_with_intercept, enrollee_ids=processed_data['enrollee_id'])

        # Display predictions
        st.header("Churn Probabilities")
        st.dataframe(predictions, use_container_width=True)

if __name__ == "__main__":
    main()
