# ETL Swarm
ETL Swarm is a Python-based implementation designed to extract, transform, and load (ETL) data using a swarm of algorithms in conjunction with a language model (LLM) to automate data correction processes. This implementation demonstrates how to use OpenAI's GPT-4o-mini model to identify and correct issues within a dataset, and then update a SQLite database with the corrected data.

Table of Contents
Overview
Prerequisites
Installation
Setup
Usage
Code Explanation
License
Overview
ETL Swarm is a proof-of-concept that leverages AI to automate the ETL process, particularly focusing on data correction. It uses OpenAI's GPT-4o-mini model to identify and suggest corrections for data issues. The workflow involves:

Extracting data from a source.
Identifying and correcting data issues.
Loading the corrected data back into the database.
Prerequisites
Before running ETL Swarm, ensure you have the following prerequisites:

Python 3.7 or higher
OpenAI API key
Installation

Follow these steps to run the ETL Swarm code in Google Colab or your local environment:

Install Required Libraries:

python
Copy code
!pip install openai pandas sqlalchemy dask
Initialize OpenAI API and Import Libraries:

python
Copy code
from openai import OpenAI
import pandas as pd
from sqlalchemy import create_engine
Set OpenAI API Key:

python
Copy code
client = OpenAI(api_key='YOUR_API_KEY')
Create Sample Data and Save to SQLite Database:

python
Copy code
# Sample data
data = {
    "ID": [1, 2, 3, None, 5, 6, 7, 8, 9, 10],
    "NAME": ["John Doe", "Prediction = \"Mary Smith\"", "Jim Brown", "Alex Johnson", "Alan Smithee", "LUCY BLACK", "Eve White", "Michael Taylor", "Sarah Green", "TIM BLUE"],
    "DATE": ["2023-07-01", "7/2/2023", "2023-07-03", "2023/07/04", "07-05-2023", "2023-07-06", "07/07/2023", "2023-07-08", "2023-07-09", "2023-07-10"],
    "AMOUNT": [150, 175, 200, 150.25, 200, 350.75, -20, None, 500.5, 50.75],
    "EMAIL": ["JOHN.DOE@EXAMPLE.COM", "MARY.SMITH@EXAMPLE.COM", "JIM.BROWN@EXAMPLE.COM", "NANCY@DOMAIN.COM", "ALAN.SMITHEE@DOMAIN", "Prediction = \"LUCY.BLACK@DOMAIN.COM\"", "EVE.WHITE@DOMAIN.COM", "MICHAEL.TAYLOR@EXAMPLE.COM", "sarah.green@example.com", ""]
}

df = pd.DataFrame(data)
db_path = '/content/example.db'
engine = create_engine(f'sqlite:///{db_path}')
df.to_sql('customers', engine, if_exists='replace', index=False)
Define Functions:

python
Copy code
def map_data():
    engine = create_engine(f'sqlite:///{db_path}')
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, engine)

    issues = []

    for index, row in df.iterrows():
        if pd.isnull(row['ID']):
            issues.append(f"Row {index+1}: Missing ID")
        if "Prediction =" in str(row['NAME']):
            issues.append(f"Row {index+1}: Predicted Name")
        if "Prediction =" in str(row['EMAIL']):
            issues.append(f"Row {index+1}: Predicted Email")
        if row['AMOUNT'] is None or row['AMOUNT'] < 0:
            issues.append(f"Row {index+1}: Invalid Amount")
        if not isinstance(row['DATE'], str) or not any(char in row['DATE'] for char in ['-', '/']):
            issues.append(f"Row {index+1}: Invalid Date Format")

    return df, issues

def generate_correction_code(issues, client):
    prompt = f"The following data issues were found:\n{', '.join(issues)}\nGenerate only the Python code to correct these issues in a pandas DataFrame named `df`."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_code = response.choices[0].message.content.strip()

    start = generated_code.find("import pandas as pd")
    end = generated_code.rfind("```")
    if start != -1 and end != -1:
        generated_code = generated_code[start:end].strip()

    return generated_code

def apply_corrections(df, correction_code):
    local_vars = {'df': df}
    exec(correction_code, globals(), local_vars)
    return local_vars['df']

def update_database(df):
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    engine = create_engine(f'sqlite:///{db_path}')
    df.to_sql('customers', engine, if_exists='replace', index=False)
Run ETL Swarm:

python
Copy code
client = OpenAI(api_key='YOUR_API_KEY')

df, issues = map_data()

if issues:
    correction_code = generate_correction_code(issues, client)
    print("Correction Code:\n", correction_code)

    corrected_df = apply_corrections(df, correction_code)

    update_database(corrected_df)
    print("Database updated successfully with corrected data.")
else:
    print("No issues found in the data.")
Code Explanation
Install Required Libraries: Install the necessary Python libraries.
Initialize OpenAI API: Set up the OpenAI client with your API key.
Create Sample Data: Create a sample DataFrame and save it to a SQLite database.
Define Functions:
map_data(): Extracts data from the database and identifies issues.
generate_correction_code(): Uses the LLM to generate Python code for correcting identified issues.
apply_corrections(): Applies the generated correction code to the DataFrame.
update_database(): Updates the database with the corrected data, ensuring no duplicate columns.
Run ETL Swarm: Executes the ETL process by identifying issues, generating corrections, applying those corrections, and updating the database.
License
This project is licensed under the MIT License.
