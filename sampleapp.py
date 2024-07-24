# Install Required Libraries
!pip install openai
!pip install pandas
!pip install sqlalchemy
!pip install dask

# Initialize OpenAI API
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key='YOUR_API_KEY')

import pandas as pd
from sqlalchemy import create_engine

# Sample data
data = {
    "ID": [1, 2, 3, None, 5, 6, 7, 8, 9, 10],
    "NAME": ["John Doe", "Prediction = \"Mary Smith\"", "Jim Brown", "Alex Johnson", "Alan Smithee", "LUCY BLACK", "Eve White", "Michael Taylor", "Sarah Green", "TIM BLUE"],
    "DATE": ["2023-07-01", "7/2/2023", "2023-07-03", "2023/07/04", "07-05-2023", "2023-07-06", "07/07/2023", "2023-07-08", "2023-07-09", "2023-07-10"],
    "AMOUNT": [150, 175, 200, 150.25, 200, 350.75, -20, None, 500.5, 50.75],
    "EMAIL": ["JOHN.DOE@EXAMPLE.COM", "MARY.SMITH@EXAMPLE.COM", "JIM.BROWN@EXAMPLE.COM", "NANCY@DOMAIN.COM", "ALAN.SMITHEE@DOMAIN", "Prediction = \"LUCY.BLACK@DOMAIN.COM\"", "EVE.WHITE@DOMAIN.COM", "MICHAEL.TAYLOR@EXAMPLE.COM", "sarah.green@example.com", ""]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to SQLite database in the correct path
db_path = '/content/example.db'
engine = create_engine(f'sqlite:///{db_path}')
df.to_sql('customers', engine, if_exists='replace', index=False)

def map_data():
    engine = create_engine(f'sqlite:///{db_path}')
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, engine)
    
    issues = []
    
    # Identify issues
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
    
    # Extract only the code from the response
    start = generated_code.find("import pandas as pd")  # Start of actual code
    end = generated_code.rfind("```")
    if start != -1 and end != -1:
        generated_code = generated_code[start:end].strip()
    
    return generated_code

def apply_corrections(df, correction_code):
    local_vars = {'df': df}
    exec(correction_code, globals(), local_vars)
    return local_vars['df']

def update_database(df):
    # Standardize column names and remove duplicates
    df.columns = df.columns.str.lower()  # Convert all column names to lower case
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    engine = create_engine(f'sqlite:///{db_path}')
    df.to_sql('customers', engine, if_exists='replace', index=False)

# Initialize the OpenAI API client
client = OpenAI(api_key='YOUR_API_KEY')

# Step 1: Map data and identify issues
df, issues = map_data()

# Step 2: Generate correction code using LLM
if issues:
    correction_code = generate_correction_code(issues, client)
    print("Correction Code:\n", correction_code)
    
    # Step 3: Apply corrections based on the generated Python code
    corrected_df = apply_corrections(df, correction_code)
    
    # Step 4: Update the database with corrected data
    update_database(corrected_df)
    print("Database updated successfully with corrected data.")
else:
    print("No issues found in the data.")
