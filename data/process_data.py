import sys
import pandas as pd   
from sqlite3 import connect

def load_data(messages_filepath, categories_filepath):
    '''
    Load the csv files and merge these files together

    Parameters:
    messages_filepath (str): The file path of "disaster_message.csv"
    categories_filepath (str): The file path of "disaster_categories.csv"

    Returns:
    Pandas Dataframe. 
    '''
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    df = df_messages.merge(df_categories,on="id")
    print(df)
    return df


def get_categories_detail_table(df):
    '''
    Get the dataframe that split the categories column into detailed value column 
    eg: 
        categories
        related-1;water-1;...
    to:
        related|water|...
            1  |  1  |...
    
    Parameter:
    df (Pandas Dataframe): Input dataframe

    Return:
    Pandas Dataframe
    '''
    output = []
    for row in df.categories:
        ls_categories = row.split(";")
        output.append({value.split("-")[0]:value.split("-")[1] for value in ls_categories})
    return pd.DataFrame(output)

def clean_data(df):
    '''
    Process and clean the data 

    Parameters:
    df (Pandas Dataframe): input dataframe

    Returns:
    Pandas Dataframe
    '''
    # The original column missing so many value, so I eliminate this column, and drop ID column: 
    df_clean = df.drop("original", axis=1)
    df_clean.drop("id", inplace=True, axis=1)
    # Then remove the duplicate row:
    df_clean = df_clean.drop_duplicates(keep='first',ignore_index=True)
    df_categories_detail = get_categories_detail_table(df_clean)

    df_clean = pd.concat([df_clean,df_categories_detail], axis=1)
    # Drop the old categories ones.
    df_clean.drop("categories", inplace=True, axis=1)
    # Delete the row that exist related equals 2: 
    df_clean = df_clean.drop(df_clean[df_clean['related'] == "2"].index)

    return df_clean

def save_data(df, database_filename):
    '''
    Save the dataframe to SQL file (file .db)

    Parameters:
    df (Pandas Dataframe): Input dataframe
    database_filename (str): The output path of database file
    '''
    conn = connect(database_filename)
    curr = conn.cursor()
    curr.execute('CREATE TABLE IF NOT EXISTS disaster_data (message TEXT, genre TEXT, related NUMBER, request NUMBER, offer NUMBER,aid_related NUMBER, medical_help NUMBER, medical_products NUMBER, search_and_rescue NUMBER,security NUMBER, military NUMBER, child_alone NUMBER, water NUMBER, food NUMBER, shelter NUMBER,clothing NUMBER, money NUMBER, missing_people NUMBER, refugees NUMBER, death NUMBER, other_aid NUMBER,infrastructure_related NUMBER, transport NUMBER, buildings NUMBER, electricity NUMBER,tools NUMBER, hospitals NUMBER, shops NUMBER, aid_centers NUMBER, other_infrastructure NUMBER, weather_related NUMBER, floods NUMBER, storm NUMBER, fire NUMBER, earthquake NUMBER, cold NUMBER, other_weather NUMBER, direct_report NUMBER)')
    conn.commit()
    df.to_sql('disaster_data', conn, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories \
              datasets as the first and second argument respectively, as \
              well as the filepath of the database to save the cleaned data \
              to as the third argument. \n\nExample: python data/process_data.py \
              data/disaster_messages.csv data/disaster_categories.csv \
              DisasterResponse.db')


if __name__ == '__main__':
    main()