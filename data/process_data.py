import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    
    """
    Loads data from messages and categories datasets, returns merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on = 'id', right_on = 'id')
    return df


def clean_data(df):
    
    """
    Takes dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    category_colnames = categories.loc[0].str.split('-').apply(lambda x: x[0]).tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        #convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))
    
    #related column contains some values of '2', re-assign them as '1'
    categories.loc[categories['related']==2,'related']=1
    
    #'child_alone' category is empty, we would not be able to train on it, drop it
    categories = categories.drop(['child_alone'], axis = 1)
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df = df[~df.duplicated()]
    
    return df


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Data shape: {} columns, {} rows'.format(df.shape[1], df.shape[0]))

        print('Cleaning data...')
        df = clean_data(df)
        print('Data shape: {} columns, {} rows'.format(df.shape[1], df.shape[0]))
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()