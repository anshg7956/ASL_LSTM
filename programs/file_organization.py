import pandas as pd
import re

def filter_csv_data(df):
    """
    Filter CSV data to keep only rows with the smallest second number
    for each unique combination of ID and first number.
    
    Args:
        df: pandas DataFrame with SENTENCE_NAME column
    
    Returns:
        tuple: (kept_df, dropped_df) - both pandas DataFrames
    """
    
    # Create a copy to avoid modifying the original
    df_filtered = df.copy()
    
    # Extract components from SENTENCE_NAME
    # Pattern: ID_firstnum-secondnum-rgb_front
    def extract_components(sentence_name):
        # Match pattern: anything_number-number-rgb_front
        pattern = r'(.+)_(\d+)-(\d+)-rgb_front'
        match = re.match(pattern, sentence_name)
        
        if match:
            video_id = match.group(1)
            first_num = int(match.group(2))
            second_num = int(match.group(3))
            return video_id, first_num, second_num
        else:
            # Return None values if pattern doesn't match
            return None, None, None
    
    # Apply extraction to create helper columns
    df_filtered[['extracted_id', 'first_num', 'second_num']] = df_filtered['SENTENCE_NAME'].apply(
        lambda x: pd.Series(extract_components(x))
    )
    

    df_na = df_filtered[df_filtered['extracted_id'].isna()| df_filtered['first_num'].isna()| df_filtered['second_num'].isna()]
    # Remove rows where extraction failed
    df_filtered = df_filtered.dropna(subset=['extracted_id', 'first_num', 'second_num'])
    
    # Get the indexes of rows to keep (minimum second_num in each group)
    keep_indexes = df_filtered.groupby(['extracted_id', 'first_num'])['second_num'].idxmin()
    
    # Create kept DataFrame
    df_kept = df_filtered.loc[keep_indexes].copy()
    
    # Create dropped DataFrame (all rows NOT in keep_indexes)
    df_dropped = df_filtered.loc[~df_filtered.index.isin(keep_indexes)].copy()
    
    # Drop the helper columns from both DataFrames
    df_kept = df_kept.drop(['extracted_id', 'first_num', 'second_num'], axis=1)
    df_dropped = df_dropped.drop(['extracted_id', 'first_num', 'second_num'], axis=1)
    
    # Reset indexes
    df_kept = df_kept.reset_index(drop=True)
    df_dropped = df_dropped.reset_index(drop=True)
    
    return df_kept, df_dropped, df_na

# Example usage:
if __name__ == "__main__":
    # Load your CSV file
    # df = pd.read_csv('your_file.csv')
    
    # Or use your existing DataFrame
    # df_kept, df_dropped = filter_csv_data(df)
    
    # Save both results
    # df_kept.to_csv('kept_rows.csv', index=False)
    # df_dropped.to_csv('dropped_rows.csv', index=False)
    
    # Example with sample data to demonstrate
    
    validation_df = pd.read_csv("validation_csv/validation_data_translations.csv", sep="\t")
    print(validation_df.head())
    print(validation_df.shape)

    df_kept, df_dropped, df_na = filter_csv_data(validation_df)
    
    print("\n" + "="*50)
    print("KEPT ROWS:")
    print(df_kept.head())
    print(df_kept.shape)
    print("\n" + "="*50)
    print("DROPPED ROWS:")
    print(df_dropped.head())
    print("\n" + "="*50)
    print("NA ROWS:")
    print(df_na.head())


    df_kept.to_csv('validation_csv/df_no-duplicates_v1.csv', index=False) 
