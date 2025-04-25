from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    # split into train and temp (test + val)
    train_df, temp_df = train_test_split(
        df, 
        test_size=test_size+val_size,
        random_state=random_state
    )
    
    # split temp into test and val
    val_size_adjusted = val_size / (test_size + val_size)
    test_df, val_df = train_test_split(
        temp_df,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    return train_df, val_df, test_df
