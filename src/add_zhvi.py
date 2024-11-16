import pandas as pd

def add_zhvi_values(train_df):
    """
    Add ZHVI values to training dataframe by merging with ZHVI data.
    
    Args:
        train_df (pd.DataFrame): Input training dataframe
        
    Returns:
        pd.DataFrame: Training dataframe with added ZHVI values
    """
    # Read ZHVI data
    zhvi_df = pd.read_csv('il_zhvi.csv')
    
    # Convert CloseDate to datetime and extract year-month
    train_df = train_df.copy()
    train_df['Listing.Dates.CloseDate'] = pd.to_datetime(train_df['Listing.Dates.CloseDate'])
    train_df['year_month'] = train_df['Listing.Dates.CloseDate'].dt.strftime('%Y-%m')

    # Convert ZHVI date columns to datetime and get year-month
    date_cols = [col for col in zhvi_df.columns if col not in ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 
                                                              'StateName', 'State', 'City', 'Metro', 'CountyName', 'BaseDate']]

    zhvi_melted = pd.melt(zhvi_df,
                          id_vars=['RegionName'],
                          value_vars=date_cols,
                          var_name='date',
                          value_name='zhvi_value')

    zhvi_melted['date'] = pd.to_datetime(zhvi_melted['date'])
    zhvi_melted['year_month'] = zhvi_melted['date'].dt.strftime('%Y-%m')

    # Convert postal codes to string for matching
    train_df['Location.Address.PostalCode'] = train_df['Location.Address.PostalCode'].astype(str)
    zhvi_melted['RegionName'] = zhvi_melted['RegionName'].astype(str)

    # Merge train data with ZHVI values based on postal code and year-month
    merged_df = train_df.merge(
        zhvi_melted,
        left_on=['Location.Address.PostalCode', 'year_month'],
        right_on=['RegionName', 'year_month'],
        how='left'
    )

    # Drop redundant columns from merge
    merged_df = merged_df.drop(['RegionName', 'date', 'year_month'], axis=1)

    return merged_df
