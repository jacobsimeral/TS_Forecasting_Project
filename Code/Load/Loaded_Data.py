import pandas as pd

"""
Loaded_Data.py

This module loads various datasets for the project.

"""

regions = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "South": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"]
}

def assign_region(state):
    for region, states in regions.items():
        if state in states:
            return region
    return None
def melt_merge_data(data: dict):
    combined_data = pd.DataFrame()
    for value, dataframe in data.items():
        dataframe.columns = dataframe.columns.str.strip().str.lower()

        df_melted = pd.melt(dataframe, id_vars=['regionid', 'sizerank', 'regionname', 'regiontype', 'statename'],
                            var_name='date', value_name=value.lower())
        df_melted['date'] = pd.to_datetime(df_melted['date'], errors='coerce')

        if combined_data.empty:
            combined_data = df_melted
        else:
            combined_data.columns = combined_data.columns.str.strip().str.lower()

            combined_data = combined_data.merge(df_melted,
                                                on=['regionid', 'sizerank', 'regionname', 'regiontype', 'statename',
                                                    'date'],
                                                how='outer', suffixes=('', '_dup'))

            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
            combined_data = combined_data.sort_values(by='date')
            combined_data.index = combined_data['date']
            combined_data = combined_data.drop(columns=['date'])
            combined_data['region'] = combined_data['statename'].apply(assign_region)

    return combined_data[['regionid', 'sizerank', 'regionname', 'regiontype','region', 'statename',
       'new construction sales count', 'new construction mean sales price ($)',
        'mean days to pending', 'mean price cut ($)',
       'mean price cut (%)', 'median sales price ($)',
       'percent sold above list (%)', 'zillow home value index',
       'zillow observed rent index']]


new_construction_sales_count = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/New_Construction/Metro_invt_fs_uc_sfrcondo_sm_month.csv')
new_construction_mean_sales_price = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/New_Construction/Metro_new_con_mean_sale_price_uc_sfrcondo_month.csv')

mean_days_to_pending = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Days_and_Price_Cuts/Metro_mean_doz_pending_uc_sfrcondo_sm_month.csv')
mean_price_cut_dol = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Days_and_Price_Cuts/Metro_mean_listings_price_cut_amt_uc_sfrcondo_sm_month.csv')
mean_price_cut_perc = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Days_and_Price_Cuts/Metro_mean_listings_price_cut_perc_uc_sfrcondo_sm_month.csv')

median_sales_price = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Sales/Metro_median_sale_price_uc_sfrcondo_sm_month.csv')
perc_sold_above_list = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Sales/Metro_pct_sold_above_list_uc_sfrcondo_sm_month.csv')

zillow_HVI = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Home_Values/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')

zillow_ORI = pd.read_csv('/Users/hassaanulhaq/Downloads/github_repo_project/TS_Forecasting_Project/Data/Input/Rentals/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv')

monthly_data = {
    'New Construction Sales Count': new_construction_sales_count,
    'New Construction Mean Sales Price ($)': new_construction_mean_sales_price,
    'Mean Days to Pending': mean_days_to_pending,
    'Mean Price Cut ($)': mean_price_cut_dol,
    'Mean Price Cut (%)': mean_price_cut_perc,
    'Median Sales Price ($)': median_sales_price,
    'Percent Sold Above List (%)': perc_sold_above_list,
    'Zillow Home Value Index' : zillow_HVI,
    'Zillow Observed Rent Index' : zillow_ORI
}

combined_data = melt_merge_data(monthly_data)





