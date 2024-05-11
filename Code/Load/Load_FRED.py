import pandas_datareader.data as web
import pandas as pd
from pathlib import Path

DATA_DIR = Path('../../Data/Input/Factors')
DATA_DIR.mkdir(parents=True, exist_ok=True)

series_to_pull = {
    'ALPCPI': 'Alabama', 'AKPCPI': 'Alaska', 'AZPCPI': 'Arizona',
    'ARPCPI': 'Arkansas', 'CAPCPI': 'California', 'COPCPI': 'Colorado',
    'CTPCPI': 'Connecticut', 'DEPCPI': 'Delaware', 'DCPCPI': 'District of Columbia',
    'FLPCPI': 'Florida', 'GAPCPI': 'Georgia', 'HIPCPI': 'Hawaii',
    'IDPCPI': 'Idaho', 'ILPCPI': 'Illinois', 'INPCPI': 'Indiana',
    'IAPCPI': 'Iowa', 'KSPCPI': 'Kansas', 'KYPCPI': 'Kentucky',
    'LAPCPI': 'Louisiana', 'MEPCPI': 'Maine', 'MDPCPI': 'Maryland',
    'MAPCPI': 'Massachusetts', 'MIPCPI': 'Michigan', 'MNPCPI': 'Minnesota',
    'MSPCPI': 'Mississippi', 'MOPCPI': 'Missouri', 'MTPCPI': 'Montana',
    'NEPCPI': 'Nebraska', 'NVPCPI': 'Nevada', 'NHPCPI': 'New Hampshire',
    'NJPCPI': 'New Jersey', 'NMPCPI': 'New Mexico', 'NYPCPI': 'New York',
    'NCPCPI': 'North Carolina', 'NDPCPI': 'North Dakota', 'OHPCPI': 'Ohio',
    'OKPCPI': 'Oklahoma', 'ORPCPI': 'Oregon', 'PAPCPI': 'Pennsylvania',
    'RIPCPI': 'Rhode Island', 'SCPCPI': 'South Carolina', 'SDPCPI': 'South Dakota',
    'TNPCPI': 'Tennessee', 'TXPCPI': 'Texas', 'UTPCPI': 'Utah',
    'VTPCPI': 'Vermont', 'VAPCPI': 'Virginia', 'WAPCPI': 'Washington',
    'WVPCPI': 'West Virginia', 'WIPCPI': 'Wisconsin', 'WYPCPI': 'Wyoming'
}

regions = {
    "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"],
    "Midwest": ["Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"],
    "South": ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "District of Columbia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"]
}
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
state_abbreviations_rev = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}

population_series_ids = {abbrev + 'POP' : state for state, abbrev in state_abbreviations.items()}
def assign_region(state_name):
    for region, states in regions.items():
        if state_name in states:
            return region
    return None

def pull_all(start_date='2000-01-01', end_date='2024-01-01', pull_API=False):
    if pull_API:
        economic_data = web.DataReader(list(series_to_pull.keys()), 'fred', start_date, end_date)
        population_data = web.DataReader(list(population_series_ids.keys()), 'fred', start_date, end_date)
        df = pd.concat([economic_data, population_data], axis=1)
    else:
        df = pd.read_csv(DATA_DIR / 'raw_fred_personal_income.csv')
        df.index = pd.to_datetime(df['DATE'])
        df.drop(columns='DATE', inplace=True)
    df_resampled = df.resample('MS').interpolate(method='linear')
    melted_df = pd.melt(df_resampled.reset_index(), id_vars=['DATE'], var_name='Variable', value_name='Value')
    melted_df['State'] = melted_df['Variable'].str.extract('([A-Z]{2})')  # Assumes state codes are two letters
    melted_df['Type'] = melted_df['Variable'].str.extract('(PCPI|POP)$')
    population_df = melted_df[melted_df['Type'] == 'POP'].copy()
    population_df['Region'] = population_df['State'].apply(lambda x: assign_region(state_abbreviations_rev[x]))
    regional_population = pd.DataFrame(population_df.groupby(['Region', 'DATE'])['Value'].sum())
    regional_population = regional_population.rename(columns={'Value':'Population'})
    pcpi_df = melted_df[melted_df['Type'] == 'PCPI']
    merged_df = pd.merge(pcpi_df, population_df, on=['DATE', 'State'], suffixes=('_PCPI', '_POP'))
    merged_df['Region'] = merged_df['State'].apply(lambda x: assign_region(state_abbreviations_rev[x]))
    merged_df['Weight'] = merged_df['Value_POP'] / merged_df.groupby('DATE')['Value_POP'].transform('sum')
    merged_df['Weighted_PCPI'] = merged_df['Weight'] * merged_df['Value_PCPI']
    regionally_weighted_pcpi = merged_df.groupby(['Region', 'DATE'])['Weighted_PCPI'].sum() / \
                               merged_df.groupby(['Region', 'DATE'])['Weight'].sum()
    regionally_weighted_pcpi = pd.DataFrame(regionally_weighted_pcpi, columns=['w_PCPI'])
    return regionally_weighted_pcpi, regional_population


def main(pull_API=False):
    regionally_weighted_pcpi, regional_population = pull_all(pull_API=pull_API)
    regionally_weighted_pcpi.to_csv(DATA_DIR / 'regionally_weighted_pcpi.csv')
    regional_population.to_csv(DATA_DIR / 'regional_population.csv')
    return regionally_weighted_pcpi, regional_population




