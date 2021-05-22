import pandas as pd

# Read the data for a single study and single metric (e.g., "Novelty_Combined")
# Optionally shuffles the data before returning dataframe
# Returns a df with a column named 'text' and a column named 'label'
def get_data(study, metric, shuffle=True):
  sheet_df = pd.read_excel("Idea Ratings_Berg_2019_OBHDP.xlsx", sheet_name=study-1) 
  sheet_df.dropna(inplace=True)
  data_df = sheet_df[['Final_Idea', metric]].rename(columns={'Final_Idea': 'text', metric: 'label'})

  if shuffle:
    data_df = data_df.sample(frac=1)
  return data_df

# Take a list with the numbers of studies, and a specific metric
# Extract multiple datasets with get_data and concatenate them
def get_multiple_datasets(study_list, metric, shuffle = True):
  dfs = [get_data(study, metric, shuffle) for study in study_list]
  return pd.concat(dfs)