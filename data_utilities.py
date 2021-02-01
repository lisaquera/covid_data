'''
'''
import pandas as pd
import numpy as np
import datetime
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

cols_indicators = ['country_name', 'region_name', 'date',
                'confirmed_cases', 'deaths',
                'density', 'median_age', 'population', 'urban_perc',
                'stringency_index',
                'school_closing_flag',
                'workplace_closing_flag',
                'cancel_public_events_flag',
                'restrictions_on_gatherings_flag',
                'close_public_transit_flag',
                'stay_at_home_requirements_flag',
                'restrictions_on_internal_movement_flag',
                'international_travel_controls',
                'income_support_flag',
                'debt_contract_relief',
                'fiscal_measures',
                'international_support',
                'public_information_campaigns_flag',
                'testing_policy',
                'contact_tracing',
                'emergency_healthcare_investment',
                'vaccine_investment',
               ]

cols_ordinals = ['country_name', 'region_name', 'date',
                'confirmed_cases', 'deaths',
                'density', 'median_age', 'population', 'urban_perc',
        'stringency_index',
       'school_closing', 'workplace_closing',
       'cancel_public_events', 'restrictions_on_gatherings',
       'close_public_transit', 'stay_at_home_requirements',
       'restrictions_on_internal_movement',
       'international_travel_controls', 'income_support',
       'debt_contract_relief', 'fiscal_measures', 'international_support',
       'public_information_campaigns', 'testing_policy',
       'contact_tracing', 'emergency_healthcare_investment',
       'vaccine_investment']

def get_merged_df(filestring1, filestring2, merge_field):
    '''Merge the covid related data with the country demographic data
    Input: filestring1 = string for filepath of first csv
        filestring2 = string for filepath of second csv
        merge_field = string representing field to join on
    Output: dataframe with data from both input csv files
    '''
    df1 = pd.read_csv(filestring1)
    df2 = pd.read_csv(filestring2)
    df1 = pd.merge(df1, df2, how='inner', on=merge_field)
    df1 = df1.drop(columns=['Unnamed: 0'], axis=1)
    return df1

def get_cleaned_stats(df):
    '''Clean the data to remove missing data rows and create useful features.
    Input: df = dataframe to clean, in this case covid-related data
    Output: df = updated dataframe
    '''
    # drop all rows missing the target labels
    df = df.dropna(subset=['deaths','confirmed_cases'], axis=0)
    # drop all rows where target labels are zero
    df = df.loc[(df['confirmed_cases'] > 0.0) & (df['deaths'] > 0.0)]
    # drop all rows missing summary indicator
    df = df.loc[df['stringency_index'].notnull()]
    # get the change in stringency_index
    df['stringency_change'] = np.round(df['stringency_index'].pct_change(), decimals=2)
    # catch calculation infinites
    df = df.drop(df.index[list(np.where(np.isfinite(df['stringency_change']) == False))])
    # ensure date field is datetime to manipulate more easily
    df['date'] = pd.to_datetime(df['date'])
    # make regional data consistent: if NA, cumulative for the country
    df['region_name'] = df['region_name'].fillna('Total')
    # create a combined field to break up larger countries into regions
    # (particularly US and Brazil)
    df['geo'] = df['country_name'] + df['region_name']
    # sort by country_name, region_name, date
    df.sort_values(['country_name', 'region_name', 'date'],
                    ascending=True,
                    inplace=True,
                    na_position='last')
    return df

def get_normed_targets(df):
    '''Original data provided has only absolute data while relative data is
    needed for comparisons. Normalize by population or change in absolute.
    Input: df = cleaned dataframe of covid data
    Output: dataframe with additional normed targets and change in targets
    '''
    # get the changes in target data over time
    df['case_perc_change'] = np.round(df['confirmed_cases'].pct_change(), decimals=2)
    # catch calculation infinites
    df = df.drop(df.index[list(np.where(np.isfinite(df['case_perc_change']) == False))])
    df['death_perc_change'] = np.round(df['deaths'].pct_change(), decimals=2)
    # catch calculation infinites
    df = df.drop(df.index[list(np.where(np.isfinite(df['death_perc_change']) == False))])
    # normalize the raw statistics by population
    df['cases_perc_capita'] = (df['confirmed_cases'] / df['population'])
    df['deaths_perc_capita'] = (df['deaths'] / df['population'])
    return df

def get_working_df(filestring1, filestring2, merge_field, cols_to_use):
    '''Convert csv files to cleaned, normed dataframe for analysis & modeling.
    Input: filestring1 = string for filepath of first csv
        filestring2 = string for filepath of second csv
        merge_field = string representing field to join on
    Output: dataframe of covid statistics ready for analysis and modeling
    '''
    cov = get_merged_df(filestring1, filestring2, merge_field)
    cov_stats = cov[list(cols_to_use)]
    cov_stats = get_cleaned_stats(cov_stats)
    cov_stats = get_normed_targets(cov_stats)
    return cov_stats

# how many unique instances for each variable?
def get_value_diversity(df):
    '''
    Understand the diversity of a dataframe by getting the value counts for each column.
    Input: dataframe
    Output: new dataframe with value counts for each column
    '''
    val_cts = pd.DataFrame(columns = ['Name', 'Length'])
    for c in list(df.columns):
        l = len(df[c].value_counts().index)
        val_cts = val_cts.append({'Name': c, 'Length' : l}, ignore_index=True)
    return val_cts

def get_monthly_data(df, group_cols=['geo','month']):
    '''
    Get the data for months instead of daily.
    Input: dataframe with date column with values for 2020
    Output: write a csv with the geographic monthly data for 2020
        return a dataframe with the geographic monthly data for 2020
    '''
    df_2020 = df.loc[(df['date'] > '2019-12-31') & (df['date'] < '2021-01-01')]
    df_2020['month'] = pd.DatetimeIndex(df_2020['date']).month
    df_2020_mths = df_2020.groupby(group_cols).max()
    df_2020_mths.to_csv('stats_by_mth_2020.csv')
    return df_2020_mths

def get_weekly_data(df, group_cols=['geo','week']):
    '''
    Get the data for weeks instead of daily.
    Input: dataframe with date column with values for 2020
    Output: write a csv with the geographic weekly data for 2020
        return a dataframe with the geographic weekly data for 2020
    '''
    df_2020 = df.loc[(df['date'] > '2019-12-31') & (df['date'] < '2021-01-01')]
    df_2020['week'] = pd.DatetimeIndex(df_2020['date']).week
    df_2020_weeks = df_2020.groupby(group_cols).max()
    df_2020_weeks.to_csv('stats_by_week_2020.csv')
    return df_2020_weeks

def get_shifted_data(df, num_shifts, cols_to_shift):
    '''Create columns of previous values to show lagged impact. For example,
    for disease data, it may take 1 or 2 weeks for a policy to impact the
    target.
    Input: df = dataframe with time-relevant data
        num_shifts = number of periods to lag, eg. 1 month, 2 weeks, etc.
        cols_to_shift = fields with data impacting lag
    '''
    for col in cols_to_shift:
        colname = 'prev_'+col
        df[colname] = df[col].shift(num_shifts)
    df = df.drop(columns=cols_to_shift)
    return df

def check_df_nulls(df, fill_val=0):
    '''Deal with missing values in dataframe to prevent crashes.
    Input: df = dataframe to check
        fill_val = value to enter if data is missing, default=0
    Output: dataframe with no missing values
    '''
    if df.isnull().sum().sum() > 0:
        df = df.fillna(value=fill_val)
    return df

def check_dataset_nulls(X,y):
    '''Prevent training crashes by filling any null values in the datasets
    Input: X as training data matrix or dataframe
        y as series
    Output: validated X and y
    '''
    X_problems = np.where(np.isfinite(X) == False)
    if len(X_problems) > 0:
        X_problem_rows = list(set(list(X_problems)[0]))
        X_problem_cols = list(set(list(X_problems)[1]))
        for r in X_problem_rows:
            if len(X_problem_cols) > 10:
                X = X.drop(X.index[r])
            else:
                for c in X_problem_cols:
                    X.iloc[r, c] = 0

    y_problems = np.where(np.isfinite(y) == False)
    if len(y_problems) > 0:
        for i in y_problems:
            if np.isfinite(y.iloc[i]).any() == False:
                y.iloc[i] = 0

    return X, y

# get Pearson's coefficients
def get_pearsons_corr(df, cols_to_check, outcome, country=None):
    '''Get the Pearson's coefficients using scipy.stats function
    Input: df = dataframe
        cols_to_check = list of feature columns to evaluate against target
        outcome = string representing target column to evaluate correlations
        country = optional string representing country to subset
    Output: dataframe of features and Pearson correlations to target
    '''
    if country:
        df = df.loc[df['country_name'] == country]
    corr_dict = []
    for col in cols_to_check:
        corr, _ = pearsonr(df[col], df[outcome])
        corr_dict.append({'feature': col, 'corr': corr})
    corr_df = pd.DataFrame.from_dict(corr_dict)
    return corr_df

# get Spearman's coefficients
def get_spearmans_corr(df, cols_to_check, outcome, country=None):
    '''Get the Spearman's coefficients using scipy.stats function
    Input: df = dataframe
        cols_to_check = list of feature columns to evaluate against target
        outcome = string representing target column to evaluate correlations
        country = optional string representing country to subset
    Output: dataframe of features and Spearman correlations to target
    '''
    if country:
        df = df.loc[df['country_name'] == country]
    corr_dict = []
    for col in cols_to_check:
        corr, _ = spearmanr(df[col], df[outcome])
        corr_dict.append({'feature': col, 'corr': corr})
    corr_df = pd.DataFrame.from_dict(corr_dict)
    return corr_df

# Compare two countries outcomes
def compare_country_outcomes(df, country1, country2, target_to_compare):
    '''Visualize a performance metric over time in 2020 for two countries.
    Input: df = dataframe with statistics
        country1 = string representing the first country to compare
        country2 = string representing the second country to compare
        target_to_compare = string representing the metric in the dataframe
    Output: a line chart comparing the monthly max metric for the two countries
    '''

    comp_df = df[['country_name', 'date', target_to_compare]]
    comp_df['date'] = pd.to_datetime(comp_df['date'])
    comp_df = comp_df.loc[(comp_df['date'] > '2019-12-31') & (comp_df['date'] < '2021-01-01')]
    comp_df['month'] = pd.DatetimeIndex(comp_df['date']).month
    comp_df = comp_df.groupby(['country_name','month']).max()
    country_one = comp_df.loc[country1]
    country_two = comp_df.loc[country2]
    #print(country_one.index)
    pic_filestring = country1+'_'+country1+'_'+target_to_compare+'.png'
    plt.plot(country_one.index, country_one[target_to_compare], label=country1)
    plt.plot(country_two.index, country_two[target_to_compare], label=country2)
    plt.xlabel('Month in 2020')
    plt.ylabel(target_to_compare)
    plt.title('Performance in 2020')
    plt.legend()
    plt.show()
    plt.savefig(pic_filestring)

#
# Compare policies with outcomes
def compare_policy_outcomes(df, country, policy, target_to_compare):
    '''Visualize a policy and an outcome performance metric over time in 2020.
    The target outcome data numbers are so small in comparison to the policies
    some scaling is necessary to make them comparable. The y label is thus a
    trend indication rather than specific number.
    Input: df = dataframe with statistics
        country = string representing the country to evaluate
        policy = string representing a government policy metric
        target_to_compare = string representing the performance metric
    Output: a line chart comparing the monthly average metrics normalized to show trends
    '''

    comp_df = df[['country_name', 'date', policy, target_to_compare]]
    comp_df['date'] = pd.to_datetime(comp_df['date'])
    comp_df = comp_df.loc[(comp_df['date'] > '2019-12-31') & (comp_df['date'] < '2021-01-01')]
    comp_df['month'] = pd.DatetimeIndex(comp_df['date']).month
    comp_df = comp_df.groupby(['country_name','month']).mean()
    policy_data = comp_df[policy].loc[country]
    policy_scaled = policy_data / policy_data.sum()
    outcome_data = comp_df[target_to_compare].loc[country]
    outcome_scaled = outcome_data / outcome_data.sum()
    #print(policy_normed)

    pic_filestring = country+'_'+policy+'_'+target_to_compare+'.png'
    plt.plot(policy_scaled.index, policy_scaled, label=policy)
    plt.plot(outcome_scaled.index, outcome_scaled, label=target_to_compare)
    plt.xlabel('Month in 2020')
    plt.ylabel('Trend')
    plt.title('Policy Performance in 2020')
    plt.legend()
    plt.show()
    plt.savefig(pic_filestring)

'''GET STARTED WITH THIS:
filestring1 = 'DSND_covid19_policy_tracker.csv'
filestring2 = 'worldmeter_info2020.csv'
merge_field = 'country_name'
covid = get_working_df(filestring1, filestring2, merge_field)
'''
