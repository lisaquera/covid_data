import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime



def get_world_stats(filestring):
    '''Scrape the data table from worldometers to get country demographics
    Input: filestring = string representing filepath to store csv
    Output: write a csv file to store the scraped and cleaned data
    '''
    urlstring = 'https://www.worldometers.info/world-population/population-by-country/'

    # get additional data by scraping www.worldometers.info statistics
    table_df = pd.DataFrame()
    page = requests.get(urlstring)

    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        #print(soup.prettify())
        table = soup.find_all('tbody')[0]
        #print(table)
        row_marker = 0
        for row in table.find_all('tr'):
            columns = row.find_all('td')
            country_name= columns[1].get_text()
            population= columns[2].get_text()
            density= columns[5].get_text()
            median_age= columns[9].get_text()
            urban_perc= columns[10].get_text()
            table_df = table_df.append({'country_name':country_name,
                                        'population':population,
                                        'density':density,
                                        'median_age':median_age,
                                        'urban_perc':urban_perc},
                                        ignore_index=True)
    else:
        print("Page request failed with code", page.status_code)

    # clean the scraped data
    if table_df.empty == False:
        # strip nonnumeric characters and convert to floats
        table_df['density'] = table_df['density'].apply(lambda x: x.replace('N.A.','0'))
        table_df['density'] = table_df['density'].apply(lambda x: x.replace(',','')).astype(float)
        table_df['median_age'] = table_df['median_age'].apply(lambda x: x.replace('N.A.','0'))
        table_df['median_age'] = table_df['median_age'].astype(float)
        table_df['population'] = table_df['population'].apply(lambda x: x.replace(',','')).astype(float)
        table_df['urban_perc'] = table_df['urban_perc'].apply(lambda x: x.replace('N.A.','0'))
        table_df['urban_perc'] = table_df['urban_perc'].apply(lambda x: x[0:2]).fillna(0).astype(float)
    else:
        print("The dataframe updates failed.")

    # output the cleaned data
    if table_df['urban_perc'].dtype == np.float64:
        table_df.to_csv(filestring)
    else:
        print("The dataframe cleaning failed.")
