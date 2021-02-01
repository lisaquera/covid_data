# README

## Associated site to view the blog post:

Blog link: http://lquera.com/capstone_blog.html


### Instructions for running these files in local environment:
1. Download the files
2. cd into your local folder and double_click capstone_blog.html to open it in your default browser
3. Go to http://0.0.0.0:3001/

## Motivation

This project is motivated by the need to understand whether active government
interventions have any impact on the COVID19 cases and deaths in that country.
The Oxford University dataset tracks policies by country and date with the
associated confirmed cases and deaths. Additional demographic data was taken
from worldometers.info to add population, density, urban percentage, and
median age.


## Libraries and files
Libraries required: pandas, requests, datetime, bs4, scipy.stats, Scikit-learn, matplotlib

Files:
* DSND_covid19_policy_tracker.csv and worldmeter_info2020.csv provide the policy and demographic data.
* Blog_COVID_Data.ipynb shows the process flow
* capture_utilities.py holds the helper function to scrape demographic data
* data_utilities.py holds the helper functions for data preprocessing and analysis
* model_utilities.py holds Univariate Feature Selection class definition and helper functions for creating feature selections and feature importances
* capstone_blog.html holds the blogpost
* surreabral.css holds the styling for the blogpost
* numerous images necessary for the blogpost visualizations (all visualizations created from the notebook and thus viewable there as well.)


## Results
### Data analysis
The correlations were disappointingly weak, too weak to make confident recommendations for which policies should be pursued.  Please see capstone_blog for full discussion.


## Acknowledgements
Thank you to Oxford University and Google Cloud Platform for the policy data and worldometers.info for the demographic data.
