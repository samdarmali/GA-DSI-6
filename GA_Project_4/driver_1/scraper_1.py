# Import scraping libraries
from bs4 import BeautifulSoup
import requests
import urllib
import os
from selenium import webdriver
from time import sleep
import getpass

# Import standard libraries
import pickle
import random
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def get_links(search='', page_start=0, page_end=0):
    # Set path from cromedriver
    chromedriver = "/Users/samdarmali/Desktop/GA/materials-duplicate/projects/project-4/chromedriver/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver

    # Create a driver called "driver."
    driver = webdriver.Chrome(executable_path=chromedriver)
    
    # Dictionary of links to keep
    links = []
    
    # Ask user for first url and number of pages
    url_frag = search.replace(' ','%20')
    
    count = 250
    
    for page in range(int(page_start), int(page_end)):
        # Use url
        url = 'https://www.mycareersfuture.sg/search?search={}&sortBy=new_posting_date&page={}'.format(url_frag, page)
        print('Getting links from {} ...'.format(url))
        
        # Get page and wait 5 to 8 seconds
        driver.get(url)
        sleep(random.randint(5,8))

        # Get the html and convert to beautiful soup
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        for i in soup.find_all('a', {'class':'bg-white'}):
            
            # Get link for each post, if job link then scrape it, if not then continue
            post_link = i.get('href')
            
            # Complete url
            if post_link.startswith('/job/'):
                post_link = 'https://www.mycareersfuture.sg' + post_link
                links.append(post_link)
            
            # Save to new txt file every 250 links
            if len(links) % 250 == 0:
                with open('links{}_{}-{}.txt'.format(count, page_start, page_end), 'wb') as fp:
                    pickle.dump(links, fp)
                print('First {} links stored in temporary txt file ...'.format(count))
                count += 250
        
        print('----\nLinks from page {} received\n-----'.format(page))
    
    # Close driver
    driver.close()
    
    return links


def mycareersfuture_scraper():
    # Set path from cromedriver
    chromedriver = "/Users/samdarmali/Desktop/GA/materials-duplicate/projects/project-4/chromedriver/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver

    # Create a driver called "driver."
    driver = webdriver.Chrome(executable_path=chromedriver)
    
    # Dictionary of information to keep
    details = {'title': [],
               'company': [],
               'description': [],
               'company_overview': [],
               'location': [],
               'role': [],
               'hours': [],
               'industry': [],
               'applications': [],
               'posted': [],
               'closing': [],
               'salary_type': [],
               'salary': []}
    
    # Ask user for first url and number of pages
    search = input('Search: ')
    url_frag = search.replace(' ','%20')
    page_start = input('Page Start: ')
    page_end = input('Page End: ')
    
    # Get the links for search and pages parameters
    links = get_links(search=search, page_start=page_start, page_end=page_end)
    
    # Counter for csv file name
    count = 250
    
    for link in links:
        print('Scraping {} ...'.format(link))
        
        # Go to post page
        driver.get(link)
        sleep(random.randint(5,8))
        post_html = driver.page_source
        post_soup = BeautifulSoup(post_html, 'lxml')

        # Get title
        if not post_soup.find_all('h1', {'id':'job_title'}):
            details['title'].append(None)
        for title in post_soup.find_all('h1', {'id':'job_title'}):
            details['title'].append(title.text)

        # Get company
        if not post_soup.find_all('p',  {'name':'company'}):
            details['company'].append(None)
        for company in post_soup.find_all('p', {'name':'company'}):
            details['company'].append(company.text)

        # Get description
        if not post_soup.find_all('div', {'class':'jobDescription'}):
            details['description'].append(None)
        for desc in post_soup.find_all('div', {'class':'jobDescription'}):
            details['description'].append(desc.text)

        # Get company overview
        xpath = '//*[@id="job_details"]/div[2]/div[1]/div/div/section/section/div[2]/div/div[1]'
        try:
            details['company_overview'].append(driver.find_element_by_xpath(xpath).text)   
        except:
            details['company_overview'].append(None)
        
        # Get location
        if not post_soup.find_all('p', {'id':'address'}):
            details['location'].append(None)
        for loc in post_soup.find_all('p', {'id':'address'}):
            details['location'].append(loc.text)

        # Get role
        if not post_soup.find_all('p', {'id':'seniority'}):
            details['role'].append(None)
        for role in post_soup.find_all('p', {'id':'seniority'}):
            details['role'].append(role.text)

        # Get hours
        if not post_soup.find_all('p', {'id':'employment_type'}):
            details['hours'].append(None)
        for hours in post_soup.find_all('p', {'id':'employment_type'}):
            details['hours'].append(hours.text)

        # Get industry
        if not post_soup.find_all('p', {'id':'job-categories'}):
            details['industry'].append(None)
        for ind in post_soup.find_all('p', {'id':'job-categories'}):
            details['industry'].append(ind.text)

        # Get applications
        if not post_soup.find_all('span', {'id':'num_of_applications'}):
            details['applications'].append(None)
        for apps in post_soup.find_all('span', {'id':'num_of_applications'}):
            details['applications'].append(apps.text)

        # Get posted
        if not post_soup.find_all('span', {'id':'last_posted_date'}):
            details['posted'].append(None)
        for posted in post_soup.find_all('span', {'id':'last_posted_date'}):
            details['posted'].append(posted.text)

        # Get closing
        if not post_soup.find_all('span', {'id':'expiry_date'}):
            details['closing'].append(None)
        for closing in post_soup.find_all('span', {'id':'expiry_date'}):
            details['closing'].append(closing.text)
            
        # Get salary type
        if not post_soup.find_all('span', {'class':'salary_type'}):
            details['salary_type'].append(None)
        for sal in post_soup.find_all('span', {'class':'salary_type'}):
            details['salary_type'].append(sal.text.replace('to','-'))

        # Get salary
        if not post_soup.find_all('span', {'class':'salary_range'}):
            details['salary'].append(None)
        for sal in post_soup.find_all('span', {'class':'salary_range'}):
            details['salary'].append(sal.text.replace('to','-'))

        print('---')
        
        # Safeguard, save every 250 scraped job postings into a temporary df
        if len(details['title']) % 250 == 0:
            temp_df = pd.DataFrame(details)
            temp_df.to_csv('temp_df{}_{}-{}.csv'.format(count, page_start, page_end))
            print('First {} postings stored in temporary dataframe ...'.format(count))
            count += 250

    # Close driver
    driver.close()
    
    try:
        details_df = pd.DataFrame(details)
        print('Dataframe created!')
    except:
        print('')
        print('error: arrays must all be same length')
        print('length title:', len(details['title']))
        print('length company:', len(details['company']))
        print('length description:', len(details['description']))
        print('length company_overview:', len(details['company_overview']))
        print('length location:', len(details['location']))
        print('length role:', len(details['role']))
        print('length hours:', len(details['hours']))
        print('length industry:', len(details['industry']))
        print('length applications:', len(details['applications']))
        print('length posted:', len(details['posted']))
        print('length closing:', len(details['closing']))
        print('length salary_type:', len(details['salary_type']))
        print('length salary:', len(details['salary']))

    return details_df, links


# Put all scraped jobs into csv file
details, links = mycareersfuture_scraper()
details.to_csv('data_{}-{}.csv'.format(0, 40))

# Put all scraped links into txt file
with open('links_{}-{}.txt'.format(0, 40), 'wb') as fp:
    pickle.dump(links, fp)



    