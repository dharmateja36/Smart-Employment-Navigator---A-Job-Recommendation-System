#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install selenium


# In[17]:


from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd
from tqdm import tqdm
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException,WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import json
import urllib
import time
import re
import math


# In[18]:


def clicknext(driver, job_count):
    num_times_click_next = math.floor(job_count/30)
    print(num_times_click_next)

    # Define the XPath for the 'Next' button
    next_button_xpath = "//*[@id='left-column']/div[2]/div/button"
    
    while num_times_click_next > 0:
        try:
            # Wait for the next button to be clickable before clicking
            next_elem = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.XPATH, next_button_xpath))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", next_elem)
            next_elem.click()
            
            time.sleep(3)  # It's better to use explicit waits instead of sleep
            
            WebDriverWait(driver, 20).until(
                lambda d: d.find_element(By.XPATH, next_button_xpath).is_displayed()
            )
            
            # Handle potential pop-ups after clicking
            try:
                driver.find_element(By.XPATH, "/html/body/div[11]/div[2]/div[1]/button").click()
            except NoSuchElementException:
                pass# If the pop-up isn't there, ignore it

            print("Clicked 'Next'")
            num_times_click_next -= 1
        
        except TimeoutException as e:
            print("Encountered TimeoutException:", e)
            break
        
        except ElementClickInterceptedException:
            # Handle the click interception
            print("Element click intercepted, trying JavaScript click.")
            driver.execute_script("arguments[0].click();", next_elem)

        except StaleElementReferenceException:
            # If the element becomes stale, wait until it's clickable again
            #print("Encountered a stale element, re-finding the 'Next' button.")
            time.sleep(1)  # Again, using sleep is not best practice

    print("Finished clicking 'Next'")


# In[19]:


def geturl(driver,url):
    url = set()
    resp = driver.page_source
    soup1 = BeautifulSoup(resp, "html.parser")
    main = soup1.find("ul",{"class":"JobsList_jobsList__Ey2Vo"})
    allJobs = main.find_all("li",{"class":"JobsList_jobListItem__JBBUV"})
    print(len(allJobs))
    for m in allJobs:
        url.add('https://www.indeed.com{}'.format(m.find('a',{"data-test":"job-link"})['href']))
    return list(url)


# Data engineer , united states

# In[5]:


target_url = "https://www.glassdoor.com/Job/united-states-data-engineer-jobs-SRCH_IL.0,13_IN1_KO14,27.htm"


# In[20]:


driver = webdriver.Chrome()


# In[7]:


driver.get(target_url)
driver.maximize_window()
time.sleep(10)
resp = driver.page_source


# In[8]:


element_text = driver.find_element(By.XPATH, "//*[@id='left-column']/div[1]/h1")
text = element_text.text

first_part = text.split(' ', 1)[0]

number_str_no_commas = first_part.replace(",", "")

job_count = 0

# Convert the string to an integer
if number_str_no_commas:
    job_count = int(number_str_no_commas)

print(job_count)


# In[9]:


url = []


# In[10]:


clicknext(driver,job_count)
time.sleep(3)


# In[11]:


with open('url_data_engineer_loc_US.json','w') as f:
    json.dump(geturl(driver,url),f, indent = 4)
    print("file created")


# Data Analyst, united states

# In[12]:


target_url = "https://www.glassdoor.com/Job/united-states-data-analyst-jobs-SRCH_IL.0,13_IN1_KO14,26.htm"


# In[13]:


driver = webdriver.Chrome()


# In[14]:


driver.get(target_url)
driver.maximize_window()
time.sleep(10)
resp = driver.page_source


# In[15]:


element_text = driver.find_element(By.XPATH, "//*[@id='left-column']/div[1]/h1")
text = element_text.text

first_part = text.split(' ', 1)[0]

number_str_no_commas = first_part.replace(",", "")

job_count = 0

# Convert the string to an integer
if number_str_no_commas:
    job_count = int(number_str_no_commas)

print(job_count)


# In[16]:


url = []


# In[17]:


clicknext(driver,job_count)
time.sleep(3)


# In[18]:


with open('url_data_analyst_loc_US.json','w') as f:
    json.dump(geturl(driver,url),f, indent = 4)
    print("file created")


# Data Scientist, united states

# ML Engineer, united states

# AI Engineer, united states

# Software Engineer, United states

# In[19]:


target_url = "https://www.glassdoor.com/Job/united-states-software-engineer-jobs-SRCH_IL.0,13_IN1_KO14,31.htm"


# In[20]:


driver = webdriver.Chrome()


# In[21]:


driver.get(target_url)
driver.maximize_window()
time.sleep(10)
resp = driver.page_source


# In[22]:


element_text = driver.find_element(By.XPATH, "//*[@id='left-column']/div[1]/h1")
text = element_text.text

first_part = text.split(' ', 1)[0]

number_str_no_commas = first_part.replace(",", "")

job_count = 0

# Convert the string to an integer
if number_str_no_commas:
    job_count = int(number_str_no_commas)

print(job_count)


# In[23]:


url = []


# In[24]:


clicknext(driver,job_count)
time.sleep(3)


# In[25]:


with open('url_software_engineer_loc_US.json','w') as f:
    json.dump(geturl(driver,url),f, indent = 4)
    print("file created")


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


with open('/Users/saranyagondeli/url_data_engineer_loc_US.json','r') as f:
    url = json.load(f)



data ={}    
i = 1
jd_df = pd.DataFrame()


# In[14]:


pip install selenium_stealth


# In[15]:


pip install fake_useragent


# In[22]:


from fake_useragent import UserAgent


# In[23]:


ua = UserAgent()


# In[ ]:





# In[ ]:





# In[31]:


#driver = webdriver.Chrome()


# In[24]:


for u in tqdm(url):
    try:
    
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": ua.random})

        driver.wait = WebDriverWait(driver, 2)
        driver.maximize_window()
        driver.get(u)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        position = None
        company = None
        location = None
        jd = None

        try:
            driver.find_element(By.XPATH, '//*[@id="challenge-stage"]/div/label').click()
            print("check")
            time.sleep(5)
        except NoSuchElementException:
            pass

        try:

            header = soup.find("div",{"id":"PageContent"})
            position = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[2]').text
            company = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[1]/div').text
            location = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[3]/span').text
            jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
            jd = jd_temp.text
            #info = soup.find_all("infoEntity")
        except IndexError:
            print('IndexError: list index out of range')
        except NoSuchElementException:
            pass
    except WebDriverException as e:
        print("Error occurred:", e)
        driver.quit()  # Close the current driver
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": ua.random})
        driver.wait = WebDriverWait(driver, 2)
        driver.maximize_window()
        driver.get(u)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        header = soup.find("div",{"id":"PageContent"})
        position = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[2]').text
        company = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[1]/div').text
        location = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[3]/span').text
        jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
        jd = jd_temp.text
        
    data[i] = {
        'url' :u,
        'Position':position,
        'Company': company,
        'Location' :location,
        'Job_Description' :jd
    }
    i+=1   


# In[ ]:


with open('/Users/saranyagondeli/url_data_analyst_loc_US.json','r') as f:
    url_da = json.load(f)


# In[ ]:


driver = webdriver.Chrome()
for u in tqdm(url_da):
    try:
    
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": ua.random})

        driver.wait = WebDriverWait(driver, 2)
        driver.maximize_window()
        driver.get(u)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        position = None
        company = None
        location = None
        jd = None

        try:
            driver.find_element(By.XPATH, '//*[@id="challenge-stage"]/div/label').click()
            print("check")
            time.sleep(5)
        except NoSuchElementException:
            pass

        try:

            header = soup.find("div",{"id":"PageContent"})
            position = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[2]').text
            company = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[1]/div').text
            location = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[3]/span').text
            jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
            jd = jd_temp.text
            #info = soup.find_all("infoEntity")
        except IndexError:
            print('IndexError: list index out of range')
        except NoSuchElementException:
            pass
    except WebDriverException as e:
        print("Error occurred:", e)
        driver.quit()  # Close the current driver
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": ua.random})
        driver.wait = WebDriverWait(driver, 2)
        driver.maximize_window()
        driver.get(u)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        header = soup.find("div",{"id":"PageContent"})
        position = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[2]').text
        company = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[1]/div').text
        location = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[3]/span').text
        jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
        jd = jd_temp.text
        
    data[i] = {
        'url' :u,
        'Position':position,
        'Company': company,
        'Location' :location,
        'Job_Description' :jd
    }
    i+=1   


# In[ ]:


with open('/Users/saranyagondeli/url_software_engineer_loc_US.json','r') as f:
    url_se = json.load(f)


# In[ ]:


driver = webdriver.Chrome()
for u in tqdm(url_se):
    try:
    
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": ua.random})

        driver.wait = WebDriverWait(driver, 2)
        driver.maximize_window()
        driver.get(u)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        position = None
        company = None
        location = None
        jd = None

        try:
            driver.find_element(By.XPATH, '//*[@id="challenge-stage"]/div/label').click()
            print("check")
            time.sleep(5)
        except NoSuchElementException:
            pass

        try:

            header = soup.find("div",{"id":"PageContent"})
            position = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[2]').text
            company = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[1]/div').text
            location = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[3]/span').text
            jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
            jd = jd_temp.text
            #info = soup.find_all("infoEntity")
        except IndexError:
            print('IndexError: list index out of range')
        except NoSuchElementException:
            pass
    except WebDriverException as e:
        print("Error occurred:", e)
        driver.quit()  # Close the current driver
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": ua.random})
        driver.wait = WebDriverWait(driver, 2)
        driver.maximize_window()
        driver.get(u)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        header = soup.find("div",{"id":"PageContent"})
        position = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[2]').text
        company = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[1]/div').text
        location = driver.find_element(By.XPATH, '//*[@id="PageContent"]/div[1]/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/div/div/div[3]/span').text
        jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
        jd = jd_temp.text
        
    data[i] = {
        'url' :u,
        'Position':position,
        'Company': company,
        'Location' :location,
        'Job_Description' :jd
    }
    i+=1   


# In[ ]:


driver.quit()
jd_df = pd.DataFrame(data)
jd = jd_df.transpose()

jd = jd[['url','Position','Company','Location','Job_Description']]
jd.to_csv(r'/Users/saranyagondeli/Downloads/jd_unstructured_data.csv')
print('file created')


# In[ ]:


pip install nltk


# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stopw  = set(stopwords.words('english'))


# In[ ]:


unstructured_df=pd.read_csv('/Users/saranyagondeli/Downloads/jd_unstructured_data.csv')


# In[ ]:


unstructured_df.info()


# In[ ]:


unstructured_df['Company_Name'] = unstructured_df['Company'].str.split('\n',1).str[0]

unstructured_df['Rating'] = unstructured_df['Company'].str.split('\n').str[1]
# Now, you may want to convert the Rating to a numeric type
unstructured_df['Rating'] = pd.to_numeric(unstructured_df['Rating'])

# The original 'company_info' column can be dropped if no longer needed
unstructured_df = unstructured_df.drop('Company', axis=1)


# In[ ]:


unstructured_df.head()


# In[ ]:


unstructured_df['Processed_JD']=unstructured_df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))


# In[ ]:


unstructured_df.head()


# In[ ]:


unstructured_df=unstructured_df.drop(['Unnamed: 0','Job_Description'],axis=1)


# In[ ]:


unstructured_df.head()


# In[ ]:


unstructured_df.to_csv(r'/Users/saranyagondeli/Downloads/jd_structured_data.csv', index=False)


# In[ ]:




