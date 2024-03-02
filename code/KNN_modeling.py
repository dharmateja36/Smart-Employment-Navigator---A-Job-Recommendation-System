#!/usr/bin/env python
# coding: utf-8
import spacy
from spacy.matcher import Matcher
import PyPDF2
import os
import re
from word2number import w2n
import datetime
from datetime import datetime
import streamlit as st
import pandas as pd
from pyresparser import ResumeParser
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from ftfy import fix_text
import numpy as np
import nltk
from nltk.corpus import stopwords
from docx import Document

# Define a function to categorize the job scope
def categorize_job_scope(keyword):
    if keyword in ['data analyst', 'business analyst']:
        return 'Data Analyst'
    elif keyword in ['data engineer', 'big data engineer', 'business intelligence']:
        return 'Data Engineer'
    elif keyword in ['data scientist', 'machine learning engineer']:
        return 'Data Scientist'
    elif keyword in ['cloud engineer', 'cloud architect', 'python developer', 'database engineer', 'sql developer']:
        return 'Software Engineer'
    else:
        return 'Other'

def preprocessing():
    # Load the Spacy English model
    nlp = spacy.load('en_core_web_sm')

    # Read skills from CSV file
    file_path=r'/Users/saranyagondeli/Downloads/skills.csv'

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        skills = [row for row in csv_reader]

    # Create pattern dictionaries from skills
    skill_patterns = [[{'LOWER': skill}] for skill in skills[0]]

    # Create a Matcher object
    matcher = Matcher(nlp.vocab)

    # Add skill patterns to the matcher
    for pattern in skill_patterns:
        matcher.add('Skills', [pattern])

    stopw  = set(stopwords.words('english'))

    jd_df=pd.read_csv(r'/Users/saranyagondeli/Downloads/combined_dataset.csv')

    jd_df.info()

    # Group by job description and filter out groups with more than one entry
    duplicate_descriptions = jd_df[jd_df.duplicated('Job_description', keep=False)]

    # Group these duplicates by the job description itself and list their indices
    grouped_duplicates = duplicate_descriptions.groupby('Job_description').apply(lambda x: list(x.index))


    # Remove duplicates by keeping only the first occurrence of each job description
    jd_df = jd_df.drop_duplicates(subset='Job_description', keep='first')

    # duplicates = jd_df.duplicated(subset=['Search_Keyword', 'Search_Location','Job_Title', 'Company_Name','Location', 'Salary','Job_rating', 'Job_review_count','Job_Type','Job_description','Email','Job_Posted','Job_valid_through'])

    # # To check how many duplicates there are
    # number_of_duplicates = duplicates.sum()
    
    # jd_df = jd_df.drop_duplicates(subset=['Search_Keyword', 'Search_Location','Job_Title', 'Company_Name','Location', 'Salary','Job_rating', 'Job_review_count','Job_Type','Job_description','Email','Job_Posted','Job_valid_through'])

    # Apply the function to the 'Search_Keyword' column to create a new 'job_scope' column
    jd_df['job_scope'] = jd_df['Search_Keyword'].apply(categorize_job_scope)

    return nlp, matcher, jd_df

# Function to extract skills from text
def extract_skills(nlp, matcher, text):
    doc = nlp(text)
    #print(doc)
    matches = matcher(doc)
    #print(matches)
    skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text
        #print(skill)
        skills.add(skill)
    return skills

# Function to extract text from PDF
def extract_text_from_pdf(file_path:str):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def find_experience_section(text, possible_titles):
    for title in possible_titles:
        start = text.lower().find(title.lower())
        #print(start)
        if start != -1:
            return text[start:start+10000]  # Adjust length based on expected section size
    return ''

def parse_date(date_str):
    for fmt in ("%b %Y", "%m/%Y", "%d/%m/%Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None  # Return None if no format matches

def calculate_experience(experience_section):
    # Updated regex to capture a wide range of date formats
    date_pattern = r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}/\d{4}|\d{2})[ \/-](\d{4})?) - (\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}/\d{4}|\d{2})[ \/-](\d{4})?|Present|Current)'
    date_ranges = re.findall(date_pattern, experience_section, re.IGNORECASE)
    #print(date_ranges)

    total_duration = 0
    for start_date_str, start_year, end_date_str, end_year in date_ranges:
        start_date = parse_date(start_date_str)
        end_date = datetime.now() if end_date_str in ['Present', 'Current'] else parse_date(end_date_str)
        #print(start_date)
        #print(end_date)
        if start_date and end_date:
            duration = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
            total_duration += duration

    years = total_duration // 12
    months = total_duration % 12
    return years, months

def skills_extractor(nlp, matcher, file_path):
        # Extract text from PDF
        path=r'/Users/saranyagondeli/Downloads'
        full_file_path = os.path.join(path, file_path)
        resume_text = extract_text_from_pdf(full_file_path)
        #print(resume_text)
        possible_titles = ['Experience', 'Work History', 'Professional Background', 'Career Summary', 'Employment History']
        experience_section = find_experience_section(resume_text, possible_titles)

        years, months = calculate_experience(experience_section)
        print(f"Total Experience: {years} years and {months} months")
        
        # Extract skills from resume text
        skills = list(extract_skills(nlp, matcher, resume_text))
        return skills

def extract_experience(description):
    # Regex pattern to match numeric and certain word representations of years
    pattern = r'(\d+(-\d+)?\s*(years|year|yrs|yr))|(\d+\+?\s*(years|year|yrs|yr))|(at least\s*(\w+)\s*(years|year|yrs|yr))'
    match = re.search(pattern, description, re.IGNORECASE)

    if match:
        # Extract the matched string
        matched_string = match.group()

        # Convert textual number to numeric if necessary
        if re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', matched_string, re.IGNORECASE):
            textual_number = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', matched_string, re.IGNORECASE).group()
            numeric_value = w2n.word_to_num(textual_number)
            return f'{numeric_value} years'
        else:
            return matched_string

    return 'Not specified'

def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def getNearestN(query,vectorizer,nbrs):
    queryTFIDF_ = vectorizer.transform(query)
    #print(queryTFIDF_)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices



def process_resume(file_path):
    
    nlp, matcher, jd_df = preprocessing()

    # Extract text from PDF resume
    resume_skills= skills_extractor(nlp, matcher,file_path)

    # Perform job recommendation based on parsed resume data
    skills=[]
    skills.append(' '.join(word for word in resume_skills))
    print(skills)
    
    
    # Feature Engineering:
    #vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False, stop_words='english', max_features=5000)
    vectorizer = TfidfVectorizer(min_df=1, lowercase=False, stop_words='english', max_features=10000)
    #tfidf = vectorizer.fit_transform(skills)

    #jd_df['Processed_JD'] = jd_df['Processed_JD'].fillna('')
    jd_test = (jd_df['Job_description'].values.astype('U'))
    tfidf = vectorizer.fit_transform(jd_test)

    #tfidf = vectorizer.fit_transform(jd_df['Skills'])
    
    
    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(tfidf)
    
    #print(jd_test)

    #distances, indices = getNearestN(jd_test,vectorizer,nbrs)
    
    distances, indices = getNearestN(skills,vectorizer,nbrs)
    
    # test = list(jd_test) 
    matches = []

    # for i,j in enumerate(indices):
    #     dist=round(distances[i][0],2)
    #     temp = [dist]
    #     matches.append(temp)
    
    # matches = pd.DataFrame(matches, columns=['Match confidence'])

    # # Following recommends Top 5 Jobs based on candidate resume:
    # jd_df['match']=matches['Match confidence']

    for i, index in enumerate(indices):
        temp = []
        for j, idx in enumerate(index):
            dist = round(distances[i][j], 2)  # Use `j` instead of `idx` for indexing
            job_desc = jd_df.iloc[idx]['Job_description']  # `idx` is correct here
            temp.append((idx, dist))
        matches.append(temp)

    top_job_descriptions = [] 
    for match in matches:
        for idx, dist in match[:10]:  # Assuming you want the top 5 matches
            row = jd_df.iloc[idx]  # Get the row from the original dataframe
            #row['Match Confidence'] = dist  # Optionally, add the match confidence to the row
            top_job_descriptions.append(row)

    top_jobs_df = pd.DataFrame(top_job_descriptions)  

    #matches_df = pd.DataFrame(matches, columns=[f'Job {i+1}' for i in range(len(matches[0]))])
    
    # duplicates = jd_df.duplicated()

    # # To check how many duplicates there are
    # number_of_duplicates = duplicates.sum()
    
    top_jobs_df = top_jobs_df.drop_duplicates()

    # jd_df = jd_df.sort_values('match',ascending=False)
    
    #return jd_df
    return top_jobs_df

def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    

    if uploaded_file is not None:
        # Process resume and recommend jobs
        file_path=uploaded_file.name
        print(file_path)
        df_jobs = process_resume(file_path)
        #print(df_jobs[['Skills']])

        # Display recommended jobs as DataFrame
        st.write("Recommended Jobs:")
        #st.dataframe(df_jobs[['Job_Title','Job_Detail_Link','Location']].head())
        st.dataframe(df_jobs)

# Run the Streamlit app
if __name__ == '__main__':
    main()





