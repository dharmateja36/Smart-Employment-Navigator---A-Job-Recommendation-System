#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import itertools
import collections
import pandas as pd
import matplotlib.pyplot as plt
import fitz
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os


# Function to filter words based on TF-IDF vocabulary
def filter_words(description, tfidf_vocab):
    return ' '.join([word for word in description.split() if word in tfidf_vocab])

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

    pd.set_option('display.max_colwidth', 50)

    df = pd.read_csv('/Users/saranyagondeli/Downloads/combined_dataset.csv')

    null_counts = df.isnull().sum()

    # Drop rows where 'Job_description', 'Job_Title', or 'Job_Detail_Link' is null
    df = df.dropna(subset=['Job_description', 'Job_Title', 'Job_Detail_Link'])
    null_counts = df.isnull().sum()

    # Group by job description and filter out groups with more than one entry
    duplicate_descriptions = df[df.duplicated('Job_description', keep=False)]

    # Group these duplicates by the job description itself and list their indices
    grouped_duplicates = duplicate_descriptions.groupby('Job_description').apply(lambda x: list(x.index))


    # Remove duplicates by keeping only the first occurrence of each job description
    df = df.drop_duplicates(subset='Job_description', keep='first')

    # Apply the function to the 'Search_Keyword' column to create a new 'job_scope' column
    df['job_scope'] = df['Search_Keyword'].apply(categorize_job_scope)

    df['Job_description'] = df['Job_description'].str.lower()

    df['Job_description'] = df['Job_description'].str.replace(r"http\S+","",regex = True)

    df['Job_description'] = df['Job_description'].str.replace('[^\w\s]', '', regex=True)

    nltk.download('punkt')
    df['Job_description'] = df['Job_description'].apply(nltk.word_tokenize)

    nltk.download('stopwords')
    stop = stopwords.words('english')
    df['Job_description'] = df['Job_description'].apply(lambda x: [item for item in x if item not in stop])

    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    df['Job_description'] = df['Job_description'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    df['word_length'] = df['Job_description'].apply(len)

    # Assuming df['Job_description'] contains lists of lemmatized tokens
    lemmatized_tokens = list(df['Job_description'])
    token_list = list(itertools.chain(*lemmatized_tokens))
    counts_no = collections.Counter(token_list)

    # Create a DataFrame with the most common words
    clean_tweets = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count'])

    df1 = df.copy()

    # df1['Job_description'] contains lists of preprocessed tokens, convert them back to strings
    df1['Job_description_str'] = df1['Job_description'].apply(' '.join)
    initial_vocab = set()
    df1['Job_description_str'].str.split().apply(initial_vocab.update)

    # Now use TfidfVectorizer on the joined strings
    vectorizer = TfidfVectorizer(min_df=0.05)  # Adjust min_df to suit your dataset
    tfidf_matrix = vectorizer.fit_transform(df1['Job_description_str'])

    # Count the number of words in each original job description
    df1['word_length'] = df1['Job_description_str'].str.split().str.len()

    # Get the number of words that survived the TF-IDF vectorization (features)
    num_features = len(vectorizer.get_feature_names_out())
    tfidf_vocab = set(vectorizer.get_feature_names_out())

    removed_words = initial_vocab - tfidf_vocab

    # Apply the function to each job description
    df1['job_description_idf'] = df1['Job_description_str'].apply(lambda x: filter_words(x, tfidf_vocab))

    df1['job_description_idf'] = df1['job_description_idf'].apply(nltk.word_tokenize)


    pd.set_option('display.max_colwidth', None)

    pd.set_option('display.max_colwidth', 50)

    return df1

def extract_content_with_strict_check(pdf_path, start_section_keywords, end_section_keywords):
    doc = fitz.open(pdf_path)
    full_text = ""
    extracting = False

    for page in doc:
        text = page.get_text()
        lines = text.split('\n')

        for line in lines:
            # Normalize the line for comparison
            normalized_line = line.strip().lower()

            # Check if the line matches any of the start section keywords as standalone headings
            if any(normalized_line == keyword.lower() for keyword in start_section_keywords):
                extracting = True
                continue

            # Check if the line matches any of the end section keywords as standalone headings
            if extracting and any(normalized_line == keyword.lower() for keyword in end_section_keywords):
                extracting = False
                break

            # Extract text if we are in the right section
            if extracting:
                full_text += line + '\n'

    return full_text.strip()

def count_tokens(tokenizer,text):
    return len(tokenizer.encode(text, add_special_tokens=True))

def encode_text(text, tokenizer, model):
    # Truncate and encode the text
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)

    # Calculate the mean across the first axis (tokens) and keep the result as 2D tensor
    return torch.mean(outputs[0], dim=1)

def calculate_similarity(job_embedding, resume_embedding):
    job_embedding_reshaped = job_embedding.reshape(1, -1)
    resume_embedding_reshaped = resume_embedding.reshape(1, -1)
    return cosine_similarity(job_embedding_reshaped, resume_embedding_reshaped)[0][0]

# Get the top 5 recommendations
def get_top_5_recommendations(df1, resume_embedding):
    df1['similarity'] = df1['job_description_embedding'].apply(lambda x: calculate_similarity(x, resume_embedding))
    top_5_jobs = df1.sort_values(by='similarity', ascending=False).head(5)
    # Returning a list of dictionaries for each job
    return top_5_jobs[['Job_Title', 'Job_Detail_Link', 'Location', 'job_scope']].to_dict('records')


def job_recommend(file_path):

    df1 = preprocessing()

    # Keywords to identify the start and end of the experience section
    start_section_keywords = ["Professional Experience", "Experience", "Work Experience"]
    end_section_keywords = ["Projects", "Certifications", "Skills", "Education"]

    path=r'/Users/saranyagondeli/Downloads'
    full_file_path = os.path.join(path, file_path)
    # Extract the content with strict check for section headings
    pdf_path = full_file_path
    exp_extracted_content = extract_content_with_strict_check(pdf_path, start_section_keywords, end_section_keywords)
    # Create a DataFrame with the extracted content
    df_resume = pd.DataFrame({'Experience': [exp_extracted_content]})

    # Keywords to identify the start and end of the skills section
    start_section_keywords_skills = ["Skills", "Professional Skills", "Technical Skills", "Programming Skills"]
    end_section_keywords_skills = ["Projects", "Certifications", "Skills", "Education","Experience","Professional Experience","Work Experience","About","Career Objective"]

    # Extract the content with strict check for section headings
    skills_extracted_content = extract_content_with_strict_check(pdf_path, start_section_keywords_skills, end_section_keywords_skills)

    df_resume['Experience'] = df_resume['Experience'] + " " + skills_extracted_content

    df_resume['Experience'] = df_resume['Experience'].str.lower()

    df_resume['Experience'] = df_resume['Experience'].str.replace(r"http\S+","",regex = True)

    df_resume['Experience'] = df_resume['Experience'].str.replace('[^\w\s]', '', regex=True)

    # Create a regular expression pattern to match years from 1970 to 2023
    year_pattern = r'\b(19[7-9]\d|20[0-2]\d|2023)\b'

    # Remove years in the specified range
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: re.sub(year_pattern, '', x))

    # List of month names
    months = ["january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december","jan","feb","mar","apr","may","jun",
            "jul","aug","sep","oct","nov","dec"]

    # Example list of common cities and countries (extend this list as needed)
    cities_countries = ["new york","hyderabad", "india", "usa", "canada"]

    # Combine lists and create a regex pattern (adjust as needed)
    names_to_remove = months + cities_countries
    pattern = r'\b(?:' + '|'.join(names_to_remove) + r')\b'

    # Remove these names from the 'Experience' column
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: re.sub(pattern, '', x, flags=re.IGNORECASE))

    # Replace newline characters with a space
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: re.sub(r'\n', ' ', x))

    # Replace multiple spaces with a single space
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: re.sub(r'\s+', ' ', x))

    nltk.download('punkt')
    df_resume['Experience'] = df_resume['Experience'].apply(nltk.word_tokenize)

    nltk.download('stopwords')
    stop = stopwords.words('english')
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: [item for item in x if item not in stop])

    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    df2 = df_resume.copy()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to count the number of tokens

    # Apply the function to your dataframe and create a new column with token counts
    df2['token_count'] = df2['Experience'].apply(lambda x: count_tokens(tokenizer, x))

    # Filter to get descriptions with more than 512 tokens
    long_descriptions = df2[df2['token_count'] > 512]

    # Load the embeddings
    embeddings = np.load('/Users/saranyagondeli/Downloads/job_description_embeddings.npy')

    # Reshape the embeddings to remove the middle dimension
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])

    # Ensure that the number of embeddings matches the number of rows in df1
    if len(embeddings) != len(df1):
        print('The number of embeddings does not match the number of rows in the dataframe.')
        raise ValueError("The number of embeddings does not match the number of rows in the dataframe.")

    # Add embeddings to the dataframe
    df1['job_description_embedding'] = list(embeddings)
    df2['Experience_embedding'] = df2['Experience'].apply(lambda x: encode_text(x, tokenizer, model).numpy())

    # Iterate over each resume and get top 5 recommendations
    all_recommendations = {}
    for index, row in df2.iterrows():
        resume_embedding = row['Experience_embedding']
        top_5_recommendations = get_top_5_recommendations(df1, resume_embedding)
        all_recommendations[index] = top_5_recommendations

    return all_recommendations

    # # Print the recommendations for each resume in the desired format
    # for resume_idx, recommendations in all_recommendations.items():
    #     print(f"Recommendations for Resume Index {resume_idx}:")

    #     for i, job in enumerate(recommendations):
    #         print(f"  Recommendation {i + 1}:")
    #         print(f"    Job Title: {job['Job_Title']}")
    #         print(f"    Job Link: {job['Job_Detail_Link']}")
    #         print(f"    Job Description: {job['Job_description_str']}")
    #         print()

    #     print("--------------------------------------------------\n")

def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    

    if uploaded_file is not None:
        # Process resume and recommend jobs
        file_path=uploaded_file.name
        print(file_path)
        top_n_recommendations = job_recommend(file_path)
        top_n_recommendations = pd.DataFrame(top_n_recommendations)

        # Display recommended jobs as DataFrame
        st.write("Recommended Jobs:")
        all_recommendations = []
        for resume_idx, recommendations in top_n_recommendations.items():
            for rec in recommendations:
                rec['Resume_Index'] = resume_idx  # Optional: add resume index
                all_recommendations.append(rec)

        combined_df = pd.DataFrame(all_recommendations)
        combined_df.rename(columns={'Location': 'Company_&_Location'}, inplace=True)
        combined_df = combined_df.drop('Resume_Index', axis=1)
        st.dataframe(combined_df)

# Run the Streamlit app
if __name__ == '__main__':
    main()