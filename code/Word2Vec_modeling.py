#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import itertools
import collections
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import fitz
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st
import os

# Function to filter words based on TF-IDF vocabulary
def filter_words(description, tfidf_vocab):
    return ' '.join([word for word in description.split() if word in tfidf_vocab])

def vectorize_description(description, model_w2v_sg):
    # Convert each word in the description to a vector and then average them
    vectors = [model_w2v_sg.wv[word] for word in description if word in model_w2v_sg.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)

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
    df = pd.read_csv('/Users/saranyagondeli/Downloads/combined_dataset.csv')

    null_counts = df.isnull().sum()

    # Drop rows where 'Job_description', 'Job_Title', or 'Job_Detail_Link' is null
    df = df.dropna(subset=['Job_description', 'Job_Title', 'Job_Detail_Link'])

    null_counts = df.isnull().sum()

    # Group by job description and filter out groups with more than one entry
    duplicate_descriptions = df[df.duplicated('Job_description', keep=False)]

    # Group these duplicates by the job description itself and list their indices
    grouped_duplicates = duplicate_descriptions.groupby('Job_description').apply(lambda x: list(x.index))

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

    lemmatizer = WordNetLemmatizer()
    df['Job_description'] = df['Job_description'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    df['word_length'] = df['Job_description'].apply(len)

    # Assuming df['Job_description'] contains lists of lemmatized tokens
    lemmatized_tokens = list(df['Job_description'])
    token_list = list(itertools.chain(*lemmatized_tokens))
    counts_no = collections.Counter(token_list)

    # Create a DataFrame with the most common words
    clean_tweets = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count'])


    clean_tweets.head(10)

    df1 = df.copy()

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

    #Train a Word2Vec model using cbow
    model_cbow = Word2Vec(sentences=df1['job_description_idf'], vector_size=100, window=5, min_count=1)

    # Load the model
    model_cbow = Word2Vec.load("/Users/saranyagondeli/Downloads/word2vec_job_descriptions.model")

    # Train a Word2Vec model using cbow
    model_w2v_sg = Word2Vec(sentences=df1['job_description_idf'], vector_size=100, window=5, min_count=1,sg=1)

    # Load the model
    model_w2v_sg = Word2Vec.load("/Users/saranyagondeli/Downloads/word2vec_sg_job_descriptions.model")

    # Vectorize each job description
    df1['job_description_vector'] = df1['job_description_idf'].apply(lambda x: vectorize_description(x, model_w2v_sg))

    # Option 2: Save as Pickle (preserves numpy array format)
    df1.to_pickle("/Users/saranyagondeli/Downloads/job_descriptions_with_vectors.pkl")


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

# Function to vectorize the tokenized resume data
def vectorize_resume(resume_tokens, model_w2v_sg):
    # Convert each word in the resume to a vector and then average them
    vectors = [model_w2v_sg.wv[word] for word in resume_tokens if word in model_w2v_sg.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)  # Return a zero vector if the resume has no words in the model


def job_recommend(file_path):

    df1 = preprocessing()
    
    # Keywords to identify the start and end of the experience section
    start_section_keywords = ["Professional Experience", "Experience", "Work Experience"]
    end_section_keywords = ["Projects", "Certifications", "Skills", "Education"]
    
    path=r'/Users/saranyagondeli/Downloads'
    full_file_path = os.path.join(path, file_path)
    pdf_path = full_file_path
    exp_extracted_content = extract_content_with_strict_check(pdf_path, start_section_keywords, end_section_keywords)
    # Create a DataFrame with the extracted content
    df_resume = pd.DataFrame({'Experience': [exp_extracted_content]})
    
    # Keywords to identify the start and end of the skills section
    start_section_keywords_skills = ["Skills", "Professional Skills", "Technical Skills"]
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
    
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    df_resume['Experience'] = df_resume['Experience'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    
    df2 = df_resume.copy()
    
    # Load your Word2Vec model
    model_w2v_sg = Word2Vec.load("/Users/saranyagondeli/Downloads/word2vec_sg_job_descriptions.model")
    
    df2['resume_vector'] = df2['Experience'].apply(lambda x: vectorize_resume(x, model_w2v_sg))
    
    df2.to_pickle("/Users/saranyagondeli/Downloads/resume_with_vectors.pkl")
    
    # Ensure the vectors are in the correct format (numpy arrays)
    df1['job_description_vector'] = df1['job_description_vector'].apply(np.array)
    df2['resume_vector'] = df2['resume_vector'].apply(np.array)

    # Create a cosine similarity matrix
    similarity_matrix = cosine_similarity(list(df2['resume_vector']), list(df1['job_description_vector']))

    # For each resume, find the top N job descriptions with the highest cosine similarity
    N = 5  # Number of top recommendations to extract
    top_n_recommendations = {}

    for idx, row in enumerate(similarity_matrix):
        # Get the indices of the top N similarities
        top_indices = row.argsort()[-N:][::-1]

        # Get the corresponding job details from df1
        top_jobs = df1.iloc[top_indices][['Job_Title', 'Job_Detail_Link', 'Location', 'job_scope']]

        # Store the recommendations for this resume along with the resume index
        top_n_recommendations[idx] = top_jobs.reset_index(drop=True).to_dict(orient='records')

    # top_n_recommendations now contains the top N job recommendations for each resume along with the resume index
    
    return top_n_recommendations

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


