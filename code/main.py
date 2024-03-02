import streamlit as st
import os
import Bert_modeling as bert
import Word2Vec_modeling as w2v
import KNN_modeling as knn
import pandas as pd

def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    

    if uploaded_file is not None:
        # Process resume and recommend jobs
        file_path=uploaded_file.name
        print(file_path)

        #KNN
        df_jobs = knn.process_resume(file_path)
        #print(df_jobs[['Skills']])

        # Display recommended jobs as DataFrame
        st.write("KNN Recommendations:")
        df_jobs['Job_Detail_Link'] = df_jobs['Job_Detail_Link'].astype(str)
        #df_jobs['Job_Detail_Link'] = df_jobs['Job_Detail_Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        df_jobs.rename(columns={'Location': 'Company_&_Location'}, inplace=True)
        st.dataframe(df_jobs[['Job_Title','Job_Detail_Link','Company_&_Location']].head())
        #html = df_jobs.to_html(escape=False, index=False)
        #st.markdown(html, unsafe_allow_html=True)

        #Word2Vec
        st.write("Word2Vec Recommendations:")
        top_n_recommendations = w2v.job_recommend(file_path)
        top_n_recommendations = pd.DataFrame(top_n_recommendations)
        all_recommendations = []
        for resume_idx, recommendations in top_n_recommendations.items():
            for rec in recommendations:
                rec['Resume_Index'] = resume_idx  # Optional: add resume index
                all_recommendations.append(rec)

        combined_df = pd.DataFrame(all_recommendations)
        combined_df.rename(columns={'Location': 'Company_&_Location'}, inplace=True)
        combined_df = combined_df.drop('Resume_Index', axis=1)
        combined_df['Job_Detail_Link'] = combined_df['Job_Detail_Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        #st.dataframe(combined_df)
        html = combined_df.to_html(escape=False, index=False)
        st.markdown(html, unsafe_allow_html=True)

        #Bert
        st.write("BERT Recommendations:")
        top_n_recommendations = bert.job_recommend(file_path)
        top_n_recommendations = pd.DataFrame(top_n_recommendations)
        all_recommendations = []
        for resume_idx, recommendations in top_n_recommendations.items():
            for rec in recommendations:
                rec['Resume_Index'] = resume_idx  # Optional: add resume index
                all_recommendations.append(rec)

        combined_df = pd.DataFrame(all_recommendations)
        combined_df.rename(columns={'Location': 'Company_&_Location'}, inplace=True)
        combined_df = combined_df.drop('Resume_Index', axis=1)
        combined_df['Job_Detail_Link'] = combined_df['Job_Detail_Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        #st.dataframe(combined_df)

        html = combined_df.to_html(escape=False, index=False)
        st.markdown(html, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()