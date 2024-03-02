# Smart Employment Navigator - A Job Recommendation System


This project is designed to recommend jobs to candidates based on their resumes using different modeling approaches: KNN, Word2Vec, and BERT.


## Directory Structure


- `Bert_modeling.py`: This script contains the BERT model's implementation for generating job recommendations.
- `Word2Vec_modeling.py`: This script applies the Word2Vec model to generate job recommendations.
- `KNN_modeling.py`: This script utilizes the K-nearest neighbors algorithm for job recommendations.
- `Bert_accuracy.py`: Computes the accuracy and MAP@K for the BERT model's recommendations.
- `Word2Vec_accuracy.py`: Computes the accuracy and MAP@K for the Word2Vec model's recommendations.
- `KNN_accuracy.py`: Computes the accuracy and MAP@K for the KNN model's recommendations.
- `main.py`: The main script that interfaces with Streamlit to upload resumes and display job recommendations from all three models.
- `DM_PROJECT_DATA_EXTRACTION.py`: A data extraction utility used in the project.
- `job_description_embeddings.npy`: Precomputed job description embeddings used in BERT model recommendations.
- `word2vec_job_descriptions.model`: A pre-trained Word2Vec model file for job descriptions.
- `word2vec_sg_job_descriptions.model`: A pre-trained Word2Vec model file using the Skip-gram architecture.


## How to Run


1. Ensure that you have Streamlit installed in your Python environment. If not, install it using `pip install streamlit`.
2. Ensure to download the .npy data and model files from the given drive links and save them in an accessible directory.
3. Place the resume PDFs you wish to process in an accessible directory.
4. Run the `main.py` script using Streamlit by executing the command `streamlit run main.py`.
5. Use the Streamlit interface to upload a resume PDF file.
6. View the job recommendations from each of the three models displayed on the Streamlit app.


## Recommendations Output


- The output will be presented in a tabular format where each table corresponds to a different model.
- Each table will display the top job recommendations along with the job titles, detailed links, and company locations.
- The job detail links are clickable and will open the job posting in a new browser tab.


## Note


- Make sure that all the necessary Python packages are installed (`pandas`, `numpy`, `nltk`, `gensim`, `transformers`, etc.).
- The paths to the folders containing the resumes should be updated according to your local environment setup.
- The models rely on precomputed data and model files, ensuring that these files are present in the correct directory as per the project structure.


Thank you for using our Job Recommendation System!


.model and .npy files link : 
https://drive.google.com/drive/folders/1mUFJx_vFCO-zBv4l5qwyDsEk1Ojm_vmw?usp=share_link


Dataset link:
Combined jobs: https://drive.google.com/file/d/1UnQzZZ6kXZgOj6B0tzpuSbXuKEd75jX_/view?usp=share_link
Skills.csv: https://drive.google.com/file/d/1k_lKCYEdXAg8wEF-OIQ4ixHfyprkaFmn/view?usp=share_link
Resumes:
https://drive.google.com/drive/folders/1F-RcsK_Eq9HLFU7vzipkLWzB93OdRNXm?usp=share_link
