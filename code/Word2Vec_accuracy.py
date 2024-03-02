import os
import numpy as np
import Word2Vec_modeling as w2v
import pandas as pd

# Function to calculate accuracy
def calculate_accuracy(recommended_jobs, resume_folder):
    if recommended_jobs:  # Check if the dictionary is not empty
        total_jobs = 0
        relevant_jobs_count = 0

        for job_list in recommended_jobs.values():  # Iterate through lists in the values
            for job in job_list:
                if isinstance(job, dict) and job.get('job_scope') == resume_folder:
                    relevant_jobs_count += 1
                total_jobs += 1

        accuracy = (relevant_jobs_count / total_jobs) * 100 if total_jobs > 0 else 0
    else:
        accuracy = 0  # Set accuracy to 0 if there are no recommended jobs

    return accuracy

def calculate_precision_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]

    if actual in predicted:
        return 1 / (predicted.index(actual) + 1)
    return 0

def calculate_mapk(actuals, predicted_lists, k):
    return np.mean([calculate_precision_at_k(actual, predicted, k) for actual, predicted in zip(actuals, predicted_lists)])

# Paths to the folders
folders = {
    'Data Scientist': '/Users/saranyagondeli/Downloads/resumes/Data Scientist',
    'Software Engineer': '/Users/saranyagondeli/Downloads/resumes/Software Engineer',
    'Data Analyst': '/Users/saranyagondeli/Downloads/resumes/Data Analyst',
    'Data Engineer': '/Users/saranyagondeli/Downloads/resumes/Data Engineer'
}

accuracies = []
mapk_scores = []

for job_scope, folder_path in folders.items():
    actuals = []
    predicted_lists = []

    for resume_file in os.listdir(folder_path):
        if resume_file == '.DS_Store':
            continue
        else:
            resume_path = os.path.join(folder_path, resume_file)
            recommended_jobs = w2v.job_recommend(resume_path)
            accuracy = calculate_accuracy(recommended_jobs, job_scope)
            accuracies.append(accuracy)

            # For MAP@K calculation
            actuals.append(job_scope)
            job_scopes = [job['job_scope'] for job_list in recommended_jobs.values() for job in job_list]
            predicted_lists.append(job_scopes)

    # Calculate MAP@K for each job category
    k = 5  # Adjust K as required
    mapk_score = calculate_mapk([job_scope] * len(predicted_lists), predicted_lists, k)
    mapk_scores.append(mapk_score)
    print(f"MAP@{k} for {job_scope}: {mapk_score}%")

# Calculate overall accuracy and MAP@K
overall_accuracy = np.mean(accuracies)
overall_mapk = np.mean(mapk_scores)

print(f"Overall System Accuracy: {overall_accuracy}%")
print(f"Overall System MAP@{k}: {overall_mapk}%")
