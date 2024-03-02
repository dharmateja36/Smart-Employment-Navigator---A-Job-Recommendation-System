import os
import numpy as np
import KNN_modeling as knn
import pandas as pd

# Function to calculate accuracy
def calculate_accuracy(recommended_jobs, resume_folder):
    if not recommended_jobs.empty:
        relevant_jobs = recommended_jobs[recommended_jobs['job_scope'].str.contains(resume_folder, na=False)]
        accuracy = len(relevant_jobs) / len(recommended_jobs) * 100
    else:
        accuracy = 0
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
            recommended_jobs = knn.process_resume(resume_path)
            accuracy = calculate_accuracy(recommended_jobs, job_scope)
            accuracies.append(accuracy)

            # For MAP@K calculation
            actuals.append(job_scope)
            predicted_lists.append(recommended_jobs['job_scope'].tolist())

    # Calculate MAP@K for each job category
    k = 5  # You can adjust K based on your requirement
    mapk_score = calculate_mapk([job_scope] * len(predicted_lists), predicted_lists, k)
    mapk_scores.append(mapk_score)
    print(f"MAP@{k} for {job_scope}: {mapk_score}%")

# Calculate overall accuracy and MAP@K
overall_accuracy = np.mean(accuracies)
overall_mapk = np.mean(mapk_scores)

print(f"Overall System Accuracy: {overall_accuracy}%")
print(f"Overall System MAP@{k}: {overall_mapk}%")
