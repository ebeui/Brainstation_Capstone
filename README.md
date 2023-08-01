# Tasty Trail - Capstone Project

## Overview <img align="right" src="TastyTrail_logo_small.svg">
Welcome to my capstone project for BrainStation — a user-friendly restaurant recommendation system. The primary goal of this project is to create a data-driven recommendation system that understands each user's preferences and provides personalized restaurant suggestions, leading to delightful dining experiences and increased customer loyalty. The system utilizes user-based collaborative filtering techniques to offer targeted and customized recommendations, which can potentially attract new patrons and benefit restaurant owners.

## 💭 Why a Restaurant Recommender System?
In today's fast-paced world, people often rely on online platforms to find places to eat. However, with an overwhelming number of restaurants and diverse individual preferences, finding the perfect dining spot can be challenging. A personalized restaurant recommendation system can revolutionize the way people discover new places to eat. By leveraging data science and machine learning techniques, we can analyze users' past restaurant preferences, ratings, and other relevant information to offer tailored suggestions, saving users time and effort while enhancing their dining experiences.

## 🚀 Key Phases of the Project
This capstone project follows a structured approach, comprising the following key phases:

1. **[Data Collection](https://www.kaggle.com/yelp-dataset/yelp-dataset)**: The foundation of any data-driven project is access to high-quality data. For this project, we acquire the necessary data from the Yelp dataset (Version 3) available on Kaggle.

2. **[Data Preprocessing](https://github.com/ebeui/Brainstation_Capstone/blob/fdde097b59a9b6a4cfefabb86a05ef1b70fc1dd4/Notebooks/1_data_preprocessing.ipynb)**: In this notebook, we're gonna talk about Data Cleaning. It's all about fixing stuff like mistakes and missing values in raw data so that we can use it properly for analysis and modeling. This step is super important because it helps us get accurate and reliable results later on. Our main goals here are to get rid of any missing values, remove duplicates, and fill in the gaps in the data as needed.

3. **[Data Merging](https://github.com/ebeui/Brainstation_Capstone/blob/fdde097b59a9b6a4cfefabb86a05ef1b70fc1dd4/Notebooks/2_data_merge.ipynb)**: In this notebook, our main focus is to merge the cleaned datasets and create the final datasets. We've put in the effort to clean the individual datasets, and now it's time to bring them all together to form the complete and ready-to-use data.

3. **[Exploratory Data Analysis (EDA)](https://github.com/ebeui/Brainstation_Capstone/blob/fdde097b59a9b6a4cfefabb86a05ef1b70fc1dd4/Notebooks/3_EDA_and_visualizations.ipynb)**: In this notebook, we shall embark on the journey of Exploratory Data Analysis (EDA), a pivotal phase in our project. EDA assumes a critical role by affording us the opportunity to gain profound insights, identify prevailing patterns, and unveil underlying trends within the dataset. Employing a diverse array of visualizations and robust statistical techniques, we shall diligently explore the data, extracting meaningful and valuable information that will substantially influence our decision-making process throughout the development of the recommendation system.

4. **[Content Based Model](https://github.com/ebeui/Brainstation_Capstone/blob/fdde097b59a9b6a4cfefabb86a05ef1b70fc1dd4/Notebooks/4_content_based.ipynb)**: In this notebook, where we'll be building a content-based recommender system for restaurants. Our goal is to create a system that suggests restaurants to users based on their preferences and past interactions with restaurants. By leveraging the content and features of the restaurants, we can provide personalized recommendations that align with each user's taste.

5. **[Matrix Factorization Model](https://github.com/ebeui/Brainstation_Capstone/blob/fdde097b59a9b6a4cfefabb86a05ef1b70fc1dd4/Notebooks/5_matrix_factorization.ipynb)**: In this notebook, our objective is to construct a Matrix Factorization Model. We shall employ hyperparameter tuning through grid search to optimize the model's performance. Once we have the tuned model, we will proceed to compare its predictions with the actual ratings to assess its accuracy and effectiveness.

## 📑 Resources
I have organized the following resources for easy access:

- **Data Source**: The original Yelp dataset (Version 3) can be found on [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset).

- **Data Documentation**: Documentation of the Yelp dataset can be found [here](https://www.yelp.com/dataset/documentation/main).

- **Standup Progress Presentation**: This is a [presentation](https://github.com/ebeui/Brainstation_Capstone/blob/fdde097b59a9b6a4cfefabb86a05ef1b70fc1dd4/Reports/progress/DianeLu_ProgressStandup.pdf) of the progress made during development.

## 📬 Contact
Thank you for your interest in my restaurant recommendation system project! If you have any questions, or suggestions, or simply want to discuss anything related to the project, please don't hesitate to reach out through this GitHub repository or via email at [dianengalu@gmail.com](mailto:dianengalu@gmail.com). I'm excited to share this project with the community and look forward to receiving valuable feedback.

Happy dining! 🍔🍕🍣
