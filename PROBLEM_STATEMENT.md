# Smart Student Study Planner

## Problem Statement

The Smart Student Study Planner is designed to use machine learning to allow students to learn more about how well they are doing academically.

The use of factors related to students (study time, previous failure, absence, family support) will help us determine what a student's final academic grade will be.

The system will additionally give students feedback about their learning and personalized advice on how to plan their studies so they can make more informed choices regarding their education.

## Project Details

- **Problem Type:** Regression
- **Target Variable:** G3 (Final Grade)
- **Dataset:** Student Performance Dataset – Mathematics
- **End Users:** Students
- **Technology:** Python, Machine Learning, Pandas, Scikit-learn and Streamlit

## Project Objective

To develop a machine learning-based application that predicts student academic performance and provides useful insights to support better study planning.

## Scope and data decisions

- **Target population:** the planner predicts the final math grade (G3) for students who sit the final exam. It does not predict dropout.
- **Rows removed:** 38 of the 395 students have G3 = 0, and all 38 have exactly 0 absences. This suggests they dropped out or did not take the exam, so their zero means "no result" rather than a low score. I removed them, leaving 357 students. The raw CSV is unchanged and the filtering is done in the notebook.
- **Features excluded:** G1 and G2 (earlier period grades) are not used, because a student would not have them when planning early in the term. Using them would leak information about the final grade.
- **Limitations:** the dataset is small (357 rows after cleaning), comes from two Portuguese schools, and is largely self-reported, so results may not generalize. The model cannot flag students at risk of dropping out.