# Assessing Software Bug Severity using Sentiment Analysis
In software development, managing and prioritizing bug reports from various sources, such as GitHub repositories, is a critical yet challenging task. Traditional methods of handling bug reports often involve manual processes that are time-consuming and prone to human error. Moreover, accurately assessing the severity and sentiment of bug reports can significantly impact the efficiency and effectiveness of the development cycle.

To address these challenges, we have developed an application that automates the extraction and analysis of bug reports from GitHub repositories.

1.  This application takes the URL of any GitHub repository and converts the issues tab into a structured CSV format.
  
2.  The descriptions in the CSV are then preprocessed, lemmatized, and analyzed using TextBlob to determine sentiment scores.
   
3.  Additionally, an ensemble machine learning model comprising:

       1. K-Nearest Neighbors (KNN)
       2. Support Vector Machine (SVM)
       3. Decision Tree
       4. Random Forest algorithms
    is utilized to evaluate the precision, accuracy, F1 score, and recall of the bug report predictions.

This automated approach not only streamlines the bug reporting process but also enhances the accuracy of severity assessments, enabling developers to prioritize and address critical issues more effectively. The integration of natural language processing (NLP) and machine learning (ML) techniques offers a sophisticated solution to improve the overall quality and efficiency of software maintenance and development.
