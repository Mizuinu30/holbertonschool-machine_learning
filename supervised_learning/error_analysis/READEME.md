# Error Analysis Project

## Overview

This project focuses on error analysis in machine learning, encompassing various metrics and concepts essential for evaluating and improving model performance. Key topics include confusion matrix, type I and type II errors, sensitivity, specificity, precision, recall, F1 score, bias, variance, and Bayes error. The project adheres to specific coding and documentation standards and is implemented using Python with numpy.

## Learning Objectives

By the end of this project, you will be able to explain the following concepts without external references:

### General Concepts
- **Confusion Matrix:** Understand what a confusion matrix is and how to create it.
- **Type I and Type II Errors:** Define and differentiate between type I (false positive) and type II (false negative) errors.
- **Sensitivity and Specificity:** Explain sensitivity (true positive rate) and specificity (true negative rate).
- **Precision and Recall:** Understand precision (positive predictive value) and recall (sensitivity).
- **F1 Score:** Calculate and interpret the F1 score, which balances precision and recall.
- **Bias and Variance:** Define bias, variance, and understand their trade-off.
- **Irreducible Error:** Explain what irreducible error is in the context of machine learning.
- **Bayes Error:** Define Bayes error and how it represents the theoretical lower bound of the error rate.
- **Approximation of Bayes Error:** Understand methods to approximate Bayes error.
- **Calculation of Bias and Variance:** Learn how to calculate bias and variance for a given model.

## Requirements

### General
- Use allowed editors: `vi`, `vim`, `emacs`.
- Files will be interpreted/compiled on Ubuntu 20.04 LTS using Python 3.9.
- Use numpy (version 1.25.2) for numerical operations.
- Ensure all files end with a new line.
- The first line of all files should be `#!/usr/bin/env python3`.
- Include a mandatory `README.md` file at the root of the project folder.
- Code should follow the pycodestyle style (version 2.11.1).
- All modules, classes, and functions should have appropriate documentation.
- Only numpy is allowed for import unless otherwise noted.
- All files must be executable.
- The length of the files will be tested using `wc`.

## Resources

### Read or Watch
- **Confusion Matrix:** [Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- **Type I and Type II Errors:** [Type I and type II errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
- **Sensitivity and Specificity:** [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- **Precision and Recall:** [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- **F1 Score:** [F1 score](https://en.wikipedia.org/wiki/F1_score)
- **Confusion Matrix in ML:** [What is a Confusion Matrix in Machine Learning?](https://towardsdatascience.com/what-is-a-confusion-matrix-in-machine-learning-9d51a775891e)
- **Confusion Matrix Terminology:** [Simple guide to confusion matrix terminology](https://towardsdatascience.com/simple-guide-to-confusion-matrix-terminology-8b6e2437d7aa)
- **Bias-Variance Tradeoff:** [Bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- **Bias and Variance:** [What is bias and variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- **Bayes Error Rate:** [Bayes error rate](https://en.wikipedia.org/wiki/Bayes_error_rate)
- **Bayes Error in ML:** [What is Bayes Error in machine learning?](https://towardsdatascience.com/what-is-bayes-error-in-machine-learning-9c347cd0b2f)
- **Bias/Variance Video:** [Bias/Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
- **ML Recipe Video:** [Basic Recipe for Machine Learning](https://www.youtube.com/watch?v=Qc5IyLW_hns)
- **Human Level Performance Video:** [Why Human Level Performance](https://www.youtube.com/watch?v=qO_NLVjD7qg)
- **Avoidable Bias Video:** [Avoidable Bias](https://www.youtube.com/watch?v=I7wcXIsAN_c)
- **Understanding Human-Level Performance Video:** [Understanding Human-Level Performance](https://www.youtube.com/watch?v=I7wcXIsAN_c)

## Project Structure

```
error_analysis/
│
├── README.md
├── confusion_matrix.py
├── type_errors.py
├── sensitivity_specificity.py
├── precision_recall.py
├── f1_score.py
├── bias_variance.py
├── bayes_error.py
└── tests/
    ├── test_confusion_matrix.py
    ├── test_type_errors.py
    ├── test_sensitivity_specificity.py
    ├── test_precision_recall.py
    ├── test_f1_score.py
    ├── test_bias_variance.py
    └── test_bayes_error.py
```

## How to Run the Project

1. Ensure you have Python 3.9 and numpy (version 1.25.2) installed.
2. Clone the repository.
3. Navigate to the project directory.
4. Run the individual scripts to perform the specific error analysis tasks.
5. Use the test files in the `tests/` directory to validate the functions.

## Example Usage

```bash
# Run a specific analysis script
./confusion_matrix.py

# Run tests
python3 -m unittest discover tests
```

