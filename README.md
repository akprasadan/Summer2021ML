# Building a ML Library From Scratch

My goal for the Summer of 2021 is to upgrade my Python and data science skills, to complement what I have already learned in R. I want to particularly improve my abilities with object oriented programming, numpy, numba, sklearn, and general data analysis/visualization. In addition, I want to explore the 'ecosystem' of software engineering: using version control, good code organization, detailed and effective documentation, and usability (for others).

To do so, I will create a unified framework (similar to Tidymodels in R or Sklearn in Python) to perform ML in Python. I will begin with base classes for Classification, Regression, Clustering, for instance, from which I will derive classes for each algorithm of interest. I will provide functionality for train/test and cross-validation splitting, model tuning, model evaluation (with a variety of metrics), and automatically generated plots.

You can find all the documentation [here](https://akprasadan.github.io/Summer2021ML/index.html), produced using Sphinx and Read the Docs.

## What's Completed so far?

- Able to use basics of Git/Github 
- Barebones classes for Regression and Classification, with drafts of k-Nearest Neighbor classification, k-Means clustering, and linear regression.
- Train/test split functionality

## File Organization

I will store the main functionality in generalclassifier. The other folders (knn, kmeans) will eventually be merged into it as I adapt them to inherit from my base classes.

 - aklearn
     - regression 
     - classification.py
     - linearreg.py
     - logisticreg.py
     - evaluation_metrics.py 
     - norms.py
     - preprocessing.py
     - \_\_init\_\_.py
 - k_means_project
    - k_means.py
    - test_k_means.py
 - knn_classify_project
    - distance_func.py
    - knn_classify.py
    - test_knn.py
    - weight_kernels.py
 - .gitignore
 - LICENSE.txt
 - README.md


## License
[MIT](https://choosealicense.com/licenses/mit/)
