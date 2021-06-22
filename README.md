# Building a Machine Learning (ML) Library From Scratch: [Aklearn](https://akprasadan.github.io/aklearn/index.html)

This repository contains my Summer 2021 project, to build a library to perform machine learning (ML) in Python, in the spirit of Tidymodels in R or Sklearn in Python. As much as possible, I will implement all algorithms from scratch (no calling sklearn, except maybe for tests). The algorithms will be organized using Python classes to keep the code D-R-Y, and will be supplemented by classes offering data processing, train-test splitting, cross-validating, and model evaluation functionality.

My motivation for this project is to upgrade my skills in data science and programming in Python, to complement what I have already learned in R. I want to particularly improve my abilities with object oriented programming, numpy, numba, and general data analysis/visualization. In addition, I want to explore the 'ecosystem' of software engineering: using version control, good code organization, detailed and effective documentation, and usability (for others). Thus, for the first time on my own volition, I will use Git on my computer for version control, using a 2 branch (development and main) workflow, and push changes to Github. I will use Sphinx to automatically carry over my documentation to the website linked below, which was produced by both Sphinx and Read the Docs. 

You can find all the documentation [here](https://akprasadan.github.io/aklearn/index.html), produced using Sphinx and Read the Docs. Example usage is also given, along with additional references.

### I Want to Get Started NOW

Great. Check out the [website!](https://akprasadan.github.io/aklearn/index.html)


## Progress So Far 

The labellings below are as follows:
 - A single +: code is running without error
 - A pair of ++: code is running and somewhat tested 
 - A triple +++: code is running/comprehensively tested
 - If applicable, an * indicates the algorithm agrees with sklearn

### Algorithms

- Classification: KNN (+), Logistic (++, *), QDA 
- Regression: Linear (+++, *), Poisson (++, *), KNN (+)
- Clustering: K-Means (+)
- Model evaluation techniques: accuracy (+++, *), confusion matrix (+++), MSE (+++)

### Data Engineering and Preparation

- Train/test splitting (+++)
- Cross-validation folds (+++)
- Data Standardization (+++)

### Workflow

- Consistent use of Git/Github for version control and website generation.
- Increased familiarity with 'virtual environments' and organizing Python files into packages.
- Autogenerated documentation: this website; essentially, I write documentation in my code and include text files with additional content, and I run 4 commands (build the HTML website using Sphinx and 3 Git commands) and the webpage automatically collects all of the documentation into this elegant form.


## To Do

Immediate concerns:
- QDA is not able to run
- Need to decide on how classification labels should be required to be structured, or what preprocessing may still be needed

By the middle of July, 2021, I would like to have:

- Linear discriminant analysis (re-using my QDA code) (easy)
- LASSO or other penalized estimators (harder)
- Bootstrapping functionality (easy-ish)
- Abstract tuning class and incorporation into child classes
- Start to add more exposition and examples into the website, for (hypothetical) users

Eventually...

- Classification trees, bagging, random forests (the latter two are easy once I do the first)
- Boosting
- Support vector machines
- Stacking functionality (fairly easy)
- A neural network implementation (hard)
- Awesome docs!


## License
[MIT](https://choosealicense.com/licenses/mit/)

![](https://github.com/akprasadan/aklearn/workflows/Project%20Tests/badge.svg)

![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=akprasadan&show_icons=true)

