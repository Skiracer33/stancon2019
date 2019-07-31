## Handling missing data, censored values and measurement error in machine learning models using multiple imputation for early stage drug discovery

This is the notebook for [STANCON 2019](https://mc-stan.org/events/stancon2019Cambridge/)

Abstract is available [here.](https://mc-stan.org/events/stancon2019Cambridge/abstracts.html#19)

The notebook contains three sections:

1) An introduction to multiple imputation
2) Simulation experiments of multiple imputation compared to other imputation techniques
3) An introduction to a library for  multiple imputation of censored data

To install the requirements for the notebook using conda on linux run:

	conda env create -f environment.yml

Precomputed simulation experiments are provided as pickled files.
To perform the simulation experiments yourself cd into the computation directory and run

	./submit_all -d my_experiments -b yes -s 52
    
The notebook looks best when read in html or as the .ipynb a pdf has been provided for compatability. 
