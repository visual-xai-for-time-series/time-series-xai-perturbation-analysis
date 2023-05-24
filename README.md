# A Deep Dive into Perturbations as Evaluation Technique for Time Series XAI

Code and result repository for "A Deep Dive into Perturbations as Evaluation Technique for Time Series XAI".

## Results

The results can be seen in the HTML files in the main directory of the repository.
Alternatively, the results directory contains the experiment results for the different datasets.  
The `results.json` can be loaded via Python to explore the raw results.  
All other perturbation analysis cards can be found in the corresponding directories.  

## Reproducibility

For reproducibility, please install Python as mentioned in the version below and the requirements.txt.  
Afterward, you can use the jupyter notebooks as they are or convert them to HTML files:
`jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True attributions-metrics-forda.ipynb`  

## Extensions

The juypter notebooks can be used as a guideline for future extensions of the experiments.  
The dataset needs to be exchanged and the models, but all the different analyses should work for other datasets.

## Libraries

-   Python v3.10
-   Pytorch (https://pytorch.org/)
-   Captum (https://captum.ai/)
-   Numpy (https://numpy.org/)
-   Scipy (https://scipy.org/)
-   Pandas (https://pandas.pydata.org/)

## License

Released under MIT License. See the LICENSE file for details.

## Reference

```
@conference{,
 author = {Schlegel, Udo and Keim, Daniel A.},
 booktitle = {1st World Conference on eXplainable Artificial Intelligence 2023},
 title = {A Deep Dive into Perturbations as Evaluation Technique for Time Series XAI},
 year = {2023}
}
```