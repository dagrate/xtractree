# xtractree
Extract Tree from Bagging Classifiers

<p align="middle">
  <img src="https://github.com/dagrate/xtractree/blob/master/plots/gitdt.png" width="400"/>       
  <img src="https://github.com/dagrate/xtractree/blob/master/plots/gitrf.png" width="400"/>
</p>

XtracTree is a Python library that proposes to convert a bagging classifer into a set of ''if-then'' rules satisfying the requirements of model validation. XtracTree is also capable of (i) performing accurate predictions based on the extracted set of ''if-then'' rules and (ii) to highlight the decision path for each individual sample. XtracTree allows non machine learning experts to understand the decision of a machine learning bagging classifier by using only ''if-then'' rules with business taxonomy.

The notebook xtractree_demo illustrates the usage of XtracTree and its output. 

----------------------------

## Dependencies

The library uses **Python 3.6** with the following modules:
- numpy (Python 3) >= '1.18.2'
- pandas (Python 3) >= '1.0.3'

To run the demos and the experiments of our publications, the following modules are also required:
- seaborn (Python 3) >= '0.10.0'
- matplotlib (Python 3) >= '3.2.1'
- scipy (Python 3) >= '1.4.1'
- sklearn (Python 3) >= '0.22.2.post1'

----------------------------

## Usage

The class XtracTree is in the file xtractree.py. <br>

To replicate the experiments of our paper publication:
- Create the data from the script xtractreecreatedata.py stored in the data folder
- Execute xtractreeroccurve.py for the ROC curves
- Execute xtractree.py for the ''if-then'' decision rules

----------------------------

## Citing

If you use the repository, please cite:

```bibtex
@article{charlier2020xtractree,
  title={XtracTree for Regulator Validation of Bagging Methods Used in Retail Banking},
  author={Charlier, Jeremy and Makarenkov, Vladimir},
  journal={arXiv preprint},
  year={2020}
}
```
