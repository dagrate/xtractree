# xtractree
Extract Tree from Bagging Classifiers

<p align="middle">
  <img src="https://github.com/dagrate/xtractree/blob/master/plots/gitdt.png" width="400"/>
  <img src="https://github.com/dagrate/xtractree/blob/master/plots/gitrf.png" width="400"/>
</p>

XtracTree is a Python library that proposes to convert a bagging classifer into a set of ''if-then'' rules satisfying the requirements of model validation. XtracTree is also capable of (i) performing accurate predictions based on the extracted set of ''if-then'' rules and (ii) to highlight the decision path for each individual sample. XtracTree allows non machine learning expert to understand the decision of a machine learning bagging classifier.


## Dependencies

The library uses **Python 3** and **R** with the following modules:
- numpy (Python 3) == '1.18.2'
- pandas (Python 3) == '1.0.3'
- seaborn (Python 3) == '0.10.0'
- matplotlib (Python 3) == '3.2.1'
- scipy (Python 3) == '1.4.1'
- sklearn (Python 3) == '0.22.2.post1'

----------------------------

## Citing

If you use the repository, please cite:

```bibtex
@inproceedings{charlier2020xtractree,
  title={XtracTree for Regulator Validation of Bagging Methods Used in Retail Banking},
  author={Charlier, Jeremy and Makarenkov, Vladimir},
  journal={arXiv preprint},
  year={2020}
}
```
