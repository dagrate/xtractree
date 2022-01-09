## Lending Club Loan Data Set Download

To demonstrate the xtractree algorithm, we use the following Kaggle dataset: wordsforthewise/lending-club

### Using the Kaggle API
First, please read the official Kaggle API documentation if you never used the API before in order to install the API and get an authentification token: wordsforthewise/lending-club

Once the Kaggle API is installed and you authentification token is ready, please download the dataset using the following command:
```bash
kaggle datasets download -d wordsforthewise/lending-clubs
```


### From Kaggle's website
The lending club loan data set can be directly downloaded from: https://www.kaggle.com/wendykan/lending-club-loan-data

## Replicate the Data Set Used in our Experiments

The size of the data set we used in our experiments is too large to be uploaded on GitHub. You can easily create the data set used in our paper publication by executing the script xtractreeCreateData.py to create all the required features.
