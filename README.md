


# Regressit
---
I built an automated regression machine which prints out an analyst-worthy PDF report. This can be used to analyze new datasets very quickly, determine if we have enough data to move forward, 
and if we do, how the variables are interacting with the target!

I wrote functions which can produce a reasonable regression model for any data we feed it. It's then able to generate a report about the independent variables and their statistical relationship to the target.

---

## How to use Regressit :

You'll find a nice importable script in this directory. Just call... 

```from regressit import * 
model = model_workflow(X, y, features_desired)
generate_report(model, title = "fitting title")
```



---



## Model workflow goes....

### Derive features
 -    Use training data to derive features though cleaning and engineering, and then save the recipe we figured out

### Create a column info DF
 -    Make a dataframe full of information about those columns so we can convey it to the audience with tables and graphs

### Select top features
 -    Take the top (n) features from the original data to be used in an OLS model. 

### Apply feature engineering
 -    Take those top performing features we found, get them from the source again, and apply those best transformations

### Model workflow
 -    Do it all in one go. Train test split, build a model, transform the test data, test the new model, and make a final one for production. Export a dictionary full of the model items, like the column_df and the model itself

### Generate report
 -    Create a PDF 



## Citations:

- https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists-2/
- https://stackabuse.com/rotate-axis-labels-in-matplotlib/
- https://stackoverflow.com/questions/46664082/python-how-to-save-statsmodels-results-as-image-file  huge thanks to this thread
- https://pyfpdf.readthedocs.io/en/latest/reference/image/index.html
- https://stackoverflow.com/questions/48145924/different-colors-for-points-and-line-in-seaborn-regplot
- https://stackoverflow.com/questions/17477979/dropping-infinite-values-from-dataframes-in-pandas
- https://datagy.io/python-get-dictionary-key-with-max-value/
- #https://www.statology.org/matplotlib-line-thickness/#:~:text=By%20default%2C%20the%20line%20width,any%20value%20greater%20than%200
