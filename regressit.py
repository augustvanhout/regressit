


# ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
#▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀█░█▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ 
#▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌               ▐░▌          ▐░▌     
#▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄      ▐░▌          ▐░▌     
#▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌          ▐░▌     
#▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌     ▐░▌          ▐░▌     
#▐░▌     ▐░▌  ▐░▌          ▐░▌       ▐░▌▐░▌     ▐░▌  ▐░▌                    ▐░▌          ▐░▌     ▐░▌          ▐░▌     
#▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄█░█▄▄▄▄      ▐░▌     
#▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌     
# ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀       ▀      
                                                                                                                     
    
    
# Model workflow goes....

# Derive features
#     Use training data to derive features though cleaning and engineering, and then save the recipe we figured out

# Create a column info DF
#     Make a dataframe full of information about those columns so we can convey it to the audience with tables and graphs

# Select top features
#     Take the top (n) features from the original data to be used in an OLS model. 

# Apply feature engineering
#     Take those top performing features we found, get them from the source again, and apply those best transformations

# Model workflow
#     Do it all in one go. Train test split, build a model, transform the test data, test the new model, and make a final one
#     for production. Export a dictionary full of the model items, like the column_df and the model itself

# Generate report
#     Crease a PDF 



# In short... run the following:

# from regressit import *
# model = model_workflow(X, y, features_desired)
# generate_report(model, title = "fitting title")

# and then check the output folder.


# Happy regressing
# - Augie


# Package importation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
import sklearn 
pd.options.display.max_columns = 100

import warnings
warnings.filterwarnings('ignore')

def derive_features(dataset, y):
    scratch_dataset = dataset.copy()
    column_dict = {}
    numeric_engineering_dict = {} # this stores the optimal transformation so it can be repeated on new data
    categorical_value_dict = {} # this stores the optimal transformation so it can be mapped to new data

    for column in scratch_dataset.select_dtypes("number"):
        candidate_scratch = pd.DataFrame()
        mean = scratch_dataset[column].mean()
        candidate_scratch["mean_imputed"] = scratch_dataset[column].fillna(float(mean))
        candidate_scratch["zero_imputed"] = scratch_dataset[column].fillna(0)
        non_null_col = scratch_dataset.loc[scratch_dataset[column].isna() == False][column]
        mode = non_null_col.mode()
        candidate_scratch["mode_imputed"] = scratch_dataset[column].fillna(float(min(mode)))

        poly = PolynomialFeatures()
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        candidate_array = poly.fit_transform(candidate_scratch)
        candidate_scratch = pd.DataFrame(candidate_array, columns = [poly.get_feature_names_out(candidate_scratch.columns)])

        for cand_col in candidate_scratch.columns:
            cand_col_name = str(cand_col).replace("(", "").replace(")", "").replace("'", "").replace(",", "")
            log_adjusted = np.log(candidate_scratch[cand_col]).replace([np.inf, -np.inf], 0)
            candidate_scratch[cand_col_name + "_log"] = log_adjusted.fillna(np.mean(log_adjusted) ) # not super happy to replace these with 0
            #https://stackoverflow.com/questions/17477979/dropping-infinite-values-from-dataframes-in-pandas

        candidate_scratch = candidate_scratch.drop(columns = candidate_scratch.filter(like = " ").columns)
        candidate_scratch = candidate_scratch.drop(columns = candidate_scratch.filter(like = "^2_log").columns)

        # and now select a best fitting column
        r2_dict = {}
        for cand_column in candidate_scratch.columns:
            cand_column_name = str(cand_column).replace("(", "").replace(")", "").replace("'", "").replace(",", "").replace(" ", "_")
            X = candidate_scratch[[cand_column]]
            lr = LinearRegression()
            score = lr.fit(X, y).score(X, y)
            r2_dict[cand_column_name] = score
        top_col = max(r2_dict, key = r2_dict.get) # https://datagy.io/python-get-dictionary-key-with-max-value/
        top_col_score = r2_dict[top_col]
        #print("TOP SCORE: ", column,"| engineering: ", top_col, "| r2 score: ", top_col_score)
        column_dict[(column + "_" + top_col)] = [column, top_col, "numeric", score] # I'm going to make this into a dataframe later.
        numeric_engineering_dict[column] = [(column + "_" + top_col), top_col]

    # this concludes numerics. Let's do categoricals!
    # categoricals
    for column in scratch_dataset.select_dtypes("object"):
        candidate_scratch = pd.DataFrame()
        candidate_scratch[column] = scratch_dataset[column]
        mode = scratch_dataset[column].mode()
        candidate_scratch["mode_imputed"] = scratch_dataset[column].fillna(scratch_dataset[column].value_counts().idxmax())
        candidate_scratch["na_val_imputed"] = scratch_dataset[column].fillna("NA")
        # columns with mode and na imputation complete.

        candidate_scratch["y"] = y
        candidate_rank_df = pd.DataFrame(candidate_scratch.groupby("mode_imputed")["y"].mean())
        candidate_rank_dict = candidate_rank_df.to_dict()["y"]
        candidate_scratch["mode_imputed"] = candidate_scratch["mode_imputed"].map(candidate_rank_dict).fillna(0)

        candidate_rank_df = pd.DataFrame(candidate_scratch.groupby("na_val_imputed")["y"].median())
        candidate_rank_dict = candidate_rank_df.to_dict()["y"]
        candidate_scratch["na_val_imputed"] = candidate_scratch["na_val_imputed"].map(candidate_rank_dict).fillna(0)
        # variables have been ranked with both median and mode.

        poly = PolynomialFeatures()
        candidate_numeric = candidate_scratch.select_dtypes("number")
        candidate_array = poly.fit_transform(candidate_numeric)
        candidate_scratch = pd.DataFrame(candidate_array, columns = [poly.get_feature_names_out(candidate_numeric.columns)])
        #candidate_scratch = candidate_numeric.drop(columns = "1")

        for cand_col in candidate_numeric.columns:
            cand_col_name = str(cand_col).replace("(", "").replace(")", "").replace("'", "").replace(",", "")
            candidate_scratch[cand_col_name + "_log"] = np.log(candidate_numeric[cand_col]).replace([np.inf, -np.inf], 0)
        candidate_scratch = candidate_numeric.drop(columns = candidate_numeric.filter(like = " ").columns)
        candidate_scratch = candidate_numeric.drop(columns = candidate_numeric.filter(like = "^2_log").columns)

        # and now select a best fitting column
        r2_dict = {}
        for cand_column in candidate_scratch.drop(columns = "y").columns:
            cand_column_name = str(cand_column).replace("(", "").replace(")", "").replace("'", "").replace(",", "").replace(" ", "_")
            X = candidate_scratch[[cand_column]]
            lr = LinearRegression()
            score = lr.fit(X, y).score(X, y)
            r2_dict[cand_column_name] = score
        top_col = max(r2_dict, key = r2_dict.get) # https://datagy.io/python-get-dictionary-key-with-max-value/
        top_col_score = r2_dict[top_col]
        #print("TOP SCORE: ", column, "| engineering: ", top_col, "| r2 score: ", top_col_score)
        column_dict[(column + "_" + top_col)] = [column, top_col, "categorical", score] # I'm going to make this into a dataframe later.
        # and this next line saves values to use when converting test data using our findings here.
        categorical_value_dict[column] = [column + "_" + top_col, candidate_rank_dict]

    return column_dict, numeric_engineering_dict, categorical_value_dict

def create_column_info_df(column_dict):
    column_info_df = pd.DataFrame.from_dict(column_dict).T.reset_index().rename(columns = {"index" :  "feature",
                                                                                      0 : "original_feature",    
                                                                                      1 : "engineering",
                                                                                      2 : "type",
                                                                                      3 : "r2_score"}).sort_values(by = "r2_score", ascending = False)
    column_info_df.reset_index(inplace = True, drop = True)
    column_info_df.head(50)
    return column_info_df

# create model with top (n) columns
def select_top_features(new_df, column_info_df, n):
    selected_dataset = new_df.filter(items = [feature for feature in column_info_df["original_feature"][:n]])
    return selected_dataset

# clean and engineer columns using the prescribed methods
def prescribed_transform(dataframe, column, t_dict):
    # cleaning
    if "mean_imputed" in t_dict[column][1]:
        new_column = dataframe[column].fillna(dataframe[column].mean())
        name = column + "_mean_imputed"
    if "mode_imputed" in t_dict[column][1]:
        new_column = dataframe[column].fillna(dataframe[column].mode())
        name = column + "_mode_imputed"
    if "zero_imputed" in t_dict[column][1]:    
        new_column = dataframe[column].fillna(0)
        name = column + "_zero_imputed"
        
    # and a little transformation
    if t_dict[column][1][-4:] == "_log":
        new_column = np.log(new_column)
        name = name + "_log"
    if t_dict[column][1][-2:] == "^2":
        new_column = new_column **2
        name = name + "^2"
    return name, new_column

def prescribed_map(dataframe, column, m_dict):
    # coding defensively by making sure no non-matching(new data) map values get left in to ruin the numeric column
    new_column = []
    name = m_dict[column][0]
    for value in dataframe[column]:
        if value not in m_dict[column][1].keys():
            new_column.append(np.nan)
        else:
            new_column.append(m_dict[column][1][value])
    return name, new_column

def apply_feature_engineering(dataframe, numeric_dict, categorical_dict):
    for column in dataframe.select_dtypes("number").columns:
        name_and_values = prescribed_transform(dataframe, column, numeric_dict)
        dataframe[column] = name_and_values[1]
        dataframe.rename(columns = {column : name_and_values[0]}, inplace = True)
        
    for column in dataframe.select_dtypes("object").columns:
        mapped_column = prescribed_map(dataframe, column, categorical_dict)
        dataframe[column] = mapped_column[1]
        dataframe[column] = dataframe[column].fillna(dataframe[column].mean()).astype(float) # and fill whatever new categoricals with mean
        dataframe.rename(columns = {column : mapped_column[0]}, inplace = True)
    return dataframe

def model_workflow(full_X, full_y, feature_count):
    X_train, X_test, y_train, y_test = train_test_split(full_X, full_y)
    # derive features
    column_dict, numeric_engineering_dict, categorical_value_dict = derive_features(X_train, y_train)
    # create column info df
    column_info_df = create_column_info_df(column_dict)
    # select top features
    selected_dataset = select_top_features(X_train, column_info_df, feature_count)
    # re-apply column engineering
    X = apply_feature_engineering(selected_dataset, numeric_engineering_dict, categorical_value_dict)
    y = y_train
    # create ols model
    lr = LinearRegression()
    lr.fit(X, y)
    # evaluate model
    print("Training Score: ", lr.score(X, y))
    # create test
    selected_dataset = select_top_features(X_test, column_info_df, feature_count)
    X = apply_feature_engineering(selected_dataset, numeric_engineering_dict, categorical_value_dict)
    y = y_test
    # evaluate test
    print("Testing Score: ", lr.score(X, y))
    
    # Creation of Production Model
    # derive features
    column_dict, numeric_engineering_dict, categorical_value_dict = derive_features(full_X, full_y)
    # create column info df
    column_info_df = create_column_info_df(column_dict)
    # select top features
    selected_dataset = select_top_features(full_X, column_info_df, feature_count)
    # re-apply column engineering
    X = apply_feature_engineering(selected_dataset, numeric_engineering_dict, categorical_value_dict)
    y = full_y
    # create ols model
    lr = LinearRegression()
    lr.fit(X, y)
    # evaluate model
    print("Production Score: ", lr.score(X, y))
    
    return {"model" : lr, 
            "column_dict" : column_dict,
            "column_info_df" : column_info_df,
            "numeric_engineering_dict" : numeric_engineering_dict, 
            "categorical_value_dict" : categorical_value_dict,
            "X" : X,
            "y" : y,
            "full_X" : full_X}

def generate_report(model_workflow_object, title):
    model = model_workflow_object
    import os
    import fpdf
    import statsmodels.api as sm
    
    if os.path.isdir("report") == False:
        os.mkdir("report")
    pdf = fpdf.FPDF()
    
    plt.figure()
    preds = model["model"].predict(model["X"])
    plt.figure(figsize = (7, 5))
    plt.scatter(model["y"], preds)
    plt.xlabel("Actual Values", size = 12)
    plt.ylabel("Predicted Values", size = 12)
    plt.title(size = 18, label = "Price Model")
    plt.plot(model["y"], model["y"],  linewidth = 1.5, c = "black") #https://www.statology.org/matplotlib-line-thickness/#:~:text=By%20default%2C%20the%20line%20width,any%20value%20greater%20than%200.
    plt.savefig("report/model_predictions_vs_actials.png", bbox_inches = "tight", facecolor = "white")

    plt.clf()
    import dataframe_image as dfi
    model_variable_df = model["column_info_df"][:len(model["X"].columns)]
    dfi.export(model_variable_df, 'report/variable_df.png')

    pdf.add_page()
    pdf.set_font("Arial", size = 28)
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = title,
             ln = 1, align = 'C')
    pdf.set_font("Arial", size = 12)
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.image("report/model_predictions_vs_actials.png", x = 23, type = 'png')
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.image('report/variable_df.png', x = 42, w = 128, h =len(model["X"].columns) * 7.5, type = 'png')

    pdf.add_page()
    plt.figure()
    results = sm.OLS(model["y"],model["X"]).fit()

    # these guys right here https://stackoverflow.com/questions/46664082/python-how-to-save-statsmodels-results-as-image-file
    plt.tight_layout(pad = 0)
    plt.figure(figsize = (8, 5))
    plt.text(0.01, 0.05, str(results.summary()), {'fontsize': 8}, fontproperties = 'monospace') 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('report/main_summary.png', facecolor = "white")

    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = "Main Model Summary",
             ln = 2, align = 'C')
    pdf.cell(200, 10, txt = " ",
             ln = 2, align = 'C')
    pdf.image("report/main_summary.png", x = 23, type = 'png')
    
    plt.clf()
    plt.figure()
    plt.tight_layout(w_pad = 0.5)
    plt.figure(figsize = (len(model["X"].columns) * 1.5,4.5))
    model_variable_df = model["column_info_df"][:len(model["X"].columns)]
    plt.bar(model_variable_df["original_feature"], model_variable_df["r2_score"] * 100)
    plt.title("Impact of Variable", size = 24)
    plt.ylabel("% Change Explained", size = 18)
    plt.xticks(size = 10)

    plt.savefig("report/main_model_variables.png", bbox_inches = "tight", facecolor = "white")
    pdf.image("report/main_model_variables.png", x = 30, w = len(model["X"].columns) * 18, type = 'png')

    for column in model["X"].columns:
        pdf.add_page()

        y_new = model["y"]
        X_new = model["X"][[column]]
        original_colname = model["column_dict"][column][0]
        lr = LinearRegression()
        lr.fit(X_new,y_new)
        score = lr.score(X_new, y_new)

        plt.clf()
        plt.figure()
        plt.tight_layout(w_pad = 0.5)
        plt.figure(figsize = (7,4.5))
        if model["column_dict"][column][2] == "numeric":
            sns.regplot(model["X"][column], model["y"], ci = False, line_kws= {"color" : "black"}); #https://stackoverflow.com/questions/48145924/different-colors-for-points-and-line-in-seaborn-regplot
        else:
            df = pd.DataFrame({original_colname : model["full_X"][original_colname], 
                               y_new.name: model["y"]})
            sns.boxplot(data = df, 
                x = model["column_dict"][column][0], 
                y = y_new.name,
               order = df.groupby(original_colname).median()[y_new.name].sort_values(ascending = False).reset_index()[original_colname].tolist()
               );
            if len(df.groupby(original_colname).median()[y_new.name].sort_values(ascending = False).reset_index()[original_colname].tolist()) > 8:
                plt.xticks(rotation = 45)
        plt.xlabel(original_colname, size = 16)
        plt.ylabel(y_new.name, size = 16)
        plt.title(original_colname + "  vs  " + y_new.name, size = 20)
        plt.savefig("report/" + column + "_variable.png", bbox_inches = "tight", facecolor = "white")
        pdf.image("report/" + column + "_variable.png", x = 18, type = 'png')

        pdf.cell(200, 10, txt = ("Variable Type: " + model["column_dict"][column][2]),
             ln = 2, align = 'C')
        pdf.cell(200, 10, txt = ("Engineering: " + model["column_dict"][column][1]),
             ln = 2, align = 'C')

        plt.clf()
        plt.figure()
        results = sm.OLS(X_new, y_new).fit()

        plt.tight_layout(w_pad = 0.5)
        plt.figure(figsize = (7, 5.5))
        plt.text(0.01, 0.05, str(results.summary()), {'fontsize': 10}, fontproperties = 'monospace') 
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('report/' + column +'_summary.png')
        pdf.image("report/" + column + "_summary.png", x = 24, w = 155, type = 'png')
        plt.clf()
        
    pdf.output("report/model_report.pdf")  