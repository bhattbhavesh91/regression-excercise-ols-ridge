# A Regression Exercise covering OLS & Ridge Regression

## **Rationale for the methodology used**

- Loaded the data into a dataframe df
- The dataset has 155 rows and 20 columns
- The target column **Response Variable** is continuous in nature, thus it is a **Regression** task and not a **Classification**
- The data has no missing values.
- I drop the date column since it has 155 unique values thus it carries no predictive power.
- I split my data into X &amp; Y where X contain the features and Y contains my target variable &quot; **Response Variable**&quot;
- After splitting my data into X &amp; Y, I split my data into train and test datasets with 70% of data being used for training and 30% data being used for testing.
- Now, I start by analysing the target variable distribution.
  - The target variable is positively skewed with a mean value close to 891, so we will have to apply transformation to convert it to near normal distribution.
  - I apply log10 transformation thrice to make the overall distribution near normal with a skew value of 0.25.

- Next up, I try to analyse the relationship of features with response variables using Seaborn&#39;s jointplot and this is what I observe
  - Feature 1 &amp; 2 have a slightly significant negative relationship with the response variable.
  - Feature 4 has no relationship with the response variable.
  - Feature 3, 5, 6, 7, 8, 9, 10, 11, 12, 13 &amp; 17 have a highly significant positive relationship with the response variable.
  - Feature 14 &amp; 18 have a highly significant negative relationship with the response variable.
  - Feature 15 &amp; 16 have a slightly significant positive relationship with the response variable.

- Next up, I plot a heatmap of features to find out their correlation strength with the target features and also try to infer if there is any multicollinearity in the data.
  - The features - Feature 9, 11, 10, 17, 13, 12 &amp; 6 are heavily correlated with the target variable.
  - Feature 9, 11 &amp; 10 exhibit multicollinearity.
  - Feature 12 &amp; 6 are also heavily correlated with each other.
  - There are many such combinations that you fill find of collinearity.

- I now use Variance inflation factor (VIF) for each feature to find out if the feature can be described using other features and a general rule of thumb commonly used in practice is if a VIF is \&gt; 5, you have high multicollinearity. So, I eliminate features with very high VIF score and I&#39;m left with features 1, 4, 5, 13 and 15.

- I again plot a heatmap to see if there is no multicollinearity in the data and I find heavy correlation between Feature 13 and Feature 5.

- Since Feature 13 is more significantly correlated with Response Variable, I drop Feature 5 from the training dataset.

- I use the OLS function from the statsmodel package.
  - We start the exercise by using the statsmodel based OLS model to fit X &amp; Y because of the results provided by the model that explain the significance of each predictor variable.
  - The F-test of overall significance indicates whether your linear regression model provides a better fit to the data than a model that contains no independent variables. So, in our case based on the p-value of F-test or Prob (F-statistic) we conclude that the model is a good fit model.
  - R-squared value of 0.69 signifies it a descent fit model.
  - The P\&gt;|t| or the p-value of the feature signifies how significant a feature is.
  - In Linear Regression, the Null Hypothesis for a feature &amp; target variable is that there is no relationship between target &amp; the feature.
  - Alternate Hypothesis states that there is no relationship between target and feature.
  - A p-value of less than 0.05 signifies that the features **Feature 1, 4, 13** are significant predictors and are related with the target variable.
  - However, **Feature 15** has a p-value of 0.064 which makes it insignificant for the given target variable.
  - On testing dataset, the RMSE value comes to be really small i.e. 0.000244 which is a characteristic of a good model.
  - The r-squared score during testing comes out to be 0.635 which is decently good given its a simple model that we created without regularization using the limited training dataset that we had.
  - Also, I utilize the residuals of the model to validate the regression assumptions and quality of the OLS model created.
    - By plotting a scatter plot between residuals and the predicted value, the plot seems to be reasonably random.
    - By visualizing the probability plot, There is a good fit of observed and expected thus indicating that normality is a reasonable approximation.

- I also use Ridge Regression to create a model.
  - Ridge regression is a technique to reduce model complexity &amp; prevent over-fitting which may result from simple linear regression.
  - I first find out the best value of **alpha (penalty term)** using GridSearch &amp; cross validation by using **neg\_mean\_squared\_error** as the scoring criterion.
  - Based on the data and scoring criteria, the best value of alpha comes out to be 20.
  - I fit the data on the best estimator from GridSearch.
  - On testing dataset, the RMSE value is similar to the OLS model.
  - The R-squared score during testing comes out to be a bit better as compared to the OLS model.

## Do you modify/transform or remove from the feature list. If so why?

- Yes, I have removed a lot of columns due to a high value of Multicollinearity exhibited by them. Here are some values and the columns dropped
  - Dropping &#39;Feature 11&#39; with VIF value : 3942.998747788726
  - Dropping &#39;Feature 6&#39; with VIF value : 1091.6086219503093
  - Dropping &#39;Feature 16&#39; with VIF value : 974.4334324017261
  - Dropping &#39;Feature 12&#39; with VIF value : 300.6222736376661
  - Dropping &#39;Feature 10&#39; with VIF value : 202.678484487964
  - Dropping &#39;Feature 7&#39; with VIF value : 141.40042641429838
  - Dropping &#39;Feature 9&#39; with VIF value : 106.76674138525257
  - Dropping &#39;Feature 3&#39; with VIF value : 86.02264209663117
  - Dropping &#39;Feature 8&#39; with VIF value : 67.31897419929773
  - Dropping &#39;Feature 14&#39; with VIF value : 58.7676190989684
  - Dropping &#39;Feature 17&#39; with VIF value : 13.797398636780125
  - Dropping &#39;Feature 18&#39; with VIF value : 8.976374644362185
  - Dropping &#39;Feature 2&#39; with VIF value : 5.840934708316343

## How did you evaluate the model for the accuracy?

- Since, the target variable is continuous in nature, the given problem is a **regression** problem and not a classification problem.
- The metrics used in case of Linear Regression is **Root Mean Square Error (RMSE)** which is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; A RMSE score closer to 0 means the model is well fit.

## Any other insights you have gained during the building of this model.

- A lot of features were heavily correlated with the target variable.
- However, those features were correlated with other features as well.
- There was a lot of multicollinearity in the dataset which if weren&#39;t tackled would have given unreliable weights for the features.
