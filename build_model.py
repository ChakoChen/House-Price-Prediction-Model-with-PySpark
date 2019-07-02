# build_model.py -- split data without data leakage, do feature engineering for Random Forest Regression, build a model, interprete model predictions and save and load model 

import seaborn as sns
import matplotlib.pyplot as plt

# Add PySpark to sys.path at runtime
import findspark
findspark.init()

# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession
# Create or get a SparkSession
spark = SparkSession.builder.getOrCreate()


def read_data():
    """Read in CSV as Spark DataFrame (first row as column names)"""
    df = spark.read.csv('../../data/2017_StPaul_MN_Real_Estate.csv', header=True, inferSchema=True)
    return df


def feature_engineering_RFR():
    """Run through feature engineering for Random Forest Regression"""
    # (1) Dropping columns with low observations
    """After doing a lot of feature engineering it's a good idea to take a step back and look at what you've created. If you've used some automation techniques on your categorical features like exploding or One Hot Encoding you may find that you now have hundreds of new binary features. While the subject of feature selection is material for a whole other course but there are some quick steps you can take to reduce the dimensionality of your data set.
    In this exercise, we are going to remove columns that have less than 30 observations. 30 is a common minimum number of observations for statistical significance. Any less than that and the relationships cause overfitting because of a sheer coincidence!"""
    # Load data
    df = read_data()

    # Missing feature engineering
    #...

    # Get binary column names (all extracted categorical columns)
    binary_cols = newly_extracted_categorical_columns

    obs_threshold = 30
    cols_to_remove = list()
    # Inspect first 10 binary columns in list
    for col in binary_cols[0:10]:
        # Count the number of 1 values in the binary column
        obs_count = df.agg({col: 'sum'}).collect()[0][0]
        # If less than our observation threshold, remove
        if obs_count < obs_threshold:
            cols_to_remove.append(col)

    # Drop columns and print starting and ending dataframe shapes
    new_df = df.drop(*cols_to_remove)

    print('Rows: '+str(df.count())+' Columns: '+str(len(df.columns))) # 5000, 253
    print('Rows: '+str(new_df.count())+' Columns: '+str(len(new_df.columns))) # 5000, 250

    # (2) Naively handling missing and categorical values
    """Random Forest Regression is robust enough to allow us to ignore many of the more time consuming and tedious data preparation steps. While some implementations of Random Forest handle missing and categorical values automatically, PySpark's does not. The math remains the same however so we can get away with some naive value replacements.

    For missing values since our data is strictly positive, we will assign -1. The random forest will split on this value and handle it differently than the rest of the values in the same feature.

    For categorical values, we can just map the text values to numbers and again the random forest will appropriately handle them by splitting on them. In this example, we will dust off pipelines from Introduction to PySpark to write our code more concisely. Please note that the exercise will start by displaying the dtypes of the columns in the dataframe, compare them to the results at the end of this exercise."""
    # Replace missing values
    df = df.fillna(-1, subset=['WALKSCORE', 'BIKESCORE'])

    categorical_cols = ['CITY','LISTTYPE','SCHOOLDISTRICTNUMBER','POTENTIALSHORTSALE','STYLE','ASSUMABLEMORTGAGE','ASSESSMENTPENDING']

    # (a) Create list of StringIndexers using list comprehension
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_IDX").setHandleInvalid("keep") for col in categorical_cols]

    # Create pipeline of indexers
    indexer_pipeline = Pipeline(stages=indexers)
    # Fit and Transform the pipeline to the original data
    df_indexed = indexer_pipeline.fit(df).transform(df)

    # Clean up redundant columns no longer needed
    df_indexed = df_indexed.drop(*categorical_cols)

    # Inspect data transformations
    print(df_indexed.dtypes)

    # (b) One hot encode indexed values
    encoders = [OneHotEncoder(inputCol=col+'_IDX', outputCol=col+'_Vec') for col in categorical_cols]

    # Create pipeline of indexers
    encoder_pipeline = Pipeline(stages=encoders)

    # Fit and Transform the pipeline to the data
    df = encoder_pipeline.fit(df).transform(df)


def split_data():
    """
    Use VectorAssembler to put all features in one column, then 
    split data into train and test without data leakage from future
    """
    # (0) PySpark ML algorithms require all of the features to be provided in a single column of type vector.
    from pyspark.ml.feature import VectorAssembler

    # Load data
    df = read_data()  # WARNING! feature engineering needed... 

    # Define the columsn to be converted to vectors (slimmed down features)
    feature_cols = ['SQFT_TOTAL','TAXES','LIVINGAREA','SQFTABOVEGROUND','BATHSTOTAL','YEARBUILT','FIREPLACES','BATHSHALF','walkscore']  

    # Create teh vector assembler transformer
    vec = VectorAssembler(inputCols=feature_cols, outputCol='features')

    # Apply the vector transformer to data ('features' column will be added)
    df = vec.transform(df)

    # (1) Creating time splits
    """In the video, we learned why splitting data randomly can be dangerous for time series as data from the future can cause overfitting in our model. Often with time series, you acquire new data as it is made available and you will want to retrain your model using the newest data. In the video, we showed how to do a percentage split for test and training sets but suppose you wish to train on all available data except for the last 45days which you want to use for a test set."""
    from datetime import timedelta
    from pyspark.sql.functions import datediff, to_date, lit

    def train_test_split_date(df, split_col, test_days=45):
        """Calculate the date to split test and training sets"""
        # Find how many days our data spans
        max_date = df.agg({split_col: 'max'}).collect()[0][0]
        min_date = df.agg({split_col: 'min'}).collect()[0][0]
        # Subtract an integer number of days from the last date in dataset
        split_date = max_date - timedelta(days=test_days)
        return split_date
 
    # Convert OFFMKTDATE to date
    df = df.withColumn('OFFMKTDATE', to_date('OFFMKTDATE', 'MM/dd/yyyy HH:mm')) 

    # Find the date to use in spitting test and train
    split_date = train_test_split_date(df, 'OFFMKTDATE')

    # Create Sequential Test and Training Sets
    train_df = df.where(df['OFFMKTDATE'] < split_date)
    test_df = df.where(df['OFFMKTDATE'] >= split_date).where(df['LISTDATE'] <= split_date)

    # (2) Adjusting time features (a test to get 'right' DAYSONMARKET)
    """
    We have mentioned throughout this course some of the dangers of leaking information to your model during training. Data leakage will cause your model to have very optimistic metrics for accuracy but once real data is run through it the results are often very disappointing.
    In this exercise, we are going to ensure that DAYSONMARKET only reflects what information we have at the time of predicting the value. I.e., if the house is still on the market, we don't know how many more days it will stay on the market. We need to adjust our test_df to reflect what information we currently have as of 2017-12-10.
    NOTE: This example will use the lit() function. This function is used to allow single values where an entire column is expected in a function call.
    """
    # Recalculate DAYSONMARKET from what we know on our split date
    test_df = test_df.withColumn('DAYSONMARKET', datediff(lit(split_date), 'LISTDATE'))

    # Review the difference
    test_df[['LISTDATE', 'OFFMKTDATE', 'DAYSONMARKET']].show()

    """
    # Convert string to a pyspark date by calling the literal function first
    split_date = to_date(lit('2017-12-10'))

    # Convert LISTDATE to date
    df = df.withColumn('LISTDATE', to_date('LISTDATE', 'MM/dd/yyyy HH:mm')) 

    # Create Sequential Test set
    test_df = df.where(df['OFFMKTDATE']>=split_date).where(df['LISTDATE']<=split_date).select(['OFFMKTDATE', 'DAYSONMARKET', 'LISTDATE'])

    # Create a copy of DAYSONMARKET to review later
    test_df = test_df.withColumn('DAYSONMARKET_Original', test_df['DAYSONMARKET'])

    # Recalculate DAYSONMARKET from what we know on our split date
    test_df = test_df.withColumn('DAYSONMARKET', datediff(split_date, 'LISTDATE'))

    # Review the difference
    test_df[['LISTDATE', 'OFFMKTDATE', 'DAYSONMARKET_Original', 'DAYSONMARKET']].show()
    # +-------------------+-------------------+---------------------+------------+
    # |           LISTDATE|         OFFMKTDATE|DAYSONMARKET_Original|DAYSONMARKET|
    # +-------------------+-------------------+---------------------+------------+
    # |2017-10-06 00:00:00|2018-01-24 00:00:00|                  110|          65|
    # |2017-09-18 00:00:00|2017-12-12 00:00:00|                   82|          83|
    # |2017-11-07 00:00:00|2017-12-12 00:00:00|                   35|          33|
    # ...
    """
    """Thinking critically about what information would be available at the time of prediction is crucial in having accurate model metrics and saves a lot of embarassment down the road if decisions are being made based off your results!"""



def build_model()
    """Build regressin models, evaluate and compare algorithms"""
    # (1.1) Training a Random Forest Regressor (RFR) model
    """One of the great things about PySpark ML module is that most algorithms can be tried and tested without changing much code. Random Forest Regression is a fairly simple ensemble model, using bagging to fit. Another tree based ensemble model is Gradient Boosted Trees which uses a different approach called boosting to fit. In this exercise let's train a GBTRegressor."""
    from pyspark.ml.regression import RandomForestRegressor, GBTRegressor

    # initialize a RFR model with columns to utilize
    rfr = RandomForestRegressor(featuresCol="features",
                                labelCol="SALESCLOSEPRICE",
                                predictionCol="Prediction_Price",
                                seed=42)
    # train the RFR model
    rfr_model = rf.fit(train_df)

    # make predictions
    rfr_predictions = rfr_model.transform(test_df)

    # (1.2) Train a Gradient Boosted Trees (GBT) model
    gbt = GBTRegressor(featuresCol='features',
                               labelCol='SALESCLOSEPRICE',
                               predictionCol="Prediction_Price",
                               seed=42)
    # train the GBT model
    gbt_model = gbt.fit(train_df)
    
    # make predictions
    gbt_predictions = gbt_model.transform(test_df)

    # (2) Inspect results
    rfr_predictions.select('Prediction_Price','SALESCLOSEPRICE').show()
    gbt_predictions.select('Prediction_Price','SALESCLOSEPRICE').show()


    # (2) Evaluate and compare algorithms
    """Now that we've created a new model with GBTRegressor its time to compare it against our baseline of RandomForestRegressor. To do this we will compare the predictions of both models to the actual data and calculate RMSE and R^2."""
    from pyspark.ml.evaluation import RegressionEvaluator

    # Select columns to compute test error
    evaluator = RegressionEvaluator(labelCol='SALESCLOSEPRICE', 
                                    predictionCol='Prediction_Price')
    # Dictionary of model predictions to loop over
    models = {'Gradient Boosted Trees': gbt_predictions, 'Random Forest Regression': rfr_predictions}
    for key, preds in models.items():
        # Create evaluation metrics
        rmse = evaluator.evaluate(preds, {evaluator.metricName: 'rmse'})
        r2 = evaluator.evaluate(preds, {evaluator.metricName: 'r2'})
      
        # Print Model Metrics
        print(key + ' RMSE: ' + str(rmse))
        print(key + ' R^2: ' + str(r2))


def save_load_model():
    """Interpret model results, save and load model"""
    # (1) Interpreting results
    """It is almost always important to know which features are influencing your prediction the most. Perhaps its counterintuitive and that's an insight? Perhaps a hand full of features account for most of the accuracy of your model and you don't need to perform time acquiring or massaging other features.

    In this example we will be looking at a model that has been trained without any LISTPRICE information. With that gone, what influences the price the most?"""
    # Convert feature importances to a pandas column
    #importances = model.featureImportances.toArray()
    fi_df = pd.DataFrame(importances, columns=['importance'])

    # Convert list of feature names to pandas column
    fi_df['feature'] = pd.Series(feature_cols)

    # Sort the data based on feature importance
    fi_df.sort_values(by=['importance'], ascending=False, inplace=True)

    # Inspect Results
    fi_df.head(10)
    #     importance             feature
    # 36    0.256598          SQFT_TOTAL
    # 4     0.212320               TAXES
    # 6     0.166661          LIVINGAREA
    # ...

    # (2) Saving and loading models
    """Often times you may find yourself going back to a previous model to see what assumptions or settings were used when diagnosing where your prediction errors were coming from. Perhaps there was something wrong with the data? Maybe you need to incorporate a new feature to capture an unusual event that occurred?"""
    from pyspark.ml.regression import RandomForestRegressionModel

    # Save model
    model.save('rfr_no_listprice')

    # Load model
    loaded_model = RandomForestRegressionModel.load('rfr_no_listprice')


def main():
    #feature_engineering_RFR() ?
    #split_data()
    #build_model() ?
    #save_load_model() ?

if __name__ == '__main__':
    main()
