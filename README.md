# ML-data-analysis

# Car Sales Gender Prediction

# This project utilizes machine learning to predict the gender of car buyers based on various features of car sales data. The RandomForestClassifier from the sklearn library is used to train and predict gender based on the data.

## Dependencies

# This project requires the following Python libraries:
#
# - Pandas
# - NumPy
# - Matplotlib
# - Seaborn
# - scikit-learn

# You can install these dependencies using pip:
#
# ```bash
# pip install pandas numpy matplotlib seaborn scikit-learn
# ```

## Project Files

# - `car_sales_prediction.py`: The main Python script that contains all the code for preprocessing data, training the RandomForest model, and making predictions.
#
# - `Car Sales.xlsx - car_data.csv`: The dataset file which should be placed in the same directory as the script.

## Data Preprocessing

# The script performs several preprocessing steps on the car sales data:
#
# 1. Drops unnecessary columns that are not used in the prediction.
# 2. Scales numerical features like 'Annual Income' and 'Price' using MinMaxScaler.
# 3. Encodes categorical variables like 'Company', 'Model', 'Engine', 'Transmission', 'Color', and 'Body Style' into numerical values.
# 4. Splits the data into training and testing sets.

## Model Training

# The script trains a RandomForestClassifier with the training data. It also evaluates the model on the test data and prints the accuracy and a confusion matrix.

## Prediction

# The script demonstrates how to encode and scale new data points for making predictions using the trained model. It prints out the predicted gender based on the model.

## Usage

# To run this script, ensure that the dataset is in the correct location and simply execute the Python file:
#
# ```bash
# python car_sales_prediction.py
# ```

## Output

# The script will output:
#
# - The accuracy of the model on the test data.
# - A confusion matrix and classification report.
# - Predictions for any new data points provided in the script.
