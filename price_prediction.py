from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

data_cars = pd.read_csv('Car Sales.xlsx - car_data.csv')


# Printing the dataset

data_cars = data_cars.drop(['Gender','Car_id','Date','Customer Name','Annual Income','Dealer_No','Phone'],axis=1)



#Train test splitting
X = data_cars.drop('Price',axis=1)
Y = data_cars['Price']
print(X)
print(Y)
####

# OneHotEncoding

columns_to_encode = ['Dealer_Name', 'Model','Dealer_Region']

# Initialize an empty DataFrame to store the encoded columns
encoded_columns_df = pd.DataFrame()

# Loop through each column and apply one-hot encoding
for column in columns_to_encode:
    # Create a OneHotEncoder instance
    oh_encoder = OneHotEncoder()
    
    # Extract the column you want to encode and reshape it to a 2D array
    column_to_encode = X[[column]]
    
    # Fit and transform the column
    encoded_column = oh_encoder.fit_transform(column_to_encode)
    
    # Convert the sparse matrix to a dense array for easier handling
    encoded_column_array = encoded_column.toarray()
    
    # Get the feature names for the encoded columns
    encoded_column_names = oh_encoder.get_feature_names_out([column])
    
    # Create a DataFrame with the encoded column
    encoded_df = pd.DataFrame(encoded_column_array, columns=encoded_column_names)
    
    # Concatenate the encoded DataFrame with the previous ones
    encoded_columns_df = pd.concat([encoded_columns_df, encoded_df], axis=1)

# Drop the original columns from the DataFrame
data_cars_encoded = X.drop(columns=columns_to_encode)

# Concatenate the encoded DataFrame with the original DataFrame
X_encoded = pd.concat([data_cars_encoded, encoded_columns_df], axis=1)


# LabelEncoding
markalar = [
    "Dodge", "Cadillac", "Toyota", "Acura", "Mitsubishi", "Chevrolet","Ford", 
    "Nissan", "Mercury", "BMW", "Chrysler", "Subaru", "Hyundai", 
    "Honda", "Infiniti", "Audi", "Porsche", "Volkswagen", "Buick", 
    "Saturn", "Mercedes-B", "Jaguar", "Volvo", "Pontiac", "Lincoln", 
    "Oldsmobile", "Lexus", "Plymouth", "Saab", "Jeep"
]

# Özel label encoding değerleri
ozel_label_encoding = {
    "Porsche": 30,
    "BMW": 29,
    "Mercedes-B": 28,
    "Dodge": 27,
    "Cadillac": 26,
    "Lexus": 25
}

# Geri kalan markalar
kalan_markalar = [marka for marka in markalar if marka not in ozel_label_encoding]

# Geri kalan markalar için rastgele benzersiz label encoding değerleri oluşturma
random.seed(42)  # Rastgeleliği kontrol etmek için sabit bir seed kullanıyoruz
max_value = len(markalar)
available_values = list(range(7, max_value + 1))
random.shuffle(available_values)

# Kalan markalara rastgele değerler atama
rastgele_label_encoding = {marka: available_values[i] for i, marka in enumerate(kalan_markalar)}

# Tüm label encoding değerlerini birleştirme
tum_label_encoding = {**ozel_label_encoding, **rastgele_label_encoding}

# Markaları sıralı şekilde label encoding ile dönüştürme
label_encoded_markalar = [tum_label_encoding[marka] for marka in markalar]


X_encoded['Company'] = X_encoded['Company'].map(tum_label_encoding)


class_encoding = {
    "Manual": 0,
    "Auto": 1
}

# Class sütununa label encoding uygulama
X_encoded['Transmission'] = X_encoded['Transmission'].map(class_encoding)


class_encoding_engine = {
    "Overhead Camshaft": 0,
    "DoubleA OverHead Camshaft": 1
}
X_encoded['Engine'] = X_encoded['Engine'].map(class_encoding_engine)

class_encoding_color = {
    "Black" : 2,
    "Red" : 1,
    "Pale White" : 0
}
X_encoded['Color'] = X_encoded['Color'].map(class_encoding_color)

class_encoding_body_style = {
    "Hatchback" : 0,
    "Sedan" : 1,
    "SUV" : 2,
    "Passenger" : 3,
    "Hardtop" : 4
}

X_encoded['Body Style'] = X_encoded['Body Style'].map(class_encoding_body_style)

unique_body_styles = X_encoded['Body Style'].unique()
missing_values = [style for style in unique_body_styles if style not in class_encoding_body_style]

print("These Body Style values are not in the encoding dictionary:", missing_values)



X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.20, random_state=42)

print(X_train)
print(Y_train)
# ML algorithm to apply on the dataset
model = RandomForestRegressor()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate MAE
mae = mean_absolute_error(Y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate R-squared
r2 = r2_score(Y_test, y_pred)
print(f'R-squared: {r2}')

print('The accuracy for this model is : % {}'.format(round(r2*100,2)))

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Train Data Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(Y_train, Y_train_pred, alpha=0.5, color='blue', edgecolors='k', label='Train Data')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Data: Actual vs Predicted Price')
plt.legend()
plt.grid(True)

# Scatter plot for Test Data: Actual vs Predicted Price
plt.subplot(1, 2, 2)
plt.scatter(Y_test, Y_test_pred, alpha=0.5, color='red', edgecolors='k', label='Test Data')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Test Data: Actual vs Predicted Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()