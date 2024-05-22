import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Öznitelikleri Kodlamak için Yardımcı Fonksiyon
def encode_column(dataframe, column_name, encoding_dict):
    dataframe[column_name] = dataframe[column_name].map(encoding_dict)
    return dataframe

# Veri Setini Yükleme
data_cars = pd.read_csv('Car Sales.xlsx - car_data.csv')

# Gereksiz Sütunları Çıkarma
data_cars = data_cars.drop(['Car_id', 'Date', 'Customer Name', 'Dealer_Name', 'Dealer_No', 'Phone', 'Dealer_Region'], axis=1)

# Bağımsız Değişkenler (Öznitelikler) ve Bağımlı Değişken (Hedef)
X = data_cars.drop('Gender', axis=1)
Y = data_cars['Gender']

# Veri Ölçeklendirme İşlemleri
scaler = MinMaxScaler()
X[['Annual Income', 'Price']] = scaler.fit_transform(X[['Annual Income', 'Price']])

# Model ve Marka İsimlerini Kodlama
models = ['Expedition', 'Durango', 'Eldorado', 'Celica', 'TL', 'Diamante', 'Corolla',
 'Galant', 'Malibu', 'Escort', 'RL', 'Pathfinder', 'Grand Marquis', '323i',
 'Sebring Coupe', 'Forester', 'Accent', 'Land Cruiser', 'Accord', '4Runner',
 'I30', 'A4', 'Carrera Cabrio', 'Jetta', 'Viper', 'Regal', 'LHS', 'LW', '3000GT',
 'SLK230', 'Civic', 'S-Type', 'S40', 'Mountaineer', 'Park Avenue',
 'Montero Sport', 'Sentra', 'S80', 'Lumina', 'Bonneville', 'C-Class', 'Altima',
 'DeVille', 'Stratus', 'Cougar', 'SW', 'C70', 'SLK', 'Tacoma', 'M-Class', 'A6',
 'Intrepid', 'Sienna', 'Eclipse', 'Contour', 'Town car', 'Focus', 'Mustang',
 'Cutlass', 'Corvette', 'Impala', 'Cabrio', 'Dakota', '300M', '328i', 'Bravada',
 'Maxima', 'Ram Pickup', 'Concorde', 'V70', 'Quest', 'ES300', 'SL-Class',
 'Explorer', 'Prizm', 'Camaro', 'Outback', 'Taurus', 'Cavalier', 'GS400',
 'Monte Carlo', 'Sonata', 'Sable', 'Metro', 'Voyager', 'Cirrus', 'Avenger',
 'Odyssey', 'Intrigue', 'Silhouette', '5-Sep', '528i', 'LS400', 'Aurora',
 'Breeze', 'Beetle', 'Elantra', 'Continental', 'RAV4', 'Villager', 'S70', 'LS',
 'Ram Van', 'S-Class', 'E-Class', 'Grand Am', 'SC', 'Passat', 'Xterra',
 'Frontier', 'Crown Victoria', 'Camry', 'Navigator', 'CL500', 'Escalade', 'Golf',
 'Ranger', 'Prowler', 'Windstar', 'GTI', 'Passport', 'Boxter', 'LX470', 'CR-V',
 'Sunfire', 'Caravan', 'Ram Wagon', 'Neon', 'Wrangler', 'Integra', 'Grand Prix',
 'Grand Cherokee', 'F-Series', 'A8', 'Mystique', '3-Sep', 'Cherokee',
 'Carrera Coupe', 'Catera', 'Seville', 'CLK Coupe', 'LeSabre', 'Sebring Conv.',
 'GS300', 'Firebird', 'V40', 'Montero', 'Town & Country', 'SL', 'Alero', 'Mirage',
 'Century', 'RX300', 'Avalon']

encoding_for_models = {model: i for i, model in enumerate(models)}
X['Model'] = X['Model'].map(encoding_for_models)

brands = ["Dodge", "Cadillac", "Toyota", "Acura", "Mitsubishi", "Chevrolet", "Ford", 
          "Nissan", "Mercury", "BMW", "Chrysler", "Subaru", "Hyundai", "Honda", 
          "Infiniti", "Audi", "Porsche", "Volkswagen", "Buick", "Saturn", "Mercedes-B", 
          "Jaguar", "Volvo", "Pontiac", "Lincoln", "Oldsmobile", "Lexus", "Plymouth", 
          "Saab", "Jeep"]

encoding_for_brands = {brand: i for i, brand in enumerate(brands)}
X['Company'] = X['Company'].map(encoding_for_brands)

# Diğer Kategorik Değişkenleri Kodlama
class_encoding = {"Manual": 0, "Auto": 1}
X = encode_column(X, 'Transmission', class_encoding)

class_encoding_engine = {"Overhead Camshaft": 0, "DoubleA OverHead Camshaft": 1}
X = encode_column(X, 'Engine', class_encoding_engine)

class_encoding_color = {"Black": 2, "Red": 1, "Pale White": 0}
X = encode_column(X, 'Color', class_encoding_color)

class_encoding_body_style = {"Hatchback": 0, "Sedan": 1, "SUV": 2, "Passenger": 3, "Hardtop": 4}
X = encode_column(X, 'Body Style', class_encoding_body_style)

# Veri Setini Eğitim ve Test Olarak Ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# RandomForest Modelini Oluşturma ve Eğitme
rfr = RandomForestClassifier(random_state=42)
rfr.fit(X_train, Y_train)

# Test Verisi Üzerinde Tahmin Yapma
y_pred = rfr.predict(X_test)

# Başarı Metriklerini Hesaplama ve Yazdırma
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: %{accuracy * 100:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Sınıflandırma Raporu
print("Classification Report:")
print(classification_report(Y_test, y_pred, target_names=['Female', 'Male']))

# Yeni Veri Noktasında Tahmin Yapma
new_data = {'Annual Income': [2000], 'Price': [10000], 'Company': ['Ford'], 'Model': ['Escort'], 'Engine': ['DoubleA OverHead Camshaft'], 
            'Transmission': ['Auto'], 'Color': ['Red'], 'Body Style': ['Passenger']}

new_df = pd.DataFrame(new_data)

# Yeni Veri Noktasını Kodlama
new_df['Company'] = new_df['Company'].map(encoding_for_brands)
new_df['Model'] = new_df['Model'].map(encoding_for_models)
new_df = encode_column(new_df, 'Engine', class_encoding_engine)
new_df = encode_column(new_df, 'Transmission', class_encoding)
new_df = encode_column(new_df, 'Color', class_encoding_color)
new_df = encode_column(new_df, 'Body Style', class_encoding_body_style)

# Ölçeklendirme İşlemini Uygulama
new_df[['Annual Income', 'Price']] = scaler.transform(new_df[['Annual Income', 'Price']])

# Eğitimde Kullanılan Özniteliklerle Uyumlu Hale Getirme
for feature in X_train.columns:
    if feature not in new_df.columns:
        new_df[feature] = 0

# Sütunları Doğru Sırayla Yeniden Düzenleme
new_df = new_df[X_train.columns]

# Tahmin Yapma
prediction = rfr.predict(new_df)
prediction_label = 'Female' if prediction == 0 else 'Male'

print(f"Tahmin edilen cinsiyet: {prediction_label}")