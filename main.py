import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ChromeDriver yolunu belirtin
driver_path = "resources/driver/chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Hedef URL
url = "https://www.epey.com/laptop/"
driver.get(url)

# Sayfa numarasını başlat
page_number = 1
max_pages = 5  # Çekilecek maksimum sayfa sayısı

# Verileri tutacak boş listeler
laptop_names = []
prices = []
inches = []
ram_sizes = []
processors = []
ratings = []

while page_number <= max_pages:
    try:
        # Sayfa HTML içeriğini al
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "urunadi")))

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        # Verileri çıkart
        laptops = soup.find_all("a", class_="urunadi")
        prices_html = soup.find_all("li", class_="fiyat cell")
        inches_html = soup.find_all("li", class_="ozellik ozellik1426 cell")
        ram_html = soup.find_all("li", class_="ozellik ozellik1027 cell")
        processors_html = soup.find_all("li", class_="ozellik ozellik1364 cell")
        ratings_html = soup.find_all("div", class_="circliful")

        for i in range(len(laptops)):

            # Fiyat bilgisini güvenli bir şekilde alalım
            price = "Fiyat Bulunamadı"
            if prices_html[i].find("a"):
                price_tag = prices_html[i].find("a")
                if price_tag:
                    price = price_tag.get_text(strip=True).split("TL")[0] + " TL"
            else:
                if prices_html[i].find("strong"):
                    price = prices_html[i].find("strong").get_text(strip=True)

            price = re.sub(r'[^\d,]', '', price)
            try:
                price = float(price.replace(",", "."))  # Virgül yerine nokta kullanarak sayıya dönüştür
            except ValueError:
                continue
            laptop_names.append(laptops[i].get_text(strip=True))
            prices.append(price)

            inch = inches_html[i].get_text(strip=True) if i < len(inches_html) else "İnç Bilgisi Bulunamadı"
            ram_size = ram_html[i].get_text(strip=True) if i < len(ram_html) else "RAM Bilgisi Bulunamadı"
            processor = processors_html[i].get_text(strip=True) if i < len(
                processors_html) else "İşlemci Bilgisi Bulunamadı"
            rating = ratings_html[i].find("span", class_="circle-text").get_text(strip=True) if i < len(
                ratings_html) else "Puan Bilgisi Bulunamadı"

            inches.append(inch)
            ram_sizes.append(ram_size)
            processors.append(processor)
            ratings.append(rating)

        print(f"Sayfa {page_number} verileri çekildi.")

        # Sonraki sayfa butonunu kontrol et
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "ileri"))
            )
            next_button.click()
            page_number += 1
            time.sleep(3)
        except Exception:
            print("Sonraki sayfa bulunamadı veya yüklenemedi.")
            break

    except Exception as e:
        print(f"Hata oluştu: {e}")
        break

# Veriyi bir DataFrame'e dönüştür
df = pd.DataFrame({
    'Laptop Adı': laptop_names,
    'Fiyat': prices,
    'İnç': inches,
    'RAM': ram_sizes,
    'İşlemci': processors,
    'Puan': ratings
})

# Fiyat Aralığını kategorilere ayırma
price_bins = [0, 100000, 140000, 170000, 200000, float('inf')]
price_labels = ['0-100k', '100k-140k', '140k-170k', '170k-200k', '200k+']
df['Fiyat Aralığı'] = pd.cut(df['Fiyat'], bins=price_bins, labels=price_labels)

# İnç değerini sayısal verilere dönüştürme (örneğin '18.0 İnç' → 18.0)
df['İnç'] = df['İnç'].str.extract('(\\d+\\.?\\d*)').astype(float)

# RAM'leri sayısal verilere dönüştürme (örneğin '8 GB' → 8)
df['RAM'] = df['RAM'].str.extract('(\\d+)').astype(float)

# İşlemciyi sayısal verilere dönüştürme
label_encoder = LabelEncoder()
df['İşlemci'] = label_encoder.fit_transform(df['İşlemci'])

# Puanları sayısal verilere dönüştürme
df['Puan'] = pd.to_numeric(df['Puan'], errors='coerce')

# Eksik verileri temizleme (NaN değerlerini ortalama ile dolduruyoruz)
df['İnç'] = df['İnç'].fillna(df['İnç'].mean())
df['RAM'] = df['RAM'].fillna(df['RAM'].mean())
df['İşlemci'] = df['İşlemci'].fillna(df['İşlemci'].mode()[0])
df['Puan'] = df['Puan'].fillna(df['Puan'].mean())

# Özellikler (X) ve hedef değişkeni (y) ayırma
X = df[['İnç', 'RAM', 'İşlemci', 'Puan']]  # Kullanacağınız özellikler
y = df['Fiyat Aralığı']  # Hedef değişken olarak Fiyat Aralığı kullanıyoruz

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bias-Variance Analizi için sonuçları tutacak listeler
biases = []
variances = []
total_errors = []

# Modelin karmaşıklığını değiştirme (C parametresi ile)
C_values = [0.1, 1, 10, 100, 1000]

for C_value in C_values:
    model = SVC(C=C_value, kernel='rbf', gamma=0.1)
    model.fit(X_train_scaled, y_train)

    # Tahminler
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Bias Hesaplama: Ortalama hata
    bias = mean_squared_error(y_test, y_pred_test)  # Bias^2
    # Varyans Hesaplama: Test tahminlerinin varyansı
    variance = np.var(y_pred_test)  # Variance
    # Toplam Hata
    total_error = bias + variance  # Total Error

    biases.append(bias)
    variances.append(variance)
    total_errors.append(total_error)

# Bias-Variance Tradeoff Grafiği
plt.figure(figsize=(10, 6))
plt.plot(C_values, biases, label="Bias", color="blue", linestyle="--")
plt.plot(C_values, variances, label="Variance", color="red", linestyle="-.")
plt.plot(C_values, total_errors, label="Total Error", color="green", linestyle="-")
plt.xscale('log')
plt.xlabel("C Parametresi (Model Karmaşıklığı)")
plt.ylabel("Hata")
plt.title("Bias-Variance Tradeoff (SVC Modeli)")
plt.legend()
plt.grid(True)
plt.show()

# Tarayıcıyı kapat
driver.quit()