import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

# Fungsi untuk mengekstrak fitur GLCM dari gambar
def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    
    # Convert RGBA to RGB if needed
    if img.shape[-1] == 4:  # If image has 4 channels (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Convert to grayscale using cv2 instead of skimage
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    ASM = graycoprops(glcm, 'ASM')
    
    # Mengembalikan hasil fitur sebagai array
    features = [np.mean(contrast), np.mean(dissimilarity), np.mean(homogeneity), np.mean(energy), np.mean(correlation), np.mean(ASM)]
    return features

# Fungsi untuk memproses gambar dalam folder (latih atau uji)
def process_images_in_folder(folder_path):
    features_list = []
    file_names = []
    
    # Menyimpan nama file dan fitur-fitur yang diekstrak
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Menangani gambar JPG dan PNG
            img_path = os.path.join(folder_path, filename)
            features = extract_glcm_features(img_path)
            features_list.append(features)
            file_names.append(filename)

    # Mengonversi daftar fitur menjadi DataFrame untuk analisis lebih lanjut
    df_features = pd.DataFrame(features_list, columns=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'])
    df_features['filename'] = file_names
    return df_features

# Folder tempat gambar berada
train_folder = 'dataset/train/'
test_folder = 'dataset/test/'

# Memproses data latih
df_train = process_images_in_folder(train_folder)

# Menampilkan fitur pertama dari data latih untuk memastikan ekstraksi berhasil
print("Data Latih:")
print(df_train.head())

# Menggunakan K-Means untuk klasterisasi (2 klaster: busuk dan tidak busuk) pada data latih
kmeans = KMeans(n_clusters=2, random_state=0)
df_train['cluster'] = kmeans.fit_predict(df_train[['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']])

# Menentukan label berdasarkan klaster (asumsi klaster 0 adalah busuk, klaster 1 adalah tidak busuk)
df_train['label'] = df_train['cluster'].apply(lambda x: 'Busuk' if x == 0 else 'Tidak Busuk')

# Menampilkan hasil klasterisasi data latih
print("Hasil Klasterisasi Data Latih:")
print(df_train)

# Menyimpan hasil klasterisasi data latih ke CSV
df_train.to_csv('glcm_train_results.csv', index=False)

# Memproses data uji
df_test = process_images_in_folder(test_folder)

# Menampilkan fitur pertama dari data uji untuk memastikan ekstraksi berhasil
print("\nData Uji:")
print(df_test.head())

# Menggunakan K-Means untuk klasterisasi (menggunakan model latih) pada data uji
df_test['cluster'] = kmeans.predict(df_test[['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']])

# Menentukan label berdasarkan klaster (asumsi klaster 0 adalah busuk, klaster 1 adalah tidak busuk)
df_test['label'] = df_test['cluster'].apply(lambda x: 'Busuk' if x == 0 else 'Tidak Busuk')

# Menampilkan hasil klasterisasi data uji
print("Hasil Klasterisasi Data Uji:")
print(df_test)

# Menyimpan hasil klasterisasi data uji ke CSV
df_test.to_csv('glcm_test_results.csv', index=False)

# Visualisasi hasil klasterisasi data latih dan uji menggunakan PCA
pca = PCA(n_components=2)

# PCA untuk data latih
pca_result_train = pca.fit_transform(df_train[['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_result_train[:, 0], pca_result_train[:, 1], c=df_train['cluster'], cmap='viridis')
plt.title("PCA Data Latih")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# PCA untuk data uji
pca_result_test = pca.transform(df_test[['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']])
plt.subplot(1, 2, 2)
plt.scatter(pca_result_test[:, 0], pca_result_test[:, 1], c=df_test['cluster'], cmap='viridis')
plt.title("PCA Data Uji")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()
plt.show()
