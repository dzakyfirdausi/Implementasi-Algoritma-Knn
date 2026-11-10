import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import time

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) **2)) # Mengikuti rumus d = √Σ(xi − yi)2

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2)) # Mengikuti rumus d = Σ|xi − yi|

def minkowski_distance(x1, x2, p):
    return (np.sum(np.abs(x1 - x2) ** p))**(1/p) # Mengikuti rumus d = (Σ|xi − yi|p)^(1/p)

def min_max_scaler(x_train, x_test=None):
    x_train = np.array(x_train, dtype=float)

    # Hitung parameter dari data latih
    min_scaler = np.min(x_train, axis=0)
    max_scaler = np.max(x_train, axis=0)

    # Akan menghindari pembagian 0 jjika semua nilai sama
    min_max = np.where(max_scaler - min_scaler == 0, 1, max_scaler - min_scaler)

    # Normalisasi data latih
    data_latih = (x_train - min_scaler) / min_max

    # Normalisasi data uji Jika data uji-nya ada
    data_tes = None
    if x_test is not None:
        x_test = np.array(x_test, dtype=float)
        data_tes = (x_test - min_scaler) / min_max

    return data_latih, data_tes, min_scaler, max_scaler

def z_score_scaler(x_train, x_test=None):
    x_train = np.array(x_train, dtype=float)

    # Hitung parameter data latih
    nilai_mean = np.mean(x_train, axis=0)
    nilai_std = np.std(x_train, axis=0)

    # Akan menghindari pembagian nol jika std = 0
    nilai_aman_std = np.where(nilai_std == 0, 1, nilai_std)

    # Normalisasi data latih
    data_latih = (x_train - nilai_mean) / nilai_aman_std

    # Normalisasi data uji jika ada
    data_tes = None
    if x_test is not None:
        x_test = np.array(x_test, dtype=float)
        data_tes = (x_test - nilai_mean) / nilai_aman_std

    return data_latih, data_tes, nilai_mean, nilai_std

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean', p=2, weights='uniform'):
        self.k = k
        self.metric = metric
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return euclidean_distance(a, b)
        elif self.metric == 'manhattan':
            return manhattan_distance(a, b)
        elif self.metric == 'minkowski':
            return minkowski_distance(a, b, self.p)
        else:
            raise ValueError(f"Metrik tidak dikenal: {self.metric}")

    def predict(self, X):
        X = np.array(X, dtype=float)
        predictions = []

        for x in X:
            # Hitung jarak ke seluruh data latih
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
            # Urutkan indeks berdasarkan jarak terkecil
            sorted_indices = np.argsort(distances)
            # Mengambil K tetangga terdekat
            k_indices = sorted_indices[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]
            if self.weights == 'uniform':
                unique_classes, counts = np.unique(k_nearest_labels, return_counts=True)
                max_count = np.max(counts)
                best_classes = unique_classes[counts == max_count]
                # Menggunakan Aturan tie-breaking: pilih kelas dari tetangga terdekat secara urut
                if len(best_classes) > 1:
                    for idx in k_indices:
                        if self.y_train[idx] in best_classes:
                            pred_class = self.y_train[idx]
                            break
                else:
                    pred_class = best_classes[0]
            elif self.weights == 'distance':
                # Mencegah division by zero jika jarak bernilai 0
                epsilon = 1e-5
                weights = 1.0 / (k_distances ** 2 + epsilon)
                class_weights = {}
                for label, weight in zip(k_nearest_labels, weights):
                    class_weights[label] = class_weights.get(label, 0.0) + weight
                pred_class = max(class_weights, key=class_weights.get)
            else:
                raise ValueError(f"Mode bobot tidak dikenal: {self.weights}")
            predictions.append(pred_class)
        return np.array(predictions)

if __name__ == "__main__":
    # Data Studi Kasus (Verifikasi)
    X_train_studi = [
        [85, 80], # A
        [60, 55], # B
        [75, 70], # C
        [50, 65], # D
        [90, 85], # E
        [65, 50], # F
        [70, 75], # G
        [55, 60]  # H
    ]
    y_train_studi = ['Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus']
    X_test_studi = [[68, 64]] # Mahasiswa X

    # Uji K = 3
    knn_k3 = KNNClassifier(k=3, metric='euclidean', weights='uniform')
    knn_k3.fit(X_train_studi, y_train_studi)
    print("Hasil Prediksi K=3:", knn_k3.predict(X_test_studi)[0]) # Target: Lulus

    # Uji K = 5
    knn_k5 = KNNClassifier(k=5, metric='euclidean', weights='uniform')
    knn_k5.fit(X_train_studi, y_train_studi)
    print("Hasil Prediksi K=5:", knn_k5.predict(X_test_studi)[0]) # Target: Tidak Lulus

    # Uji K = 5 dengan Weighted Distance (Target: Lulus)
    knn_k5_weighted = KNNClassifier(k=5, metric='euclidean', weights='distance')
    knn_k5_weighted.fit(X_train_studi, y_train_studi)
    print("Hasil Prediksi K=5 (Weighted):", knn_k5_weighted.predict(X_test_studi)[0])  # Target: Lulus



if __name__ == "__main__":
    # BAGIAN B & C: EKSPERIMEN & VALIDASI DATASET WINE (NIM ...35)
    dua_digit_nim = 35
    data = load_wine()
    X, y = data.data, data.target
    print(f"\n--- Menggunakan Dataset: Wine (Jumlah Data: {X.shape[0]}, Fitur: {X.shape[1]}) ---")

    # 1. Split data 80:20 secara stratified dengan random_state = 35
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=dua_digit_nim, stratify=y
    )

    # 2. Validasi dengan Scikit-learn (Bagian B)
    X_train_s, X_test_s, _, _ = min_max_scaler(X_train, X_test)
    my_knn = KNNClassifier(k=3, metric='euclidean', weights='uniform')
    my_knn.fit(X_train_s, y_train)
    y_pred_my = my_knn.predict(X_test_s)

    sk_knn = SklearnKNN(n_neighbors=3, metric='euclidean', weights='uniform')
    sk_knn.fit(X_train_s, y_train)
    y_pred_sk = sk_knn.predict(X_test_s)

    kesamaan = np.mean(y_pred_my == y_pred_sk) * 100
    print(f"[Bagian B] Kesamaan prediksi dengan Scikit-learn (K=3): {kesamaan:.2f}%")

    # 3. Eksperimen Kombinasi Parameter (Bagian C)
    k_values = [3, 5, 7, 9]
    metrics = ['euclidean', 'manhattan']
    scalers = {
        'Tanpa Normalisasi': (X_train, X_test),
        'Min-Max Scaler': min_max_scaler(X_train, X_test)[:2],
        'Z-Score Scaler': z_score_scaler(X_train, X_test)[:2]
    }

    print("\n[Bagian C] Hasil Ringkasan Eksperimen:")
    print(f"{'Normalisasi':<20} | {'Metrik':<10} | {'K':<3} | {'Akurasi':<8}")
    print("-" * 50)

    best_acc = 0
    best_config = None

    for scaler_name, (tr_scaled, te_scaled) in scalers.items():
        for metric in metrics:
            for k in k_values:
                clf = KNNClassifier(k=k, metric=metric, weights='uniform')
                clf.fit(tr_scaled, y_train)
                preds = clf.predict(te_scaled)

                acc = accuracy_score(y_test, preds)
                print(f"{scaler_name:<20} | {metric:<10} | {k:<3} | {acc * 100:.2f}%")

                if acc > best_acc:
                    best_acc = acc
                    best_config = (scaler_name, metric, k, preds)

    # 4. Evaluasi Metrik Lengkap untuk Konfigurasi Terbaik (Bagian C.4)
    print("\n--- Detail Evaluasi Konfigurasi Terbaik ---")
    b_scaler, b_metric, b_k, b_preds = best_config
    print(f"Terbaik: {b_scaler} | Metrik: {b_metric} | K={b_k} | Akurasi: {best_acc * 100:.2f}%")

    cm = confusion_matrix(y_test, b_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, b_preds, average='macro', zero_division=0)

    print("Confusion Matrix:\n", cm)
    print(f"Presisi (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")

# 1. Waktu Prediksi dengan Implementasi Sendiri
start_my = time.time()
y_pred_my = my_knn.predict(X_test_s)
end_my = time.time()
waktu_my = end_my - start_my

# 2. Waktu Prediksi dengan Scikit-Learn
start_sk = time.time()
y_pred_sk = sk_knn.predict(X_test_s)
end_sk = time.time()
waktu_sk = end_sk - start_sk

print(f"Waktu prediksi implementasi sendiri: {waktu_my:.6f} detik")
print(f"Waktu prediksi scikit-learn: {waktu_sk:.6f} detik")