import cv2
import numpy as np

def kina_var_mi(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü yüklenemedi.")
        return

    # BGR'dan HSV'ye dönüştür
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Kına için renk aralığını belirle (örnek değerler, ayarlanması gerekebilir)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])

    # Renk aralığına göre maske oluştur
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Maskeyi kullanarak konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Konturların toplam alanını hesapla
    total_area = sum(cv2.contourArea(contour) for contour in contours)

    # Alan büyüklüğüne göre kına varlığına karar ver
    if total_area > 250:  # Alan eşiği, ayarlanabilir
        print("Kına algılandı.")
    else:
        print("Kına algılanmadı.")

# Fonksiyonu test et
kina_var_mi('avuc-ici-terlemesi.jpg')