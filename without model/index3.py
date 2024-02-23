import cv2
import numpy as np

def detect_lines(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü yüklenemedi.")
        return

    # Görüntüyü griye çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azalt
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenar tespiti
    edges = cv2.Canny(blurred, 50, 150)

    # Çizgi tespiti
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Çizgi sayısına göre koşullu ifadeler
    if lines is not None:
        print(f"Toplam çizgi sayısı: {len(lines)}")
        if len(lines) < 20:
            print("Normal El.")
        elif 5 <= len(lines) > 21:
            print("Kınalı El .")
    else:
        print("Hiç çizgi tespit edilmedi.")

# Fonksiyonu test et
detect_lines('avuc-ici-terlemesi.jpg')