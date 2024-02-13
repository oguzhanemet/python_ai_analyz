from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


import numpy as np
# Modelin yüklenmesi
model = load_model('model path')
# Yeni fotoğrafın yolu
img_path = 'test img path'

# Fotoğrafı yükleme ve ön işleme
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Tahmin yapma
prediction = model.predict(img_tensor)

# Tahmini yorumlama
if prediction < 0.5:
    print("this photo belongs the woman hand")
else:
    print("this photo belongs the man hand")