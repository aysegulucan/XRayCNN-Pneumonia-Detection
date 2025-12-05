import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

# HEDEF: Toplam 8000 görüntü (Orijinal + Sentetik)
HEDEF_SAYI = 8000

def apply_advanced_preprocessing(img_pil):
    """
    Raporundaki Pipeline 2 Ön İşleme Adımları:
    1. Grayscale
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    # PIL -> OpenCV formatına (numpy array) çevir
    img_np = np.array(img_pil.convert('L')) # 'L' = Grayscale

    # --- CLAHE Uygulama ---
    # Rapordaki ayarlar: clipLimit=2.0, tileGridSize=(8,8) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_np)

    # (Opsiyonel) NLM Denoising çok yavaş çalışır, bu yüzden offline işlemde
    # sadece çok gürültülü resimler için önerilir. Şimdilik CLAHE yeterli.
    
    # Tekrar PIL formatına çevir
    return Image.fromarray(img_clahe)

def augment_folder_advanced(folder_path):
    extensions = ('.jpeg', '.jpg', '.png')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    
    current_count = len(files)
    needed = HEDEF_SAYI - current_count
    
    print(f"\nKlasör: {folder_path} | Mevcut: {current_count} | Hedef: {HEDEF_SAYI}")
    
    if needed <= 0:
        print("✅ Hedef sayıya zaten ulaşılmış.")
        return

    # --- PIPELINE 2 AUGMENTATION AYARLARI [cite: 85] ---
    augmentor = T.Compose([
        # Dönme: ±15 derece (Rapora uygun) [cite: 86]
        T.RandomRotation(degrees=15),
        
        # Kaydırma: %10 (Rapora uygun) [cite: 90]
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)), # Scale: [cite: 94]
        
        # Parlaklık/Kontrast: %12 (Rapora uygun) [cite: 98]
        T.ColorJitter(brightness=0.12, contrast=0.12),
        
        # Yatay Çevirme: Evet (Dokuyu öğrenmesi için)
        T.RandomHorizontalFlip(p=0.5),
    ])

    print(f"🚀 Pipeline 2 (CLAHE + Advanced Aug.) ile {needed} fotoğraf üretiliyor...")

    for i in tqdm(range(needed)):
        random_file = random.choice(files)
        img_path = os.path.join(folder_path, random_file)
        
        try:
            with Image.open(img_path) as img:
                # 1. Önce CLAHE uygula (Görüntüyü netleştir)
                processed_img = apply_advanced_preprocessing(img)
                
                # 2. Sonra Augmentation uygula (Çeşitle)
                aug_img = augmentor(processed_img)
                
                # Kaydet
                new_filename = f"aug_adv_{i}_{random_file}"
                save_path = os.path.join(folder_path, new_filename)
                aug_img.save(save_path)
                
        except Exception as e:
            print(f"Hata: {e}")

if __name__ == "__main__":
    # --- AYARLAR ---
    # Buraya kendi TRAIN yolunu yapıştır
    train_ana_yolu = r"C:\\Users\\Ayşegül Uçan\\Desktop\\DL_Project\\One_O_One\\changable_dataset\\train" 
    
    augment_folder_advanced(os.path.join(train_ana_yolu, "NORMAL"))
    augment_folder_advanced(os.path.join(train_ana_yolu, "PNEUMONIA"))