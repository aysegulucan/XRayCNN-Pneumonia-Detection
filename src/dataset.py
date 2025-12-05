import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    """
    Önceden işlenmiş (augmented) verileri diskten okur ve DataLoader'a çevirir.
    
    Args:
        data_dir (str): 'train', 'val' ve 'test' klasörlerinin olduğu ana dizin.
        batch_size (int): Her adımda kaç resim yükleneceği.
    
    Returns:
        train_loader, val_loader (veya test_loader opsiyonel eklenebilir)
    """
    
    # 1. Standartlaştırma (Tüm resimler için AYNI işlem)
    # Not: Burada döndürme/kırpma yapmıyoruz çünkü onları zaten 'augment_data.py' ile 
    # fiziksel olarak yapıp kaydettik. Burası sadece formatlama yeri.
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Modelin giriş boyutu
        transforms.ToTensor(),         # 0-255 arasını 0-1 arasına çeker
        transforms.Normalize(          # ImageNet standartları (Renk dağılımını dengeler)
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Klasör Yolları
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    # test_dir = os.path.join(data_dir, 'test') # İstersen açabilirsin

    # 3. Veri Setlerini Oluştur (ImageFolder sihirbazı)
    # Bu fonksiyon klasör isimlerini (NORMAL, PNEUMONIA) otomatik etiket (0, 1) yapar.
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)

    # 4. DataLoader (Veri Taşıyıcılar)
    # Train için shuffle=True (Karıştır ki ezberlemesin)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Val için shuffle=False (Sıralı olsun ki hatayı takip edebilelim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Veri Setleri Yüklendi:")
    print(f"   - Eğitim Görsel Sayısı: {len(train_dataset)} (Orijinal + Augmented)")
    print(f"   - Doğrulama Görsel Sayısı: {len(val_dataset)}")
    print(f"   - Sınıflar: {train_dataset.classes}") # ['NORMAL', 'PNEUMONIA']

    return train_loader, val_loader

# --- Test Bloğu ---
if __name__ == "__main__":
    # Test etmek için kendi yolunu buraya yaz
    path = r"C:\\Users\\Ayşegül Uçan\\Desktop\\DL_Project\\One_O_One\\changable_dataset"
    try:
        t_loader, v_loader = get_data_loaders(path)
        images, labels = next(iter(t_loader))
        print(f"   - Batch Boyutu: {images.shape}") # [32, 3, 224, 224] olmalı
    except Exception as e:
        print(f"Hata: {e}")