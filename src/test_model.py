import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Kendi modelini çağır
from model import XRayCNN

# --- AYARLAR ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def get_test_loader(data_dir):
    """Sadece TEST klasörünü okur ve hazırlar."""
    test_dir = os.path.join(data_dir, 'test')
    
    # Test verisi için SADECE standartlaştırma yapılır (Augmentation YOK)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_loader, test_dataset.classes

def load_best_model(model_path):
    print(f"📂 Model yükleniyor: {model_path}")
    model = XRayCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Test modu (Dropout kapalı!)
    return model

def main():
    # 1. Yolları Tanımla (BURAYI GÜNCELLE!)
    VERI_YOLU = r"C:\\Users\\Ayşegül Uçan\\Desktop\\DL_Project\\One_O_One\\changable_dataset" 
    MODEL_YOLU = "best_model.pth" # train.py ile aynı klasörde oluşmuştu
    
    # 2. Veriyi ve Modeli Yükle
    print("⏳ Test verisi hazırlanıyor...")
    test_loader, class_names = get_test_loader(VERI_YOLU)
    print(f"   - Test görsel sayısı: {len(test_loader.dataset)}")
    
    try:
        model = load_best_model(MODEL_YOLU)
    except FileNotFoundError:
        print("❌ HATA: 'best_model.pth' dosyası bulunamadı! Lütfen dosya yolunu kontrol et.")
        return

    # 3. Tahminleri Topla
    all_preds = []
    all_labels = []
    
    print("🚀 Test işlemi başladı...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy() # 0 veya 1
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 4. Metrikleri Hesapla
    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*30)
    print("📊 SINIFLANDIRMA RAPORU")
    print("="*30)
    # Target names: 0 -> Normal, 1 -> Pneumonia
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 5. Görselleştirme (Confusion Matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('Confusion Matrix (Test Sonuçları)')
    plt.savefig('test_sonuclari_matrix.png')
    print("✅ Grafik 'test_sonuclari_matrix.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()