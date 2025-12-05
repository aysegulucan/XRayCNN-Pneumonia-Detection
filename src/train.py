import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # İlerleme çubuğu

# Kendi yazdığımız modülleri çağırıyoruz
# (Eğer hata alırsan terminalde 'src' klasörünün içine girip çalıştır)
from model import XRayCNN
from dataset import get_data_loaders

# --- AYARLAR (HYPERPARAMETERS) ---
LEARNING_RATE = 0.0001  # Modelin öğrenme hızı (Çok büyük olursa ezberler, küçük olursa öğrenemez)
BATCH_SIZE = 32         # Her seferde kaç resim birden işlenecek
EPOCHS = 10             # Tüm veri seti üzerinden kaç kez geçilecek
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train() # Modeli "Eğitim Modu"na al (Dropout ve BatchNorm çalışsın)
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False) # İlerleme çubuğu
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Etiketleri float yap (BCELoss için gerekli) ve boyutunu düzelt
        labels = labels.float().unsqueeze(1) 
        
        # 1. İLERİ (Forward Pass)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. GERİ (Backward Pass - Öğrenme Anı)
        optimizer.zero_grad() # Eski gradyanları temizle
        loss.backward()       # Hatanın kaynağını bul
        optimizer.step()      # Ağırlıkları güncelle
        
        # İstatistikler
        running_loss += loss.item()
        
        # Tahminleri 0 veya 1'e çevir (Sigmoid > 0.5 ise 1, değilse 0)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval() # Modeli "Test Modu"na al (Dropout KAPANIR)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Test ederken gradyan hesaplama (Hafıza tasarrufu)
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    print(f"Kullanılan Cihaz: {DEVICE}")
    
    # 1. Veriyi Yükle
    # BURAYA DİKKAT: Veri setinin olduğu yolu buraya yazmalısın!
    VERI_YOLU = r"C:\\Users\\Ayşegül Uçan\\Desktop\\DL_Project\\One_O_One\\changable_dataset" 
    
    print("Veri yükleniyor...")
    train_loader, val_loader = get_data_loaders(VERI_YOLU, BATCH_SIZE)
    
    # 2. Modeli Kur
    model = XRayCNN().to(DEVICE)
    
    # 3. Silahları Seç (Loss ve Optimizer)
    # BCEWithLogitsLoss: Hem Sigmoid uygular hem hata hesaplar (Daha stabil)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Grafikler için listeler
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf') # En iyi sonucu takip etmek için
    
    print("Eğitim Başlıyor! 🚀")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Eğitim Turu
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Doğrulama Turu
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # Kaydet
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: %{train_acc:.2f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: %{val_acc:.2f}")
        
        # Eğer bu model öncekilerden iyiyse kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model iyileşti ve kaydedildi (best_model.pth)")
            
    # --- GRAFİK ÇİZME BÖLÜMÜ ---
    plt.figure(figsize=(12, 5))
    
    # 1. Loss Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Eğitim ve Doğrulama Hatası')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Accuracy Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Eğitim ve Doğrulama Başarısı')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("egitim_sonuclari.png")
    print("\nGrafik 'egitim_sonuclari.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()