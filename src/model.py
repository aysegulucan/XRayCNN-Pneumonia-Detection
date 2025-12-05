import torch
import torch.nn as nn

class XRayCNN(nn.Module):
    def __init__(self):
        super(XRayCNN, self).__init__()
        
        # --- BLOK 1: Özellik Çıkarma (Feature Extraction) ---
        # Girdi: (Batch_Size, 3, 224, 224) -> RGB Resim
        self.features = nn.Sequential(
            # Katman 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Eğitim hızını artırır ve kararlı hale getirir
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Boyut yarıya iner: 112x112

            # Katman 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Boyut: 56x56

            # Katman 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Boyut: 28x28
            
            # Katman 4 (Daha derin detaylar için)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Boyut: 14x14
        )
        
        # --- BLOK 2: Sınıflandırma (Classifier) ---
        self.classifier = nn.Sequential(
            nn.Flatten(), # 256 kanal x 14 x 14 pikseli düzleştirir
            nn.Linear(256 * 14 * 14, 512), 
            nn.ReLU(),
            
            # --- OVERFITTING ÖNLEYİCİ SİGORTA ---
            nn.Dropout(0.5), # Nöronların %50'sini rastgele kapatır. (Ezberlemeyi engeller)
            
            nn.Linear(512, 1) # Çıkış: 1 Nöron (0: Normal, 1: Pnömoni olasılığı)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Mühendislik Testi (Sanity Check) ---
# Bu dosyayı doğrudan çalıştırdığında modelin mimarisini ve çıktı boyutunu kontrol eder.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XRayCNN().to(device)
    
    # Rastgele bir tensör üret (Batch=1, Kanal=3, Boyut=224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Model Mimarisi:\n{model}")
    print(f"\nTest Girdisi Boyutu: {dummy_input.shape}")
    print(f"Çıktı Boyutu: {output.shape}") # torch.Size([1, 1]) görmelisin
    
    # Parametre Sayısı Hesaplama
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nToplam Parametre Sayısı: {total_params:,}")