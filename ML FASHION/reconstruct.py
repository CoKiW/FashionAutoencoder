"""
reconstruct.py
Opsi A: Rekonstruksi citra Fashion-MNIST menggunakan autoencoder penuh
(encoder -> latent vector -> decoder).

Contoh menjalankan dari terminal:
    python reconstruct.py --model autoencoder_fashion_mnist_dim2.pth --latent_dim 2 --index 10
    python reconstruct.py --model autoencoder_fashion_mnist_dim32.pth --latent_dim 32 --index 25

Program ini akan:
1. Memuat arsitektur Autoencoder dengan latent_dim yang sesuai dengan saat training.
2. Memuat bobot model dari file .pth.
3. Mengambil satu data Fashion-MNIST test berdasarkan --index.
4. Menjalankan forward pass (encoder -> decoder).
5. Menyimpan original.png, reconstructed.png, dan comparison.png.
"""

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Arsitektur model — HARUS identik dengan arsitektur saat training di Kaggle
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def get_test_image(index):
    """Mengambil satu gambar Fashion-MNIST test set berdasarkan index.
    Otomatis mengunduh dataset (via torchvision) jika belum ada di ./data.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    img, label = test_dataset[index]
    return img, label


def main():
    parser = argparse.ArgumentParser(description="Rekonstruksi citra Fashion-MNIST dengan Autoencoder")
    parser.add_argument("--model", type=str, required=True, help="Path ke file .pth autoencoder")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension saat training (2/8/32)")
    parser.add_argument("--index", type=int, default=0, help="Index data test Fashion-MNIST yang dipakai")
    parser.add_argument("--outdir", type=str, default=".", help="Folder untuk menyimpan hasil")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1 & 2: muat arsitektur + bobot
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model berhasil dimuat dari: {args.model} (latent_dim={args.latent_dim})")

    # 3: ambil data berdasarkan index
    img, label = get_test_image(args.index)
    print(f"Index {args.index} -> label kelas: {label}")
    img_batch = img.unsqueeze(0).to(device)

    # 4: forward pass
    with torch.no_grad():
        recon = model(img_batch)

    original_np = img.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()

    # 5: simpan hasil
    plt.imsave(f"{args.outdir}/original.png", original_np, cmap="gray")
    plt.imsave(f"{args.outdir}/reconstructed.png", recon_np, cmap="gray")

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original_np, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recon_np, cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    plt.suptitle(f"Label: {label} | latent_dim={args.latent_dim} | index={args.index}")
    plt.savefig(f"{args.outdir}/comparison.png", dpi=150, bbox_inches="tight")

    print("Selesai. File tersimpan: original.png, reconstructed.png, comparison.png")


if __name__ == "__main__":
    main()
