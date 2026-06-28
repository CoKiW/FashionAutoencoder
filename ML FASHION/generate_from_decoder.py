"""
generate_from_decoder.py
Opsi B: Generasi citra Fashion-MNIST hanya menggunakan bagian decoder.
Mahasiswa memberi nilai latent vector lewat terminal, decoder menghasilkan gambar.

Contoh untuk latent dimension 2:
    python generate_from_decoder.py --decoder decoder_fashion_mnist_dim2.pth --latent_dim 2 --z1 0.5 --z2 -1.2

Contoh untuk latent dimension lebih dari 2 (8 atau 32), gunakan --latent dengan daftar angka:
    python generate_from_decoder.py --decoder decoder_fashion_mnist_dim8.pth --latent_dim 8 --latent "0.5,-1.2,0.3,0.8,0.1,-0.4,0.2,0.0"
    python generate_from_decoder.py --decoder decoder_fashion_mnist_dim32.pth --latent_dim 32 --latent "0.1,0.2,...(32 angka)"

Jika --latent tidak diberikan, latent vector akan di-random (acak) dengan ukuran latent_dim.
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


def build_latent_vector(args):
    if args.latent is not None:
        values = [float(v) for v in args.latent.split(",")]
        if len(values) != args.latent_dim:
            raise ValueError(
                f"Jumlah nilai pada --latent ({len(values)}) tidak sama dengan --latent_dim ({args.latent_dim})"
            )
        return torch.tensor(values, dtype=torch.float32)

    if args.latent_dim == 2 and (args.z1 is not None or args.z2 is not None):
        z1 = args.z1 if args.z1 is not None else 0.0
        z2 = args.z2 if args.z2 is not None else 0.0
        return torch.tensor([z1, z2], dtype=torch.float32)

    print(f"Tidak ada nilai latent diberikan, menggunakan latent vector acak (latent_dim={args.latent_dim}).")
    return torch.randn(args.latent_dim)


def main():
    parser = argparse.ArgumentParser(description="Generate citra Fashion-MNIST dari decoder")
    parser.add_argument("--decoder", type=str, required=True, help="Path ke file .pth decoder")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension saat training (2/8/32)")
    parser.add_argument("--z1", type=float, default=None, help="Nilai latent dim ke-1 (khusus latent_dim=2)")
    parser.add_argument("--z2", type=float, default=None, help="Nilai latent dim ke-2 (khusus latent_dim=2)")
    parser.add_argument("--latent", type=str, default=None, help='Daftar nilai latent dipisah koma, misal "0.5,-1.2,0.3"')
    parser.add_argument("--outdir", type=str, default=".", help="Folder untuk menyimpan hasil")
    parser.add_argument("--output_name", type=str, default="generated_image.png", help="Nama file output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # muat arsitektur + bobot decoder
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    state_dict = torch.load(args.decoder, map_location=device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    print(f"Decoder berhasil dimuat dari: {args.decoder} (latent_dim={args.latent_dim})")

    z = build_latent_vector(args).to(device)
    print("Latent vector yang dipakai:", z.cpu().numpy())

    z_batch = z.unsqueeze(0)
    with torch.no_grad():
        generated = decoder(z_batch)

    generated_np = generated.squeeze().cpu().numpy()

    out_path = f"{args.outdir}/{args.output_name}"
    plt.imsave(out_path, generated_np, cmap="gray")
    print(f"Selesai. Gambar hasil generate disimpan di: {out_path}")


if __name__ == "__main__":
    main()
