# Autoencoder Fashion-MNIST

## 1. Deskripsi
Proyek ini melatih Autoencoder pada dataset Fashion-MNIST dengan tiga ukuran latent
dimension (2, 8, 32), lalu melakukan rekonstruksi/generasi citra melalui terminal.

## 2. Versi Python & Library
- Python: 3.x (isi versi yang kamu pakai)
- torch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm

Install dependencies:
```
pip install torch torchvision numpy pandas matplotlib tqdm
```

## 3. Training di Kaggle
1. Upload/attach dataset Fashion-MNIST ke notebook Kaggle.
2. Buka dan jalankan `autoencoder-fashion-mnist.ipynb` dari atas ke bawah.
3. Notebook akan otomatis melatih model untuk latent_dim = 2, 8, 32 dan menyimpan:
   - `autoencoder_fashion_mnist_dim{2,8,32}.pth`
   - `encoder_fashion_mnist_dim{2,8,32}.pth`
   - `decoder_fashion_mnist_dim{2,8,32}.pth`
   - `loss_comparison.png`
   - `reconstruction_comparison.png`
4. Download semua file tersebut dari panel Output Kaggle.

## 4. Menjalankan Rekonstruksi dari Terminal

### Shell 1 — Autoencoder penuh
```
!python reconstruct.py --model output/autoencoder_fashion_mnist_dim2.pth --latent_dim 2 --index 10
```
Menghasilkan: `original.png`, `reconstructed.png`, `comparison.png`

### Shell 2 — Decoder saja
```
!python generate_from_decoder.py --decoder output/decoder_fashion_mnist_dim2.pth --latent_dim 2 --z1 0.5 --z2 -1.2
```
Menghasilkan: `generated_image.png`

## 5. Daftar File Output
- `autoencoder-fashion-mnist.ipynb`
- `autoencoder_fashion_mnist_dim{2,8,32}.pth`
- `reconstruct.py`
- `generate_from_decoder.py`
- `original.png`, `reconstructed.png`, `comparison.png`
- `generated_image.png`
- `laporan.pdf`
