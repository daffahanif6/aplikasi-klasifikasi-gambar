Nama : Daffa Hanif Durachman

NIM : 21.11.4470

Mata Kuliah : Kecerdasan Buatan Lanjut

Kelas : 22S1IF-Kecerda5(ST164)
___

# Fitur Utama
* Klasifikasi Gambar:
Model ResNet-50 memproses gambar dan menghasilkan prediksi dari 1.000 kategori objek (ImageNet).
Menampilkan probabilitas dari 5 kategori teratas.

* Interpretasi Model (Explainable AI):
Menggunakan Integrated Gradients untuk menunjukkan area gambar yang paling berkontribusi terhadap prediksi model.
Visualisasi heatmap untuk memahami keputusan model secara intuitif.

* Antarmuka yang Interaktif:
Komponen upload gambar.
Visualisasi prediksi (diagram batang).
Tampilan gambar asli dan interpretasi XAI berdampingan.
Penjelasan kategori prediksi teratas berdasarkan hasil klasifikasi.

* Teknologi yang Digunakan:
Streamlit untuk antarmuka aplikasi.
PyTorch dan TorchVision untuk model ResNet-50.
Captum untuk interpretasi model.
Matplotlib untuk visualisasi probabilitas dan interpretasi.

# Tentang ResNet-50
ResNet-50 adalah model deep learning dengan arsitektur Residual Neural Network yang terkenal karena kemampuannya menangani pelatihan jaringan yang sangat dalam. Model ini menggunakan "residual connections" untuk mempercepat aliran gradien selama pelatihan, sehingga mengatasi masalah "vanishing gradients". Dilatih pada dataset ImageNet, ResNet-50 dapat mengenali hingga 1.000 kelas objek yang mencakup hewan, kendaraan, alat sehari-hari, dan lainnya.

# Cara Kerja
1. Preprocessing Gambar:
Menggunakan pipeline yang disediakan oleh TorchVision untuk memastikan gambar sesuai format yang diharapkan oleh model.

2. Inferensi dengan ResNet-50:
Model mengubah input gambar menjadi prediksi probabilitas untuk setiap kategori.

3. Visualisasi Probabilitas:
Menampilkan kategori dengan probabilitas tertinggi dalam bentuk diagram batang.

4. Explainable AI:
Menjelaskan keputusan model dengan heatmap yang menunjukkan fitur gambar yang relevan.

# Referensi
[1] S. Solanki, “PyTorch: Image Classification using Pre-Trained Models.” Accessed: Jan. 23, 2025. [Online]. Available: https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-image-classification-using-pre-trained-models
[2] S. Solanki, “Captum: Interpret Predictions Of PyTorch Image Classification Networks.” Accessed: Jan. 23, 2025. [Online]. Available: https://coderzcolumn.com/tutorials/artificial-intelligence/captum-for-pytorch-image-classification-networks
[3] Petru Potrimba, “What is ResNet-50?,” Robotflow. Accessed: Jan. 24, 2025. [Online]. Available: https://blog.roboflow.com/what-is-resnet-50/
[4] M. Ibrahim, “The Basics of ResNet50 | ml-articles – Weights & Biases.” Accessed: Jan. 24, 2025. [Online]. Available: https://wandb.ai/mostafaibrahim
