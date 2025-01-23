import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import captum
import PIL
from PIL import Image

from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from captum import attr
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Fungsi preprocessing untuk ResNet-50 menggunakan bobot pretrained
# Transformasi ini termasuk dalam pipeline AI untuk memastikan gambar yang diunggah sesuai format yang diharapkan oleh model
preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

# Fungsi untuk memuat model ResNet-50 (pretrained pada ImageNet)
@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()  # Mengatur model ke mode evaluasi untuk memastikan tidak ada perubahan parameter selama inferensi
    return model

# Fungsi untuk membuat prediksi berdasarkan gambar yang telah diproses
# Aspek AI: Model melakukan inferensi dengan menghasilkan probabilitas untuk setiap kelas berdasarkan input gambar

def make_prediction(model, processed_img):
    # Menghitung probabilitas keluaran model
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)  # Menggunakan fungsi softmax untuk menghasilkan probabilitas
    probs = probs[0].detach().numpy()

    # Mengambil 5 kelas dengan probabilitas tertinggi
    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

# Fungsi untuk interpretasi prediksi model menggunakan Integrated Gradients
# Aspek AI: Menggunakan Explainable AI (XAI) untuk memberikan interpretasi visual terhadap prediksi model

def interpret_prediction(model, processed_img, target):
    # Menggunakan Integrated Gradients untuk menghitung feature importance
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1, 2, 0)  # Mengubah dimensi menjadi (height, width, channels)

    return feature_imp

# Antarmuka dashboard Streamlit
st.title("ResNet-50 Image Classifier :tea: :coffee:")

# Subtitle menjelaskan ResNet-50
st.subheader("Apa itu ResNet-50?")
st.write("ResNet-50 adalah model deep learning berbasis arsitektur Residual Neural Network (ResNet) yang sangat populer dalam bidang Computer Vision. Model ini dilatih pada dataset ImageNet dan mampu mengenali 1.000 kelas objek yang berbeda, termasuk hewan, kendaraan, dan objek sehari-hari.")
st.write("ResNet-50 bekerja dengan memperkenalkan \"residual connections\", yaitu jalur pintas yang memungkinkan gradien mengalir dengan lebih efektif selama proses pelatihan. Hal ini membantu mengatasi masalah \"vanishing gradients\" dan memungkinkan pelatihan model yang sangat dalam, seperti ResNet-50 dengan 50 lapisan.")

# Komponen untuk mengunggah gambar
upload = st.file_uploader(label="Upload Image:", type=["png", "jpg", "jpeg"])

if upload:
    # Membaca gambar yang diunggah
    img = Image.open(upload)

    # Memuat model ResNet-50
    model = load_model()

    # Preprocessing gambar yang diunggah
    # Aspek AI: Transformasi gambar mencakup normalisasi nilai pixel, resizing, dan memastikan format tensor yang sesuai
    preprocessed_img = preprocess_func(img)

    # Membuat prediksi
    # Aspek AI: Model deep learning memproses gambar dan menghasilkan probabilitas kelas berdasarkan pelatihan sebelumnya
    probs, idxs = make_prediction(model, preprocessed_img)

    # Menginterpretasi prediksi model
    # Aspek AI: Menggunakan Integrated Gradients untuk memberikan visualisasi bagian gambar yang berkontribusi terhadap keputusan model
    feature_imp = interpret_prediction(model, preprocessed_img, idxs[0])

    # Visualisasi probabilitas 5 kelas teratas
    main_fig = plt.figure(figsize=(12, 3))
    ax = main_fig.add_subplot(111)
    plt.barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"]*4 + ["tomato"])
    plt.title("Top 5 Probabilities", loc="center", fontsize=15)
    st.pyplot(main_fig, use_container_width=True)

    # Visualisasi interpretasi model (Integrated Gradients)
    interp_fig, ax = viz.visualize_image_attr(feature_imp, show_colorbar=True, fig_size=(6, 6))

    # Membagi antarmuka menjadi dua kolom
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Menampilkan gambar asli
        main_fig = plt.figure(figsize=(6, 6))
        ax = main_fig.add_subplot(111)
        plt.imshow(img)
        plt.xticks([], [])  # Menghilangkan sumbu x
        plt.yticks([], [])  # Menghilangkan sumbu y
        st.pyplot(main_fig, use_container_width=True)

    with col2:
        # Menampilkan interpretasi gambar
        st.pyplot(interp_fig, use_container_width=True)