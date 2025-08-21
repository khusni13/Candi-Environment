# app.py

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from rembg import remove

# ===================== FUNGSI CACHE =====================
@st.cache_resource
def load_model_dummy():
    return "Dummy model jika ada, misal load_model('model.h5')"

@st.cache_data
def read_dataset(csv_path):
    return pd.read_csv(csv_path)

# ===================== LOGIKA SUDUT =====================
def vector_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_angle_edges(pil_img, visualize=False, top_n=20):
    # === STEP 1: Hapus background ===
    pil_img = pil_img.convert("RGBA")
    no_bg = remove(pil_img).convert("RGB")
    img = np.array(no_bg)

    # === STEP 2: Grayscale & Edge ===
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # === STEP 3: Harris Corner ===
    harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    coords = np.argwhere(harris > 0.01 * harris.max())
    if len(coords) < 4:
        return None, img

    # Urutkan berdasarkan nilai Harris
    values = harris[harris > 0.01 * harris.max()]
    sorted_indices = np.argsort(values)[::-1]
    top_coords = coords[sorted_indices[:top_n]]

    # Filter corner hanya di tepi
    final_corners = []
    for y, x in top_coords:
        if edges[y, x] != 0:
            final_corners.append((x, y))

    if len(final_corners) < 3:
        return None, img

    # === STEP 4: Titik ekstrem ===
    final_coords = np.array(final_corners)
    top_pt = final_coords[np.argmin(final_coords[:, 1])]
    bottom_pt = final_coords[np.argmax(final_coords[:, 1])]
    left_pt = final_coords[np.argmin(final_coords[:, 0])]
    right_pt = final_coords[np.argmax(final_coords[:, 0])]

    # Hitung sudut
    angle = vector_angle(left_pt, top_pt, right_pt)

    # === STEP 5: Visualisasi ===
    vis_img = img.copy()
    if visualize:
        pts = np.array([left_pt, top_pt, right_pt])
        for pt in pts:
            cv2.circle(vis_img, tuple(pt), 5, (0, 255, 0), -1)
        cv2.polylines(vis_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return angle, vis_img

# ===================== STREAMLIT APP =====================
st.title("Perhitungan Sudut Puncak Candi")
st.markdown("Upload gambar candi untuk mendeteksi besar sudut puncak")
st.markdown("### 📤 Upload Gambar Candi")
st.markdown(
    """
    ⚠️ **Pastikan gambar diambil dari sudut depan dan posisi tegak lurus terhadap puncak candi**  
    Agar sistem dapat mendeteksi sudut secara akurat, hindari foto miring, terlalu dekat, atau dengan latar belakang yang kompleks.
    """
)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
range_min = st.slider("Batas bawah sudut Candi Hindu (°)", 0, 90, 32)
range_max = st.slider("Batas atas sudut Candi Hindu (°)", 0, 90, 75)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if st.button("Hitung Sudut"):
        with st.spinner("⏳ Sedang memproses gambar..."):
            angle, vis_img = extract_angle_edges(image, visualize=True)

        if angle is not None:
            pred_label = "Hindu" if range_min <= angle <= range_max else "Buddha"

            st.image(vis_img, caption="Citra dengan sudut terdeteksi", channels="RGB", width=300)
            st.markdown(f"### Sudut terdeteksi: `{angle:.2f}°`")
            st.success(f"Hasil (Rule-Based) Tergolong Dalam Rentang Sudut: **Candi {pred_label}**")

            st.markdown("#### Catatan:")
            st.write("- Candi Hindu cenderung memiliki sudut puncak lebih kecil (runcing)")
            st.write("- Candi Buddha cenderung memiliki sudut lebih besar (menumpul)")
        else:
            st.error("❌ Sudut tidak dapat terdeteksi dari gambar. Coba gunakan gambar dengan puncak yang lebih jelas.")
