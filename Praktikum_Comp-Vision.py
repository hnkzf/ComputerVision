import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(BASE_DIR, "dataset") 
save_base_path = os.path.join(BASE_DIR, "results") 


if not os.path.exists(input_dir):
    print(f"ERROR: Folder dataset tidak ditemukan di {input_dir}")
    exit()

os.makedirs(save_base_path, exist_ok=True)

def show_gray(title, im):
  plt.imshow(im, cmap='gray')
  plt.title(title)
  plt.axis('off')
  plt.show()

def save_transformed_image(original_img_name, transformed_image, transform_name):
  base_name = os.path.splitext(original_img_name)[0]
  filename = os.path.join(save_base_path, f"{base_name}_{transform_name}.jpg")
  cv2.imwrite(filename, transformed_image)
  print(f"Gambar '{base_name}_{transform_name}' disimpan di: {filename}")

def apply_translation(image, tx, ty):
  matrix_translasi = np.float32([[1, 0, tx], [0, 1, ty]])
  transformed_img = cv2.warpAffine(image, matrix_translasi, (image.shape[1], image.shape[0]))
  return transformed_img

def apply_rotation(image, angle, scale=1.0):
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  matrix_rotasi = cv2.getRotationMatrix2D(center, angle, scale)
  transformed_img = cv2.warpAffine(image, matrix_rotasi, (w, h))
  return transformed_img

def apply_scaling(image, fx, fy):
  transformed_img = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR if fx > 1 else cv2.INTER_AREA)
  return transformed_img

def apply_shearing_x(image, shear_factor_x):
  (h, w) = image.shape[:2]
  matrix_shear_x = np.float32([[1, shear_factor_x, 0], [0, 1, 0]])
  transformed_img = cv2.warpAffine(image, matrix_shear_x, (int(w + h * shear_factor_x), h))
  return transformed_img

def apply_shearing_y(image, shear_factor_y):
  (h, w) = image.shape[:2]
  matrix_shear_y = np.float32([[1, 0, 0], [shear_factor_y, 1, 0]])
  transformed_img = cv2.warpAffine(image, matrix_shear_y, (w, int(h + w * shear_factor_y)))
  return transformed_img

def apply_reflection(image, flip_code): 
  transformed_img = cv2.flip(image, flip_code)
  return transformed_img

def apply_affine_transform(image, pts1, pts2):
  (h, w) = image.shape[:2]
  matrix_affine = cv2.getAffineTransform(pts1, pts2)
  transformed_img = cv2.warpAffine(image, matrix_affine, (w, h))
  return transformed_img

def apply_perspective_transform(image, pts1, pts2):
  (h, w) = image.shape[:2]
  matriks_perspektif = cv2.getPerspectiveTransform(pts1, pts2)
  transformed_img = cv2.warpPerspective(image, matriks_perspektif, (w, h))
  return transformed_img

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Gambar {filename} tidak ditemukan atau tidak dapat dibaca. Melanjutkan ke gambar berikutnya.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tinggi, lebar = gray.shape[:2] 

        print(f"\nMemproses gambar: {filename}")

        # 0. Simpan Grayscale Original
        save_transformed_image(filename, gray, "grayscale")

        # 1. Translasi
        hasil_translasi_gray = apply_translation(gray, 50, 30)
        save_transformed_image(filename, hasil_translasi_gray, "translasi")

        # 2. Rotasi
        hasil_rotasi_gray = apply_rotation(gray, 45)
        save_transformed_image(filename, hasil_rotasi_gray, "rotasi")

        # 3. Scaling
        hasil_skala_besar_gray = apply_scaling(gray, 1.5, 1.5)
        save_transformed_image(filename, hasil_skala_besar_gray, "skala_besar")

        hasil_skala_kecil_gray = apply_scaling(gray, 0.5, 0.5)
        save_transformed_image(filename, hasil_skala_kecil_gray, "skala_kecil")

        # 4. Shearing
        hasil_shear_x_gray = apply_shearing_x(gray, 0.3)
        save_transformed_image(filename, hasil_shear_x_gray, "shearing_x")

        hasil_shear_y_gray = apply_shearing_y(gray, 0.3)
        save_transformed_image(filename, hasil_shear_y_gray, "shearing_y")

        # 5. Refleksi
        hasil_flip_horizontal_gray = apply_reflection(gray, 1)
        save_transformed_image(filename, hasil_flip_horizontal_gray, "flip_horizontal")

        hasil_flip_vertikal_gray = apply_reflection(gray, 0)
        save_transformed_image(filename, hasil_flip_vertikal_gray, "flip_vertikal")

        hasil_flip_dua_arah_gray = apply_reflection(gray, -1)
        save_transformed_image(filename, hasil_flip_dua_arah_gray, "flip_dua_arah")

        # 6. Transformasi Affine
        pts1_affine = np.float32([[50,50], [150,50], [50,150]])
        pts2_affine = np.float32([[10,70], [150,50], [70,170]])
        hasil_affine_gray = apply_affine_transform(gray, pts1_affine, pts2_affine)
        save_transformed_image(filename, hasil_affine_gray, "affine")

        # 7. Transformasi Proyektif (Perspective Transform)
        pts1_perspektif = np.float32([[50,50], [170,50], [50,170], [170,170]])
        pts2_perspektif = np.float32([[30,70], [190,50], [70,190], [150,150]])
        hasil_perspektif_gray = apply_perspective_transform(gray, pts1_perspektif, pts2_perspektif)
        save_transformed_image(filename, hasil_perspektif_gray, "perspektif")