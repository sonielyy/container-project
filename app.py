import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

# Model yolları
MODEL1_PATH = "runs/detect/train15_prev-train_120-10-bw/weights/best.pt"
MODEL2_PATH = "runs/detect/train16_prev-train3_120-10-colored/weights/best.pt"

# Modelleri yükle
model1 = YOLO(MODEL1_PATH)  # siyah-beyaz (front)
model2 = YOLO(MODEL2_PATH)  # renkli (side)

# Global kamera değişkenleri
camera_active = False
cap = None

def preprocess_image(img):
    """
    Ön işleme: Gri tonlama, Gaussian Blur, CLAHE ve 3 kanallı BGR'ye dönüştürme.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred_img)
    bgr_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    return bgr_img

def get_container_prediction(results):
    """
    YOLO sonuçlarını 'ContainerID' bölgesi altındaki harf kutularını gruplayarak tahmine dönüştürür.
    """
    sections = []
    letters = []
    if results and len(results) > 0 and hasattr(results[0], "boxes"):
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf.cpu().numpy()[0])
            cls_idx = int(box.cls.cpu().numpy()[0])
            class_label = results[0].names[cls_idx]
            if class_label == "ContainerID":
                sections.append({'box': (x1, y1, x2, y2), 'letters': []})
            elif class_label != "FrontFace":
                letters.append({'label': class_label, 'box': (x1, y1, x2, y2), 'conf': conf})
        for sec in sections:
            sec_x1, sec_y1, sec_x2, sec_y2 = sec['box']
            for let in letters:
                lx1, ly1, lx2, ly2 = let['box']
                center_x = (lx1 + lx2) // 2
                center_y = (ly1 + ly2) // 2
                if sec_x1 <= center_x <= sec_x2 and sec_y1 <= center_y <= sec_y2:
                    sec['letters'].append(let)
            sec['letters'].sort(key=lambda l: l['box'][0])
        if sections:
            prediction = "".join([l['label'] for l in sections[0]['letters']])
            return prediction, sections[0]['letters']
    return "", []

def run_inference(model, img):
    """
    Model ile tahmin yapar ve harf tahmini yoksa "XXXX" ekler.
    """
    results = model(img, show=False)
    prediction, letters_list = get_container_prediction(results)
    if not any(l['label'].isalpha() for l in letters_list) and len(prediction) == 7:
        prediction = "XXXX" + prediction
    return prediction, letters_list

def is_valid_container_id(text):
    """
    Container ID: 11 karakter, ilk 4'ü harf, son 7'si sayı.
    """
    if len(text) != 11:
        return False
    letters_part = text[:4]
    numbers_part = text[4:]
    return letters_part.isalpha() and numbers_part.isdigit()

def filter_prediction(letters_list, target_len=11):
    """
    Tahmin edilen karakter sayısı hedef uzunluktan fazla ise en düşük confidence'leri çıkarır.
    """
    if len(letters_list) == target_len:
        return "".join([l['label'] for l in letters_list])
    if len(letters_list) < target_len:
        return None
    letters_sorted = sorted(letters_list, key=lambda l: l['conf'])
    num_to_remove = len(letters_list) - target_len
    filtered_letters = letters_list.copy()
    for i in range(num_to_remove):
        filtered_letters.remove(letters_sorted[i])
    prediction = "".join([l['label'] for l in filtered_letters])
    return prediction if len(prediction) == target_len else None

def process_image(image_path):
    """
    Resim dosyasını okuyup tahmin yapar.
    """
    img = cv2.imread(image_path)
    if img is None:
        print_console("Hata: Resim okunamadi!")
        messagebox.showerror("Hata", "Resim okunamadi!")
        return
    preprocessed = preprocess_image(img)
    pred1, letters1 = run_inference(model1, preprocessed)
    pred2, letters2 = run_inference(model2, img)
    valid1 = is_valid_container_id(pred1)
    valid2 = is_valid_container_id(pred2)
    final_prediction = ""
    if valid1 and not valid2:
        final_prediction = pred1
    elif valid2 and not valid1:
        final_prediction = pred2
    elif valid1 and valid2:
        avg1 = np.mean([l['conf'] for l in letters1]) if letters1 else 0
        avg2 = np.mean([l['conf'] for l in letters2]) if letters2 else 0
        final_prediction = pred1 if avg1 >= avg2 else pred2
    else:
        filtered1 = filter_prediction(letters1)
        filtered2 = filter_prediction(letters2)
        if filtered1 and is_valid_container_id(filtered1):
            final_prediction = filtered1
        elif filtered2 and is_valid_container_id(filtered2):
            final_prediction = filtered2
        else:
            final_prediction = "Tahmin yapilamadi"
    output = f"Container ID: {final_prediction}"
    print_console(output)
    messagebox.showinfo("Tahmin Sonucu", output)

def start_camera_thread():
    """
    Kameradan görüntü alır ve anlık tahmin yapar.
    Bu fonksiyon ayrı bir thread içerisinde çalıştırılmalıdır.
    """
    global camera_active, cap
    if camera_active:
        print_console("Kamera zaten acik!")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_console("Hata: Kamera acilamadi!")
        messagebox.showerror("Hata", "Kamera acilamadi!")
        return
    camera_active = True
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        display_frame = frame.copy()
        preprocessed_frame = preprocess_image(frame)
        pred1, letters1 = run_inference(model1, preprocessed_frame)
        pred2, letters2 = run_inference(model2, frame)
        valid1 = is_valid_container_id(pred1)
        valid2 = is_valid_container_id(pred2)
        final_prediction = ""
        if valid1 and not valid2:
            final_prediction = pred1
        elif valid2 and not valid1:
            final_prediction = pred2
        elif valid1 and valid2:
            avg1 = np.mean([l['conf'] for l in letters1]) if letters1 else 0
            avg2 = np.mean([l['conf'] for l in letters2]) if letters2 else 0
            final_prediction = pred1 if avg1 >= avg2 else pred2
        else:
            final_prediction = "Gecerli tahmin yok"
        cv2.putText(display_frame, final_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv2.imshow("Kamera - Orijinal Goruntu", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print_console(f"Anlik Tahmin: {final_prediction}")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    camera_active = False
    print_console("Kamera kapatildi.")

def stop_camera():
    """
    Kamerayı kapatmak için global flag'i degistirir.
    """
    global camera_active
    if camera_active:
        camera_active = False
        print_console("Kamera kapatiliyor...")
    else:
        print_console("Kamera acik degil.")

def print_console(text):
    """
    Konsola metin yazdırır.
    """
    console_text.insert(tk.END, text + "\n")
    console_text.see(tk.END)

def clear_console():
    """
    Konsol temizleme fonksiyonu.
    """
    console_text.delete("1.0", tk.END)

def upload_image():
    """
    Resim dosyası yükler ve önizleme yapar.
    """
    file_path = filedialog.askopenfilename(title="Resim Sec", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        try:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Hata", "Resim okunamadi!")
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(pil_img)
            preview_label.config(image=photo)
            preview_label.image = photo  # Referansı saklamak gerekli
        except Exception as e:
            messagebox.showerror("Hata", f"Resim önizlemesi gösterilemedi: {e}")
        process_image(file_path)

# Arayüz Oluşturma
root = tk.Tk()
root.title("Container ID Tahmin Uygulamasi")

# Üst buton çerçevesi
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

btn_upload = tk.Button(top_frame, text="Resim Yukle", command=upload_image, width=15, height=2)
btn_upload.grid(row=0, column=0, padx=5)

btn_camera = tk.Button(top_frame, text="Kamera Ac", command=lambda: threading.Thread(target=start_camera_thread, daemon=True).start(), width=15, height=2)
btn_camera.grid(row=0, column=1, padx=5)

btn_stop_camera = tk.Button(top_frame, text="Kamera Kapat", command=stop_camera, width=15, height=2)
btn_stop_camera.grid(row=0, column=2, padx=5)

btn_clear_console = tk.Button(top_frame, text="Konsolu Temizle", command=clear_console, width=15, height=2)
btn_clear_console.grid(row=0, column=3, padx=5)

# Alt çerçeve: Önizleme ve Konsol
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10)

preview_label = tk.Label(bottom_frame)
preview_label.grid(row=0, column=0, padx=5)

console_text = tk.Text(bottom_frame, height=15, width=50)
console_text.grid(row=0, column=1, padx=5)

root.mainloop()
