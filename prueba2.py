import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Variables globales
drawing = False
points = []

def click_and_draw(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.polylines(img_copy, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.imshow('Selecciona el área', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        cv2.polylines(img, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.imshow('Selecciona el área', img)

def select_area():
    global img
    # Cargar una imagen
    filepath = filedialog.askopenfilename()
    img = cv2.imread(filepath)
    cv2.imshow('Selecciona el área', img)
    cv2.setMouseCallback('Selecciona el área', click_and_draw)
    cv2.waitKey(0)

# Crear la ventana de Tkinter
root = tk.Tk()
root.title("Selector de Área de Detección")

select_button = tk.Button(root, text="Cargar Imagen", command=select_area)
select_button.pack()

# Iniciar el bucle de la interfaz gráfica
root.mainloop()

# Ahora puedes usar los puntos seleccionados para filtrar las detecciones
# Cargar el modelo
model = YOLO('yolov8n.pt')

# Detección de objetos
results = model(img)

# Filtrar las detecciones basadas en el área seleccionada
for deteccion in results[0].boxes:
    x1, y1, x2, y2 = map(int, deteccion.xyxy[0])  # Coordenadas del recuadro
    # Verificar si alguna esquina del rectángulo está dentro del polígono
    if any(cv2.pointPolygonTest(np.array(points), (x, y), False) >= 0 for x, y in [(x1, y1), (x2, y2), (x1, y2), (x2, y1)]):
        # Dibuja la detección
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        clase = int(deteccion.cls[0])  # Clase detectada
        confianza = deteccion.conf[0].item()  # Confianza
        cv2.putText(img, f'Clase: {clase}, Conf: {confianza:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Mostrar y guardar la imagen con las detecciones
cv2.imshow('Detecciones', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
