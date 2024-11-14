import cv2
from ultralytics import YOLO

# Cargar el modelo preentrenado
model = YOLO('yolov8n.pt')  # Cambia a 'yolov8s.pt', etc., si es necesario

# Ruta de la imagen
image_path = 'images\image_3.jpeg'  # Usa '/' o '\\' para evitar problemas de escape

# Umbral de confianza
confidence_threshold = 0.2  # Cambia este valor según tus necesidades

# Detección de objetos en la imagen
results = model(image_path)

# Cargar la imagen original para dibujar los recuadros
img = cv2.imread(image_path)

# Obtener las detecciones
detecciones = results[0].boxes

for deteccion in detecciones:
    confianza = deteccion.conf[0].item()  # Confianza de la detección
    if confianza >= confidence_threshold:  # Verificar si la confianza supera el umbral
        x1, y1, x2, y2 = map(int, deteccion.xyxy[0])  # Coordenadas del recuadro
        clase = int(deteccion.cls[0])  

        # Dibujar el recuadro
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Recuadro en color rojo
        # Añadir texto con la clase y confianza
        cv2.putText(img, f'Clase: {clase}, Conf: {confianza:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


cv2.imshow('Detecciones', img)
cv2.waitKey(0)  # Espera hasta que se presione una tecla
cv2.destroyAllWindows()  # Cierra la ventana

# Guardar imagen con detecciones
cv2.imwrite('imagen_con_detecciones.jpg', img)  # Guarda la imagen con recuadros
