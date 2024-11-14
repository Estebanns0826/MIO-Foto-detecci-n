from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Cargar el modelo

# Diccionario para mapear los números de clase a etiquetas
class_labels = {
    2: "auto",          # Clase para autos
    3: "motocicleta",   # Clase para motocicletas
    5: "bus"            # Clase para buses
}

# Filtramos solo las clases de interés: motocicletas, autos y buses
target_classes = {2, 3, 5}  # Clases: auto, motocicleta, bus

@app.route('/')
def index():
    return render_template('front.html')  # Renderiza front.html

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    area = data['area']
    image_data = data['image'].split(',')[1]  # Extraer la parte de base64

    # Decodificar la imagen
    img_data = base64.b64decode(image_data)
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convierte los puntos a un formato adecuado
    points = np.array(area, dtype=np.int32)

    # Detección de objetos
    results = model(img)

    detecciones = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas del recuadro
        class_id = int(detection.cls[0])  # ID de clase

        # Filtrar solo las clases de interés
        if class_id in target_classes:
            # Verificar si alguna esquina está dentro del polígono
            if any(cv2.pointPolygonTest(points, (x, y), False) >= 0 for x, y in [(x1, y1), (x2, y2), (x1, y2), (x2, y1)]):
                detecciones.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection.conf[0].item(),
                    'class': class_labels[class_id],  # Usar etiqueta en lugar de número de clase
                })

    return jsonify(detecciones)

if __name__ == '__main__':
    app.run(debug=True)
