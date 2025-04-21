from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image, ImageDraw
import base64

app = FastAPI()

# === Cargar modelo y redefinir nombres ===
modelo_path = "runs/detect/train2/weights/best.pt"
model = YOLO(modelo_path)
model.model.names = {
    0: 'Unhealthy',
    1: 'Healthy'
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen como PIL
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Realizar inferencia usando PIL (¡no NumPy!)
        results = model.predict(image, conf=0.25)
        boxes = results[0].boxes

        # Procesar resultados
        detecciones = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.model.names[cls]
                detecciones.append({
                    "class_id": cls,
                    "label": label,
                    "confidence": round(conf, 3)
                })
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="lime")

        # Convertir imagen a base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print(f"[INFO] Detecciones: {detecciones}")
        return JSONResponse(content={
            "detections": detecciones,
            "image_base64": img_base64
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
