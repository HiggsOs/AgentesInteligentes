
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://basilprediction.ddns.net:8000/predict"
image_path = "/home/higgsos/.cache/kagglehub/datasets/csafrit2/plant-leaves-for-image-classification/versions/2/Plants_2/test/Bael diseased (P4b)/0016_0010.JPG"




with open(image_path, "rb") as f:
    files = {"file": ("imagen.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

# === Leer respuesta JSON
data = response.json()
print("Detecciones:", data.get("detections"))

# === Mostrar imagen decodificada desde base64
if "image_base64" in data:
    image_data = base64.b64decode(data["image_base64"])
    image = Image.open(BytesIO(image_data))
    image.show()  # abre la imagen en visor del sistema

    # O si estás en un Jupyter Notebook:
    # from IPython.display import display
    # display(image)
else:
    print("No se recibió la imagen procesada.")
