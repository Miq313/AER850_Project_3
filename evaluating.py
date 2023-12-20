from ultralytics import YOLO

image_paths = [
    "ardmega.jpg",
    "arduno.jpg",
    "rasppi.jpg"
]
image_paths = ["data/evaluation/" + image_path for image_path in image_paths]

model = YOLO("model.pt")

results = model(image_paths)
for image_path in image_paths:
    model.predict(image_path, save=True, imgsz=320, conf=0.5)
