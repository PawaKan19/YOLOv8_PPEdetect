from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="D:\InternProject\YOLOv8_PPEdetect\Dataset\data.yaml",
        epochs=250,
        imgsz=320,
        batch=8,
        workers=0
    )


if __name__ == '__main__':
    main()