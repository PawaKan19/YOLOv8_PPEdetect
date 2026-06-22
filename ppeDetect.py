import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
from collections import defaultdict


class PPEDetector:
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(r'D:\InternProject\YOLOv8_PPEdetect\runs\detect\train-4\weights\best.pt')
        self.conf_threshold = conf_threshold
        
        self.ppe_classes = {
            'Helmet': 0,
            'Vest': 1
        }
        
        self.colors = {
            'Helmet': (0, 255, 0),
            'Vest': (0, 165, 255)
        }
        
    def count_detections(self, results):
        counts = defaultdict(int)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf >= self.conf_threshold:
                    for class_name, class_id in self.ppe_classes.items():
                        if cls_id == class_id:
                            counts[class_name] += 1
                            break
        
        return counts
    
    def draw_detections(self, frame, results, counts):
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_name = None
                    color = (255, 255, 255)
                    
                    for name, class_id in self.ppe_classes.items():
                        if cls_id == class_id:
                            class_name = name
                            color = self.colors.get(name, (255, 255, 255))
                            break
                    
                    if class_name:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f'{class_name}: {conf:.2f}'
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1),
                                    color, -1)
                        
                        cv2.putText(annotated_frame, label,
                                  (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 0), 2)
        
        self.draw_count_summary(annotated_frame, counts)
        
        return annotated_frame
    
    def draw_count_summary(self, frame, counts):
        h, w = frame.shape[:2]
        
        summary_height = 100
        summary_width = 250
        overlay = frame.copy()
        
        cv2.rectangle(overlay,
                     (w - summary_width - 10, 10),
                     (w - 10, summary_height + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "PPE Detection Count",
                   (w - summary_width, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)
        
        cv2.line(frame,
                (w - summary_width, 45),
                (w - 20, 45),
                (255, 255, 255), 1)
        
        y_offset = 70
        for class_name in ['Helmet', 'Vest']:
            count = counts.get(class_name, 0)
            color = self.colors.get(class_name, (255, 255, 255))
            
            text = f"{class_name}: {count}"
            cv2.putText(frame, text,
                       (w - summary_width, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       color, 2)
            y_offset += 30
    
    def detect_image(self, image_path, output_path=None, show=True):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        counts = self.count_detections(results)
        annotated_frame = self.draw_detections(frame, results, counts)
        
        if output_path:
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"Saved result to: {output_path}")
        
        if show:
            cv2.imshow('PPE Detection - Image', annotated_frame)
            print("\nDetection Summary:")
            print(f"  Helmet: {counts.get('Helmet', 0)}")
            print(f"  Vest: {counts.get('Vest', 0)}")
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return counts
    
    def detect_video(self, video_path, output_path=None, show=True):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_counts = defaultdict(int)
        
        print(f"\nProcessing video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print("\nProcessing... Press 'q' to quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            counts = self.count_detections(results)
            
            for class_name, count in counts.items():
                total_counts[class_name] += count
            
            annotated_frame = self.draw_detections(frame, results, counts)
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
            
            if writer:
                writer.write(annotated_frame)
            
            if show:
                cv2.imshow('PPE Detection - Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        if writer:
            writer.release()
            print(f"\nSaved result to: {output_path}")
        cv2.destroyAllWindows()
        
        avg_counts = {k: v / frame_count for k, v in total_counts.items()}
        
        print("\n" + "="*50)
        print("Video Processing Complete")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"\nAverage Detection Counts:")
        print(f"  Helmet: {avg_counts.get('Helmet', 0):.2f}")
        print(f"  Vest: {avg_counts.get('Vest', 0):.2f}")
        print(f"\nTotal Detections:")
        print(f"  Helmet: {total_counts.get('Helmet', 0)}")
        print(f"  Vest: {total_counts.get('Vest', 0)}")
        
        return avg_counts


def main():
    image_path = r'D:\InternProject\YOLOv8_PPEdetect\4253CCC96C4C1915_2026-06-22T00-43-28-821Z.png'
    output_path = r'D:\InternProject\YOLOv8_PPEdetect\output_image.jpg'
    
    # video_path = r'D:\InternProject\4253CCC96C4C1915_2026-06-22T00-25-01-166Z.webm'
    # output_path = r'D:\InternProject\YOLOv8_PPEdetect\output_video.mp4'
    
    print("\n" + "="*50)
    print("PPE Detection System - YOLOv8n")
    print("="*50)
    
    detector = PPEDetector(conf_threshold=0.12)
    
    print(f"Confidence threshold: 0.12")
    print("="*50 + "\n")
    
    detector.detect_image(image_path, output_path, show=True)
    # detector.detect_video(video_path, output_path, show=True)


if __name__ == '__main__':
    main()
