import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

MODEL_PPE    = r'D:\InternProject\YOLOv8_PPEdetect\runs\detect\train-4\weights\best.pt'
MODEL_PERSON = 'yolov8n.pt'

def select_roi_interactive(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image from {image_path}")
        return None
    
    print("\n=== ROI Selection ===")
    print("1. Click and drag to select ROI")
    print("2. Press SPACE or ENTER to confirm")
    print("3. Press 'c' to clear and reselect")
    print("4. Press ESC to cancel\n")
    
    roi = cv2.selectROI("Select ROI (SPACE=confirm, c=clear, ESC=cancel)", img, False, False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        roi_coords = (x, y, x + w, y + h)
        print(f"ROI selected: {roi_coords}")
        return roi_coords
    else:
        print("No ROI selected")
        return None

class PPEDetector:
    
    def __init__(self, conf_ppe=0.05, conf_person=0.005, use_sahi=True, use_sahi_ppe=True, roi=None):
        self.model_ppe    = YOLO(MODEL_PPE)
        self.model_person = YOLO(MODEL_PERSON)
        self.conf_ppe     = conf_ppe
        self.conf_person  = conf_person
        self.unique_ids   = set()
        self.use_sahi     = use_sahi
        self.use_sahi_ppe = use_sahi_ppe
        self.roi          = roi
        
        if use_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=MODEL_PERSON,
                confidence_threshold=conf_person,
                device='cuda:0'
            )
        
        if use_sahi_ppe:
            self.sahi_model_ppe = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=MODEL_PPE,
                confidence_threshold=conf_ppe,
                device='cuda:0'
            )

        self.ppe_classes = {0: 'Helmet', 1: 'Vest'}

        self.colors = {
            'Helmet': (0, 255, 0),
            'Vest':   (0, 165, 255),
        }
        
    def get_ppe_detections(self, frame):
        if self.use_sahi_ppe and self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
            result = get_sliced_prediction(
                roi_frame,
                self.sahi_model_ppe,
                slice_height=64,
                slice_width=64,
                overlap_height_ratio=0.8,
                overlap_width_ratio=0.8,
                postprocess_type="NMS",
                postprocess_match_threshold=0.2,
                verbose=1
            )
            ppe_boxes = []
            for obj in result.object_prediction_list:
                cls_id = obj.category.id
                class_name = self.ppe_classes.get(cls_id)
                if class_name:
                    bbox = obj.bbox
                    px1, py1, px2, py2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                    ppe_boxes.append({
                        'class':  class_name,
                        'conf':   obj.score.value,
                        'box':    (px1 + x1, py1 + y1, px2 + x1, py2 + y1),
                        'center': ((px1 + px2) // 2 + x1, (py1 + py2) // 2 + y1),
                    })
        else:
            if self.roi:
                x1, y1, x2, y2 = self.roi
                roi_frame = frame[y1:y2, x1:x2]
                results = self.model_ppe(roi_frame, conf=self.conf_ppe, iou=0.3, verbose=False)
            else:
                results = self.model_ppe(frame, conf=self.conf_ppe, iou=0.4, verbose=False)
            
            ppe_boxes = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf  = float(box.conf[0])
                    class_name = self.ppe_classes.get(cls_id)
                    if class_name:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        if self.roi:
                            bx1 += x1
                            by1 += y1
                            bx2 += x1
                            by2 += y1
                        ppe_boxes.append({
                            'class':  class_name,
                            'conf':   conf,
                            'box':    (bx1, by1, bx2, by2),
                            'center': ((bx1 + bx2) // 2, (by1 + by2) // 2),
                        })
        return ppe_boxes

    def get_person_detections(self, frame, track=False):
        if self.use_sahi and not track:
            return self._get_person_sahi(frame)
        
        if track:
            results = self.model_person.track(
                frame, conf=self.conf_person, iou=0.4,
                classes=[0], tracker='bytetrack.yaml',
                persist=True, verbose=False
            )
        else:
            results = self.model_person(
                frame, conf=self.conf_person, iou=0.4,
                classes=[0], verbose=False
            )
        persons = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf     = float(box.conf[0])
                track_id = int(box.id[0]) if (track and box.id is not None) else None
                persons.append({'box': (x1, y1, x2, y2), 'conf': conf, 'id': track_id})
        return persons
    
    def _get_person_sahi(self, frame):
        h, w = frame.shape[:2]
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
            result = get_sliced_prediction(
                roi_frame,
                self.sahi_model,
                slice_height=128,
                slice_width=128,
                overlap_height_ratio=0.75,
                overlap_width_ratio=0.75,
                postprocess_type="NMS",
                postprocess_match_threshold=0.2,
                verbose=1
            )
            persons = []
            for obj in result.object_prediction_list:
                if obj.category.id == 0:
                    bbox = obj.bbox
                    px1, py1, px2, py2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                    persons.append({
                        'box': (px1 + x1, py1 + y1, px2 + x1, py2 + y1),
                        'conf': obj.score.value,
                        'id': None
                    })
        else:
            result = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=128,
                slice_width=128,
                overlap_height_ratio=0.75,
                overlap_width_ratio=0.75,
                postprocess_type="NMS",
                postprocess_match_threshold=0.2,
                verbose=1
            )
            persons = []
            for obj in result.object_prediction_list:
                if obj.category.id == 0:
                    bbox = obj.bbox
                    x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                    persons.append({'box': (x1, y1, x2, y2), 'conf': obj.score.value, 'id': None})
        return persons
    
    def _is_in_roi(self, box):
        if self.roi is None:
            return True
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2
    
    def _is_in_roi_center(self, center):
        if self.roi is None:
            return True
        cx, cy = center
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2
    
    def draw_roi(self, frame):
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, 'ROI', (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    def check_ppe_per_person(self, persons, ppe_boxes):
        person_ppe = []
        used_ppe = set()
        
        for person in persons:
            if not self._is_in_roi(person['box']):
                continue
            px1, py1, px2, py2 = person['box']
            has_helmet = False
            has_vest   = False
            for i, ppe in enumerate(ppe_boxes):
                cx, cy = ppe['center']
                if px1 < cx < px2 and py1 < cy < py2:
                    used_ppe.add(i)
                    if ppe['class'] == 'Helmet':
                        has_helmet = True
                    elif ppe['class'] == 'Vest':
                        has_vest = True
            person_ppe.append({
                **person,
                'has_helmet': has_helmet,
                'has_vest':   has_vest,
                'compliant':  has_helmet and has_vest,
            })
        
        valid_ppe = [ppe for i, ppe in enumerate(ppe_boxes) if i in used_ppe]
        return person_ppe, valid_ppe

    def draw_label(self, frame, text, x1, y1, color):
        label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    def draw_results(self, frame, person_ppe, ppe_boxes, counts):
        annotated = frame.copy()
        self.draw_roi(annotated)
        
        overlay = annotated.copy()
        for ppe in ppe_boxes:
            if self.roi and not self._is_in_roi_center(ppe['center']):
                continue
            x1, y1, x2, y2 = ppe['box']
            color = self.colors[ppe['class']]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            self.draw_label(overlay, f"{ppe['conf']:.2f}", x1, y1, color)
        
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        self.draw_count_summary(annotated, counts)
        return annotated

    def draw_count_summary(self, frame, counts):
        h, w = frame.shape[:2]
        summary_height = 100
        summary_width  = 200
        overlay = frame.copy()
        y_start = h - summary_height - 10
        cv2.rectangle(overlay,
                     (w - summary_width - 10, y_start),
                     (w - 10, h - 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, 'PPE Count',
                   (w - summary_width, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.line(frame, (w - summary_width, y_start + 28), (w - 20, y_start + 28), (255, 255, 255), 1)

        items = [
            (f"Person   : {counts['person']}",    (255, 255, 255)),
            (f"Helmet   : {counts['helmet']}",    (0, 255, 0)),
            (f"Vest     : {counts['vest']}",      (0, 165, 255)),
        ]
        y = y_start + 45
        for text, color in items:
            cv2.putText(frame, text, (w - summary_width, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 20

    def _build_counts(self, person_ppe):
        return {
            'person':    len(person_ppe),
            'helmet':    sum(1 for p in person_ppe if p['has_helmet']),
            'vest':      sum(1 for p in person_ppe if p['has_vest']),
        }

    def detect_image(self, image_path, output_path=None, show=True):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        ppe_boxes  = self.get_ppe_detections(frame)
        persons    = self.get_person_detections(frame, track=False)
        person_ppe, valid_ppe = self.check_ppe_per_person(persons, ppe_boxes)
        counts     = self._build_counts(person_ppe)
        annotated  = self.draw_results(frame, person_ppe, valid_ppe, counts)

        if output_path:
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved result to: {output_path}")

        if show:
            cv2.imshow('PPE Detection - Image', annotated)
            print("\nDetection Summary:")
            print(f"  Total People   : {counts['person']}")
            print(f"  Helmet         : {counts['helmet']}")
            print(f"  Vest           : {counts['vest']}")
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return counts

    def detect_video(self, video_path, output_path=None, show=True):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return None

        fps         = int(cap.get(cv2.CAP_PROP_FPS))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        self.unique_ids = set()
        frame_count = 0

        print(f"\nProcessing video: {video_path}")
        print(f"Total frames: {total_frames} | FPS: {fps}")
        print("Processing... Press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            ppe_boxes  = self.get_ppe_detections(frame)
            persons    = self.get_person_detections(frame, track=True)
            person_ppe, valid_ppe = self.check_ppe_per_person(persons, ppe_boxes)
            counts     = self._build_counts(person_ppe)

            for p in person_ppe:
                if p['id'] is not None:
                    self.unique_ids.add(p['id'])

            annotated = self.draw_results(frame, person_ppe, valid_ppe, counts)
            cv2.putText(annotated,
                       f"Frame: {frame_count} | Unique People: {len(self.unique_ids)}",
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow('PPE Detection - Video', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | In frame: {counts['person']} | "
                      f"Unique total: {len(self.unique_ids)} | Compliant: {counts['compliant']}")

        cap.release()
        if writer:
            writer.release()
            print(f"\nSaved result to: {output_path}")
        cv2.destroyAllWindows()

        print("\n" + "="*50)
        print("Video Processing Complete")
        print("="*50)
        print(f"Total frames processed  : {frame_count}")
        print(f"Total unique people     : {len(self.unique_ids)}")

        return {'unique_people': len(self.unique_ids)}


def main():
    image_path = r'D:\InternProject\YOLOv8_PPEdetect\4253CCC96C4C1915_2026-06-22T00-43-28-821Z.png'
    output_path = r'D:\InternProject\YOLOv8_PPEdetect\output_image.jpg'

    # video_path  = r'D:\InternProject\4253CCC96C4C1915_2026-06-22T00-25-01-166Z.webm'
    # output_path = r'D:\InternProject\YOLOv8_PPEdetect\output_video.mp4'

    print("\n" + "="*50)
    print("PPE Detection System - Two-Stage Tracking")
    print("="*50)

    #roi = select_roi_interactive(image_path) # if you don't know the ROI, use this function to select it
    roi = (220, 47, 524, 222) # (x1, y1, x2, y2)

    detector = PPEDetector(conf_ppe=0.05, conf_person=0.005, use_sahi=True, use_sahi_ppe=True, roi=roi)

    roi_status = f"ROI: {roi}" if roi else "ROI: Full Image"
    print(f"conf_ppe: 0.05 | conf_person: 0.005 | SAHI-PPE: 64x64 Ultra-Aggressive")
    print(f"Person SAHI: ON (128x128, 75% overlap)")
    print(f"PPE: ROI-Cropped Detection")
    print(roi_status)
    print("="*50 + "\n")

    detector.detect_image(image_path, output_path, show=True)
    # detector.detect_video(video_path, output_path, show=True)


if __name__ == '__main__':
    main()
