import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

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

class KYTDetector:
    
    def __init__(self, conf_person=0.005, use_sahi=True, roi=None, kyt_threshold=3):
        self.model_person = YOLO(MODEL_PERSON)
        self.conf_person  = conf_person
        self.unique_ids   = set()
        self.use_sahi     = use_sahi
        self.roi          = roi
        self.kyt_threshold = kyt_threshold
        
        if use_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=MODEL_PERSON,
                confidence_threshold=conf_person,
                device='cuda:0'
            )
        
    def count_people_in_roi(self, persons):
        count = 0
        for person in persons:
            if self._is_in_roi(person['box']):
                count += 1
        return count

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
    
    def draw_roi(self, frame):
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, 'ROI', (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    def draw_kyt_alert(self, frame, person_count):
        if person_count >= self.kyt_threshold:
            h, w = frame.shape[:2]
            alert_text = "Have KYT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            text_size, _ = cv2.getTextSize(alert_text, font, font_scale, thickness)
            text_w, text_h = text_size
            
            padding = 10
            box_x = w - text_w - padding * 2 - 10
            box_y = 10
            box_w = text_w + padding * 2
            box_h = text_h + padding * 2
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            text_x = box_x + padding
            text_y = box_y + text_h + padding
            cv2.putText(frame, alert_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    def draw_results(self, frame, person_count):
        annotated = frame.copy()
        self.draw_roi(annotated)
        self.draw_kyt_alert(annotated, person_count)
        return annotated


    def detect_image(self, image_path, output_path=None, show=True):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        persons = self.get_person_detections(frame, track=False)
        person_count = self.count_people_in_roi(persons)
        annotated = self.draw_results(frame, person_count)

        if output_path:
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved result to: {output_path}")

        if show:
            cv2.imshow('KYT Detection - Image', annotated)
            print("\nDetection Summary:")
            print(f"  People in ROI: {person_count}")
            if person_count >= self.kyt_threshold:
                print(f"  Status: Have KYT (>={self.kyt_threshold} people)")
            else:
                print(f"  Status: Normal (<{self.kyt_threshold} people)")
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {'person_count': person_count, 'has_kyt': person_count >= self.kyt_threshold}

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

        self.unique_ids = set()
        frame_count = 0
        kyt_frames = 0

        print(f"\nProcessing video: {video_path}")
        print(f"Total frames: {total_frames} | FPS: {fps}")
        print("Processing... Press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            persons = self.get_person_detections(frame, track=True)
            person_count = self.count_people_in_roi(persons)
            
            for person in persons:
                if person['id'] is not None and self._is_in_roi(person['box']):
                    self.unique_ids.add(person['id'])
            
            if person_count >= self.kyt_threshold:
                kyt_frames += 1

            annotated = self.draw_results(frame, person_count)
            cv2.putText(annotated,
                       f"Frame: {frame_count} | Unique People: {len(self.unique_ids)}",
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow('KYT Detection - Video', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            if frame_count % 30 == 0:
                status = "Have KYT" if person_count >= self.kyt_threshold else "Normal"
                print(f"Frame {frame_count} | People in ROI: {person_count} | "
                      f"Unique total: {len(self.unique_ids)} | Status: {status}")

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
        print(f"Frames with KYT alert   : {kyt_frames} ({kyt_frames/frame_count*100:.1f}%)")

        return {'unique_people': len(self.unique_ids), 'kyt_frames': kyt_frames, 'total_frames': frame_count}


def main():
    image_path = r'D:\InternProject\YOLOv8_PPEdetect\4253CCC96C4C1915_2026-06-22T00-43-28-821Z.png'
    output_path = r'D:\InternProject\YOLOv8_PPEdetect\output_kyt.jpg'

    # video_path  = r'D:\InternProject\4253CCC96C4C1915_2026-06-22T00-25-01-166Z.webm'
    # output_path = r'D:\InternProject\YOLOv8_PPEdetect\output_kyt_video.mp4'

    print("\n" + "="*50)
    print("KYT Detection System - People Counter")
    print("="*50)

    #roi = select_roi_interactive(image_path)
    roi = (220, 47, 524, 222) # (x1, y1, x2, y2)

    detector = KYTDetector(conf_person=0.005, use_sahi=True, roi=roi, kyt_threshold=3)

    roi_status = f"ROI: {roi}" if roi else "ROI: Full Image"
    print(f"conf_person: 0.005 | Person SAHI: ON (128x128, 75% overlap)")
    print(f"KYT Threshold: >= {detector.kyt_threshold} people")
    print(roi_status)
    print("="*50 + "\n")

    detector.detect_image(image_path, output_path, show=True)
    # detector.detect_video(video_path, output_path, show=True)


if __name__ == '__main__':
    main()
