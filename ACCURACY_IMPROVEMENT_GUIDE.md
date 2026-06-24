# PPE Detection Accuracy Improvement Guide

## Current Performance (24 Jun 2026)
- **Person Detection**: 51/52 (98.1%) ✅ EXCELLENT
- **Helmet Detection**: 15/51 (29.4%) ❌ POOR
- **Vest Detection**: 19/51 (37.3%) ❌ POOR

## Root Cause Analysis

### 1. PPE Model Limitation (PRIMARY ISSUE)
- Current: YOLOv8n trained on generic PPE dataset
- Problem: Dataset doesn't match this use case
  - Different viewing angle (bird's eye vs. ground level)
  - Different scale (small PPE in crowded scene)
  - Different occlusion patterns

### 2. Technical Constraints
- ROI Size: 304×175 px for 51 people
- Average person size: ~30-50 px
- Average PPE size: ~10-20 px (very small!)
- Occlusion: Heavy overlap in crowded areas

---

## Improvement Strategies (Ranked by Impact)

### Tier 1: Model-Level Solutions (Highest Impact)

#### Option 1: Retrain with Similar Data ⭐⭐⭐⭐⭐
**Impact**: +50-100% accuracy
**Effort**: High (1-2 weeks)
**Cost**: Medium (annotation time)

**Steps**:
1. Collect 500-1000 images similar to this scene:
   - Bird's eye view angle
   - Crowded construction sites
   - Small PPE objects
   - Similar lighting

2. Annotate with high quality:
   - Use labelImg or CVAT
   - Mark even partially visible PPE
   - Include occluded cases

3. Train with YOLOv8m or YOLOv8l:
   ```bash
   yolo train model=yolov8m.pt data=ppe_custom.yaml epochs=100 imgsz=640
   ```

4. Use augmentation:
   - Random occlusion (10-30%)
   - Scale (0.5-1.5x)
   - Brightness/Contrast
   - Rotation (±15°)

**Expected Result**:
- Helmet: 15 → 30-40 (100-167% improvement)
- Vest: 19 → 35-45 (84-137% improvement)

---

#### Option 2: Upgrade to YOLOv8m/l ⭐⭐⭐⭐
**Impact**: +20-40% accuracy
**Effort**: Low (1 hour)
**Cost**: Low (slower inference)

**Implementation**:
```python
# In ppeDetect.py, line 5:
MODEL_PPE = r'D:\InternProject\YOLOv8_PPEdetect\runs\detect\train-yolov8m\weights\best.pt'
```

**Train new model**:
```bash
# Retrain with yolov8m
yolo train model=yolov8m.pt data=ppe.yaml epochs=100 imgsz=640

# Or yolov8l for maximum accuracy
yolo train model=yolov8l.pt data=ppe.yaml epochs=100 imgsz=640
```

**Comparison**:
| Model | mAP | Speed | Size |
|-------|-----|-------|------|
| YOLOv8n | 37.3 | 80 FPS | 6 MB |
| YOLOv8m | 50.2 | 25 FPS | 52 MB |
| YOLOv8l | 52.9 | 12 FPS | 87 MB |

**Expected Result**:
- Helmet: 15 → 20-25 (+33-67%)
- Vest: 19 → 25-30 (+32-58%)

---

#### Option 3: Two-Stage PPE Detection ⭐⭐⭐
**Impact**: +15-30% accuracy
**Effort**: Medium (1 day)
**Cost**: 2x slower

**Concept**: Crop each person box → detect PPE in cropped region

**Implementation**:
```python
def get_ppe_per_person(self, frame, person_box):
    """Detect PPE within a single person's bounding box"""
    x1, y1, x2, y2 = person_box
    # Add padding
    pad = 10
    person_crop = frame[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]
    
    # Detect PPE in cropped region
    results = self.model_ppe(person_crop, conf=0.05, imgsz=320)
    # ... process results
```

**Expected Result**:
- Helmet: 15 → 20-25
- Vest: 19 → 25-30
- Reduced false positives

---

### Tier 2: Parameter Tuning (Medium Impact)

#### Option 4: Ultra-Aggressive SAHI ⭐⭐⭐
**Impact**: +10-20% accuracy
**Effort**: Low (5 minutes)
**Cost**: 3-5x slower

**Current Settings**:
```python
slice_height=80, slice_width=80
overlap=0.75
conf=0.08
```

**New Settings**:
```python
slice_height=64          # Smaller slices
slice_width=64
overlap_height_ratio=0.8  # More overlap
overlap_width_ratio=0.8
conf_ppe=0.05            # Lower threshold
postprocess_match_threshold=0.2  # Accept more overlaps
```

**Expected Result**:
- Helmet: 15 → 18-22
- Vest: 19 → 23-28
- Slices: 78 → 120-150

---

#### Option 5: Multi-Scale Detection ⭐⭐
**Impact**: +5-15% accuracy
**Effort**: Medium (2 hours)
**Cost**: 2-3x slower

**Concept**: Run detection at multiple image scales

```python
def multi_scale_detection(self, frame):
    scales = [0.8, 1.0, 1.2]
    all_detections = []
    
    for scale in scales:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
        detections = self.get_ppe_detections(resized)
        # Scale back coordinates
        scaled_detections = self._scale_boxes(detections, 1/scale)
        all_detections.extend(scaled_detections)
    
    # NMS to merge
    return self._nms(all_detections)
```

---

### Tier 3: Post-Processing (Low Impact)

#### Option 6: Spatial Filtering ⭐⭐
**Impact**: +5-10% accuracy (reduce false positives)
**Effort**: Low (30 minutes)

**Already Implemented**: ✅
- Filter PPE outside ROI
- Filter PPE not in person boxes

**Additional**:
```python
# Filter by size
min_helmet_area = 50  # pixels
min_vest_area = 100

# Filter by aspect ratio
helmet_aspect_ratio = 0.8-1.5  # roughly square
vest_aspect_ratio = 0.6-1.2    # slightly vertical
```

---

#### Option 7: Temporal Smoothing (Video Only) ⭐
**Impact**: +10-20% for video
**Effort**: Medium (3 hours)

```python
# Track PPE across frames
# If PPE detected in 3/5 consecutive frames → confirm
# Reduce flickering false positives
```

---

## Recommended Action Plan

### Phase 1: Quick Wins (Today)
1. ✅ Reduce opacity to 75% (DONE)
2. ✅ Filter PPE outside person boxes (DONE)
3. Try Option 4: Ultra-Aggressive SAHI
   - Expected: +10-15 detections
   - Time: 5 minutes

### Phase 2: Medium Effort (This Week)
1. Option 2: Retrain with YOLOv8m
   - Expected: +5-10 detections
   - Time: 2-3 hours (training overnight)

2. Option 3: Two-Stage PPE Detection
   - Expected: +5-10 detections
   - Time: 1 day

### Phase 3: Long-term (Next 1-2 Weeks)
1. Option 1: Collect & Retrain with Similar Data
   - Expected: +15-25 detections (BEST RESULT)
   - Time: 1-2 weeks

---

## Expected Final Results

| Approach | Helmet | Vest | Total Effort |
|----------|--------|------|--------------|
| Current | 15 | 19 | - |
| + Ultra SAHI | 18-22 | 23-28 | 5 min |
| + YOLOv8m | 20-25 | 25-30 | 3 hours |
| + Two-Stage | 25-30 | 30-35 | 1 day |
| + Custom Dataset | 35-45 | 40-48 | 1-2 weeks |

---

## Technical Limitations

Even with best practices, some cases are **fundamentally undetectable**:
1. PPE completely occluded by other people
2. PPE outside camera view
3. PPE too small (<5 pixels)
4. Extreme lighting conditions

**Realistic Maximum**: 80-90% detection rate (40-45 helmets, 43-48 vests)

---

## Conclusion

**For immediate improvement** (today):
- Run Ultra-Aggressive SAHI (Option 4)
- Expected: Helmet 15→20, Vest 19→25

**For best long-term results** (1-2 weeks):
- Collect similar data + Retrain with YOLOv8m (Option 1+2)
- Expected: Helmet 15→35-40, Vest 19→40-45
- This is the ONLY way to achieve 70-80%+ accuracy

**Current bottleneck**: PPE model not trained for this specific use case
