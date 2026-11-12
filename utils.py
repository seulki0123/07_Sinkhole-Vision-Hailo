import cv2
import numpy as np

def get_dtype(format_type):
    return {
        "UINT8": np.uint8,
        "FLOAT32": np.float32,
    }[format_type.name]
    
def load_image(img, shape, dtype):
    img = cv2.imread(img) if isinstance(img, str) else img
    img = cv2.resize(img, (shape[1], shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img, dtype=dtype)
    
def print_result(results):
    print("- Class Number:", len(results))
    print("- Detected:")
    for class_id, detections in enumerate(results):
        if len(detections) > 0:
            print(f"Class ID: {class_id}")
            print(detections)
            
def draw_result(img, results, threshold=0.25):
    
    h, w = img.shape[:2]

    for class_id, detections in enumerate(results):
        for y1, x1, y2, x2, conf in detections:
            if conf < threshold:
                continue

            x1, x2 = int(x1*w), int(x2*w)
            y1, y2 = int(y1*h), int(y2*h)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(img, f"{class_id}: {conf:.2f}",
                        (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1)

    return img