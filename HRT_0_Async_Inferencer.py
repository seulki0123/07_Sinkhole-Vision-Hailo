
import cv2
import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm

class HailoDetector:
    def __init__(self, hef_path):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.device = VDevice(params)
        self.model = self.device.create_infer_model(hef_path)
        self.config = self.model.configure()

    def run(self, img):
        bindings = self.config.create_bindings()

        input_buf = self._load_image(
            img, self.model.input().shape,
            dtype=self._get_dtype(self.model.input().format.type)
        )
        output_buf = np.empty(
            self.model.output().shape,
            dtype=self._get_dtype(self.model.output().format.type)
        )

        bindings.input().set_buffer(input_buf)
        bindings.output().set_buffer(output_buf)

        self.config.run([bindings], 1000)

        result = bindings.output().get_buffer()
        self._print_result(result)

        return self._draw_result(img, result, threshold=0.25)

    def close(self):
        self.config.shutdown()
        self.device.release()


    """
    utility functions for image processing and result visualization
    """
    def _get_dtype(self, format_type):
        return {
            "UINT8": np.uint8,
            "FLOAT32": np.float32,
        }[format_type.name]
        
    def _load_image(self, img, shape, dtype):
        img = cv2.imread(img) if isinstance(img, str) else img
        img = cv2.resize(img, (shape[1], shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img, dtype=dtype)
        
    def _print_result(self, results):
        print("- Class Number:", len(results))
        print("- Detected:")
        for class_id, detections in enumerate(results):
            if len(detections) > 0:
                print(f"Class ID: {class_id}")
                print(detections)
                
    def _draw_result(self, img, results, threshold=0.25):
        
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
