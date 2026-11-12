import cv2
import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm

import utils

img = cv2.imread("test.jpg")
timeout_ms = 1000

params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

# The vdevice is used as a context manager ("with" statement) to ensure it's released on time.
with VDevice(params) as vdevice:

    # Create an infer model from an HEF:
    infer_model = vdevice.create_infer_model('hailo_tutorials/hefs/yolov8s.hef')

    # Configure the infer model and create bindings for it
    with infer_model.configure() as configured_infer_model:
        bindings = configured_infer_model.create_bindings()

        # Set input and output buffers
        buf_in = utils.load_image(
            img,
            infer_model.input().shape,
            dtype=utils.get_dtype(infer_model.input().format.type)
        )
        bindings.input().set_buffer(buf_in)
        
        buf_out = np.empty(
            infer_model.output().shape,
            dtype=utils.get_dtype(infer_model.output().format.type)
        )
        bindings.output().set_buffer(buf_out)

        # 동기식 추론
        configured_infer_model.run([bindings], timeout_ms)
        buf_res = bindings.output().get_buffer()
        
        utils.print_result(buf_res)
        cv2.imwrite("result.jpg", utils.draw_result(img, buf_res, threshold=0.25))

        # # 비동기 추론
        # job = configured_infer_model.run_async([bindings])
        # job.wait(timeout_ms)