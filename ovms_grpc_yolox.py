import argparse
import copy
import csv
import os
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from yoloa.camera_api import SensorRealsense
from yoloa.classes import COCO_CLASSES
from yoloa.utils import (
    mkdir,
    multiclass_nms,
    postprocess,
    preprocess,
    convert_log_to_csv,
    calculate_avg_cpu_usage,
    calculate_rcs0_average,
)
from yoloa.visualize import vis

home_dir = os.path.expanduser("~")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_parser():
    parser = argparse.ArgumentParser("openvino inference")
    parser.add_argument("--url", type=str, default="localhost:9000")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ensemble_yolox_tiny_coco80",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        default=f"{home_dir}/client_code/data/test",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--folder_name",
        type=str,
        default="test",
    )
    parser.add_argument(
        "-im",
        "--infer_mode",
        type=str,
        default="img",
        choices=["img", "cam"],
    )
    parser.add_argument(
        "-is",
        "--input_shape",
        type=str,
        default="416,416",
    )
    parser.add_argument(
        "-d",
        "--data_type",
        type=str,
        default="FP32",
        choices=["FP16", "FP32"],
    )

    return parser


class PerformanceLogger:
    def __init__(self):
        self.proc_gpu = None
        self.proc_cpu = None
        self.running = False
        self.thread = None
        self.time_records = []

        self.temp_gpu_log = f"./{current_time}_gpu.txt"
        self.temp_cpu_log = f"./{current_time}_cpu.txt"
        self.gpu_log = f"./log/{current_time}_gpu_log.csv"
        self.cpu_log = f"./log/{current_time}_cpu_log.csv"
        self.time_log = f"./log/{current_time}_time_log.csv"

    def _log_metrics(self):
        cmd_gpu = ["sudo", "intel_gpu_top", "-s", "100"]
        cmd_cpu = ["mpstat", "-P", "ALL", "1"]
        self.proc_gpu = subprocess.Popen(
            cmd_gpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.proc_cpu = subprocess.Popen(
            cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        log_gpu_file = open(self.temp_gpu_log, "w", newline="")
        log_cpu_file = open(self.temp_cpu_log, "w", newline="")

        while self.running:
            gpu_output = self.proc_gpu.stdout.readline()
            cpu_output = self.proc_cpu.stdout.readline()
            if gpu_output:
                log_gpu_file.write(gpu_output)
                log_gpu_file.flush()
            if cpu_output:
                log_cpu_file.write(cpu_output)
                log_cpu_file.flush()

        log_gpu_file.close()
        log_cpu_file.close()

    def start_logging(self):
        self.running = True
        self.thread = threading.Thread(target=self._log_metrics, daemon=True)
        self.thread.start()

    def stop_logging(self):
        self.running = False
        if self.proc_gpu:
            self.proc_gpu.terminate()

        if self.proc_cpu:
            self.proc_cpu.terminate()

        convert_log_to_csv(self.temp_gpu_log, self.gpu_log)
        convert_log_to_csv(self.temp_cpu_log, self.cpu_log)

        os.remove(self.temp_gpu_log)
        os.remove(self.temp_cpu_log)

        with open(self.time_log, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Preprocessing_Time(ms)",
                    "Inference_Time(ms)",
                    "Postprocessing_Time(ms)",
                ]
            )
            writer.writerows(self.time_records)

        print(f"Saved log file: {self.time_log}")
        print(f"Average CPU Usage: {calculate_avg_cpu_usage(self.cpu_log):.3f} %")
        print(f"Average GPU Usage: {calculate_rcs0_average(self.gpu_log):.3f} %")

    def log(self, image_index, total_length, infer_time):
        self.time_records.append([infer_time])
        print(
            f"\r[{image_index} / {total_length}] | Inference Time: {infer_time:.3f} ms\033[K",
            end="",
            flush=True,
        )


def infer_image(client, model, input_path, output_path, input_shape, data_type):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    origin_img = cv2.imread(input_path).astype(np.float32)
    img = origin_img[None, :, :, :].astype(d_type)

    # preproc_s = time.time()
    # img, ratio = preprocess(origin_img, input_shape)
    # img = img[None, :, :, :].astype(d_type)
    # preproc_e = time.time()

    inputs = grpcclient.InferInput("input_images", img.shape, datatype=data_type)
    inputs.set_data_from_numpy(img)
    outputs = grpcclient.InferRequestedOutput("output_results")

    infer_s = time.time()
    res = client.infer(model_name=model, inputs=[inputs], outputs=[outputs])
    infer_e = time.time()

    res = res.as_numpy("output_results")
    res_copy = np.copy(res).astype(d_type)

    # postproc_s = time.time()
    # pred = postprocess(res_copy, input_shape)[0]
    # boxes, scores = pred[:, :4], pred[:, 4:5] * pred[:, 5:]
    # boxes_xyxy = np.ones_like(boxes)
    # boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    # boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    # boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    # boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    # boxes_xyxy /= ratio

    # dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.3)
    # if dets is not None:
    #     final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    #     origin_img = vis(
    #         origin_img,
    #         final_boxes,
    #         final_scores,
    #         final_cls_inds,
    #         conf=0.3,
    #         class_names=COCO_CLASSES,
    #     )
    # postproc_e = time.time()

    if len(res_copy) != 0:
        class_id = res_copy[:, 0]
        score = res_copy[:, 1]
        box_coord = res_copy[:, 2:6]
        origin_img = vis(
            origin_img, box_coord, score, class_id, conf=0.3, class_names=COCO_CLASSES
        )

    output_path = os.path.join(output_path, os.path.basename(input_path))
    cv2.imwrite(output_path, origin_img)

    return (infer_e - infer_s) * 1000


def infer_camera(client, model_name, input, input_shape, data_type):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    img, ratio = preprocess(input, input_shape)
    img = img[None, :, :, :].astype(d_type)

    inputs = grpcclient.InferInput("images", [1, 3, 416, 416], datatype=data_type)
    outputs = grpcclient.InferRequestedOutput("output")
    inputs.set_data_from_numpy(img)

    res = client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])
    res = res.as_numpy("output")
    res_copy = np.copy(res).astype(d_type)

    pred = postprocess(res_copy, input_shape)[0]

    boxes, scores = pred[:, :4], pred[:, 4:5] * pred[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.3)
    output = copy.copy(input)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        output = vis(
            input,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=0.3,
            class_names=COCO_CLASSES,
        )

    return output


def main():
    args = make_parser().parse_args()
    client = grpcclient.InferenceServerClient(url=args.url)
    input_shape = tuple(map(int, args.input_shape.split(",")))
    logger = PerformanceLogger()

    if not (
        client.is_server_live()
        and client.is_server_ready()
        and client.is_model_ready(args.model)
    ):
        print("Triton server or model is not ready.")
        return

    model_metadata = client.get_model_metadata(args.model)
    print("model_metadata:", model_metadata)

    if args.infer_mode == "img":
        output_path = f"./output/{args.folder_name}"
        mkdir(output_path)

        image_files = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

        logger.start_logging()
        total_preproc_time, total_infer_time, total_postproc_time = 0, 0, 0
        for image_index, image_path in enumerate(image_files, start=1):
            infer_time = infer_image(
                client, args.model, image_path, output_path, input_shape, args.data_type
            )
            total_infer_time += infer_time
            logger.log(image_index, len(image_files), infer_time)

        logger.stop_logging()

        # print(f"Avg preprocess time: {total_preproc_time / len(image_files):.3f} ms")
        print(f"Avg inference time: {total_infer_time / len(image_files):.3f} ms")
        # print(f"Avg postprocess time: {total_postproc_time / len(image_files):.3f} ms")

    elif args.infer_mode == "cam":
        sensor = SensorRealsense()
        while True:
            input = sensor.get_video_from_pipeline()[1][0]
            output = infer_camera(
                client, args.model, input, input_shape, args.data_type
            )
            cv2.namedWindow("camera viewer", cv2.WINDOW_NORMAL)
            cv2.imshow("camera viewer", output)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
