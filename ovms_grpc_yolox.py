import psutil
import time
import subprocess
import threading
import csv
import argparse
import os
import cv2
import copy
import numpy as np
import tritonclient.grpc as grpcclient

from yoloa.utils import mkdir, multiclass_nms, postprocess, preprocess
from yoloa.classes import COCO_CLASSES
from yoloa.visualize import vis
from yoloa.camera_api import SensorRealsense
from datetime import datetime

home_dir = os.path.expanduser("~")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_parser():
    parser = argparse.ArgumentParser("openvino inference")
    parser.add_argument("--url", type=str, default="localhost:9000")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_tiny_coco80_ov_fp32",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        default=f"{home_dir}/YOLOX_inference/data/test",
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
        "-dt",
        "--data_type",
        type=str,
        default="FP32",
        choices=["FP16", "FP32"],
    )

    return parser


class PerformanceLogger:
    def __init__(self, log_file=f"./log/{current_time}_resource_log.csv", interval=1):
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = time.time()
        self.metrics_records = []
        self.time_records = []

    def _get_gpu_usage(self):
        try:
            process = subprocess.Popen(
                ["intel_gpu_top", "-l", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output, _ = process.communicate(timeout=1)
            render_usage, video_usage = 0, 0
            for line in output.split("\n"):
                if "Render/3D" in line:
                    render_usage = float(line.split()[1].replace("%", ""))
                elif "Video" in line:
                    video_usage = float(line.split()[1].replace("%", ""))
            return render_usage, video_usage
        except Exception:
            return 0, 0

    def _get_network_usage(self):
        net_io = psutil.net_io_counters()
        return net_io.bytes_sent / 1024, net_io.bytes_recv / 1024

    def _log_metrics(self):
        while self.running:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = psutil.cpu_percent(interval=0.5)
            mem_usage = psutil.virtual_memory().percent
            gpu_render, gpu_video = self._get_gpu_usage()
            net_sent, net_recv = self._get_network_usage()

            self.metrics_records.append(
                [
                    timestamp,
                    cpu_usage,
                    mem_usage,
                    gpu_render,
                    gpu_video,
                    net_sent,
                    net_recv,
                ]
            )
            print()
            print(
                f"[{timestamp}] CPU: {cpu_usage}% | MEM: {mem_usage}% | GPU(Render): {gpu_render}% | GPU(Video): {gpu_video}% | Net TX: {net_sent} KB | Net RX: {net_recv} KB"
            )
            print()

            time.sleep(self.interval)

    def start_logging(self):
        self.running = True
        self.thread = threading.Thread(target=self._log_metrics)
        self.thread.start()

    def stop_logging(self):
        self.running = False
        if self.thread:
            self.thread.join()

        with open(f"./log/{current_time}_resource_log.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Timestamp",
                    "CPU_Usage(%)",
                    "Memory_Usage(%)",
                    "GPU_Render(%)",
                    "GPU_Video(%)",
                    "Net_TX(KB)",
                    "Net_RX(KB)",
                ]
            )
            writer.writerows(self.metrics_records)
        print(f"Resource log saved")

        with open(f"./log/{current_time}_time_log.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Preprocessing_Time(ms)", "Inference_Time(ms)"])
            writer.writerows(self.time_records)
        print("Time log saved")

    def log(self, prep_time, infer_time):
        self.time_records.append([prep_time, infer_time])
        print(f"[Preprocess Time: {prep_time} ms | Inference Time: {infer_time} ms")


def infer_image(client, model, input_path, output_path, input_shape, data_type):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    prep_s = time.time()
    origin_img = cv2.imread(input_path)
    img, ratio = preprocess(origin_img, input_shape)
    img = img[None, :, :, :].astype(d_type)
    prep_e = time.time()

    inputs = grpcclient.InferInput("images", img.shape, datatype=data_type)
    inputs.set_data_from_numpy(img)
    outputs = grpcclient.InferRequestedOutput("output")

    infer_s = time.time()
    res = client.infer(model_name=model, inputs=[inputs], outputs=[outputs])
    infer_e = time.time()

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
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(
            origin_img,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=0.3,
            class_names=COCO_CLASSES,
        )

    output_path = os.path.join(output_path, os.path.basename(input_path))
    cv2.imwrite(output_path, origin_img)

    return (prep_e - prep_s) * 1000, (infer_e - infer_s) * 1000


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
    logger.start_logging()

    if not (
        client.is_server_live()
        and client.is_server_ready()
        and client.is_model_ready(args.model)
    ):
        print("Triton server or model is not ready.")
        logger.stop_logging()
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

        total_prep_time, total_infer_time = 0, 0
        for image_path in image_files:
            prep_time, infer_time = infer_image(
                client, args.model, image_path, output_path, input_shape, args.data_type
            )
            total_prep_time += prep_time
            total_infer_time += infer_time
            logger.log(prep_time, infer_time)

        print(f"Avg preprocess time: {total_prep_time / len(image_files):.3f} ms")
        print(f"Avg inference time: {total_infer_time / len(image_files):.3f} ms")

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

    logger.stop_logging()


if __name__ == "__main__":
    main()
