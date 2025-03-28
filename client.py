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
from dotenv import load_dotenv

from bytetrack.byte_tracker import BYTETracker
from dataclass.detect_results import DetectionResults
from dataclass.track_results import TrackResults
from utils.camera_api import SensorRealsense
from utils.classes import COCO_CLASSES
from utils.tools import (
    calculate_avg_cpu_usage,
    calculate_avg_gpu_usage,
    convert_log_to_csv,
    mkdir,
    multiclass_nms,
    postprocess,
    preprocess,
)
from utils.visualize import vis

TRITON_URL = "192.168.105.190:8001"
OVMS_URL = "192.168.105.194:9000"

home_dir = os.path.expanduser("~")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_parser():
    parser = argparse.ArgumentParser()

    # 기본 설정
    parser.add_argument("--url", default=OVMS_URL)
    parser.add_argument("--model", default="ensemble_yolox_tiny_coco80")

    # 데이터 및 경로 관련
    parser.add_argument("--input_path", default=f"{home_dir}/client_code/data/test")
    parser.add_argument("--output_path", default=f"{home_dir}/client_code/output/test")

    # 모델 및 추론 설정
    parser.add_argument("--model_shape", default="416,416")
    parser.add_argument("--infer_mode", choices=["img", "cam"], default="img")
    parser.add_argument("--data_type", choices=["FP16", "FP32"], default="FP32")

    # 기타 옵션
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--tracker", action="store_true")

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
            writer.writerow(["Inference_Time(ms)"])
            writer.writerows(self.time_records)

        if self.gpu_log:
            if calculate_avg_gpu_usage(self.gpu_log):
                print(
                    f"[✓] Average GPU Usage: {calculate_avg_gpu_usage(self.gpu_log):.3f} %"
                )

        if self.cpu_log:
            if calculate_avg_cpu_usage(self.cpu_log):
                print(
                    f"[✓] Average CPU Usage: {calculate_avg_cpu_usage(self.cpu_log):.3f} %"
                )

    def log(self, infer_time):
        self.time_records.append([infer_time])


def infer_image(
    client, model, input_path, output_path, model_shape, data_type, tracker
):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    origin_img = cv2.imread(input_path).astype(np.uint8)
    image_byte = cv2.imencode(".jpg", origin_img)[1].tobytes()
    img_bt = np.array([image_byte], dtype=np.object_)

    inputs = [grpcclient.InferInput("input_images", [1], "BYTES")]
    inputs[0].set_data_from_numpy(img_bt)
    outputs = grpcclient.InferRequestedOutput("output_results")

    infer_s = time.time()
    infer_result = client.infer(model_name=model, inputs=inputs, outputs=[outputs])
    infer_result = infer_result.as_numpy("output_results").astype(d_type)
    infer_e = time.time()

    detect_result = DetectionResults.from_array(infer_result)
    print(detect_result)

    track_s, track_e = 0, 0
    track_id = None

    if detect_result:
        class_id = [det.class_id for det in detect_result]
        score = [det.score for det in detect_result]
        box_coord = [(det.x1, det.y1, det.x2, det.y2) for det in detect_result]

        if tracker:
            track_s = time.time()
            track_result = tracker.update(
                class_id=class_id, score=score, box_coord=box_coord
            )
            track_e = time.time()

            final_result = TrackResults.from_list(track_result)
            print(final_result)

            if final_result:
                origin_img = vis(origin_img, final_result)
            else:
                origin_img = vis(origin_img, class_id, track_id, score, box_coord)

        else:
            origin_img = vis(origin_img, class_id, track_id, score, box_coord)

    output_path = os.path.join(output_path, os.path.basename(input_path))
    cv2.imwrite(output_path, origin_img.astype(np.uint8))

    if tracker:
        return (infer_e - infer_s) * 1000, (track_e - track_s) * 1000
    else:
        return (infer_e - infer_s) * 1000


def infer_camera(client, model_name, input, model_shape, data_type):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    img, ratio = preprocess(input, model_shape)
    img = img[None, :, :, :].astype(d_type)

    inputs = grpcclient.InferInput("images", [1, 3, 416, 416], datatype=data_type)
    outputs = grpcclient.InferRequestedOutput("output")
    inputs.set_data_from_numpy(img)

    res = client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])
    res = res.as_numpy("output")
    res_copy = np.copy(res).astype(d_type)

    pred = postprocess(res_copy, model_shape)[0]

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
    model_shape = tuple(map(int, args.model_shape.split(",")))
    logger = PerformanceLogger() if args.logger else None

    load_dotenv("./bytetrack/tracker_config.env")
    tracker_args = argparse.Namespace()
    tracker_args.track_thresh = float(os.getenv("TRACK_THRESH"))
    tracker_args.track_buffer = int(os.getenv("TRACK_BUFFER"))
    tracker_args.mot20 = os.getenv("MOT20", "False").lower()
    tracker_args.match_thresh = float(os.getenv("MATCH_THRESH"))
    tracker = BYTETracker(tracker_args) if args.tracker else None

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
        mkdir(args.output_path)

        image_files = sorted(
            [
                os.path.join(args.input_path, f)
                for f in os.listdir(args.input_path)
                if f.lower().endswith(("png", "jpg", "jpeg"))
            ]
        )

        if logger:
            logger.start_logging()

        total_infer_time = 0
        total_track_time = 0

        for image_index, image_path in enumerate(image_files, start=1):
            print(f"[{image_index} / {len(image_files)}]\n")
            if tracker:
                infer_time, track_time = infer_image(
                    client,
                    args.model,
                    image_path,
                    args.output_path,
                    model_shape,
                    args.data_type,
                    tracker,
                )

                total_infer_time += infer_time
                total_track_time += track_time

                print(
                    f"\rInference Time: {infer_time:.3f} ms | Tracking Time: {track_time:.3f} ms\033[K",
                    flush=True,
                )
                print()

            else:
                infer_time = infer_image(
                    client,
                    args.model,
                    image_path,
                    args.output_path,
                    model_shape,
                    args.data_type,
                    tracker,
                )

                total_infer_time += infer_time

                print(
                    f"\rInference Time: {infer_time:.3f} ms\033[K",
                    flush=True,
                )
                print()

            if logger:
                logger.log(infer_time)

        if tracker:
            print(f"[✓] Inference complete")
            print(
                f"[✓] Avg detection time: {total_infer_time / len(image_files):.3f} ms"
            )
            print(
                f"[✓] Avg tracking time: {total_track_time / len(image_files):.3f} ms"
            )

        else:
            print(f"[✓] Inference complete.")
            print(
                f"[✓] Avg detection time: {total_infer_time / len(image_files):.3f} ms"
            )

        if logger:
            logger.stop_logging()

        command = [
            "ffmpeg",
            "-r",
            "20",
            "-i",
            f"{args.output_path}/%06d.png",
            "-c:v",
            "libx264",
            f"{args.output_path}/test.mp4",
        ]
        subprocess.run(command, check=True)

    elif args.infer_mode == "cam":
        sensor = SensorRealsense()
        while True:
            input = sensor.get_video_from_pipeline()[1][0]
            output = infer_camera(
                client, args.model, input, model_shape, args.data_type
            )
            cv2.namedWindow("camera viewer", cv2.WINDOW_NORMAL)
            cv2.imshow("camera viewer", output)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
