import argparse
import copy
import os
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from yoloa.camera_api import SensorRealsense
from yoloa.classes import COCO_CLASSES
from yoloa.utils import mkdir, multiclass_nms, postprocess, preprocess
from yoloa.visualize import vis


def make_parser():
    parser = argparse.ArgumentParser("triton inference")
    parser.add_argument("--url", type=str, default="192.168.105.190:8001")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="cham_v04_c12_yolox_t_detect_fp16_416x416",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        default="./data/test",
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
        "-d",
        "--data_type",
        type=str,
        default="FP16",
        choices=["FP16", "FP32"],
    )

    return parser


def infer_image(
    client,
    model_name,
    image_path,
    output_path,
    input_shape,
    data_type,
    image_id,
    len_folder,
):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    origin_img = cv2.imread(image_path)
    img, ratio = preprocess(origin_img, input_shape)
    img = img[None, :, :, :].astype(d_type)

    inputs = grpcclient.InferInput("images", img.shape, datatype=data_type)
    inputs.set_data_from_numpy(img)
    outputs = grpcclient.InferRequestedOutput("output")

    start_time = time.time()
    res = client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])
    end_time = time.time()
    print(
        f"\r{image_id} / {len_folder} inference time: {(end_time - start_time) * 1000:.3f} ms\033[K",
        end="",
    )

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

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.7, score_thr=0.6)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(
            origin_img,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=0.6,
            class_names=COCO_CLASSES,
        )

    output_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_path, origin_img)


def infer_camera(client, model_name, input, input_shape, data_type):
    d_type = {"FP16": np.float16, "FP32": np.float32}[data_type]
    img, ratio = preprocess(input, input_shape)
    img = img[None, :, :, :].astype(d_type)

    inputs = grpcclient.InferInput("images", img.shape, datatype=data_type)
    inputs.set_data_from_numpy(img)
    outputs = grpcclient.InferRequestedOutput("output")

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
    input_shape = tuple(map(int, args.model.split("_")[-1].split("x")))

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

        for idx, image_path in enumerate(image_files, start=1):
            infer_image(
                client,
                args.model,
                image_path,
                output_path,
                input_shape,
                args.data_type,
                idx,
                len(image_files),
            )

        print()
        print("Inference completed for all images.")

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
