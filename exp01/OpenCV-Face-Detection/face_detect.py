import argparse

import numpy as np
import cv2 as cv

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

MODEL_PATH = './models/face_detection_yunet_2023mar.onnx'
MODEL_PATH = './models/face_detection_yunet_2023mar.onnx'

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default=MODEL_PATH, help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')

parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.putText(input, "F{} S:{:.2f}".format(idx, face[-1]), (coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    #Target:
    #DNN_TARGET_CPU
    #DNN_TARGET_OPENCL
    #DNN_TARGET_MYRIAD
    #DNN_TARGET_FPGA
    #DNN_TARGET_VULKAN
    #DNN_TARGET_CUDA
    #DNN_TARGET_HDDL
    #DNN_TARGET_NPU
    #Backend:
    #DNN_BACKEND_OPENCV
    #DNN_BACKEND_HALIDE
    #DNN_BACKEND_INFERENCE_ENGINE
    #DNN_BACKEND_VKCOM
    #DNN_BACKEND_CUDA
    #DNN_BACKEND_WEBNN
    #DNN_BACKEND_TIMVX

    ## [initialize_FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(
        model=args.face_detection_model,
        config="",
        input_size=(320, 320),                  #输入图片尺寸
        score_threshold=args.score_threshold,   #置信度阈值
        nms_threshold=args.nms_threshold,       #非极大值抑制阈值
        top_k=args.top_k,                       
        backend_id=cv.dnn.DNN_BACKEND_OPENCV,  
        target_id=cv.dnn.DNN_TARGET_CPU         
    )
    ## [initialize_FaceDetectorYN]

    tm = cv.TickMeter()

    # If input is an image
    if args.image1 is not None:
        img1 = cv.imread(cv.samples.findFile(args.image1))
        img1Width = int(img1.shape[1] * args.scale)
        img1Height = int(img1.shape[0] * args.scale)

        img1 = cv.resize(img1, (img1Width, img1Height))
        tm.start()

        ## [inference]
        # Set input size before inference
        detector.setInputSize((img1Width, img1Height))

        # faces1[1]    - a np.array of shape [n, 15].
        # faces1[1]    [[x1, y1, w, h, lm1_x, lm1_y, ..., score], ...]
        # If n = 0, None (no face detected)
        

        faces1 = detector.detect(img1)
        ## [inference]

        tm.stop()
        assert faces1[1] is not None, 'Cannot find a face in {}'.format(args.image1)

        # Draw results on the input image
        visualize(img1, faces1, tm.getFPS())

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', img1)

        # Visualize results in a new window
        cv.imshow("image1", img1)
       
        cv.waitKey(0)
    else: # Omit input to call default camera
        if args.video is not None:
            deviceId = args.video
        else:
            deviceId = 0
        cap = cv.VideoCapture(deviceId)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
        detector.setInputSize([frameWidth, frameHeight])

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv.resize(frame, (frameWidth, frameHeight))

            # Inference
            tm.start()
            faces = detector.detect(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS())

            # Visualize results
            cv.imshow('Live', frame)
    cv.destroyAllWindows()
