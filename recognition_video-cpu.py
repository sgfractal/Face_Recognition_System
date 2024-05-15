import paddlehub as hub
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if the index file exists
index_file = "Dataset/index.bin"
if not os.path.exists(index_file):
    print(f"Index file not found: {index_file}")
    print("Please make sure you have enrolled faces using enroll_new_faces_img.py")
    exit(1)

# Check if the dataset directory is not empty
dataset_dir = "Dataset"
if not os.listdir(dataset_dir):
    print(f"No enrolled faces found in the dataset directory: {dataset_dir}")
    print("Please make sure you have enrolled faces using enroll_new_faces_img.py")
    exit(1)

face_detector = hub.Module(name="pyramidbox_lite_mobile")
parser = face.parser()
args = parser.parse_args()

args.use_gpu = False
args.det = False
args.rec = True
args.rec_thresh = 0.45
args.index = "Dataset/index.bin"
args.rec_model = "Models/mobileface_v1.0_infer"
recognizer = face.InsightFace(args)

def detect_face(image):
    result = face_detector.face_detection(images=[image], use_gpu=False)
    box_list = result[0]['data']
    return box_list

def recognize_face(image, box_list):
    img = image[:, :, ::-1]
    res = list(recognizer.predict(img, box_list))
    
    if res:
        box_list = res[0]['box_list']
        labels = res[0]['labels']
    else:
        box_list = []
        labels = []
    
    return box_list, labels

def draw_boundary_boxes(image, box_list, labels):
    for box, label in zip(box_list, labels):
        score = "{:.2f}".format(box['confidence'])
        x_min, y_min, x_max, y_max = int(box['left']), int(box['top']), int(box['right']), int(box['bottom'])

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Put the label text near the box
        label_text = label + " " + str(score)
        cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def detection_video_file(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame to match the input size of the model
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        box_list = detect_face(resized_frame)
        box_list, labels = recognize_face(resized_frame, box_list)
        draw_boundary_boxes(resized_frame, box_list, labels)

        # Write the frame with bounding boxes to the output video
        out.write(resized_frame)

    # Release the video capture, video writer, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = "/workspaces/Face_Recognition_System/EvRollTest.mp4"  # Replace with your input video path
    output_path = "/workspaces/Face_Recognition_System/output.mp4"  # Replace with your desired output video path
    
    detection_video_file(video_path, output_path)
