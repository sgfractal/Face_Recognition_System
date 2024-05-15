import paddlehub as hub
import insightface_paddle as face
import cv2
import os
import time

camera_url = "rtsp://admin:neocombat1@rgnashville.better-than.tv:555"

parser = face.parser()
args = parser.parse_args()
args.build_index = "Dataset/index.bin"
args.img_dir = "Dataset"
args.label = "Dataset/labels.txt"

face_detector = hub.Module(name="pyramidbox_lite_mobile")
predictor = face.InsightFace(args)


def draw_bounding_boxes(image, faces):
    # Draw bounding boxes on the image
    for face in faces:
        left = int(face['left'])
        right = int(face['right'])
        top = int(face['top'])
        bottom = int(face['bottom'])
        confidence = face['confidence']

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        label = f": {confidence:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    return image
  
def crop_and_save_face(img, filepath, box_list):
  
  for box in box_list:
    xmin = int(box['left'])
    ymin = int(box['top'])
    xmax = int(box['right'])
    ymax = int(box['bottom'])
    face_img = img[ymin:ymax, xmin:xmax, :]
    cv2.imwrite(filepath, face_img)
    
def write_to_file(filepath, person_name, filename="Dataset/labels.txt"):
    modified_filepath = "./" + os.path.join(person_name, os.path.basename(filepath))
    with open(filename, 'a') as f:
        f.write("{}\t{}\n".format(modified_filepath, person_name))

def capture_face_images(camera_url, output_dir, duration):
    # Initialize the count
    cnt = 0

    # Start the webcam
    cap = cv2.VideoCapture(camera_url)

    end_time = time.time() + duration
    frame_count = 0

    while time.time() < end_time:
        
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        frame_count += 1
        # Perform face detection on the frame
        resized_frame = cv2.resize(frame, (640,480), interpolation= cv2.INTER_LINEAR)
        result = face_detector.face_detection(images=[resized_frame])
        box_list = result[0]['data']
        img = draw_bounding_boxes(resized_frame, box_list)
        cv2.imshow(f"Camera", img)
        # Crop and save the detected faces
        if frame_count % 4 == 0 and box_list:
            for i, box in enumerate(box_list):
                filename = 'face_{}_{}.jpeg'.format(cnt, i)
                filepath = os.path.join(output_dir, filename)
                crop_and_save_face(resized_frame, filepath, [box])
                write_to_file(filepath, "Unknown")
            cnt += 1
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
if __name__ == '__main__':
    
    duration = 120  # Set the duration in seconds (e.g., 2 minutes = 120 seconds)
    output_dir = "Dataset/Unknown"
    os.makedirs(output_dir, exist_ok=True)
    capture_face_images(camera_url, output_dir, duration)
    predictor.build_index()