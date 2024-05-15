import paddlehub as hub
import insightface_paddle as face
import cv2
import os

# Define the parser and args for InsightFace
parser = face.parser()
args = parser.parse_args()
args.build_index = "Dataset/index.bin"
args.img_dir = "Dataset"
args.label = "Dataset/labels.txt"

# Initialize the InsightFace predictor
predictor = face.InsightFace(args)


def write_to_file(filepath, person_name, filename="Dataset/labels.txt"):
    modified_filepath = "./" + os.path.join(person_name, os.path.basename(filepath))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as f:
        f.write("{}\t{}\n".format(modified_filepath, person_name))

def process_face_image(img_path, person_name):
    # Load the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    # Save the image in the Dataset directory
    output_dir = os.path.join("Dataset", person_name)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(img_path)  # Use the original image name
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img)
    
    # Write the file path and person's name to the labels file
    write_to_file(filepath, person_name)
        
if __name__ == '__main__':
    image_path = "/workspaces/Face_Recognition_System/evtest.jpg"  # Replace with the path to your face image
    person_name = "Evan M"  # Replace with the name of the person
    
    process_face_image(image_path, person_name)
    
    # Build the index
    predictor.build_index()
