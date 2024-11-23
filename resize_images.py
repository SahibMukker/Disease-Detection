import cv2
import os

def resize_image(input_folder, output_folder, size=(512, 512)):
    '''
    Function to resize images, and save them to the output folder, 
    default size is 512x512 since I am only making 1 model for all the images
    '''
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            
            if image is not None:
                resized_image = cv2.resize(image, size)
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, resized_image)
                print(f"Resized {file_name} to {size} and saved as {output_path}")
                
            else:
                print(f"Failed to read {file_name}")
