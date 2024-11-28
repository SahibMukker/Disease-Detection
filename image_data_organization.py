import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def load_csv(csv_path):
    '''
    Loads metadata CSV file and returns a Pandas DataFrame
    
    Parameters:
        csv_path (str): Path to the CSV file
    
    Returns:
        metadata (DataFrame): Pandas DataFrame
    '''
    metadata = pd.read_csv(csv_path)
    return metadata

def organize_images_by_disease(metadata, image_folder, output_folder):
    '''
    Organizes images into subfolders based on their disease label
    
    Parameters:
        metadata (DataFrame): Pandas DataFrame containing metadata
        image_folder (str): Path to the folder containing the images
        output_folder (str): Path to the folder where the organized images will be saved
    '''
    
    os.makedirs(output_folder, exist_ok=True)
    
    for _, row in metadata.iterrows():
        image_file = row['Image Index'] # can adjust name as needed
        disease_label = row['Finding Labels'] # can adjust name as needed
        source_path = os.path.join(image_folder, image_file)
        
        if os.path.exists(source_path):
            # create disease-specific folder
            disease_folder = os.path.join(output_folder, disease_label)
            os.makedirs(disease_folder, exist_ok=True)
            
            # moeve image to disease-specific folder
            shutil.move(source_path,os.path.join(disease_folder, image_file))



def split_dataset(output_folder, train_folder, val_folder, test_folder, test_size = 0.2, val_size = 0.1):
    '''
    Split the organized data into training, validation, and test sets
    
    Parameters:
        output_folder (str): Path to the folder containing the organized images
        train_folder (str): Path to the folder where the training images will be saved
        val_folder (str): Path to the folder where the validation images will be saved
        test_folder (str): Path to the folder where the test images will be saved
    '''
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    all_images = []
    all_labels = []
    
    # gather all images and labels
    for disease_folder in os.listdir(output_folder):
        disease_path = os.path.join(output_folder, disease_folder)
        
        if os.path.isdir(disease_path):
            for file_name in os.listdir(output_folder):
                all_images.append(os.path.join(disease_path, file_name))
                all_labels.append(disease_folder)
                
    # split into train, val, test
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size = test_size, stratify = all_labels, random_state = 42
        )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size = val_size, stratify = train_labels, random_state = 42
        )
    
    def move_files(images, labels, target_folder):
        '''
        Helper function to move files to target folder
        
        Parameters:
            images (list): List of image file paths
            labels (list): List of corresponding labels
            target_folder (str): Path to the target folder
        '''
        for image, label in zip(images, labels):
            label_folder = os.path.join(target_folder, label)
            os.makedirs(label_folder, exist_ok=True)
            shutil.copy(image, os.path.join(label_folder, os.path.basename(image)))
            
        # move files into their respective directories
        move_files(train_images, train_labels, train_folder)
        move_files(val_images, val_labels, val_folder)
        move_files(test_images, test_labels, test_folder)

if __name__ == "__main__":
    # paths (change as needed)
    csv_path = 'C:\\Users\\Sahib Mukker\\Documents\\Sahib Documents\\Coding Career Knowledge Stuff\\Projects\\4. Disease Classification\\raw data\\NIH Chest X-Rays\\Data_Entry_2017.csv'
    rescaled_folder = 'C:\\Users\\Sahib Mukker\\Documents\\Sahib Documents\\Coding Career Knowledge Stuff\\Projects\\4. Disease Classification\\scaled data\\NIH Chest X-Rays Resized'
    organized_folder = 'C:\\Users\\Sahib Mukker\\Documents\\Sahib Documents\\Coding Career Knowledge Stuff\\Projects\\4. Disease Classification\\NIH Chest X-Rays\\organized_folder'
    train_folder = 'C:\\Users\\Sahib Mukker\\Documents\\Sahib Documents\\Coding Career Knowledge Stuff\\Projects\\4. Disease Classification\\NIH Chest X-Rays\\train_folder'
    val_folder = 'C:\\Users\\Sahib Mukker\\Documents\\Sahib Documents\\Coding Career Knowledge Stuff\\Projects\\4. Disease Classification\\NIH Chest X-Rays\\val_folder'
    test_folder = 'C:\\Users\\Sahib Mukker\\Documents\\Sahib Documents\\Coding Career Knowledge Stuff\\Projects\\4. Disease Classification\\NIH Chest X-Rays\\test_folder'
    
    # load metadata
    metadata = load_csv(csv_path)
    
    # organize images by disease
    organize_images_by_disease(metadata, rescaled_folder, organized_folder)
    
    # split into train, val, test
    split_dataset(organized_folder, train_folder, val_folder, test_folder)