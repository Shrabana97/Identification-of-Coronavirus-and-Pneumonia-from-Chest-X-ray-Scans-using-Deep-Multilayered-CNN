import os, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
import numpy as np

rotation_range = 10
width_shift_range = 0.1 
height_shift_range = 0.1 
shear_range = 0.1 
zoom_range = 0.1

datagen = ImageDataGenerator(
        rescale=1./ 255,
        rotation_range = rotation_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


open_root_dir = open("path_of_train_dir.txt", "r")
root_dir = open_root_dir.read()
open_root_dir.close()
aug_dir = os.path.join(root_dir, 'classify train', 'training')

total = sum([len(files) for r, d, files in os.walk(aug_dir)])
with tqdm(total=total) as pbar:
    for foldername in os.listdir(aug_dir):
        folder = os.path.join(aug_dir, foldername)
        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)

            pbar.set_description("Performing classical augmentation on training data")
            pbar.update()

            img = cv2.imread(file)  
            x = np.expand_dims(img, axis = 0)

            i = 0
            for batch in datagen.flow(x, batch_size = 1, 
                            save_to_dir = folder,  
                            save_prefix ='aug', save_format ='jpg'):
                i += 1
                if i >= 3:
                    break