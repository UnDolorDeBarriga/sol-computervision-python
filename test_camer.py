import numpy as np
import pandas as pd
import os
import glob


def convertToGrayToHOG(imgVector: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale and return the HOG features.
    
    Parameters:
    imgVector: The input image in RGB format.
    Returns:
    np.ndarray: The HOG features of the grayscale image.
    """
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)

def crop(img, x1, x2, y1, y2, scale) -> np.ndarray:
    """
    Crop the image and resize it to the specified scale.

    Parameters:
    img: The input image.
    x1: The x-coordinate of the top-left corner.
    x2: The x-coordinate of the bottom-right corner.
    y1: The y-coordinate of the top-left corner.
    y2: The y-coordinate of the bottom-right corner.
    scale: The scale to resize the cropped image.
    Returns:
    np.ndarray: The cropped and resized image.
    """
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((scale, scale))) 
    return crp

def get_data(user_list: list, data_directory: str) -> tuple:
    """
    Get the HOG features of the images for the specified users.

    Parameters:
    user_list: The list of users.
    img_dict: The dictionary containing the images.
    Returns:
    tuple: The HOG features and the labels.
    """
    X = []
    Y = []
    for user in user_list:
        user_images = glob.glob(data_directory + user + '/*.jpg')
        boundingbox_df = pd.read_csv(data_directory + user + '/'
                        + user + '_loc.csv')
        
        print(user_images.shape)
        print(boundingbox_df.shape)
        # for rows in boundingbox_df.iterrows():
        #     cropped_img = crop(img_dict[rows[1]['image']], 
        #                     rows[1]['top_left_x'], 
        #                     rows[1]['bottom_right_x'], 
        #                     rows[1]['top_left_y'], 
        #                     rows[1]['bottom_right_y'], 
        #                     128)
        #     hogvector = convertToGrayToHOG(cropped_img)
        #     X.append(hogvector.tolist())
        #     Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y




if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_dir, 'Dataset')
    users = os.listdir(data_directory)

    X, Y = get_data(users, data_directory)
    img_dict = {}



# Y_mul = self.label_encoder.fit_transform(Y_mul)

# self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])