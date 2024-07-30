import os 
import json
import numpy as np
import matplotlib.pyplot as plt
import PIL

def random_image_and_mask(images_path:str, masks_path:str):
    """
    Show a random image and its mask from the dataset
    """
    class_map = create_class_map()
    images_dir = os.listdir(images_path)
    random_image_name = np.random.choice(images_dir)
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    fig.suptitle('Random Image and Mask', fontsize=16, fontweight='bold', y=.78)
    ax[0].imshow(PIL.Image.open(os.path.join(images_path, random_image_name)))
    ax[0].set_title('Image {}'.format(random_image_name))
    mask_img = PIL.Image.open(os.path.join(masks_path, random_image_name.replace('jpg', 'png')))
    ax[1].imshow(mask_img, cmap='tab20')
    ax[1].set_title('Mask {}'.format(random_image_name.replace('jpg', 'png')))
    image_labels = np.unique(np.array(mask_img))
    print("Classes in the mask:")
    total_pixels = mask_img.size[0] * mask_img.size[1]
    for label in image_labels:
        label = str(label)
        if label in class_map.keys():
            number_label = np.count_nonzero(np.array(mask_img) == int(label))
            percentage = (number_label / total_pixels) * 100
            print(f"Class {label}: {class_map[label]} Percentage: {percentage:.2f}%")

def create_class_map():
    """
    Create a class map for the dataset and save it as a json file
    """
    if not os.path.exists("class_map.json"):
        class_map = {"0": "Background",
                    "1": "Property Roof",
                    "2": "Secondary Structure",
                    "3": "Swimming Pool",
                    "4": "Vehicle",
                    "5": "Grass",
                    "6": "Trees / Shrubs",
                    "7": "Solar Panels",
                    "8": "Chimney",
                    "9": "Street Light",
                    "10": "Window",
                    "11": "Satellite Antenna",
                    "12": "Garbage Bins",
                    "13": "Trampoline",
                    "14": "Road/Highway",
                    "15": "Under Construction / In Progress Status",
                    "16": "Power Lines & Cables",
                    "17": "Water Tank / Oil Tank",
                    "18": "Parking Area - Commercial",
                    "19": "Sports Complex / Arena",
                    "20": "Industrial Site",
                    "21": "Dense Vegetation / Forest",
                    "22": "Water Body",
                    "23": "Flooded",
                    "24": "Boat"}
        with open("class_map.json", "w") as f:
            json.dump(class_map, f, indent=4)
    return json.load(open("class_map.json"))

def predict_and_plot(model, dataset):
    # Get the prediction and plot the image and mask
    model.eval()
    idx = np.random.choice(len(dataset), 1)[0]
    image, mask = dataset[idx]
    image = image.unsqueeze(0).cuda()
    pred = model(image.float())
    pred = pred.argmax(dim=1)
    pred = pred.cpu().numpy()
    mask = mask.cpu().numpy()
    plt.figure(figsize=(20,24))
    plt.suptitle('Imagery Land-Cover Classification', fontsize=16, y=0.63, fontweight='bold')
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(0).permute(1,2,0).cpu().numpy())
    plt.title('Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='tab20')
    plt.title('Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(pred.squeeze(0), cmap='tab20')
    plt.title('Prediction')
    plt.show()

# colors = [
#     'black',  # Background (0)
#     'brown',  # Property Roof (1)
#     'grey',   # Secondary Structure (2)
#     'aqua',   # Swimming Pool (3)
#     'blue',   # Vehicle (4)
#     'green',  # Grass (5)
#     'darkgreen', # Trees / Shrubs (6)
#     'gold',   # Solar Panels (7)
#     'red',    # Chimney (8)
#     'yellow', # Street Light (9)
#     'lightblue', # Window (10)
#     'purple', # Satellite Antenna (11)
#     'orange', # Garbage Bins (12)
#     'pink',   # Trampoline (13)
#     'darkgrey', # Road/Highway (14)
#     'lightgrey', # Under Construction (15)
#     'black',  # Power Lines (16)  (consider a different shade of black)
#     'cyan',   # Water Tank (17)
#     'darkorange', # Parking Area (18)
#     'magenta', # Sports Complex (19)
#     'darkred',  # Industrial Site (20)
#     'darkgreen', # Dense Vegetation (21)
#     'royalblue', # Water Body (22)
#     'lightblue', # Flooded (23)
#     'navy',   # Boat (24)
# ]
# # Define a dictionary to map class labels (integers) to colors
# class_to_color = dict(zip(range(25), colors))

# # Create a colormap object using ListedColormap
# cmap = c.LinearSegmentedColormap.from_list("", colors)