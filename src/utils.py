import numpy as np
import cv2


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float32)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

def load_binary_mask(df, file):
    encoded_pixels = df[df["ImageId"] == file]['EncodedPixels']
    return np.array(masks_as_color(encoded_pixels)).astype(np.int8)

def resize_image_and_mask(img, mask, target_size=(224,224)):
    # Resize the image
    img_resized = cv2.resize(img, target_size)

    # Resize the mask
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return img_resized, mask_resized