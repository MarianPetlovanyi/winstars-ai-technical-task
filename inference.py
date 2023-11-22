import tensorflow as tf
import cv2
import numpy as np
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="config", config_name="inference.yaml")
def inference(cfg: DictConfig):
    loaded_model = tf.keras.models.load_model(cfg.model.model_path)
    input_layer = loaded_model.layers[0]  # Assuming the input layer is the first layer, adjust if needed
    input_shape = input_layer.input_shape
    path = cfg.data.image_path

    image = cv2.imread(path)
    image_scaled = cv2.resize(image, input_shape[0][1:3])
    image_scaled = np.expand_dims(image_scaled, axis=0)

    y_pred = loaded_model.predict(image_scaled)

    y_pred = y_pred[0]
    y_pred = (y_pred - y_pred.min())/(y_pred.max()-y_pred.min())
    threshold = cfg.inference.threshold
    binary_y_pred = np.where(y_pred > threshold, 1, 0).astype(np.uint8) * 255

    resized_binary_y_pred = cv2.resize(binary_y_pred, image.shape[:2])
    cv2.imshow("Predicted image", resized_binary_y_pred)
    cv2.imshow("Input image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()