# train.py
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import cv2
from sklearn.model_selection import train_test_split
from utils import load_binary_mask, resize_image_and_mask, process_predicted
from unet import UNetCompiled
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
from hydra.core.config_store import ConfigStore

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(process_predicted(y_pred), 'float32'))
    print(y_true_f, y_pred_f)
    print(np.min(y_pred_f),np.max(y_pred_f))
    print(np.sum(y_true_f),np.sum(y_pred_f))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


@hydra.main(config_path="../config", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    df = pd.read_csv(cfg.data.csv_path)
    df = df.dropna()[:cfg.data.sample_size]
    images_dir = cfg.data.images_dir
    y = []
    X = []
    for filename in df["ImageId"]:
        mask = load_binary_mask(df, filename)
        img = cv2.imread(os.path.join(images_dir, filename))
        img_resized, mask_resized = resize_image_and_mask(img, mask, target_size=cfg.model.input_size[:2])
        X.append(img_resized)
        y.append(mask_resized)

    y = np.array(y)
    X = np.array(X)/255.

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=cfg.data.test_size, random_state=123)

    unet = UNetCompiled(input_size=cfg.model.input_size, n_filters=cfg.model.n_filters, n_classes=cfg.model.n_classes)
    unet.summary()

    unet.compile(
        optimizer=getattr(tf.keras.optimizers, cfg.training.optimizer)(),
        loss=getattr(tf.keras.losses, cfg.training.loss), 
        metrics=cfg.training.metrics
    )

    early_stopping = EarlyStopping(monitor=cfg.training.early_stopping.monitor,
                                   patience=cfg.training.early_stopping.patience,
                                   restore_best_weights=cfg.training.early_stopping.restore_best_weights)

    results = unet.fit(X_train, y_train, batch_size=cfg.training.batch_size, epochs=cfg.training.epochs,
                       validation_data=(X_valid, y_valid), callbacks=[early_stopping])

    y_pred = unet.predict(X_valid)
    print(dice_coef(y_valid,process_predicted(y_pred)))
    
    unet.save(cfg.output.model_path)

    print("Model trained successfully.")

if __name__ == "__main__":
    train()
