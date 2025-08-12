import math
import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import onnxruntime as ort

class RetinaSegmentationONNX:
    def __init__(self, onnx_path, providers=None):
        """
        Initialize the ONNX segmentation model.
        Args:
            onnx_path (str): Path to the ONNX model file.
            providers (list, optional): ONNX Runtime providers.
        """
        self.session = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())

    def histo_equalized_batch(self, imgs):
        """
        Apply histogram equalization to a batch of grayscale images.
        Args:
            imgs (np.ndarray): Batch of images, shape (N, 1, H, W).
        Returns:
            np.ndarray: Equalized images, same shape as input.
        """
        imgs_equalized = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype=np.uint8))
        return imgs_equalized

    def clahe_equalized_batch(self, imgs):
        """
        Apply CLAHE (adaptive histogram equalization) to a batch of grayscale images.
        Args:
            imgs (np.ndarray): Batch of images, shape (N, 1, H, W).
        Returns:
            np.ndarray: CLAHE equalized images, same shape as input.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
        return imgs_equalized

    def dataset_normalized_batch(self, imgs):
        """
        Normalize a batch of images over the dataset mean and std, then scale to [0, 255].
        Args:
            imgs (np.ndarray): Batch of images, shape (N, 1, H, W).
        Returns:
            np.ndarray: Normalized images, same shape as input.
        """
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs - imgs_mean) / imgs_std
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                        np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
        return imgs_normalized

    def adjust_gamma_batch(self, imgs, gamma=1.0):
        """
        Apply gamma correction to a batch of images.
        Args:
            imgs (np.ndarray): Batch of images, shape (N, 1, H, W).
            gamma (float): Gamma value.
        Returns:
            np.ndarray: Gamma adjusted images, same shape as input.
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        new_imgs = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
        return new_imgs

    def rgb2gray(self, rgb):
        """
        Convert a batch of RGB images to grayscale using weighted channels.
        Args:
            rgb (np.ndarray): Batch of images, shape (N, 3, H, W).
        Returns:
            np.ndarray: Grayscale images, shape (N, 1, H, W).
        """
        bn_imgs = rgb[:, 1, :, :] * 0.75 + rgb[:, 2, :, :] * 0.25
        bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
        return bn_imgs

    def preprocess_batch(self, data):
        """
        Full preprocessing pipeline for a batch of RGB images.
        Args:
            data (np.ndarray): Batch of images, shape (N, 3, H, W).
        Returns:
            np.ndarray: Preprocessed images, shape (N, 1, H, W).
        """
        train_imgs = self.rgb2gray(data)
        train_imgs = self.dataset_normalized_batch(train_imgs)
        train_imgs = self.clahe_equalized_batch(train_imgs)
        train_imgs = self.adjust_gamma_batch(train_imgs, 1.2)
        return train_imgs

    def make_patches(self, images, size=64, offset_y=0, offset_x=0):
        """
        Split images into patches for processing with optional offset.
        Args:
            images (np.ndarray): Batch of images, shape (N, C, H, W).
            size (int): Patch size.
            offset_y (int): Vertical offset.
            offset_x (int): Horizontal offset.
        Returns:
            tuple: (patches, info) where patches is (num_patches, C, size, size) and info is metadata.
        """
        images = np.asarray(images)
        N, C, H, W = images.shape
        offset_y = max(0, min(offset_y, size - 1))
        offset_x = max(0, min(offset_x, size - 1))
        nH = math.ceil((H - offset_y) / size)
        nW = math.ceil((W - offset_x) / size)
        required_H = offset_y + nH * size
        required_W = offset_x + nW * size
        padH = max(0, required_H - H)
        padW = max(0, required_W - W)
        padded = np.pad(images, ((0,0),(0,0),(0,padH),(0,padW)), mode='constant')
        patches = []
        for n in range(N):
            for i in range(nH):
                for j in range(nW):
                    y = offset_y + i * size
                    x = offset_x + j * size
                    block = padded[n, :, y:y+size, x:x+size]
                    patches.append(block)
        patches = np.stack(patches, axis=0)
        info = (H, W, nH, nW, padded.shape, offset_y, offset_x)
        return patches, info

    def reconstruct_from_patches(self, patches, info, size=64):
        """
        Reconstruct images from patches with offset support.
        Args:
            patches (np.ndarray): Patches, shape (num_patches, C, size, size).
            info (tuple): Metadata from make_patches.
            size (int): Patch size.
        Returns:
            np.ndarray: Reconstructed images, shape (N, C, H, W).
        """
        if len(info) == 7:
            H, W, nH, nW, padded_shape, offset_y, offset_x = info
        else:
            H, W, nH, nW, padded_shape = info
            offset_y, offset_x = 0, 0
        N = padded_shape[0]
        C = padded_shape[1]
        padded_H, padded_W = padded_shape[2], padded_shape[3]
        recon = np.zeros(padded_shape, dtype=patches.dtype)
        counts = np.zeros(padded_shape, dtype=np.int32)
        idx = 0
        for n in range(N):
            for i in range(nH):
                for j in range(nW):
                    y = offset_y + i * size
                    x = offset_x + j * size
                    recon[n, :, y:y+size, x:x+size] += patches[idx]
                    counts[n, :, y:y+size, x:x+size] += 1
                    idx += 1
        counts[counts == 0] = 1
        recon = recon / counts
        return recon[:, :, :H, :W]

    def remove_islands(self, image, minsize=50):
        """
        Remove small islands (connected components) from a binary image.
        Args:
            image (np.ndarray): Binary image (2D array).
            minsize (int): Minimum size of connected components to keep.
        Returns:
            np.ndarray: Cleaned binary image.
        """
        num_labels, labels_im = cv2.connectedComponents(image.astype(np.uint8), connectivity=8)
        cleaned_image = np.zeros_like(image, dtype=np.uint8)
        for label in range(1, num_labels):
            component_size = np.sum(labels_im == label)
            if component_size >= minsize:
                cleaned_image[labels_im == label] = 1
        return cleaned_image

    def process_and_predict_image(self, image_path, patch_size=64, resize_shape=None, offsets=None, plot=True, threshold=0.5):
        """
        Run ONNX model inference on a single image, with patching and postprocessing.
        Args:
            image_path (str): Path to input image.
            patch_size (int): Patch size for inference.
            resize_shape (tuple, optional): Resize shape for input image.
            offsets (list, optional): List of (x, y) offsets for patch extraction.
            plot (bool): Whether to plot results.
            threshold (float): Threshold for binary mask.
        Returns:
            np.ndarray: Final binary mask prediction.
        """
        if offsets is None:
            offsets = [(0, 0), (24, 24)]
        def get_pixel_visit_counts_from_offsets(image_shape, patch_size, offsets):
            H, W = image_shape
            dummy = np.zeros((1, 1, H, W), dtype=np.uint8)
            total_counts = np.zeros((1, 1, H, W), dtype=np.int32)
            for offset_y, offset_x in offsets:
                _, info = self.make_patches(dummy, size=patch_size, offset_y=offset_y, offset_x=offset_x)
                recon = self.reconstruct_from_patches(
                    np.ones((info[2] * info[3], 1, patch_size, patch_size), dtype=np.uint8),
                    info, size=patch_size
                )
                total_counts += recon.astype(np.int32)
            return total_counts[0, 0]
        def _infer_expected_layout_and_name(sess):
            inp = sess.get_inputs()[0]
            name = inp.name
            shape = inp.shape
            layout = "NHWC"
            if isinstance(shape, (list, tuple)) and len(shape) == 4:
                c1 = shape[1]
                c3 = shape[3]
                if c1 in (1, 3):
                    layout = "NCHW"
                elif c3 in (1, 3):
                    layout = "NHWC"
            return name, layout
        def _to_float01(x):
            x = x.astype(np.float32, copy=False)
            if x.max() > 1.5:
                x = x / 255.0
            return x
        image = cv2.imread(image_path)
        if resize_shape is not None:
            image = cv2.resize(image, resize_shape)
        image_save = image.copy()
        image = np.expand_dims(image, axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = self.preprocess_batch(image)
        visits = get_pixel_visit_counts_from_offsets(image.shape[2:], patch_size, offsets)
        input_name, expected_layout = _infer_expected_layout_and_name(self.session)
        predicted_images = []
        for (x_off, y_off) in offsets:
            patches, info = self.make_patches(image, size=patch_size, offset_y=y_off, offset_x=x_off)
            if expected_layout == "NHWC":
                to_model = patches.transpose(0, 2, 3, 1)
            else:
                to_model = patches
            to_model = _to_float01(to_model)
            outputs = self.session.run(None, {input_name: to_model})
            preds = outputs[0]
            if preds.ndim == 4:
                if preds.shape[1] in (1, 3):
                    preds_nchw = preds
                elif preds.shape[-1] in (1, 3):
                    preds_nchw = preds.transpose(0, 3, 1, 2)
                else:
                    preds_nchw = preds.transpose(0, 3, 1, 2)
            elif preds.ndim == 3:
                preds_nchw = preds[:, None, :, :]
            else:
                raise ValueError(f"Unexpected ONNX output shape: {preds.shape}")
            recon = self.reconstruct_from_patches(preds_nchw, info, size=patch_size)
            recon = recon.squeeze().astype(np.float32)
            predicted_images.append(recon)
        sum_predictions = np.sum(predicted_images, axis=0)
        visits = visits.astype(np.float32)
        visits[visits == 0] = 1.0
        full_pred = sum_predictions / visits
        full_pred = (full_pred > threshold).astype(np.float32)
        full_pred = self.remove_islands(full_pred, minsize=250)
        if plot:
            plt.figure(figsize=(10, 5))
            plt.imshow(full_pred, cmap='gray')
            plt.axis("off")
            plt.show()
            plt.figure(figsize=(10, 5))
            plt.axis("off")
            plt.imshow(image_save, cmap='gray')
            plt.imshow(full_pred, cmap='winter', alpha=0.1)
            plt.axis("off")
            plt.show()
            plt.figure(figsize=(10, 5))
            plt.imshow(image_save, cmap='gray')
            plt.axis("off")
            plt.show()
        return full_pred

if __name__ == "__main__":
    import time
    segmenter = RetinaSegmentationONNX(r"VSeg_64p.onnx", providers=[""])
    image_path = r"C:\Users\NateLy\retina\testimgs\live_rec_test\live_rec_test\Integration_CT_L1_85.tif"
    threshold = 0.1
    start = time.time()
    plot = False
    full_pred = segmenter.process_and_predict_image(image_path, threshold=threshold)
    plt.imshow(full_pred, cmap='gray')
    plt.axis("off")
    plt.show()
    print(full_pred.shape)
    print(f"Processing time: {time.time() - start:.2f} seconds")
    mean_white = np.mean(full_pred) * 100
    print("percentage of pixels predicted as positive:", mean_white)
    if mean_white > 10:
        print("This image is likely a good image.")
    else:
        print("This image is likely a bad image.")



