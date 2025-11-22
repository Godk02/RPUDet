import torch
import cv2
import numpy as np
import os
from models.experimental import attempt_load
from utils.augmentations import letterbox


class FeatureMapVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

        # Register hook to the target layer
        self.hook = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        """Save the activation (feature map) from the target layer."""
        self.activations = output.detach().cpu()

    def preprocess_image(self, img_path):
        """Preprocess the image for model input."""
        img = cv2.imread(img_path)
        img = letterbox(img, new_shape=(640, 640), auto=False)[0]  # Resize and pad
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        return tensor

    def visualize_feature_maps(self, img_path, save_dir, num_channels=16):
        """Visualize feature maps from the target layer."""
        # Preprocess the image
        tensor = self.preprocess_image(img_path)

        # Forward pass to get activations
        with torch.no_grad():
            self.model(tensor)

        # Check if activations are saved
        if self.activations is None:
            raise ValueError("No activations found. Check if the target layer is correct.")

        # Get the feature maps
        feature_maps = self.activations[0]  # Shape: [channels, height, width]

        # Normalize feature maps for visualization
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Visualize the first `num_channels` feature maps
        for i in range(min(num_channels, feature_maps.size(0))):
            feature_map = feature_maps[i].numpy()
            feature_map = cv2.resize(feature_map, (tensor.size(3), tensor.size(2)))  # Resize to original image size
            feature_map = np.uint8(255 * feature_map)

            # Save the feature map
            save_path = os.path.join(save_dir, f'feature_map_{i}.png')
            cv2.imwrite(save_path, feature_map)
            print(f"Saved feature map {i} to {save_path}")

    def release(self):
        """Remove the hook."""
        self.hook.remove()


def get_params():
    params = {
        'weight': 'runs/train/gelan_4_EGCN_UDM_PIOU_sota/weights/best_striped.pt',
        'device': 'cuda:0',
        'target_layer': 21,  # Set this to the target layer index
    }
    return params


if __name__ == '__main__':
    # Load model and set target layer
    params = get_params()
    model = attempt_load(params['weight'], device=params['device'])

    # Ensure the target layer is valid
    if isinstance(params['target_layer'], int):
        target_layer = list(model.model.children())[params['target_layer']]
    else:
        raise ValueError("Target layer must be an integer index.")

    # Initialize visualizer
    visualizer = FeatureMapVisualizer(model, target_layer)

    # Visualize feature maps
    img_path = '/home/qk/data/new_test/images/1_4118.jpg'  # Path to your image
    save_dir = 'feature_maps'  # Directory to save feature maps
    visualizer.visualize_feature_maps(img_path, save_dir, num_channels=16)

    # Release the hook
    visualizer.release()