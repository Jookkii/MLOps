import numpy as np
import bentoml
from PIL import Image
import torch

mnist_model = bentoml.pytorch.get("mnist_mlp_model:latest")

torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

@bentoml.service(
    name="mlp_classifier",
    resources={"cpu": "200m"}, 
)
class MNISTService:
    model_ref = mnist_model

    def __init__(self):
        self.model = self.model_ref.load_model(weights_only=False)
        self.model.eval() # Dobra praktyka dla serwowania

    @bentoml.api
    def classify(self, input_series: np.ndarray) -> np.ndarray:
        img = Image.fromarray(input_series.astype('uint8'))
        
        img_resized = img.resize((28, 28), resample=Image.Resampling.LANCZOS)
        
        final_array = np.array(img_resized).astype(np.float32) / 255.0
        final_array = (final_array - 0.1307) / 0.3081
        
        
        input_tensor = torch.from_numpy(final_array).reshape(1, 1, 28, 28)
        
        with torch.no_grad():
            result = self.model(input_tensor)
        
        return result.numpy()