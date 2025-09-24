"""
BitGen: Advanced Tiny Language Model for Embedded Microcontrollers
Main module and API interface
"""

from .bitgen_model import BitGenModel, BitGenConfig, create_bitgen_model
from .adaptive_loss import BitGenLoss, AdaptiveLossManager, PerformanceTracker
from .data_loader import COCODataset, RobotSelectionDataset, BitGenTokenizer, BitGenDataLoader
from .train_bitgen import BitGenTrainer
from .evaluate_bitgen import BitGenEvaluator
from .embedded_deployment import EmbeddedModelOptimizer, EmbeddedBenchmark, export_for_microcontroller
from .bitgen_configs import BitGenNanoConfig, BitGenTinyConfig, BitGenSmallConfig, get_config_for_target

__version__ = "1.0.0"
__author__ = "BitGen Development Team"
__description__ = "Advanced Tiny Language Model for Embedded Microcontrollers"

# Easy-to-use API
class BitGen:
    """Main BitGen API for easy model creation and deployment"""

    def __init__(self, model_size='tiny', target_device='cortex-m4'):
        """
        Initialize BitGen model

        Args:
            model_size: 'nano', 'tiny', or 'small'
            target_device: 'cortex-m0', 'cortex-m4', or 'cortex-m7'
        """
        self.model_size = model_size
        self.target_device = target_device

        # Get appropriate configuration
        from .bitgen_configs import get_config_for_target
        config_dict = get_config_for_target(target_device)

        # Create model configuration
        if model_size == 'nano':
            self.config = BitGenNanoConfig()
        elif model_size == 'tiny':
            self.config = BitGenTinyConfig()
        elif model_size == 'small':
            self.config = BitGenSmallConfig()
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        # Create model
        self.model = create_bitgen_model(model_size)
        self.tokenizer = BitGenTokenizer(self.config.vocab_size)

        # Training components
        self.trainer = None
        self.evaluator = None

    def train(self,
              coco_data_path: str,
              robot_data_path: str = None,
              output_dir: str = 'checkpoints',
              **kwargs):
        """Train the BitGen model"""

        self.trainer = BitGenTrainer(
            config=self.config,
            model_size=self.model_size,
            output_dir=output_dir,
            **kwargs
        )

        return self.trainer.train(
            coco_data_path=coco_data_path,
            robot_data_path=robot_data_path,
            **kwargs
        )

    def evaluate(self,
                coco_test_path: str,
                robot_test_path: str,
                reasoning_test_path: str = None,
                **kwargs):
        """Evaluate the BitGen model"""

        if self.evaluator is None:
            self.evaluator = BitGenEvaluator(self.model, self.config)

        return self.evaluator.run_comprehensive_evaluation(
            coco_test_path=coco_test_path,
            robot_test_path=robot_test_path,
            reasoning_test_path=reasoning_test_path,
            **kwargs
        )

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 0.7):
        """Generate text from prompt"""

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids])

        # Generate
        with torch.no_grad():
            generated, _ = self.model.generate_embedded(
                input_tensor,
                max_length=max_length,
                temperature=temperature
            )

        # Decode output
        output_text = self.tokenizer.decode(generated[0].tolist())
        return output_text

    def process_image_and_text(self, image_path: str, text: str):
        """Process image and text together (multimodal)"""

        from PIL import Image
        import torchvision.transforms as transforms

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)

        # Tokenize text
        input_ids = self.tokenizer.encode(text)
        input_tensor = torch.tensor([input_ids])

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor, images=image_tensor)

        return outputs

    def select_robot_for_task(self, task_description: str):
        """Select appropriate robot for given task"""

        # Tokenize task
        input_ids = self.tokenizer.encode(task_description)
        input_tensor = torch.tensor([input_ids])

        # Get robot selection
        with torch.no_grad():
            outputs = self.model(input_tensor, return_robot_selection=True)
            robot_probs = outputs['robot_selection']

        # Get best robot
        best_robot_id = robot_probs.argmax(dim=-1).item()
        confidence = robot_probs.max(dim=-1)[0].item()

        # Map to robot type (simplified)
        robot_types = [
            'manipulator', 'mobile_base', 'quadruped', 'humanoid',
            'aerial_drone', 'ground_vehicle', 'gripper_robot', 'inspection_robot'
        ]

        selected_robot = robot_types[best_robot_id % len(robot_types)]

        return {
            'selected_robot': selected_robot,
            'confidence': confidence,
            'robot_id': best_robot_id
        }

    def export_for_embedded(self, output_dir: str):
        """Export model for embedded deployment"""

        return export_for_microcontroller(
            model_path=None,  # Use current model
            output_dir=output_dir,
            target_board=self.target_device
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint"""
        import torch

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, output_path: str):
        """Save current model checkpoint"""
        import torch

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'model_size': self.model_size,
            'target_device': self.target_device
        }

        torch.save(checkpoint, output_path)
        print(f"Model saved to {output_path}")

# Convenience functions
def quick_train(coco_data_path: str,
                model_size: str = 'tiny',
                target_device: str = 'cortex-m4',
                **kwargs):
    """Quick training function"""

    bitgen = BitGen(model_size=model_size, target_device=target_device)
    return bitgen.train(coco_data_path=coco_data_path, **kwargs)

def quick_evaluate(model_path: str,
                  coco_test_path: str,
                  robot_test_path: str,
                  **kwargs):
    """Quick evaluation function"""

    bitgen = BitGen()
    bitgen.load_checkpoint(model_path)
    return bitgen.evaluate(
        coco_test_path=coco_test_path,
        robot_test_path=robot_test_path,
        **kwargs
    )

def create_embedded_deployment(model_path: str,
                             output_dir: str,
                             target_device: str = 'cortex-m4'):
    """Create embedded deployment package"""

    return export_for_microcontroller(
        model_path=model_path,
        output_dir=output_dir,
        target_board=target_device
    )

# Import torch for the API functions
import torch
