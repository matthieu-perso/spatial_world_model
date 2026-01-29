import argparse
from fastchat.model import add_model_args

class InferenceArgumentParser:
    def __init__(self, version, description="Inference arg parser."):
        self.version = version
        self.parser = argparse.ArgumentParser(description=description)

        self._add_common_args()

        if version == "lm":
            self._add_lm_args()
        elif version == "vlm":
            self._add_vlm_args()
        else:
            raise ValueError(f"Unknown version: {version}")

    def _add_common_args(self):
        # Arguments common to all models
        self.parser.add_argument("--dataset_id", type=str, default="MilaWang/SpatialEval",
                                 help="Dataset identifier for Hugging Face.")
        self.parser.add_argument("--temperature", type=float, default=0.2)
        self.parser.add_argument("--top_p", type=float, default=0.9)
        self.parser.add_argument("--repetition_penalty", type=float, default=1.0)
        self.parser.add_argument("--max_new_tokens", type=int, default=1024)
        self.parser.add_argument("--output_folder", type=str, default="outputs")
        self.parser.add_argument("--task", type=str, default="all", choices=["all", "spatialmap", "mazenav", "spatialgrid", "spatialreal"],
                                 help="Set specific task to evaluate or evaluate all tasks.")
        self.parser.add_argument("--completion", action="store_true", help="Add completion prompt.")
        self.parser.add_argument("--w_reason", action="store_true", help="Add reason prompt.")
        self.parser.add_argument("--first_k", type=int, default=None, help="Test first k samples for each question type. If not specified, test all samples.")
        self.parser.add_argument("--cache_residual_stream", action="store_true",
                                 help="Run forward with output_hidden_states=True and save layer outputs to cache_dir for further modeling.")
        self.parser.add_argument("--cache_dir", type=str, default=None,
                                 help="Directory for residual-stream cache (default: <output_folder>/residual_stream). Used only if --cache_residual_stream.")
        
    def _add_lm_args(self):
        add_model_args(self.parser)
        self.parser.add_argument("--mode", default="tqa", choices=["tqa"], 
                                 help="Set mode for test input modality (only 'tqa' allowed for language model).")
        
    def _add_vlm_args(self):
        self.parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu", "npu"], default="cuda",
                                help="The device type.")
        self.parser.add_argument("--mode", choices=["tqa", "vqa", "vtqa"], 
                                 help="Set mode for test input modality.")
        self.parser.add_argument("--random_image", action="store_true", help="Use random image for inference.")
        self.parser.add_argument("--noise_image", action="store_true", help="Use noise image for inference.")
        self.parser.add_argument("--model_path", type=str,
                                 help="Local model path for storing model checkpoints or model identifier for Hugging Face.")
        self.parser.add_argument("--model_base", type=str, default=None, help="Base model.")

    def parse_args(self):
        return self.parser.parse_args()