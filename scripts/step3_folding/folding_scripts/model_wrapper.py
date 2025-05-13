import torch
from transformers import EsmForProteinFolding, AutoTokenizer


class ESMModelWrapper:
    # Wrapper class for the ESMFold model and tokenizer. Handles loading the model and generating atomic positions.
    def __init__(self, model_name="facebook/esmfold_v1", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        self.model = EsmForProteinFolding.from_pretrained(model_name).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_positions(self, sequences):
        # Tokenize sequences and generate atomic positions using the model.
        inputs = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            # Print device information for inputs and model
            print(f"[INFO] Model is on device: {next(self.model.parameters()).device}")
            print(f"[INFO] Inputs are on device: {inputs['input_ids'].device}")
            outputs = self.model(**inputs)
        return outputs