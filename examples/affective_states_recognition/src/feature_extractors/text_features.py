import torch
from transformers import AutoModel, AutoTokenizer


class TextFeatureExtractor:
    def __init__(
        self,
        max_length: int = 48,
        model_name: str = "jinaai/jina-embeddings-v3",
        device: str | torch.device | None = None,
    ) -> None:
        self.max_length = max_length
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = str(model_name)

        if "jinaai" in self.model_name.lower():  # jinaai/jina-embeddings-v3
            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        elif "roberta" in self.model_name.lower():  # "FacebookAI/xlm-roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", add_prefix_space=True)
            self.model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
        else:
            raise NotImplementedError

        self.model = self.model.to(self.device)
        self.output_shape = (
            int(self.max_length),
            int(self.model.config.hidden_size),
        )

    def __call__(self, text: str) -> torch.Tensor:
        if text != "" and str(text) != "nan":
            inputs = self.tokenizer(
                text, return_tensors="pt", padding="max_length", truncation="longest_first", max_length=self.max_length
            )

            inputs = inputs.to(self.device)

            with torch.no_grad():
                features = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            return features.last_hidden_state.squeeze().cpu().float().detach().numpy()

        features = torch.zeros((1, self.max_length, self.model.config.hidden_size))

        return features.detach().cpu().squeeze()
