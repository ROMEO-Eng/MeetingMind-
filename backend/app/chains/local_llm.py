"""Local Mistral Nemo 4-bit model and LangChain LLM wrapper."""

from typing import Any

from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict, Field

from backend.app.config import Settings
from backend.app.chains.errors import ModelUnavailableError


class TransformersLLM(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(exclude=True)
    tokenizer: Any = Field(exclude=True)
    model_name: str
    max_new_tokens: int = 900
    max_input_tokens: int = 8192

    @property
    def _llm_type(self) -> str:
        return "local-transformers-mistral-nemo"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "max_new_tokens": self.max_new_tokens}

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for stop_text in stop or []:
            if stop_text in answer:
                answer = answer.split(stop_text, 1)[0].rstrip()
        return answer


def load_local_model(settings: Settings) -> tuple[TransformersLLM, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise ModelUnavailableError(
            "Meeting analysis needs a CUDA GPU for 4-bit Mistral Nemo. Use a CUDA-enabled GPU runtime."
        )
    try:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            quantization_config=quantization,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        llm = TransformersLLM(
            model=model,
            tokenizer=tokenizer,
            model_name=settings.model_name,
            max_new_tokens=settings.model_max_new_tokens,
            max_input_tokens=settings.max_input_tokens,
        )
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer(
            settings.embedding_model,
            device="cuda",
        )
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            "The local AI model could not be loaded. Check GPU memory, model access, and backend logs."
        ) from error
    return llm, embeddings
