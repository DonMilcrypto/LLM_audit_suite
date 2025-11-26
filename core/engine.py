from transformers import AutoTokenizer, AutoModelForCausalLM

class AuditEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("AuditEngine")
        self._load_resources()

    def _load_resources(self):
        self.logger.info(f"Initializing resources on {self.config.DEVICE}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_PATH)

            # CRITICAL: Set padding to left for Causal LM generation
            self.tokenizer.padding_side = "left"

            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_PATH,
                torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.config.DEVICE == "cuda" else None
            )

            if self.config.DEVICE == "cpu":
                self.model.to("cpu")

            self.model.eval()
            self.logger.info("Engine Ready.")
        except Exception as e:
            self.logger.critical(f"Engine Initialization Failed: {e}")
            raise

    def generate_batch(self, batch_data: list):
        """
        Processes a batch of prompt dictionaries.
        Returns a list of result dictionaries populated with outputs.
        """
        prompts = [item['prompt'] for item in batch_data]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.config.DEVICE)

        results = []

        # We run the generation loop multiple times per prompt if requested
        # Note: Ideally, we would expand the batch size by ITERATIONS factor,
        # but for simplicity, we loop the generation call here.
        for _ in range(self.config.ITERATIONS_PER_PROMPT):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.config.GEN_PARAMS,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Batch decode
            decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Align outputs with original prompt objects
            for i, text in enumerate(decoded_texts):
                # Remove the input prompt from the output for cleaner logs
                response_only = text[len(prompts[i]):]

                entry = batch_data[i].copy()
                entry['output'] = response_only
                entry['model_config'] = self.config.MODEL_PATH
                results.append(entry)

        return results
