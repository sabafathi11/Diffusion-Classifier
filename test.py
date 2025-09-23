import torch
import torch.nn as nn
from diffusers import  StableDiffusionPipeline, EulerDiscreteScheduler


class PromptLearner(nn.Module):
    def __init__(self, class_names, tokenizer, text_encoder, n_ctx=16, ctx_init=None, class_token_position="end", dtype=torch.float16):
        super().__init__()
        
        self.n_cls = len(class_names)
        self.class_names = class_names
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.class_token_position = class_token_position
        
        # Get text encoder embedding dimension from the actual model
        self.ctx_dim = self.text_encoder.get_input_embeddings().embedding_dim
        

        if ctx_init:
            # use given words to initialize context vectors
            n_ctx = len(ctx_init.split(" "))
            prompt_tokens = self.tokenizer(ctx_init, add_special_tokens=False, return_tensors="pt")
            with torch.no_grad():
                input_ids = prompt_tokens.input_ids.to(self.text_encoder.device)
                init_embeddings = self.text_encoder.get_input_embeddings()(input_ids).type(dtype)

            ctx_vectors = init_embeddings[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        name_lens = [self.tokenizer(name, return_tensors="pt").input_ids.shape[1] for name in self.class_names]
        prompts = [prompt_prefix + " " + name + "." for name in self.class_names]

        tokenized_prompts = torch.cat([self.tokenizer(p, return_tensors="pt").input_ids for p in prompts]).to(self.text_encoder.device)
        with torch.no_grad():
            embedding = self.text_encoder.get_input_embeddings()(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        position_ids = torch.arange(prompts.size(1), device=prompts.device).unsqueeze(0)
        position_embeddings = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
        hidden_states = prompts + position_embeddings
        mask = torch.ones(prompts.size(0), 1, prompts.size(1), prompts.size(1), device=prompts.device).type(hidden_states.dtype)

        encoder_outputs = self.text_encoder.text_model.encoder(
            hidden_states,
            attention_mask=mask,
            output_hidden_states=False,
        )

        last_hidden_state = self.text_encoder.text_model.final_layer_norm(encoder_outputs[0])

        return last_hidden_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
custom_cache = "/mnt/public/Ehsan/docker_private/learning2/saba/datasets/SD"
scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler", cache_dir=custom_cache)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base", scheduler=scheduler, torch_dtype=dtype, cache_dir=custom_cache
)

vae = pipe.vae.to(device)
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.to(device)
unet = pipe.unet.to(device)
torch.backends.cudnn.benchmark = True

prompt_learner = PromptLearner(['a','b','c'], tokenizer, text_encoder, 16, 'a photo of a', 'end').to(device)

x = prompt_learner()
print(x.shape)