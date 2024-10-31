import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


from datasets import FillDataset

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def main():
    dataset = FillDataset()
    prompt_list = [dataset[0]["txt"], dataset[1]["txt"]]
    img_list = [dataset[0]["jpg"], dataset[1]["jpg"]]
    hint_list = [dataset[0]["hint"], dataset[1]["hint"]]
    
    text_model1 = "openai/clip-vit-large-patch14"
    tokenizer1 = CLIPTokenizer.from_pretrained(text_model1)
    text_encoder1 = CLIPTextModelWithProjection.from_pretrained(text_model1)

    text_model2 = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    tokenizer2 = CLIPTokenizer.from_pretrained(text_model2)
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(text_model2)

    text_model3 = "google/t5-v1_1-xxl"
    tokenizer3 = T5TokenizerFast.from_pretrained(text_model3)
    text_encoder3 = T5EncoderModel.from_pretrained(text_model3)
    
    encoder_list = [text_encoder1, text_encoder2, text_encoder3]
    tokenizer_list = [tokenizer1, tokenizer2, tokenizer3]
    
    for encoder in encoder_list:
        for param in encoder.parameters():
            param.requires_grad = False
            
    
    prompt_embeds1, pooled_prompt_embeds1 = encode_prompt(encoder_list, tokenizer_list, prompt_list[0], 77)
    prompt_embeds2, pooled_prompt_embeds2 = encode_prompt(encoder_list, tokenizer_list, prompt_list[1], 77)

    
    tensor_list = [
        {'prompt_embeds': prompt_embeds1, 
        'pooled_prompt_embeds': pooled_prompt_embeds1,
        'prompt': prompt_list[0], 
        'img': img_list[0],
        'hint': hint_list[0]
        },
        {'prompt_embeds': prompt_embeds2, 
        'pooled_prompt_embeds': pooled_prompt_embeds2,
        'prompt': prompt_list[1], 
        'img': img_list[1],
        'hint': hint_list[1]
        }
    ]
    torch.save(tensor_list, 'prompt_tensors_list.pt')
    
def get_precomputed_tensors(filename='prompt_tensors_list.pt'):
    return torch.load(filename)
    
    
if __name__ == "__main__":
    main()
    