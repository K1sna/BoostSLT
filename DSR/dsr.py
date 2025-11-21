#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel

from lexmasker import LexMasker


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def encode_segments(
    segments: List[str],
    tokenizer,
    max_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    combined_text = " ".join(segments)
    
    encoded = tokenizer(
        combined_text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    return encoded['input_ids'], encoded['attention_mask']


def decode_tokens(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_with_lexmasker(
    model,
    prompt: torch.Tensor,
    lexmasker: LexMasker,
    original_segments: List[str],
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'lexical_aware',
    mask_id: int = 126336,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    use_lexmasker: bool = True,
    lexmasker_guidance_steps: Optional[List[int]] = None
) -> Tuple[torch.Tensor, Dict]:
    device = prompt.device
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)

    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    if lexmasker_guidance_steps is None:
        lexmasker_guidance_steps = list(range(steps))
    
    info = {
        'total_steps': steps * num_blocks,
        'lexmasker_applied': 0,
        'constraint_checks': []
    }

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            global_step = num_block * steps_per_block + i
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'lexical_aware' and use_lexmasker and global_step in lexmasker_guidance_steps:
                current_token_ids = x0[0].cpu().clone()
                current_tokens = lexmasker.tokenizer.convert_ids_to_tokens(current_token_ids.tolist())
                
                masked_ids, new_mask, lexmasker_info = lexmasker.remask_step(
                    current_tokens,
                    current_token_ids,
                    current_mask=mask_index[0].cpu(),
                    denoising_step=global_step,
                    total_steps=steps * num_blocks
                )
                
                new_mask = new_mask.to(device)
                if new_mask.dim() == 1:
                    new_mask = new_mask.unsqueeze(0)
                
                p = F.softmax(logits, dim=-1)
                x0_p_base = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                
                x0_p = torch.where(new_mask, x0_p_base, -np.inf)
                
                info['lexmasker_applied'] += 1
                
                constraints = lexmasker.check_constraints(
                    current_tokens,
                    original_segments,
                    named_entity_preservation=True,
                    segment_coverage=True,
                    temporal_order=True
                )
                info['constraint_checks'].append({
                    'step': global_step,
                    'constraints': constraints
                })
                
            elif remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, info


class DSRModule:
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        language: str = 'de',
        device: str = 'cuda',
        use_lexmasker: bool = True,
        **lexmasker_kwargs
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_lexmasker = use_lexmasker
        
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device).eval()
        
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        assert self.tokenizer.pad_token_id != 126336
        
        if use_lexmasker:
            from lexmasker import LexMasker
            self.lexmasker = LexMasker(
                tokenizer=self.tokenizer,
                language=language,
                **lexmasker_kwargs
            )
        else:
            self.lexmasker = None
    
    def reconstruct(
        self,
        segments: List[str],
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = 'lexical_aware',
        use_lexmasker: Optional[bool] = None,
        lexmasker_guidance_steps: Optional[List[int]] = None
    ) -> Tuple[str, Dict]:
        if use_lexmasker is None:
            use_lexmasker = self.use_lexmasker
        
        if use_lexmasker and self.lexmasker is None:
            raise ValueError("LexMasker requested but not initialized")
        
        if remasking == 'lexical_aware' and not use_lexmasker:
            remasking = 'low_confidence'
        
        input_ids, attention_mask = encode_segments(segments, self.tokenizer)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        lexmasker = self.lexmasker if use_lexmasker else None
        
        generated_ids, info = generate_with_lexmasker(
            model=self.model,
            prompt=input_ids,
            lexmasker=lexmasker,
            original_segments=segments,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=126336,
            use_lexmasker=use_lexmasker,
            lexmasker_guidance_steps=lexmasker_guidance_steps
        )
        
        output_ids = generated_ids[:, input_ids.shape[1]:]
        reconstructed_text = decode_tokens(output_ids[0], self.tokenizer)
        
        info['reconstructed_text'] = reconstructed_text
        info['original_segments'] = segments
        
        return reconstructed_text, info
    
    def batch_reconstruct(
        self,
        batch_segments: List[List[str]],
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = 'lexical_aware',
        use_lexmasker: Optional[bool] = None
    ) -> List[Tuple[str, Dict]]:
        results = []
        for segments in batch_segments:
            text, info = self.reconstruct(
                segments=segments,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                use_lexmasker=use_lexmasker
            )
            results.append((text, info))
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DSR Module')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to LLaDA model')
    parser.add_argument('--segments', type=str, nargs='+',
                       default=[],
                       help='Translation segments to reconstruct')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of denoising steps')
    parser.add_argument('--gen_length', type=int, default=None,
                       help='Generated sequence length')
    parser.add_argument('--block_length', type=int, default=None,
                       help='Block length for generation')
    parser.add_argument('--language', type=str, default='de',
                       choices=['de', 'en'],
                       help='Language for LexMasker')
    parser.add_argument('--no_lexmasker', action='store_true',
                       help='Disable LexMasker')
    
    args = parser.parse_args()
    
    dsr = DSRModule(
        model_path=args.model_path,
        language=args.language,
        use_lexmasker=not args.no_lexmasker
    )
    
    if not args.segments:
        return
    
    reconstructed, info = dsr.reconstruct(
        segments=args.segments,
        steps=args.steps if args.steps else None,
        gen_length=args.gen_length if args.gen_length else None,
        block_length=args.block_length if args.block_length else None,
        remasking='lexical_aware' if not args.no_lexmasker else 'low_confidence'
    )
    
    print(reconstructed)


if __name__ == '__main__':
    main()
