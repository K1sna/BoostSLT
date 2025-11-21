#!/usr/bin/env python3

import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np


class LexicalClassifier(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.classifier(lstm_out)
        return logits


class NamedEntityRecognizer:
    
    def __init__(self):
        self.person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b'
        ]
        self.location_patterns = [
            r'\b[A-Z][a-z]+(?:stadt|burg|land|berg)\b',
            r'\b(?:Berlin|München|Hamburg|Köln|Frankfurt|Stuttgart|Düsseldorf|Dortmund|Essen|Leipzig)\b'
        ]
        self.organization_patterns = [
            r'\b[A-Z][A-Za-z]+ (?:GmbH|AG|Inc|Corp|Ltd)\b'
        ]
        self.number_patterns = [
            r'\b\d+(?:\.\d+)?\b',
            r'\b(?:ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|hundert|tausend)\b'
        ]
        
    def extract_entities(self, text: str) -> Set[Tuple[int, int, str]]:
        entities = set()
        words = text.split()
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            
            for pattern in self.person_patterns:
                if re.match(pattern, word):
                    entities.add((i, i + 1, 'PERSON'))
                    break
            
            for pattern in self.location_patterns:
                if re.match(pattern, word, re.IGNORECASE):
                    entities.add((i, i + 1, 'LOCATION'))
                    break
            
            for pattern in self.organization_patterns:
                if re.match(pattern, word):
                    entities.add((i, i + 1, 'ORG'))
                    break
            
            for pattern in self.number_patterns:
                if re.match(pattern, word, re.IGNORECASE):
                    entities.add((i, i + 1, 'NUMERIC'))
                    break
        
        return entities


class POSBasedClassifier:
    
    def __init__(self, language: str = 'de'):
        self.language = language
        
        self.content_word_patterns = {
            'nouns': [
                r'\b(?:der|die|das|ein|eine|eines|einer|einem)\s+[A-Z][a-z]+\w*\b',
                r'\b[A-Z][a-z]+\w*\b'
            ],
            'verbs': [
                r'\b\w+(?:en|st|t|te|ten|test|tet)\b',
                r'\b(?:ist|sind|war|waren|hat|haben|wird|werden|kann|können|muss|müssen)\b'
            ]
        }
        
        self.function_words = {
            'de': {
                'articles': ['der', 'die', 'das', 'ein', 'eine', 'eines', 'einer', 'einem', 'den', 'dem', 'des'],
                'prepositions': ['in', 'auf', 'für', 'von', 'mit', 'zu', 'an', 'bei', 'über', 'unter', 'durch', 'nach', 'vor', 'zwischen'],
                'conjunctions': ['und', 'oder', 'aber', 'dass', 'weil', 'wenn', 'obwohl', 'damit', 'sodass'],
                'pronouns': ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'mich', 'dich', 'ihn', 'uns', 'euch'],
                'adverbs': ['nicht', 'auch', 'noch', 'schon', 'immer', 'oft', 'manchmal', 'sehr', 'zu', 'so', 'wie', 'wo', 'wann', 'warum']
            },
            'en': {
                'articles': ['the', 'a', 'an'],
                'prepositions': ['in', 'on', 'at', 'for', 'of', 'with', 'by', 'from', 'to', 'about', 'into', 'onto'],
                'conjunctions': ['and', 'or', 'but', 'that', 'because', 'if', 'although', 'so', 'when', 'where'],
                'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
                'adverbs': ['not', 'also', 'still', 'already', 'always', 'often', 'sometimes', 'very', 'too', 'so', 'how', 'where', 'when', 'why']
            }
        }
        
    def classify_word(self, word: str, context: Optional[List[str]] = None) -> str:
        word_lower = word.lower().strip()
        word_clean = re.sub(r'[^\w]', '', word_lower)
        
        if not word_clean:
            return 'function'
        
        func_words = self.function_words.get(self.language, self.function_words['en'])
        
        for category in func_words.values():
            if word_clean in category:
                return 'function'
        
        if word[0].isupper() and len(word) > 2:
            return 'content'
        
        if re.match(r'^\d+', word_clean):
            return 'content'
        
        if len(word_clean) > 4:
            return 'content'
        
        return 'function'


class LexMasker:
    
    def __init__(
        self,
        tokenizer,
        language: str = 'de',
        use_ner: bool = True,
        use_ml_classifier: bool = False,
        ml_classifier: Optional[LexicalClassifier] = None,
        content_word_threshold: float = 0.6,
        mask_function_words: bool = True,
        preserve_named_entities: bool = True,
        preserve_numerals: bool = True
    ):
        self.tokenizer = tokenizer
        self.language = language
        self.use_ner = use_ner
        self.use_ml_classifier = use_ml_classifier
        self.ml_classifier = ml_classifier
        self.content_word_threshold = content_word_threshold
        self.mask_function_words = mask_function_words
        self.preserve_named_entities = preserve_named_entities
        self.preserve_numerals = preserve_numerals
        
        self.pos_classifier = POSBasedClassifier(language=language)
        self.ner = NamedEntityRecognizer() if use_ner else None
        
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.unk_token_id
        if self.mask_token_id is None:
            self.mask_token_id = tokenizer.unk_token_id
        
    def identify_content_words(
        self,
        tokens: List[str],
        token_ids: Optional[torch.Tensor] = None
    ) -> Dict[int, bool]:
        content_mask = {}
        text = ' '.join(tokens)
        
        named_entities = set()
        if self.use_ner and self.ner:
            named_entities = self.ner.extract_entities(text)
            ner_indices = {start for start, end, _ in named_entities}
        else:
            ner_indices = set()
        
        for i, token in enumerate(tokens):
            is_content = False
            
            if self.use_ml_classifier and self.ml_classifier is not None and token_ids is not None:
                with torch.no_grad():
                    if token_ids.dim() == 1:
                        token_ids_batch = token_ids.unsqueeze(0)
                    else:
                        token_ids_batch = token_ids
                    
                    if i < token_ids_batch.shape[1]:
                        logits = self.ml_classifier(token_ids_batch)
                        probs = F.softmax(logits, dim=-1)
                        content_prob = probs[0, i, 1].item()
                        is_content = content_prob > self.content_word_threshold
            
            if not is_content:
                word_clean = re.sub(r'[^\w]', '', token.lower())
                
                if i in ner_indices and self.preserve_named_entities:
                    is_content = True
                elif re.match(r'^\d+', word_clean) and self.preserve_numerals:
                    is_content = True
                else:
                    word_class = self.pos_classifier.classify_word(token, tokens)
                    is_content = (word_class == 'content')
            
            content_mask[i] = is_content
        
        return content_mask
    
    def create_mask(
        self,
        tokens: List[str],
        token_ids: torch.Tensor,
        current_mask: Optional[torch.Tensor] = None,
        preserve_anchors: bool = True
    ) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = token_ids.shape
        
        if current_mask is None:
            current_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=token_ids.device)
        
        content_mask = self.identify_content_words(tokens, token_ids[0])
        
        new_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=token_ids.device)
        
        for i in range(min(len(tokens), seq_len)):
            is_content = content_mask.get(i, False)
            
            if preserve_anchors and is_content:
                new_mask[0, i] = False
            elif self.mask_function_words and not is_content:
                new_mask[0, i] = True
            elif current_mask[0, i]:
                new_mask[0, i] = True
        
        if squeeze_output:
            new_mask = new_mask.squeeze(0)
        
        return new_mask
    
    def apply_mask(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
            squeeze_output = True
        else:
            squeeze_output = False
        
        masked_ids = token_ids.clone()
        masked_ids[mask] = self.mask_token_id
        
        if squeeze_output:
            masked_ids = masked_ids.squeeze(0)
        
        return masked_ids
    
    def remask_step(
        self,
        tokens: List[str],
        token_ids: torch.Tensor,
        current_mask: Optional[torch.Tensor] = None,
        denoising_step: int = 0,
        total_steps: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        new_mask = self.create_mask(tokens, token_ids, current_mask, preserve_anchors=True)
        masked_ids = self.apply_mask(token_ids, new_mask)
        
        content_mask = self.identify_content_words(tokens, token_ids)
        num_content = sum(1 for v in content_mask.values() if v)
        num_function = len(content_mask) - num_content
        num_masked = new_mask.sum().item() if new_mask.dim() > 0 else int(new_mask.item())
        
        info = {
            'num_content_words': num_content,
            'num_function_words': num_function,
            'num_masked_tokens': num_masked,
            'mask_ratio': num_masked / len(tokens) if len(tokens) > 0 else 0.0,
            'denoising_step': denoising_step,
            'preserved_anchors': num_content - num_masked
        }
        
        return masked_ids, new_mask, info
    
    def get_semantic_anchors(
        self,
        tokens: List[str],
        token_ids: Optional[torch.Tensor] = None
    ) -> List[Tuple[int, str]]:
        content_mask = self.identify_content_words(tokens, token_ids)
        anchors = [(i, tokens[i]) for i in range(len(tokens)) if content_mask.get(i, False)]
        return anchors
    
    def check_constraints(
        self,
        tokens: List[str],
        original_segments: List[str],
        named_entity_preservation: bool = True,
        segment_coverage: bool = True,
        temporal_order: bool = True
    ) -> Dict[str, bool]:
        constraints = {
            'named_entity_preserved': True,
            'segment_coverage': True,
            'temporal_order': True
        }
        
        if named_entity_preservation and self.ner:
            text = ' '.join(tokens)
            entities = self.ner.extract_entities(text)
            constraints['named_entity_preserved'] = len(entities) > 0 or not self.preserve_named_entities
        
        if segment_coverage:
            text = ' '.join(tokens).lower()
            for seg in original_segments:
                seg_lower = seg.lower()
                if seg_lower not in text:
                    constraints['segment_coverage'] = False
                    break
        
        return constraints


def create_lexmasker(
    tokenizer,
    language: str = 'de',
    use_ml_classifier: bool = False,
    classifier_path: Optional[str] = None,
    **kwargs
) -> LexMasker:
    ml_classifier = None
    if use_ml_classifier:
        if classifier_path and os.path.exists(classifier_path):
            ml_classifier = LexicalClassifier()
            ml_classifier.load_state_dict(torch.load(classifier_path))
            ml_classifier.eval()
        else:
            print("Warning: ML classifier path not found, using rule-based classifier only")
            use_ml_classifier = False
    
    return LexMasker(
        tokenizer=tokenizer,
        language=language,
        use_ml_classifier=use_ml_classifier,
        ml_classifier=ml_classifier,
        **kwargs
    )
