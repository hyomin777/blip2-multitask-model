import math
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration
)


class Mode(Enum):
    ALL = 'all'
    SENTIMENT = 'sentiment'
    CATEGORY = 'category'
    TEXT = 'text'


class TaskSpecificAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim)
        )

    def forward(self, x):
        # Residual connection with task-specific adaptation
        return x + self.adapter(x)


class Blip2MultiTask(nn.Module):
    def __init__(
        self,
        blip_name: str = "Salesforce/blip2-opt-2.7b",
        sentiment_classes: int = 2,
        category_classes: int = 8,
        caption_max_new_tokens: int = 50,
    ):
        super().__init__()
        self.blip = Blip2ForConditionalGeneration.from_pretrained(
            blip_name,
            torch_dtype=torch.float16,
        )
        self.processor = Blip2Processor.from_pretrained(blip_name)

        for p in self.blip.parameters():
            p.requires_grad = False
        
        q_dim = self.blip.config.qformer_config.hidden_size
        print(f'Hidden dim: {q_dim}')

        self.sentiment_head = MLPHead(
            input_dim=q_dim,
            hidden_dims=[512, 256],
            num_classes=sentiment_classes,
            dropout_rate=0.2
        )
        self.category_head = MLPHead(
            input_dim=q_dim,
            hidden_dims=[512, 256],
            num_classes=category_classes,
            dropout_rate=0.2
        )

        self.caption_max_new_tokens = caption_max_new_tokens

    def extract_features(self, pixel_values: torch.Tensor):
        # Image encoding
        with torch.no_grad():
            image_embeds = self.blip.vision_model(pixel_values).last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )

            query_tokens = self.blip.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.blip.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
        return query_outputs.pooler_output  # [B, q_dim]

    def forward(self, pixel_values: torch.Tensor, mode: Mode):
        if mode == Mode.TEXT:
            generate_ids = self.blip.generate(
                pixel_values=pixel_values,
                max_new_tokens=self.caption_max_new_tokens
            )
            generated_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            return generated_text
        
        elif mode == Mode.ALL:
            features = self.extract_features(pixel_values)
            
            sentiment_logits = self.sentiment_head(features)
            category_logits = self.category_head(features)
            
            return {
                'sentiment': sentiment_logits,
                'category': category_logits
            }
        
        elif mode in [Mode.SENTIMENT, Mode.CATEGORY]:
            features = self.extract_features(pixel_values)
            
            if mode == Mode.SENTIMENT:
                return self.sentiment_head(features)
            elif mode == Mode.CATEGORY:
                return self.category_head(features)
        
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_trainable_parameters(self, which: Mode = Mode.ALL):
        params = []
        if which in [Mode.SENTIMENT, Mode.ALL]:
            params.extend(list(self.sentiment_head.parameters()))
        
        if which in [Mode.CATEGORY, Mode.ALL]:
            params.extend(list(self.category_head.parameters()))
        return params
    
    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.2):
        super().__init__()
        layers = []

        self.adapter = TaskSpecificAdapter(input_dim)

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.adapter(x)
        return self.mlp(x)
