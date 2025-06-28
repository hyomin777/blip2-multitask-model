from enum import Enum
import torch
import torch.nn as nn
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration
)

class Mode(Enum):
    ALL = 'all'
    SENTIMENT = 'sentiment'
    CATEGORY = 'category'
    TEXT = 'text'


class TaskSpecificAdapter(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Residual connection with task-specific adaptation
        adapted_features = x + self.adapter(x)
        return self.classifier(adapted_features)


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
            blip_name
        )
        self.processor = Blip2Processor.from_pretrained(blip_name)

        for p in self.blip.parameters():
            p.requires_grad = False
        embed_dim = self.blip.vision_model.config.hidden_size
        print(f'Hidden dim: {embed_dim}')

        self.sentiment_head = TaskSpecificAdapter(
            hidden_dim=embed_dim,
            num_classes=sentiment_classes
        )
        self.category_head = TaskSpecificAdapter(
            hidden_dim=embed_dim,
            num_classes=category_classes
        )

        self.caption_max_new_tokens = caption_max_new_tokens

    def extract_image_embeds(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            image_embeds = self.blip.vision_model(pixel_values).last_hidden_state
        return image_embeds

    def extract_query_embeds(self, image_embeds: torch.Tensor):
        with torch.no_grad():
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
        image_embeds = self.extract_image_embeds(pixel_values)

        if mode == Mode.TEXT:
            with torch.no_grad():
                generate_ids = self.blip.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=self.caption_max_new_tokens
                )
                generated_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            return generated_text

        elif mode in [Mode.SENTIMENT, Mode.CATEGORY]:
            pooled_features = image_embeds[:, 0, :]

            if mode == Mode.SENTIMENT:
                return self.sentiment_head(pooled_features)
            elif mode == Mode.CATEGORY:
                return self.category_head(pooled_features)

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
