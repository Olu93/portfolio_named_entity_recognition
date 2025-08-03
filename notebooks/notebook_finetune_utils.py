from transformers import DistilBertForTokenClassification
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

class WeightedDistilBertForTokenClassification(DistilBertForTokenClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_state = outputs[0]        # (bs, seq_len, hidden_size)
        logits = self.classifier(hidden_state)  # (bs, seq_len, num_labels)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits
        }



class WeightedTokenClassifier(AutoModelForTokenClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # don't use default loss
            **kwargs
        )
        logits = outputs.logits  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits
        }




class DistilBertWithHingeLoss(DistilBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.MultiMarginLoss()  # You can add margin=1.0, reduction='mean'

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # **kwargs
        )
        hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)
        logits = self.classifier(hidden_state)  # (batch_size, seq_len, num_labels)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = self.loss_fn(active_logits, active_labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


