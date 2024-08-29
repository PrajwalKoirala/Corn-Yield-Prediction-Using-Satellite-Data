import torch
import torch.nn as nn
from transformers import BertTokenizer, VisualBertModel
import torch.nn.functional as F
from torchvision.models import resnet18

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class TransformerEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        super(TransformerEncoder, self).__init__()
        
        hidden_size = 768
        
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.device = device
        self.to(device)
    
    def forward(self, json_strings, visual_embeds, visual_token_type_ids, visual_attention_mask):
        inputs = self.tokenizer(json_strings, return_tensors='pt', padding=True, truncation=True)
        B, seq_len, embed_size = visual_embeds.shape
        # visual token attention
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )
        inputs = inputs.to("cuda")
        #
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class SatInfoEncoder(nn.Module):
    def __init__(self, original_dim=36, target_dim=2048, device='cuda'):
        super(SatInfoEncoder, self).__init__()
        
        self.reduce_dim = nn.Linear(original_dim, target_dim//4)
        self.bn1 = nn.BatchNorm1d(target_dim//4)
        self.fc_final = nn.Linear(target_dim//4, target_dim//2)
        self.bn2 = nn.BatchNorm1d(target_dim//2)
        
        self.device = device
        self.to(device)

    def forward(self, x, embeddings_len):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        embeddings_len = embeddings_len.to(self.device)
        
        batch_size, seq_length, embedding_size = x.size()
        
        # Flatten to (batch_size * seq_length, embedding_size)
        x = x.view(batch_size * seq_length, embedding_size)
        
        # Mask for valid embeddings
        valid_embeddings = torch.zeros((batch_size * seq_length), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            valid_embeddings[i * seq_length : i * seq_length + embeddings_len[i]] = True
        
        x_valid = x[valid_embeddings]  # Only valid embeddings
        features = self.reduce_dim(x_valid)  # shape: (num_valid_embeddings, target_dim)
        features = self.bn1(features)
        features = F.relu(features)
        features = self.fc_final(features)
        features = self.bn2(features)
        
        # Create a full tensor with all features, fill with zeros where embeddings are not valid
        features_full = torch.zeros((batch_size * seq_length, features.size(1)), dtype=features.dtype, device=self.device)
        features_full[valid_embeddings] = features
        
        features_full = features_full.view(batch_size, seq_length, -1)  # shape: (batch_size, seq_length, target_dim)

        max_length = seq_length
        index_tensor = torch.arange(max_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        visual_token_type_ids = (index_tensor < embeddings_len.unsqueeze(1)).long().view(batch_size, seq_length)
        visual_attention_mask = (index_tensor < embeddings_len.unsqueeze(1)).float().view(batch_size, seq_length)
        
        return features_full, visual_token_type_ids, visual_attention_mask



class ResNetEncoder(nn.Module):
    def __init__(self, original_dim=512, target_dim=2048, device='cuda'):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        
        self.reduce_dim = nn.Linear(original_dim, target_dim//2)
        self.bn1 = nn.BatchNorm1d(target_dim//2)
        self.fc_final = nn.Linear(target_dim//2, target_dim//2)
        self.bn2 = nn.BatchNorm1d(target_dim//2)
        
        self.device = device
        self.to(device)

    def forward(self, x, images_len):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        images_len.to(self.device)
        
        batch_size, seq_length, channels, height, width = x.size()
        x = normalize(x.view(batch_size * seq_length, channels, height, width))
        
        valid_images = torch.zeros((batch_size * seq_length), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            valid_images[i * seq_length : i * seq_length + images_len[i]] = True
        
        x_valid = x[valid_images]  # Only valid images
        features = self.resnet(x_valid)
        features = self.reduce_dim(features)  # shape: (num_valid_images, target_dim)
        features = self.bn1(features)
        features = F.relu(features)
        features = self.fc_final(features)
        features = self.bn2(features)
        
        features_full = torch.zeros((batch_size * seq_length, features.size(1)), dtype=features.dtype, device=self.device)
        features_full[valid_images] = features
        
        features_full = features_full.view(batch_size, seq_length, -1)  # shape: (batch_size, seq_len, target_dim)

        max_length = seq_length
        index_tensor = torch.arange(max_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        visual_token_type_ids = (index_tensor < images_len.unsqueeze(1)).long().view(batch_size, seq_length) #*(index_tensor + 1)
        visual_attention_mask = (index_tensor < images_len.unsqueeze(1)).float().view(batch_size, seq_length)
        
        return features_full, visual_token_type_ids, visual_attention_mask

class CombinedModel(nn.Module):
    def __init__(self, device='cuda'):
        super(CombinedModel, self).__init__()
        self.image_encoder = ResNetEncoder()
        self.feature_encoder = TransformerEncoder()
        self.vector_encoder = SatInfoEncoder()
        self.fc1 = nn.Linear(128, 128)  # image encoding + feature encoding
        self.bn1 = nn.BatchNorm1d(128)  # BatchNorm layer after fc1
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.device = device
        self.to(device)

    def forward(self, log, images, images_info, images_len):
        # Extract embeddings
        image_embedding, visual_token_type_ids, visual_attention_mask = self.image_encoder(images, images_len)
        info_embedding, vector_token_type_ids, vector_attention_mask = self.vector_encoder(images_info, images_len)
        combined_embedding = torch.cat((image_embedding, info_embedding), dim=-1)
        feature_embedding = self.feature_encoder(log, combined_embedding, visual_token_type_ids, visual_attention_mask)
        x = F.relu(self.bn1(self.fc1(feature_embedding)))
        output = self.fc3(x)
        return output
