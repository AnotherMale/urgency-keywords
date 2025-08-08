def predict_urgency_from_github(text):
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    import requests

    class SentimentClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, input_ids, attention_mask):
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embedding = bert_output.last_hidden_state[:, 0, :]
            return self.classifier(sentence_embedding)

    def download_model_from_lfs(owner, repo, branch, path, local_path='model.pth'):
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        r = requests.get(api_url)
        r.raise_for_status()
        file_info = r.json()
        if "download_url" not in file_info or file_info["download_url"] is None:
            raise RuntimeError("Could not find download URL in GitHub API response.")
        r2 = requests.get(file_info["download_url"], stream=True)
        r2.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r2.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path

    local_model_path = download_model_from_lfs(
        owner="AnotherMale",
        repo="urgency-keywords",
        branch="main",
        path="bert_urgency_model.pth"
    )

    model = SentimentClassifier()
    state_dict = torch.load(local_model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(encoding['input_ids'], encoding['attention_mask']).item()
        return {
            'probability': output,
            'class': 'urgency keywords present' if output > 0.5 else 'no urgency keywords present'
        }

result = predict_urgency_from_github("This is your last chance to WIN cash. Click this link")
print(result)
