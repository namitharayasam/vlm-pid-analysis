from torch.utils.data import Dataset, DataLoader # Keep only what's needed for the class

class GQADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.dataset_name = "GQA"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        question = item['question']
        answer = item['answer']
        fullAnswer = item.get('fullAnswer', None) 

        # The message format for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}"}
                ]
            }
        ]

        return {
            'image': image,
            'question': question,
            'answer': answer,
            'fullAnswer': fullAnswer,
            'messages': messages
        }