import datasets
from .llm import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 2: Define the model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 160)
        self.fc2 = nn.Linear(160, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

input_dim = 50304
bin_model = BinaryClassifier(input_dim)
try:
    bin_model.load_state_dict(torch.load('binary_classification.pth'))
except:
    pass

bin_model.to(device)
model.to(device)

import logging
 
    
def calculate_gpt(text):
    bin_model.eval()
    # Calculate the AI content percentage based on the text
    score = bin_model(embed(model, tokenizer, device, [text]))
    
    # Get an instance of a logger
    logger = logging.getLogger(__name__)
    logger.warning(text)
    logger.warning(score)
    return score.cpu().item() 
