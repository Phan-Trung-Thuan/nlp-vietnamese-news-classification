import torch
import torch.nn.functional as F
from ..data.dataset import remove_stopwords

def predict(model, text, tokenizer, stop_words, device, max_length=8000):
    model.eval()
    model.to(device)
    
    # Clean text
    cleaned_text = remove_stopwords(text, stop_words)
    
    # Tokenize
    inputs = tokenizer([cleaned_text], 
                       padding='max_length', 
                       truncation=True, 
                       max_length=max_length, 
                       return_tensors="pt")
    
    input_ids = inputs['input_ids'].to(device)
    
    with torch.no_grad():
        output = model(input_ids)
        probabilities = F.softmax(output, dim=0) # output is (num_classes,) because of squeeze in model forward
        prediction = torch.argmax(probabilities).item()
        
    return prediction, probabilities
