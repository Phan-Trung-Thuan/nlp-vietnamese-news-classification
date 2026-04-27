import os
import torch
from transformers import AutoTokenizer

from config import parse_args, CATEGORY_MAP
from models import get_model
from utils.inference import predict
from data.dataset import load_stopwords

def main():
    args = parse_args()
    
    if not args.text:
        raise ValueError("Please provide text to classify using the --text argument.")
        
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading tokenizer and stopwords...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    stop_words = load_stopwords(args.stopwords)
    
    print(f"Initializing model {args.model}...")
    model = get_model(args.model, args.num_classes)
    
    weight_path = os.path.join(args.weights_dir, f"{args.model}_best_loss_weight.pth")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(args.weights_dir, f"{args.model}_best_loss_weigth.pth") # original typo in notebook

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
    print(f"Loading weights from {weight_path}...")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    print("Running inference...")
    prediction, probabilities = predict(
        model=model, 
        text=args.text, 
        tokenizer=tokenizer, 
        stop_words=stop_words, 
        device=device
    )
    
    predicted_category = CATEGORY_MAP.get(prediction + 1, "Unknown") # +1 because classes in CATEGORY_MAP are 1-indexed, prediction is 0-indexed
    
    print("\n--- Inference Result ---")
    print(f"Text: {args.text[:100]}..." if len(args.text) > 100 else f"Text: {args.text}")
    print(f"Predicted Category: {predicted_category} (Class {prediction + 1})")
    print("------------------------\n")

if __name__ == "__main__":
    main()
