import os
import torch
from torch.utils.data import DataLoader
import warnings

from config import parse_args
from models import get_model
from utils.trainer import eval_model
from data.dataset import NewsDataset # needed if loading with torch.load when class is not in __main__

# Suppress warnings from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading test dataset...")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError("Test dataset file not found.")
        
    test_dataset = torch.load(args.test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Initializing model {args.model}...")
    model = get_model(args.model, args.num_classes)
    
    weight_path = os.path.join(args.weights_dir, f"{args.model}_best_loss_weight.pth")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(args.weights_dir, f"{args.model}_best_loss_weigth.pth") # original typo in notebook

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
    print(f"Loading weights from {weight_path}...")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    print("Starting evaluation...")
    acc, f1 = eval_model(
        model=model, 
        model_name=args.model, 
        test_dataloader=test_dataloader, 
        device=device,
        save_path='docs'
    )
    
    print("Evaluation finished.")

if __name__ == "__main__":
    main()
