import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

from config import parse_args
from models import get_model
from utils.trainer import train_model
from data.dataset import NewsDataset # needed if loading with torch.load when class is not in __main__

# Suppress warnings from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    if not os.path.exists(args.train_data) or not os.path.exists(args.test_data):
        raise FileNotFoundError("Dataset files not found. Please run the prepair-data notebook first.")
        
    train_dataset = torch.load(args.train_data)
    test_dataset = torch.load(args.test_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Initializing model {args.model}...")
    model = get_model(args.model, args.num_classes)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    history = train_model(
        model=model, 
        model_name=args.model, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device,
        epochs=args.epochs,
        save_path=args.weights_dir
    )
    
    print("Training finished.")

if __name__ == "__main__":
    main()
