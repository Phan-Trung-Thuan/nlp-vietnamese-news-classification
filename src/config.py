import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Vietnamese News Classification")
    parser.add_argument("--model", type=str, default="minibert", choices=["1d_cnn", "bi_lstm", "bi_gru", "minibert"], help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training/evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--train_data", type=str, default="data/processed/train_dataset.pth", help="Path to training dataset (.pth)")
    parser.add_argument("--test_data", type=str, default="data/processed/test_dataset.pth", help="Path to testing dataset (.pth)")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save/load weights")
    parser.add_argument("--num_classes", type=int, default=12, help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--text", type=str, default="", help="Text to classify (for inference)")
    parser.add_argument("--stopwords", type=str, default="data/processed/vietnamese-stopwords.txt", help="Path to stopwords file")
    
    return parser.parse_args()

CATEGORY_MAP = {
    1: "Chính trị",
    2: "Xã hội",
    3: "Kinh tế",
    4: "Văn hóa",
    5: "Sức khỏe",
    6: "Pháp luật",
    7: "Thế giới",
    8: "Khoa học - Công nghệ",
    9: "Thể thao",
    10: "Giải trí",
    11: "Du lịch",
    12: "Giáo dục"
}
