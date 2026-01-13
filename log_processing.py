import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
LOG_FILE = 'ship_training_2775964.log'
CSV_OUTPUT = 'training_metrics.csv'

def parse_log(file_path):
    data = []
    
    # Regex to capture the final status line of each epoch
    # Pattern looks for: "dice_coef: 0.xxxx - loss: 0.xxxx - val_dice_coef: 0.xxxx - val_loss: 0.xxxx"
    # It also handles the optional learning rate at the end
    metrics_pattern = re.compile(
        r'dice_coef:\s+(?P<dice>[\d\.]+)\s+-\s+'
        r'loss:\s+(?P<loss>[\d\.]+)\s+-\s+'
        r'val_dice_coef:\s+(?P<val_dice>[\d\.]+)\s+-\s+'
        r'val_loss:\s+(?P<val_loss>[\d\.]+)'
        r'(?:\s+-\s+learning_rate:\s+(?P<lr>[\d\.]+e?-?\d*))?'
    )
    
    # Regex to capture duration (e.g., " 807s ")
    duration_pattern = re.compile(r'\s(\d+)s\s')
    
    current_epoch = 0
    current_lr = 0.001 # Default start LR
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Track Epoch Number
        if "Epoch " in line and "/" in line:
            parts = line.split('/')
            if parts[0].strip().startswith("Epoch"):
                try:
                    current_epoch = int(parts[0].split()[1])
                except:
                    pass
        
        # Track Learning Rate Changes (ReduceLROnPlateau)
        if "reducing learning rate to" in line:
            try:
                current_lr = float(line.split(" to ")[-1].strip())
            except:
                pass

        # Extract Metrics
        if "val_loss:" in line:
            # Clean ANSI colors if present
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            match = metrics_pattern.search(clean_line)
            if match:
                # Extract Duration
                dur_match = duration_pattern.search(clean_line)
                duration = int(dur_match.group(1)) if dur_match else 0
                
                # Get specific LR for this line if present, else use tracked LR
                line_lr = match.group('lr')
                final_lr = float(line_lr) if line_lr else current_lr

                row = {
                    'Epoch': current_epoch,
                    'Training Loss': float(match.group('loss')),
                    'Training Dice': float(match.group('dice')),
                    'Val Loss': float(match.group('val_loss')),
                    'Val Dice': float(match.group('val_dice')),
                    'Learning Rate': final_lr,
                    'Duration (s)': duration
                }
                data.append(row)

    return pd.DataFrame(data)

def plot_metrics(df):
    sns.set_theme(style="whitegrid")
    
    # 1. Loss Graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss', marker='o')
    plt.plot(df['Epoch'], df['Val Loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('graph_loss.png')
    print("Saved graph_loss.png")

    # 2. Dice Coefficient Graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Training Dice'], label='Training Dice', marker='o')
    plt.plot(df['Epoch'], df['Val Dice'], label='Validation Dice', marker='o')
    plt.title('Dice Coefficient (Accuracy) Over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.savefig('graph_dice.png')
    print("Saved graph_dice.png")

    # 3. Learning Rate & Duration
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate', color=color)
    ax1.step(df['Epoch'], df['Learning Rate'], where='post', color=color, marker='x')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Learning Rate Schedule & Training Duration', fontsize=14)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Duration (seconds)', color=color)
    ax2.bar(df['Epoch'], df['Duration (s)'], alpha=0.3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.savefig('graph_lr_duration.png')
    print("Saved graph_lr_duration.png")

if __name__ == "__main__":
    try:
        print(f"Processing {LOG_FILE}...")
        df = parse_log(LOG_FILE)
        
        if not df.empty:
            df.to_csv(CSV_OUTPUT, index=False)
            print(f"Successfully saved metrics to {CSV_OUTPUT}")
            print(df.head())
            plot_metrics(df)
        else:
            print("No metrics found. Please check the log file format.")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{LOG_FILE}'. Make sure it is in the same folder.")