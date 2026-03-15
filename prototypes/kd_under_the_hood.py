"""
Knowledge Distillation (KD) - Educational Prototype.
This script demonstrates the "under the hood" mechanics of Knowledge Distillation.
It trains a small, blank Student model (GPT-2 config) to mimic a pre-trained Teacher 
model (TinyLlama) using a combined loss function (Hard Cross-Entropy + Soft KL-Divergence).

Note: This is a toy example to demonstrate the training loop and mathematics of KD.
For the actual inference benchmark, we use officially distilled production checkpoints.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Config
from src.benchmark_core import CONFIG

# ----- Hyperparameters ------
distil_cfg = CONFIG['distillation']
TEACHER_MODEL   = CONFIG['model']['name']
TEMPERATURE     = distil_cfg['temperature']
ALPHA           = distil_cfg['alpha']
BETA            = distil_cfg['beta']
LR              = distil_cfg['learning_rate']
N_STEPS         = distil_cfg['n_steps']
MAX_LENGTH      = distil_cfg['max_length']

# Dummy calibration data
TRAIN_DATA = [
    "Machine learning enables computers to learn from data.",
    "Neural networks are inspired by the human brain.",
    "Deep learning uses multiple layers to process information.",
    "Transformers use attention mechanisms for sequence modeling."
]

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha, beta):
    """
    Computes the Knowledge Distillation loss.
    L_KD = (alpha * Hard_Loss) + (beta * Soft_Loss)
    """
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    soft_loss = F.kl_div(
        student_soft.view(-1, student_soft.size(-1)),
        teacher_soft.view(-1, teacher_soft.size(-1)),
        reduction="batchmean"
    ) * (TEMPERATURE ** 2)
    
    return alpha * hard_loss + beta * soft_loss, hard_loss.item(), soft_loss.item()

def run_toy_distillation():
    print("Loading Teacher model (TinyLlama)...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    
    # Load on CPU to prevent VRAM crash during this toy prototype
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, torch_dtype=torch.float32, device_map="cpu")
    teacher.eval() # Teacher is frozen
    
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("Building blank Student model (Custom 4-Layer GPT-2)...")
    student_config = GPT2Config(vocab_size=teacher_tokenizer.vocab_size, n_embd=256, n_layer=4, n_head=4, n_positions=512)
    student = GPT2LMHeadModel(student_config)
    student.train()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    
    # Tokenize data
    encodings = teacher_tokenizer(TRAIN_DATA, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100 # Ignore padding in loss
    
    print(f"Training student for {N_STEPS} steps...")
    
    # Simple training loop
    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()
        
        # We process the first sequence just for this toy example
        x = input_ids[0].unsqueeze(0)
        mask = attention_mask[0].unsqueeze(0)
        y = labels[0].unsqueeze(0)
        
        # Teacher Forward Pass (No gradients)
        with torch.no_grad():
            teacher_out = teacher(input_ids=x, attention_mask=mask)
            teacher_logits = teacher_out.logits
            
        # Student Forward Pass
        student_out = student(input_ids=x)
        student_logits = student_out.logits
        
        # Calculate KD Loss
        loss, hard, soft = distillation_loss(student_logits, teacher_logits, y, TEMPERATURE, ALPHA, BETA)
        
        # Backward Pass & Optimize
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:2d}/{N_STEPS} | Total Loss: {loss.item():.4f} | Hard: {hard:.4f} | Soft: {soft:.4f}")

if __name__ == "__main__":
    run_toy_distillation()