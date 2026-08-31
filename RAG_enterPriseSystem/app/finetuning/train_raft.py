"""
RAFT fine-tuning entrypoint for LLaMA 3.1 8B using LoRA.

This is a scaffold for prototype readiness. Run only in an environment with:
- GPU
- Hugging Face access to unsloth/llama-3.1-8b-Instruct-bnb-4bit
- unsloth / trl / peft / bitsandbytes installed
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/llama-3.1-8b-Instruct-bnb-4bit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="models/ip-sakti-llama3.1-8b-raft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()
    
    try:
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install: pip install unsloth trl peft bitsandbytes datasets"
        ) from exc
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    dataset = load_dataset("json", data_files=args.data, split="train")
    
    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    
    dataset = dataset.map(format_chat)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            output_dir=args.output_dir,
            save_strategy="epoch",
        ),
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
