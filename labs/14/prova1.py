#!/usr/bin/env python3
import argparse
import json

import peft
import torch
import transformers

import npfl139
npfl139.require_version("2526.14.1")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--llm", default="HuggingFaceTB/SmolLM2-135M-Instruct", type=str, help="LLM to use.")
parser.add_argument("--lora_rank", default=32, type=int, help="LoRA rank to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--task", default="arithmetic", type=str, help="LLM task to solve.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=..., type=float, help="Clipping epsilon.")
parser.add_argument("--dev_size", default=256, type=int, help="Number of examples to evaluate.")
parser.add_argument("--epochs", default=..., type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--max_tokens", default=64, type=int, help="Maximum length of generated outputs.")
parser.add_argument("--model_path", default="rlvr.pt", type=str, help="Path where to save the model.")
parser.add_argument("--train_dataset", default=5, type=int, help="Train dataset size per iteration.")
parser.add_argument("--train_outcomes", default=7, type=int, help="Number of outcomes per training prompt.")


class LLMAgent(torch.nn.Module):
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, task: npfl139.llm.Task, args: argparse.Namespace) -> None:
        super().__init__()
        self._task = task
        self._args = args

        # Create a suitable tokenizer.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(args.llm)

        # Load the specified causal LLM.
        self._llm = transformers.AutoModelForCausalLM.from_pretrained(args.llm)

        # Apply LoRA of the given rank to the model.
        self._llm = peft.get_peft_model(self._llm, autocast_adapter_dtype=False, peft_config=peft.LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear"))

        # Move the agent to the device.
        self.to(self.device)

        # Create the optimizer for the trainable parameters of the model with the given learning rate.
        self._optimizer = torch.optim.Adam(
            (param for param in self._llm.parameters() if param.requires_grad), lr=args.learning_rate)

    def _tokenize_prompts(self, prompts: list[str]) -> transformers.BatchEncoding:
        """Helper method to tokenize prompts with the chat template and instructions."""
        chats = self._tokenizer.apply_chat_template(
            [[{"role": "system", "content": self._task.instructions},
              {"role": "user", "content": p}] for p in prompts],
            add_generation_prompt=True, tokenize=False,
        )
        return self._tokenizer(chats, padding="longest", padding_side="left", return_tensors="pt").to(self.device)

    @torch.no_grad
    def generate(
        self, prompts: list[str], sample: bool = False, output_probs: bool = False,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor | None]:
        """Generate responses for the given prompts, optionally using sampling and returning token probabilities.

        Returns:
          responses: The generated responses as a list of strings.
          token_ids: The generated tokens as a 2D tensor, with padding tokens replaced by -100.
          probs: The probabilities of the generated `token_ids`, as a tensor of the same shape.
        """
        self._llm.eval()

        tokens = self._tokenize_prompts(prompts)
        results = self._llm.generate(
            input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, max_new_tokens=self._args.max_tokens,
            pad_token_id=self._tokenizer.eos_token_id, forced_eos_token_id=self._tokenizer.eos_token_id,
            do_sample=sample, return_dict_in_generate=True, output_logits=output_probs,
        )

        results.sequences = results.sequences[:, tokens.input_ids.shape[1]:]
        outcomes = self._tokenizer.batch_decode(results.sequences, skip_special_tokens=True)
        if output_probs:
            logits = torch.stack(results.logits, dim=1)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.gather(-1, results.sequences.unsqueeze(-1)).squeeze(-1)
        results.sequences[(results.sequences == self._tokenizer.eos_token_id).cumsum(-1) >= 2] = -100

        return outcomes, results.sequences, probs if output_probs else None

    def token_probabilities(self, prompts: list[str], token_ids: torch.Tensor) -> torch.Tensor:
        """For the given prompts and generated responses in `token_ids`, compute their probabilities.

        The padding tokens are indicated by -100.

        Returns:
          probs: The probabilities of the corresponding tokens, as a tensor of the same shape;
            arbitrary values are returned for the padding tokens.
        """
        tokens = self._tokenize_prompts(prompts)
        input_ids = torch.cat([tokens.input_ids, torch.relu(token_ids[:, :-1])], dim=-1)
        attention_mask = torch.cat([tokens.attention_mask, token_ids[:, :-1] != -100], dim=-1)

        logits = self._llm(input_ids=input_ids, attention_mask=attention_mask).logits

        logits = logits[:, tokens.input_ids.shape[1] - 1:]
        probs = torch.softmax(logits, dim=-1)
        probs = probs.gather(-1, torch.relu(token_ids).unsqueeze(-1)).squeeze(-1)
        return probs

    def train_batch(
        self, prompts: list[str], token_ids: torch.Tensor, advantages: torch.Tensor, old_probs: torch.Tensor,
    ) -> None:
        self._llm.train()

        # TODO: Implement a training step using the GRPO loss, without using the KL penalty
        # (or you can use any other algorithm you find suitable). As in the Dr. GRPO paper,
        # compute the loss for every non-padding token and then average it across all valid
        # tokens in the batch.
        ...

    # Serialization methods.
    def save_lora(self, path: str) -> None:
        torch.save(peft.get_peft_model_state_dict(self._llm), path)

    def load_lora(self, path: str) -> None:
        peft.set_peft_model_state_dict(self._llm, torch.load(path, map_location=self.device))

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_args(path: str) -> argparse.Namespace:
        with open(path, "r", encoding="utf-8-sig") as file:
            args = json.load(file)
        return argparse.Namespace(**args)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Simulate ReCodEx evaluation by running the agent on the dev set and printing the results.
    task = npfl139.llm.Task.from_name(main_args.task)(seed=main_args.seed)
    dev = task.create_dataset(main_args.dev_size)

    llm_agent = LLMAgent(task, main_args) # TODO Remove this, just testing

    # responses = []
    # for i in range(0, len(dev), main_args.batch_size):
    #     responses.extend(llm_agent.generate([example.prompt for example in dev[i:i + main_args.batch_size]])[0])

    # accuracy = task.evaluate(responses, dev)
    # print(f"Evaluation results: {100 * accuracy:.2f}%")

    # for i in range(min(10, len(dev))):
    #     print("Prompt:", dev[i].prompt)
    #     print("Correct:", dev[i].answer)
    #     print("Response:", repr(responses[i]))
    #     print("Extracted:", task.extract_answer(responses[i]))
    #     print()


    train_data = task.create_dataset(3)

    prompts = []
    correct_answers = []

    for example in train_data:
        for _ in range(main_args.train_outcomes):
            prompts.append(example.prompt)
            correct_answers.append(example.answer)

    responses, token_ids, old_probs = llm_agent.generate(
        prompts,
        sample=True,
        output_probs=True,
    )

    print("Number of responses:", len(responses))
    print("token_ids shape:", token_ids.shape)
    print("old_probs shape:", old_probs.shape)

    for i in range(len(responses)):
        extracted = task.extract_answer(responses[i])

        print("=" * 80)
        print("Prompt:", prompts[i])
        print("Correct:", correct_answers[i])
        print("Response:", repr(responses[i]))
        print("Extracted:", extracted)