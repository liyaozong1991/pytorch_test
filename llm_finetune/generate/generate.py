
import argparse
import json, os
import time
import random


generation_config = dict(
    temperature=[1.0],
    #top_k=40,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    #repetition_penalty=1.3,
    max_new_tokens=1536
)


## 需要填充的prompt模版
prompt_template = (
    "有一个女性在陌陌社交平台上发布了以下内容，请给这个动态写10条评论\n输入：\n\"\"\"\n{text}\n\"\"\"\n\n输出:\n"
)
def build_prompt_tpl(text):
    return prompt_template.format_map({'text': text})


def load_input_file(input_file, args):
    example_list = []
    with open(input_file) as fin:
        if input_file.endswith("json"):
            raw_data_list = json.load(fin)
        else:
            raw_data_list = []
            for line in fin:
                raw_data_list.append(json.loads(line))

    raw_data = []
    batch = []
    for i, data in enumerate(raw_data_list):
        input_text = data["input"]

        if args.need_build_prompt is True:
            input_text = build_prompt_tpl(input_text)

        raw_data.append(data)
        batch.append(input_text)

        if (i+1) % args.batch_size == 0:
            example_list.append((raw_data, batch))
            raw_data = []
            batch = []

    # last batch
    if len(batch) > 0:
        example_list.append((raw_data, batch))

    return example_list


def main(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, device_map='auto').half()

    if args.lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

    if device == torch.device('cpu'):
        model.float()

    model.eval()
    with torch.no_grad():

        def run_gen(inputs, gen_cfg):
            new_gen_cfg = { **gen_cfg }
            if isinstance(new_gen_cfg["temperature"], list):
                new_gen_cfg["temperature"] = random.choice(new_gen_cfg["temperature"])

            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                **new_gen_cfg
            )
            return generation_output

        if args.interactive:
            print("Start inference with interactive mode.")

            while True:
                raw_input_text = input("Input:")
                raw_input_text = raw_input_text.strip()
                if len(raw_input_text.strip()) == 0:
                    break

                if args.need_build_prompt:
                    input_text = build_prompt_tpl(raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = run_gen(inputs, generation_config)
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=False)
                response = output
                print("Response:")
                print(response)
                print("\n")

        elif args.input_file is not None:
            example_list = load_input_file(args.input_file, args)

            dirname = os.path.dirname(args.output_file)
            os.makedirs(dirname, exist_ok=True)
            with open(args.output_file + '.gen_cfg', 'w') as f:
                json.dump(generation_config, f, ensure_ascii=False, indent=2)

            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left" # for batch inference
            print("Start inference.")
            results = []
            start = time.time()
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for index, example in enumerate(example_list):
                    raw_data = example[0]
                    batch = example[1]
                    inputs = tokenizer(batch,
                                       padding=True,
                                       return_tensors="pt")  #add_special_tokens=False ?
                    inputs.to(device)
                    generation_output = run_gen(inputs, generation_config)
                    s = generation_output[:, inputs["input_ids"].shape[1]:]
                    output = tokenizer.batch_decode(s, skip_special_tokens=True)
                    response = output

                    for i, res in enumerate(response):
                        out_data = raw_data[i]
                        out_data["res"] = res
                        try:
                            out_str = json.dumps(out_data, ensure_ascii=False)
                            f.write(f"{out_str}\n")
                        except:
                            print(f"write error:{out_str}")

                    if index % 10 == 0:
                        end = time.time()
                        speed = 10 * args.batch_size / (end - start)
                        print(f"batch={index} speed={speed} samp/sec")
                        start = end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
        default=None,
        type=str,
        required=True
    )
    parser.add_argument('--tokenizer_path',
        default=None,
        type=str
    )
    parser.add_argument('--lora_path',
        default=None,
        type=str
    )
    parser.add_argument('--input_file',
        default=None,
        type=str,
        help="A json file that contains prompts '[{\"input\":\"xxx\"}, ...]', or one json per line"
    )
    parser.add_argument('--output_file',
        default=None,
        type=str
    )
    parser.add_argument('--need_build_prompt',
        action='store_true',
        help="wrap the input with the prompt automatically"
    )
    parser.add_argument('--interactive',
        action='store_true',
        help="run in the interactive mode (single-turn)"
    )
    parser.add_argument('--gpus',
        default="0",
        type=str
    )
    parser.add_argument('--only_cpu',
        action='store_true',
        help='only use CPU for inference'
    )
    parser.add_argument('--batch_size',
        default=1,
        type=int
    )

    args = parser.parse_args()

    if args.only_cpu is True:
        args.gpus = ""

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    main(args)


