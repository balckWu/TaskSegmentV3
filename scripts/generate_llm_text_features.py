import os
import re
import json
import argparse
from typing import Dict, List, Tuple

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
)


# ============================================================
# 5 个结构化语义组，必须与模型里的 text_group_summary_count=5 对齐
# ============================================================
GROUP_ORDER = ["anatomy", "appearance", "boundary", "exclusion", "task"]
GROUP_SUMMARY_COUNT = len(GROUP_ORDER)

GROUP2ID = {name: idx for idx, name in enumerate(GROUP_ORDER)}


# ============================================================
# 数据域到医学目标的映射
# 文件名会保存为：
# text_features_thyroid.pt
# text_features_TN3K.pt
# text_features_BUSI_WHU.pt
# text_features_BUS-BRA.pt
# text_features_OTU.pt
# text_features_prostate.pt
# ============================================================
DOMAINS = {
    "thyroid": "thyroid nodule",
    "TN3K": "thyroid nodule",
    "BUSI_WHU": "breast mass",
    "BUS-BRA": "breast mass",
    "OTU": "ovarian tumor",
    "prostate": "prostate gland",
}


# ============================================================
# fallback：如果 LLM 输出不是合法 JSON，就用这些兜底文本
# 避免整个生成流程中断
# ============================================================
FALLBACK_GROUP_TEXTS = {
    "thyroid nodule": {
        "anatomy": "The target is a thyroid nodule located within the thyroid gland on ultrasound.",
        "appearance": "The thyroid nodule may appear hypoechoic, isoechoic, heterogeneous, or with nonuniform internal texture.",
        "boundary": "The nodule boundary may be clear, blurred, regular, irregular, or partially obscured.",
        "exclusion": "Exclude normal thyroid parenchyma, surrounding soft tissue, speckle noise, artifacts, and non-nodule background.",
        "task": "Segment only the thyroid nodule region rather than the whole thyroid gland.",
    },
    "breast mass": {
        "anatomy": "The target is a breast mass or breast lesion visible in ultrasound imaging.",
        "appearance": "The breast mass may show low echogenicity, heterogeneous internal echoes, posterior shadowing, or acoustic enhancement.",
        "boundary": "The lesion margin may be circumscribed, indistinct, angular, microlobulated, or irregular.",
        "exclusion": "Exclude normal breast tissue, fat, glandular tissue, fibrous tissue, speckle noise, and acoustic artifacts.",
        "task": "Segment only the breast mass region and separate it from surrounding normal tissue.",
    },
    "ovarian tumor": {
        "anatomy": "The target is an ovarian tumor or ovarian lesion visible in ultrasound imaging.",
        "appearance": "The ovarian tumor may show cystic, solid, mixed, heterogeneous, or complex internal ultrasound appearance.",
        "boundary": "The tumor boundary may be smooth, irregular, partially blurred, or difficult to distinguish from adjacent tissue.",
        "exclusion": "Exclude normal ovarian tissue, pelvic background, fluid artifacts, speckle noise, and non-target structures.",
        "task": "Segment only the ovarian tumor region in the ultrasound image.",
    },
    "prostate gland": {
        "anatomy": "The target is the prostate gland in ultrasound imaging.",
        "appearance": "The prostate gland may show relatively homogeneous or heterogeneous echotexture with variable internal contrast.",
        "boundary": "The prostate boundary may be smooth, weak, partially shadowed, or locally ambiguous.",
        "exclusion": "Exclude surrounding pelvic tissue, rectal wall, acoustic artifacts, speckle noise, and non-prostate background.",
        "task": "Segment only the prostate gland region in the ultrasound image.",
    },
}


def get_fallback_group_texts(organ: str) -> Dict[str, str]:
    if organ in FALLBACK_GROUP_TEXTS:
        return dict(FALLBACK_GROUP_TEXTS[organ])

    return {
        "anatomy": f"The target is the {organ} in ultrasound imaging.",
        "appearance": f"The {organ} may show characteristic echogenicity, texture, and internal ultrasound appearance.",
        "boundary": f"The {organ} boundary may be clear, blurred, regular, irregular, or partially obscured.",
        "exclusion": "Exclude surrounding normal tissue, artifacts, speckle noise, and non-target background.",
        "task": f"Segment only the target {organ} region in the ultrasound image.",
    }


def build_structured_prompt(organ: str) -> str:
    return f"""
You are an expert radiologist.

For ultrasound segmentation of {organ}, write five short professional descriptions.

Return exactly one valid JSON object in this format:
{{
  "anatomy": "...",
  "appearance": "...",
  "boundary": "...",
  "exclusion": "...",
  "task": "..."
}}

Requirements:
- anatomy: describe the target anatomical structure or lesion identity.
- appearance: describe echogenicity, texture, internal structure, and ultrasound appearance.
- boundary: describe margin, contour, boundary clarity, and shape.
- exclusion: describe surrounding tissue, artifacts, noise, or background that should not be segmented.
- task: describe the exact segmentation target.

Each value must be one concise professional English sentence.
Do not output Markdown.
Do not output explanations.
Only output the JSON object.
""".strip()


def parse_group_json(output_text: str, organ: str) -> Dict[str, str]:
    """
    从 Qwen 输出中解析 JSON。

    Qwen 有时会输出：
    ```json
    {...}
    ```
    或者在 JSON 前后加解释文本，所以这里用正则抽取最外层 JSON。
    """
    fallback = get_fallback_group_texts(organ)

    try:
        text = output_text.strip()

        # 去掉 markdown code fence
        text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is not None:
            text = match.group(0)

        obj = json.loads(text)

        group_texts = {}
        for group_name in GROUP_ORDER:
            value = obj.get(group_name, "")
            value = str(value).strip()

            if len(value) == 0:
                value = fallback[group_name]

            group_texts[group_name] = value

        return group_texts

    except Exception as exc:
        print(f"[警告] LLM 输出 JSON 解析失败，将使用 fallback。错误: {exc}")
        print(f"[警告] 原始输出:\n{output_text}")
        return fallback


def filter_content_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    token_feats: torch.Tensor,
    valid_positions: torch.Tensor,
) -> Tuple[torch.Tensor, List[str]]:
    """
    过滤掉无意义 token，例如纯标点、空 token。
    保留 BioBERT 的子词特征作为 fine tokens。
    """
    token_ids = input_ids[valid_positions].detach().cpu().tolist()
    sub_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    kept_feats = []
    kept_tokens = []

    for feat, tok in zip(token_feats, sub_tokens):
        clean_tok = tok.replace("##", "").strip()

        if clean_tok == "":
            continue

        # 过滤纯标点
        if all(ch in "-_,.;:()[]{}\\/|?!'\"`" for ch in clean_tok):
            continue

        # 过滤太短且无意义的 token
        if len(clean_tok) <= 1 and clean_tok.lower() not in {"t", "c"}:
            continue

        kept_feats.append(feat.unsqueeze(0))
        kept_tokens.append(tok)

    if len(kept_feats) == 0:
        return token_feats.mean(dim=0, keepdim=True), ["fallback_token"]

    return torch.cat(kept_feats, dim=0), kept_tokens


@torch.no_grad()
def encode_one_group_text(
    bert_model,
    bert_tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    编码单个 group 文本。

    返回：
    group_summary: [1, 1024]
    fine_feats:    [N, 1024]
    token_texts:   List[str]
    """
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = bert_model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    # 使用最后 4 层平均，比只用最后一层更稳
    # last4: [1, L, D]
    last4 = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)

    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0].bool()

    valid_positions = attention_mask.nonzero(as_tuple=False).squeeze(1)

    # 去掉 [CLS] 和 [SEP]
    # BioBERT/BERT 格式通常是：
    # [CLS] token token token [SEP]
    if valid_positions.numel() >= 3:
        content_positions = valid_positions[1:-1]
    else:
        content_positions = valid_positions

    raw_token_feats = last4[0, content_positions, :].detach().cpu()

    fine_feats, token_texts = filter_content_tokens(
        tokenizer=bert_tokenizer,
        input_ids=input_ids.detach().cpu(),
        token_feats=raw_token_feats,
        valid_positions=content_positions.detach().cpu(),
    )

    # 当前 group 的 summary：该组 fine tokens 的 mean pooling
    group_summary = fine_feats.mean(dim=0, keepdim=True)

    return group_summary, fine_feats, token_texts


@torch.no_grad()
def encode_structured_group_texts(
    bert_model,
    bert_tokenizer,
    group_texts: Dict[str, str],
    device: torch.device,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    把五组结构化文本编码成模型需要的 text_features。

    输出 text_features 格式：

    [
      anatomy_summary,
      appearance_summary,
      boundary_summary,
      exclusion_summary,
      task_summary,
      fine_tokens...
    ]

    其中前 5 个 token 是真正有区分性的 group summary，
    后面是所有 group 文本的细粒度 BioBERT token features。
    """
    group_summary_tokens = []
    fine_token_banks = []

    token_group_ids = []
    token_texts = []

    group_summary_texts = []

    for group_name in GROUP_ORDER:
        if group_name not in group_texts:
            raise KeyError(f"group_texts 缺少必要字段: {group_name}")

    for group_id, group_name in enumerate(GROUP_ORDER):
        text = group_texts[group_name]
        group_summary_texts.append(text)

        group_summary, fine_feats, group_token_texts = encode_one_group_text(
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer,
            text=text,
            device=device,
            max_length=max_length,
        )

        group_summary_tokens.append(group_summary)
        fine_token_banks.append(fine_feats)

        token_group_ids.append(
            torch.full(
                (fine_feats.shape[0],),
                group_id,
                dtype=torch.long,
            )
        )

        token_texts.extend(group_token_texts)

    # [5, 1024]
    group_summary_tokens = torch.cat(group_summary_tokens, dim=0)

    # [N, 1024]
    fine_feats = torch.cat(fine_token_banks, dim=0)

    # [5 + N, 1024]
    text_features = torch.cat(
        [group_summary_tokens, fine_feats],
        dim=0,
    )

    total_tokens = text_features.shape[0]

    text_mask = torch.ones(total_tokens, dtype=torch.long)

    # 前 5 个 token 是 group summary，后面是 fine token
    token_is_group_summary = torch.cat(
        [
            torch.ones(GROUP_SUMMARY_COUNT, dtype=torch.long),
            torch.zeros(fine_feats.shape[0], dtype=torch.long),
        ],
        dim=0,
    )

    # 前缀 summary 的 group id 是 0..4
    # 后缀 fine token 的 group id 来自对应文本组
    token_group_ids_all = torch.cat(
        [
            torch.arange(GROUP_SUMMARY_COUNT, dtype=torch.long),
            torch.cat(token_group_ids, dim=0),
        ],
        dim=0,
    )

    token_texts_all = [
        f"<{group_name}_summary>" for group_name in GROUP_ORDER
    ] + token_texts

    return {
        # ===== 核心字段：bank.py 实际读取这个 =====
        "text_features": text_features.float(),

        # ===== 向后兼容字段 =====
        "text_mask": text_mask,
        "text_group_summary_count": GROUP_SUMMARY_COUNT,
        "token_is_group_summary": token_is_group_summary,
        "hidden_dim": int(text_features.shape[-1]),

        # ===== 额外分析字段，不影响训练/推理 =====
        "group_names": list(GROUP_ORDER),
        "group_texts": dict(group_texts),
        "group_summary_texts": group_summary_texts,
        "token_group_ids": token_group_ids_all,
        "token_texts": token_texts_all,
        "embedding_type": "llm_structured_group_biobert_v1",
        "group_summary_is_prefix": True,
    }


def generate_group_texts_with_qwen(
    qwen_model,
    qwen_processor,
    domain: str,
    organ: str,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, str]:
    prompt = build_structured_prompt(organ)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text_prompt = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = qwen_processor(
        text=[text_prompt],
        return_tensors="pt",
    )

    # 当前仓库原脚本就是把输入放到 device。
    # 如果使用 device_map="auto"，单卡 4090 场景一般没有问题。
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = qwen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    group_texts = parse_group_json(output_text, organ=organ)

    print(f"\n[{domain}] LLM 结构化文本:")
    for group_name in GROUP_ORDER:
        print(f"  {group_name}: {group_texts[group_name]}")

    return group_texts


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured LLM + BioBERT text features for TaskSegmentV3."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./text_features_llm",
        help="Directory to save generated text feature .pt files.",
    )

    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Qwen2-VL model name or local path.",
    )

    parser.add_argument(
        "--bert-model",
        type=str,
        default="dmis-lab/biobert-large-cased-v1.1",
        help="BioBERT/BERT model name or local path. Must output 1024-dim features for current config.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for Qwen generation.",
    )

    parser.add_argument(
        "--bert-max-length",
        type=int,
        default=128,
        help="Max token length for each structured group text.",
    )

    parser.add_argument(
        "--use-fallback-only",
        action="store_true",
        help="Do not load Qwen. Use predefined fallback structured texts only.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output dir: {args.output_dir}")

    # ============================================================
    # 1. 生成或读取结构化 group 文本
    # ============================================================
    domain_group_texts: Dict[str, Dict[str, str]] = {}

    if args.use_fallback_only:
        print("\n1. 使用 fallback 结构化文本，不加载 Qwen。")

        for domain, organ in DOMAINS.items():
            group_texts = get_fallback_group_texts(organ)
            domain_group_texts[domain] = group_texts

            print(f"\n[{domain}] fallback 结构化文本:")
            for group_name in GROUP_ORDER:
                print(f"  {group_name}: {group_texts[group_name]}")

    else:
        print("\n1. 正在加载 Qwen2-VL，用于生成结构化专家级超声文本先验...")

        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        qwen_model.eval()

        qwen_processor = AutoProcessor.from_pretrained(args.qwen_model)

        for domain, organ in DOMAINS.items():
            print(f"\n正在让 LLM 生成 [{domain}] 的结构化超声文本先验...")
            group_texts = generate_group_texts_with_qwen(
                qwen_model=qwen_model,
                qwen_processor=qwen_processor,
                domain=domain,
                organ=organ,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            domain_group_texts[domain] = group_texts

        # 释放 Qwen 显存，避免和 BioBERT 同时占显存
        print("\n释放 Qwen 显存...")
        del qwen_model
        del qwen_processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 2. 加载 BioBERT-large
    # ============================================================
    print("\n2. 正在加载 BioBERT-large，将结构化文本编码为 1024 维特征...")

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    bert_model = AutoModel.from_pretrained(
        args.bert_model,
        use_safetensors=True,
    ).to(device)

    bert_model.eval()

    hidden_size = int(getattr(bert_model.config, "hidden_size", 0))
    print(f"BioBERT hidden size: {hidden_size}")

    if hidden_size != 1024:
        print(
            f"[警告] 当前模型 hidden_size={hidden_size}，"
            f"但 TaskSegmentV3 默认 text_dim=1024。"
            f"如果继续使用当前模型，请确保 train.py 里 --text-dim 或 config 对应修改。"
        )

    # ============================================================
    # 3. 编码并保存 .pt
    # ============================================================
    print("\n3. 正在编码并保存结构化密集语义特征 .pt 文件...")

    for domain, group_texts in domain_group_texts.items():
        save_dict = encode_structured_group_texts(
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer,
            group_texts=group_texts,
            device=device,
            max_length=args.bert_max_length,
        )

        save_path = os.path.join(args.output_dir, f"text_features_{domain}.pt")

        torch.save(save_dict, save_path)

        text_features = save_dict["text_features"]

        print(
            f"已保存 [{domain}] -> {save_path} | "
            f"text_features shape: {tuple(text_features.shape)} | "
            f"group_summary_count: {save_dict['text_group_summary_count']}"
        )

    print("\n✅ 结构化 LLM + BioBERT 文本特征生成完毕！")
    print("✅ 前 5 个 token 现在分别对应 anatomy / appearance / boundary / exclusion / task。")
    print("✅ 后续 train.py / predict.py 继续使用 --text-dir 指向该输出目录即可。")


if __name__ == "__main__":
    main()