"""
Flux1.devモデルをFP8形式に量子化するスクリプト (HSWQ V1.2: Flux Edition)
SDXL HSWQ V1.21 (GPU Accelerated) と完全に同一の構造・アルゴリズムで実装。

アルゴリズム: (SDXL V1.21と同一)
1. Load Pipeline + comfyui_to_diffusers_map 構築
2. Calibration Loop (DualMonitor: Sensitivity & Input Importance)
3. Layer Selection (Keep Top N% Sensitive Layers)
4. HSWQ Optimization (Weighted Histogram MSE, scaled=False)
5. GPU Accelerated Quantization (comfyui_to_diffusers_map逆引きで変換)
"""

import argparse
import torch
from diffusers import FluxPipeline, AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel, AutoTokenizer, CLIPTextConfig, T5Config
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import numpy as np

# HSWQ専用モジュールをインポート
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

# C++20標準を強制
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# === SageAttention2 Integration (SDXL V1.21と同一) ===
_sage_attn_available = False
_original_sdpa = None

def try_import_sage_attention():
    global _sage_attn_available
    try:
        from sageattention import sageattn
        _sage_attn_available = True
        print("[SageAttention2] インポート成功")
        return True
    except ImportError:
        print("[SageAttention2] 未インストール。標準Attentionで実行します。")
        return False

def enable_sage_attention():
    global _original_sdpa
    if not _sage_attn_available:
        print("[SageAttention2] 有効化不可 - 利用不可")
        return False
    
    import torch.nn.functional as F
    from sageattention import sageattn
    
    _original_sdpa = F.scaled_dot_product_attention
    
    def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if attn_mask is not None or is_causal:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        try:
            return sageattn(query, key, value, is_causal=False)
        except Exception as e:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
    
    F.scaled_dot_product_attention = sage_sdpa_wrapper
    print("[SageAttention2] 有効化完了 (SDPA monkey-patch)")
    return True

def disable_sage_attention():
    global _original_sdpa
    if _original_sdpa is not None:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
        print("[SageAttention2] 無効化完了（元のSDPAに復元）")


# --- ComfyUI互換のマッピング関数群 (SDXL V1.21の unet_to_diffusers_mapping に相当) ---

def flux_to_diffusers_mapping(state_dict, key_prefix=None):
    """Flux ComfyUIキー → Diffusersキー のマッピングを構築。
    SDXL V1.21の unet_to_diffusers_mapping と同じ役割。
    
    key_prefix: None の場合は自動検出。
    
    戻り値: comfyui_to_diffusers_map = {comfy_full_key: diffusers_key}
    例: {"model.diffusion_model.double_blocks.0.img_mlp.0.weight": "transformer_blocks.0.ff.net.0.proj.weight"}
    """
    state_dict_keys = list(state_dict.keys())
    
    # プレフィックスの自動検出
    if key_prefix is None:
        has_full_prefix = any(k.startswith("model.diffusion_model.") for k in state_dict_keys)
        has_no_prefix = any(k.startswith("double_blocks.") or k.startswith("single_blocks.") for k in state_dict_keys)
        
        if has_full_prefix:
            key_prefix = "model.diffusion_model."
        elif has_no_prefix:
            key_prefix = ""
        else:
            key_prefix = "model.diffusion_model."
        
        print(f"  プレフィックス自動検出: '{key_prefix}' (prefix付き={has_full_prefix}, prefix無し={has_no_prefix})")
    
    # ブロック数を自動検出
    num_double = 0
    num_single = 0
    for k in state_dict_keys:
        if k.startswith(key_prefix):
            stripped = k[len(key_prefix):]
            if stripped.startswith("double_blocks."):
                parts = stripped.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    num_double = max(num_double, int(parts[1]) + 1)
            elif stripped.startswith("single_blocks."):
                parts = stripped.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    num_single = max(num_single, int(parts[1]) + 1)
    
    print(f"  検出: Double Blocks={num_double}, Single Blocks={num_single}")
    
    comfyui_to_diffusers_map = {}
    
    # --- Embeddings ---
    embed_map = {
        "txt_in": "context_embedder",
        "img_in": "x_embedder",
        "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "final_layer.linear": "proj_out",
    }
    for suffix in [".weight", ".bias"]:
        for comfy_name, diff_name in embed_map.items():
            comfy_key = f"{key_prefix}{comfy_name}{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
    
    # --- Double Blocks ---
    for i in range(num_double):
        # 1:1マッピング（非融合レイヤー）
        one_to_one = {
            f"double_blocks.{i}.img_mod.lin": f"transformer_blocks.{i}.norm1.linear",
            f"double_blocks.{i}.txt_mod.lin": f"transformer_blocks.{i}.norm1_context.linear",
            f"double_blocks.{i}.img_attn.proj": f"transformer_blocks.{i}.attn.to_out.0",
            f"double_blocks.{i}.txt_attn.proj": f"transformer_blocks.{i}.attn.to_add_out",
            f"double_blocks.{i}.img_mlp.0": f"transformer_blocks.{i}.ff.net.0.proj",
            f"double_blocks.{i}.img_mlp.2": f"transformer_blocks.{i}.ff.net.2",
            f"double_blocks.{i}.txt_mlp.0": f"transformer_blocks.{i}.ff_context.net.0.proj",
            f"double_blocks.{i}.txt_mlp.2": f"transformer_blocks.{i}.ff_context.net.2",
        }
        for suffix in [".weight", ".bias"]:
            for comfy_name, diff_name in one_to_one.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
        
        # 融合QKVレイヤー: ComfyUI側は1つのテンソル、Diffusers側は分割
        # → マッピング先を "FUSED:xxx" としてマーク（量子化時にDiffusersの分割モジュールではなく直接処理）
        for suffix in [".weight", ".bias"]:
            # img_attn.qkv → to_q + to_k + to_v (融合)
            comfy_key = f"{key_prefix}double_blocks.{i}.img_attn.qkv{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"FUSED:transformer_blocks.{i}.attn.img_qkv{suffix}"
            # txt_attn.qkv → add_q_proj + add_k_proj + add_v_proj (融合)
            comfy_key = f"{key_prefix}double_blocks.{i}.txt_attn.qkv{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"FUSED:transformer_blocks.{i}.attn.txt_qkv{suffix}"
        
        # Norm scales (RMSNorm)
        for suffix in [".scale"]:
            norm_map = {
                f"double_blocks.{i}.img_attn.norm.query_norm": f"transformer_blocks.{i}.attn.norm_q",
                f"double_blocks.{i}.img_attn.norm.key_norm": f"transformer_blocks.{i}.attn.norm_k",
                f"double_blocks.{i}.txt_attn.norm.query_norm": f"transformer_blocks.{i}.attn.norm_added_q",
                f"double_blocks.{i}.txt_attn.norm.key_norm": f"transformer_blocks.{i}.attn.norm_added_k",
            }
            for comfy_name, diff_name in norm_map.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}.weight"
    
    # --- Single Blocks ---
    for i in range(num_single):
        # 1:1マッピング
        one_to_one = {
            f"single_blocks.{i}.modulation.lin": f"single_transformer_blocks.{i}.norm.linear",
            f"single_blocks.{i}.linear2": f"single_transformer_blocks.{i}.proj_out",
        }
        for suffix in [".weight", ".bias"]:
            for comfy_name, diff_name in one_to_one.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
        
        # 融合Linear1: to_q + to_k + to_v + proj_mlp (4つが融合)
        for suffix in [".weight", ".bias"]:
            comfy_key = f"{key_prefix}single_blocks.{i}.linear1{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"FUSED:single_transformer_blocks.{i}.linear1{suffix}"
        
        # Norm scales
        for suffix in [".scale"]:
            norm_map = {
                f"single_blocks.{i}.norm.query_norm": f"single_transformer_blocks.{i}.attn.norm_q",
                f"single_blocks.{i}.norm.key_norm": f"single_transformer_blocks.{i}.attn.norm_k",
            }
            for comfy_name, diff_name in norm_map.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}.weight"
    
    return comfyui_to_diffusers_map


def load_flux_pipeline_from_safetensors(path, device="cuda", token=None, clip_path=None, t5_path=None, vae_path=None):
    """Fluxパイプラインをロード。SDXL V1.21の load_unet_from_safetensors と同じ役割。
    
    戻り値:
        pipeline: FluxPipeline
        original_state_dict: 元のsafetensorsのstate_dict
        comfyui_to_diffusers_map: ComfyUIキー→Diffusersキーのマッピング
    """
    print(f"Flux1モデルをロード中: {path}")
    
    # 元のstate_dictをロード（SDXL V1.21と同様に保持）
    original_state_dict = load_file(path)
    
    # キーマッピングを構築
    print("キーマッピングを作成中...")
    comfyui_to_diffusers_map = flux_to_diffusers_mapping(original_state_dict)
    print(f"  マッピング数: {len(comfyui_to_diffusers_map)}")
    
    # 外部コンポーネントのロード
    text_encoder = None
    text_encoder_2 = None
    tokenizer = None
    tokenizer_2 = None
    vae = None
    
    # ヘルパー: ファイルまたはディレクトリからモデルをロード
    def load_external_component(path, model_class, config_class_or_repo, default_repo=None, tokenizer_repo=None, is_diffusers=False, token=None):
        model = None
        tok = None
        
        if os.path.isfile(path):
            print(f"単一ファイルからロード中: {path}")
            try:
                if is_diffusers:
                    print(f"デフォルト設定を取得中: {config_class_or_repo}")
                    model = model_class.from_pretrained(config_class_or_repo, subfolder="vae", token=token)
                    sd = load_file(path)
                    m, u = model.load_state_dict(sd, strict=False)
                    print(f"重みロード完了。Missing: {len(m)}, Unexpected: {len(u)}")
                    model.to(torch.float16)
                else:
                    print(f"デフォルト設定を取得中: {default_repo}")
                    config = config_class_or_repo.from_pretrained(default_repo, token=token)
                    model = model_class(config)
                    sd = load_file(path)
                    m, u = model.load_state_dict(sd, strict=False)
                    print(f"重みロード完了。Missing: {len(m)}, Unexpected: {len(u)}")
                    model.to(torch.float16)
            except Exception as e:
                print(f"単一ファイルロード失敗: {e}")
                sys.exit(1)
            
            if tokenizer_repo:
                print(f"トークナイザーを取得中: {tokenizer_repo}")
                tok = AutoTokenizer.from_pretrained(tokenizer_repo, token=token)
                
        else:
            print(f"ディレクトリからロード中: {path}")
            try:
                if is_diffusers:
                     model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                else:
                    model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                    tok = AutoTokenizer.from_pretrained(path, token=token)
            except Exception as e:
                print(f"ディレクトリロードエラー: {e}")
                sys.exit(1)
                
        return model, tok

    # 外部CLIPロード
    if clip_path:
        text_encoder, tokenizer = load_external_component(
            clip_path, CLIPTextModel, CLIPTextConfig, "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", token=token
        )
        print("CLIPロード完了。")

    # 外部T5ロード
    if t5_path:
        text_encoder_2, tokenizer_2 = load_external_component(
            t5_path, T5EncoderModel, T5Config, "google/t5-v1_1-xxl", "google/t5-v1_1-xxl", token=token
        )
        print("T5ロード完了。")

    # 外部VAEロード
    if vae_path:
        vae, _ = load_external_component(
            vae_path, AutoencoderKL, "black-forest-labs/FLUX.1-schnell", is_diffusers=True, token=token
        )
        print("VAEロード完了。")

    # FluxPipelineをロード
    # FluxPipelineをロード
    print("FluxPipelineを構築中（手動ウェイトロード）...")
    pipeline = None
    try:
        # まずConfigのみで空のモデルを初期化
        # Configロード用のパス決定
        config_path = "black-forest-labs/FLUX.1-schnell"
        if is_diffusers:
             config_path = config_class_or_repo
        
        pipeline = FluxPipeline.from_pretrained(
             config_path,
             torch_dtype=torch.float16,
             token=token,
             text_encoder=text_encoder,
             text_encoder_2=text_encoder_2,
             tokenizer=tokenizer,
             tokenizer_2=tokenizer_2,
             vae=vae
        )

        # ターゲットのstate_dictを作成 (Diffusers形式)
        converted_state_dict = {}
        
        # 既存のマッピングを利用して単純な1:1転送と、Fused Layerの分割転送を行う
        print("    ウェイトをDiffusers形式に変換・分割中...")
        
        # 1. 1:1 マッピングの適用
        # comfyui_to_diffusers_map には "FUSED:" 以外の通常マッピングも含まれている
        for comfy_key, val in original_state_dict.items():
            if comfy_key in comfyui_to_diffusers_map:
                diff_key = comfyui_to_diffusers_map[comfy_key]
                if not diff_key.startswith("FUSED:"):
                     converted_state_dict[diff_key] = val
        
        # 2. Fused Layer の手動分割
        # Double Blocks: img_attn.qkv -> to_q, to_k, to_v
        # Double Blocks: txt_attn.qkv -> add_q_proj, add_k_proj, add_v_proj
        # Single Blocks: linear1 -> to_q, to_k, to_v, proj_mlp
        
        for key, value in original_state_dict.items():
            # Double Block Fused QKV (img)
            if "img_attn.qkv" in key and "double_blocks" in key:
                parts = key.split(".")
                # double_blocks.{i}.img_attn.qkv.{weight/bias}
                if len(parts) >= 5 and parts[0] == "double_blocks" and parts[2] == "img_attn" and parts[3] == "qkv":
                    idx = parts[1]
                    suffix = parts[4] # weight or bias
                    
                    # Split (3072, 3072, 3072)
                    q, k, v = torch.split(value, 3072, dim=0)
                    
                    base = f"transformer.transformer_blocks.{idx}.attn"
                    converted_state_dict[f"{base}.to_q.{suffix}"] = q
                    converted_state_dict[f"{base}.to_k.{suffix}"] = k
                    converted_state_dict[f"{base}.to_v.{suffix}"] = v
            
            # Double Block Fused QKV (txt)
            elif "txt_attn.qkv" in key and "double_blocks" in key:
                parts = key.split(".")
                if len(parts) >= 5 and parts[0] == "double_blocks" and parts[2] == "txt_attn" and parts[3] == "qkv":
                    idx = parts[1]
                    suffix = parts[4]
                    
                    q, k, v = torch.split(value, 3072, dim=0)
                    
                    base = f"transformer.transformer_blocks.{idx}.attn"
                    converted_state_dict[f"{base}.add_q_proj.{suffix}"] = q
                    converted_state_dict[f"{base}.add_k_proj.{suffix}"] = k
                    converted_state_dict[f"{base}.add_v_proj.{suffix}"] = v

            # Single Block Fused Linear1
            elif "linear1" in key and "single_blocks" in key:
                parts = key.split(".")
                # single_blocks.{i}.linear1.{weight/bias}
                if len(parts) >= 4 and parts[0] == "single_blocks" and parts[2] == "linear1":
                    idx = parts[1]
                    suffix = parts[3] 
                    
                    # shape is [21504, 3072] -> Split 3072, 3072, 3072, 12288
                    # Order: q, k, v, mlp
                    q, k, v, mlp = torch.split(value, [3072, 3072, 3072, 12288], dim=0)
                    
                    base = f"transformer.single_transformer_blocks.{idx}"
                    converted_state_dict[f"{base}.attn.to_q.{suffix}"] = q
                    converted_state_dict[f"{base}.attn.to_k.{suffix}"] = k
                    converted_state_dict[f"{base}.attn.to_v.{suffix}"] = v
                    converted_state_dict[f"{base}.proj_mlp.{suffix}"] = mlp

        # モデルにロード
        print(f"変換済みState Dictサイズ: {len(converted_state_dict)}")
        m, u = pipeline.load_state_dict(converted_state_dict, strict=False)
        print(f"手動ロード結果 - Missing: {len(m)}, Unexpected: {len(u)}")
        
        # 簡易チェック
        if len(m) > 100: # あまりに多い場合は警告
            print(f"警告: Missing keysが多いです ({len(m)}個)。正しくロードできていない可能性があります。")
            # print(m[:5])

    except Exception as e:
        print(f"手動ロード失敗: {e}")
        print("from_single_fileへのフォールバックを試みます...")
        try:
            pipeline = FluxPipeline.from_single_file(
                path, 
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=token,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=vae
            )
        except Exception as e2:
            print(f"フォールバックも失敗: {e2}")
            sys.exit(1)
            
    print("VRAM最適化: Model CPU Offload を有効化します...")
    pipeline.enable_model_cpu_offload()
    
    # SDXL V1.21と同じ: pipeline, original_state_dict, comfyui_to_diffusers_map を返す
    return pipeline, original_state_dict, comfyui_to_diffusers_map


# --- Dual Monitor: Sensitivity & Importance (SDXL V1.21と同一) ---
class DualMonitor:
    def __init__(self):
        # Sensitivity (出力分散)
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        # Importance (入力活性化)
        self.channel_importance = None
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            # 1. Sensitivity更新 (出力分散)
            out_detached = output_tensor.detach().float()
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()
            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean
            
            # 2. Importance更新 (入力活性化)
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4:    # Conv2d: (B, C, H, W)
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:  # Transformer: (B, T, C)
                current_imp = inp_detached.abs().mean(dim=(0, 1))
            elif inp_detached.dim() == 2:  # Linear: (B, C)
                current_imp = inp_detached.abs().mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
                
            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0: return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        return sq_mean - mean ** 2

dual_monitors = {}

def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    inp = input[0]
    out = output
    dual_monitors[name].update(inp, out)


def main():
    parser = argparse.ArgumentParser(description="Flux1.dev FP8量子化 (HSWQ V1.2: SDXL V1.21同一構造)")
    parser.add_argument("--input", type=str, required=True, help="入力safetensorsモデルのパス")
    parser.add_argument("--output", type=str, required=True, help="出力safetensorsモデルのパス")
    parser.add_argument("--calib_file", type=str, required=True, help="キャリブレーションプロンプトテキストファイルのパス")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="キャリブレーションサンプル数 (HSWQ推奨: 256)")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="推論ステップ数 (HSWQ正規: 25)")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="FP16保持レイヤーの割合")
    parser.add_argument("--sa2", action="store_true", help="SageAttention2で高速化")
    parser.add_argument("--token", type=str, help="HuggingFaceトークン（ゲートモデル用）")
    parser.add_argument("--clip_path", type=str, help="CLIPテキストエンコーダのパス")
    parser.add_argument("--t5_path", type=str, help="T5テキストエンコーダのパス")
    parser.add_argument("--vae_path", type=str, help="VAEモデルのパス")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    # === SageAttention2初期化 (SDXL V1.21と同一) ===
    if args.sa2:
        if try_import_sage_attention():
            enable_sage_attention()
        else:
            print("[警告] --sa2指定ですがSageAttention2が利用不可。標準Attentionで続行します。")

    # === SDXL V1.21と同一: pipeline, original_state_dict, comfyui_to_diffusers_map を取得 ===
    pipeline, original_state_dict, comfyui_to_diffusers_map = load_flux_pipeline_from_safetensors(
        args.input, device, token=args.token, clip_path=args.clip_path, t5_path=args.t5_path, vae_path=args.vae_path
    )

    # === キャリブレーション (SDXL V1.21と同一構造) ===
    print("キャリブレーション準備中（Dual Monitorフック登録）...")
    handles = []
    target_modules = []
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            handles.append(handle)
            target_modules.append(name)

    print("キャリブレーションデータを準備中...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[:args.num_calib_samples]
    else:
        prompts = prompts[:args.num_calib_samples]

    print(f"キャリブレーションを実行中（{args.num_calib_samples}サンプル, {args.num_inference_steps}ステップ）...")
    print("※ 感度(Sensitivity)と入力重要度(Importance)を同時計測します...")
    
    pipeline.set_progress_bar_config(disable=False)
    
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, guidance_scale=3.5, output_type="latent")
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # フック解除
    for h in handles: h.remove()
    
    # === SageAttention2 Cleanup ===
    if args.sa2:
        disable_sage_attention()

    # === レイヤー感度分析 (SDXL V1.21と同一) ===
    print("\nレイヤー感度分析を実行中...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
    
    # 感度順にソート（降順）
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    # 上位N%を特定
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
    
    print(f"総レイヤー数: {len(layer_sensitivities)}")
    print(f"FP16保持レイヤー数: {len(keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
    print("Top 5 Sensitive Layers:")
    for i in range(min(5, len(layer_sensitivities))):
        print(f"  {i+1}. {layer_sensitivities[i][0]}: {layer_sensitivities[i][1]:.4f}")

    # === HSWQ最適化 (SDXL V1.21と同一: named_modules → amax計算) ===
    print("\n[HSWQ] 完全重み付けMSE解析と量子化パラメータ計算を開始します...")
    print("※ 互換モード (scaled=False): エラーを最小化する最適なクリッピング閾値を探索...")
    weight_amax_dict = {}
    
    # HSWQ専用最適化器を初期化（SDXL V1.21と同一パラメータ: bins=4096, 200候補, 3回精錬）
    hswq_optimizer = HSWQWeightedHistogramOptimizer(
        bins=4096,
        num_candidates=200,
        refinement_iterations=3,
        device=device
    )
    
    for name, module in tqdm(pipeline.transformer.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # FP16保持レイヤーならスキップ（amax計算不要）
            if name in keep_layers:
                continue
                
            # 重要度の取得
            importance = None
            if name in dual_monitors:
                importance = dual_monitors[name].channel_importance
            
            # HSWQ: 最適amaxを探索 (scaled=False)
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance,
                scaled=False  # 重要: 互換モード
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            
            torch.cuda.empty_cache()

    first_fused_module_name = {}
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            # QKVが分割されている場合、代表的なモジュール名（最初のQなど）を保存しておく
            # transformer_blocks.{i}.attn.to_q
            # transformer_blocks.{i}.attn.add_q_proj
            # single_transformer_blocks.{i}.attn.to_q
            parts = name.split(".")
            if len(parts) >= 2 and parts[-1].startswith("to_q"):
                # transformer_blocks.{i}.attn.to_q -> {i}.attn.img_qkv
                # single_transformer_blocks.{i}.attn.to_q -> {i}.linear1 (Singleはlinear1=QKV+MLP)
                if parts[0] == "transformer_blocks":
                    block_idx = parts[1]
                    key = f"transformer_blocks.{block_idx}.attn.img_qkv"
                    first_fused_module_name[key] = name
                elif parts[0] == "single_transformer_blocks":
                    block_idx = parts[1]
                    key = f"single_transformer_blocks.{block_idx}.linear1"
                    first_fused_module_name[key] = name
            
            elif len(parts) >= 2 and parts[-1] == "add_q_proj":
                 # transformer_blocks.{i}.attn.add_q_proj -> {i}.attn.txt_qkv
                if parts[0] == "transformer_blocks":
                    block_idx = parts[1]
                    key = f"transformer_blocks.{block_idx}.attn.txt_qkv"
                    first_fused_module_name[key] = name

    # === Flux固有: 融合QKVレイヤーのamax計算 ===
    # ComfyUI形式では img_attn.qkv / txt_attn.qkv / linear1 が融合テンソル
    # Diffusersでは分割されるため named_modules には存在しない
    # → original_state_dict から直接amaxを計算するが、ImportanceはDiffusersモジュールから借りる
    fused_count = 0
    for key, value in original_state_dict.items():
        if key in comfyui_to_diffusers_map:
            diffusers_key = comfyui_to_diffusers_map[key]
            if diffusers_key.startswith("FUSED:") and diffusers_key.endswith(".weight"):
                # "FUSED:transformer_blocks.0.attn.img_qkv.weight"
                target_base = diffusers_key[6:-7] # remove FUSED: and .weight

                # Importance (入力活性化) を取得
                importance = None
                
                # target_baseに対応する代表的なDiffusersモジュールを探す
                # 例: transformer_blocks.0.attn.img_qkv -> transformer_blocks.0.attn.to_q
                if target_base in first_fused_module_name:
                    rep_name = first_fused_module_name[target_base]
                    if rep_name in dual_monitors:
                        importance = dual_monitors[rep_name].channel_importance
                        # print(f"  [FUSED Importance] {target_base} -> used {rep_name}")

                if value.dim() == 2 and value.numel() >= 1024:
                    optimal_amax = hswq_optimizer.compute_optimal_amax(
                        value, importance, scaled=False
                    )
                    weight_amax_dict[diffusers_key] = optimal_amax
                    fused_count += 1
    
    print(f"量子化対象レイヤー数: {len(weight_amax_dict)} (うち融合QKV: {fused_count})")
    
    # === VRAM最適化 (SDXL V1.21と同一) ===
    print("\n[VRAM最適化] GPU高速変換の準備中...")
    del pipeline
    del hswq_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # === 量子化変換 (SDXL V1.21と完全に同一の構造) ===
    # comfyui_to_diffusers_map を逆引きして変換する
    print(f"量子化モデルを保存中: {args.output}")
    output_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    print("重みを変換中 (GPU Accelerated)...")
    for key, value in tqdm(original_state_dict.items(), desc="Converting"):
        # SDXL V1.21と同一: comfyui_to_diffusers_mapでDiffusersキーを取得
        diffusers_key = None
        if key in comfyui_to_diffusers_map:
            diffusers_key = comfyui_to_diffusers_map[key]
        
        # diffusers_keyからモジュール名を特定（.weightを除く）
        module_name = None
        if diffusers_key:
            if diffusers_key.endswith(".weight"):
                module_name = diffusers_key[:-7]
            
        # 変換判定 (SDXL V1.21と同一ロジック)
        if module_name and module_name in keep_layers:
            # FP16保持
            new_value = value
            kept_count += 1
        elif diffusers_key:
            # 量子化対象
            weight_key = diffusers_key + ".weight"
            if diffusers_key.endswith(".weight"):
                weight_key = diffusers_key
            
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                if amax == 0: amax = 1e-6
                # GPU上でClamp→FP8変換
                val_gpu = value.float().to(device)
                clamped_value = torch.clamp(val_gpu, -amax, amax)
                new_value = clamped_value.to(torch.float8_e4m3fn).cpu()
                converted_count += 1
                del val_gpu, clamped_value
            else:
                new_value = value
        else:
            new_value = value
            
        output_state_dict[key] = new_value

    print(f"変換完了:")
    print(f"  FP8化されたレイヤー: {converted_count}")
    print(f"  FP16保持されたレイヤー: {kept_count}")
    
    save_file(output_state_dict, args.output)
    print("保存完了！")

if __name__ == "__main__":
    main()
