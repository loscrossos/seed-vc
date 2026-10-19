import os
import sys
from typing import Optional, Tuple, Union

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

DEFAULT_CHECKPOINT_DIR = "checkpoints"
_CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.environ.get("SEED_VC_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))


def get_checkpoints_dir() -> str:
    """Return the currently configured base checkpoints directory."""
    return _CHECKPOINT_DIR


def set_checkpoints_dir(path: str) -> None:
    """Set the base checkpoints directory and update related environment variables."""
    global _CHECKPOINT_DIR
    _CHECKPOINT_DIR = path
    os.environ["CHECKPOINT_DIR"] = path
    os.environ["SEED_VC_CHECKPOINT_DIR"] = path
    os.environ["HF_HUB_CACHE"] = os.path.join(path, "hf_cache")
    os.makedirs(path, exist_ok=True)


def parse_repo_and_filename(repo_id: str, filename: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Parse a Hugging Face repo ID or URL and filename.
    Returns:
        (clean_repo_id, reponame, effective_filename)

    Examples:
        - "Plachta/Seed-VC", "v2/cfm_small.pth" -> ("Plachta/Seed-VC", "Seed-VC", "v2/cfm_small.pth")
        - "huggingface.com/author/reponame/subdir1/mymodel.gguf" -> ("author/reponame", "reponame", "subdir1/mymodel.gguf")
        - "funasr/campplus", "campplus_cn_common.bin" -> ("funasr/campplus", "campplus_cn_common.bin")
    """
    s = repo_id.strip()
    for prefix in ["https://", "http://", "huggingface.co/", "huggingface.com/"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.startswith("huggingface.co/"):
        s = s[len("huggingface.co/"):]
    if s.startswith("huggingface.com/"):
        s = s[len("huggingface.com/"):]
    s = s.strip("/")

    parts = s.split("/")
    clean_repo_id = s
    reponame = parts[-1]
    extracted_filename = None

    if len(parts) >= 2:
        author = parts[0]
        repo = parts[1]
        clean_repo_id = f"{author}/{repo}"
        reponame = repo
        if len(parts) > 2:
            rest = parts[2:]
            if len(rest) >= 2 and rest[0] in ("resolve", "blob", "raw"):
                extracted_filename = "/".join(rest[2:])
            else:
                extracted_filename = "/".join(rest)
    else:
        clean_repo_id = s
        reponame = s

    effective_filename = (
        filename if (filename is not None and filename != "") else (extracted_filename or "pytorch_model.bin")
    )
    return clean_repo_id, reponame, effective_filename


def extract_repo_name(repo_id: str) -> str:
    """Extract repository name from repo ID or URL."""
    _, reponame, _ = parse_repo_and_filename(repo_id)
    return reponame


def resolve_model_path(repo_id: str, filename: str, base_dir: Optional[str] = None) -> Optional[str]:
    """
    Check if a model file exists locally under the base checkpoints directory.
    Checks:
        1. base_dir / reponame / filename (per user requirement)
        2. base_dir / clean_repo_id / filename
        3. base_dir / filename
        4. configs/presets / configs for configuration files
    """
    if not filename:
        return None
    base_dir = base_dir or get_checkpoints_dir()
    clean_repo_id, reponame, _ = parse_repo_and_filename(repo_id, filename)

    candidates = [
        os.path.join(base_dir, reponame, filename),
        os.path.join(base_dir, clean_repo_id, filename),
        os.path.join(base_dir, filename),
    ]

    # Config files may already be in configs/ or configs/presets/
    if filename.endswith((".yml", ".yaml")):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(root_dir, "configs", "presets", os.path.basename(filename)))
        candidates.append(os.path.join(root_dir, "configs", os.path.basename(filename)))
        candidates.append(os.path.join(root_dir, filename))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def resolve_pretrained_path(model_id_or_repo: str, base_dir: Optional[str] = None) -> str:
    """
    Check if a pretrained model directory exists locally under the base checkpoints directory.
    If found, return the local directory path; otherwise return the original repo/model identifier.
    Used for models loaded via .from_pretrained (e.g. BigVGAN, Whisper, HuBERT, Wav2Vec2).
    """
    if not model_id_or_repo or not isinstance(model_id_or_repo, str):
        return model_id_or_repo
    if os.path.isdir(model_id_or_repo) or os.path.isfile(model_id_or_repo):
        return model_id_or_repo

    base_dir = base_dir or get_checkpoints_dir()
    clean_repo_id, reponame, _ = parse_repo_and_filename(model_id_or_repo)

    candidates = [
        os.path.join(base_dir, reponame),
        os.path.join(base_dir, clean_repo_id),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    return model_id_or_repo


def load_custom_model_from_hf(
    repo_id: str,
    model_filename: str = "pytorch_model.bin",
    config_filename: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Union[str, Tuple[str, str]]:
    """
    Load model and optional config from local checkpoints directory if present,
    otherwise download from Hugging Face into the designated checkpoints directory.
    """
    base_dir = base_dir or get_checkpoints_dir()
    os.makedirs(base_dir, exist_ok=True)

    clean_repo_id, reponame, effective_model_filename = parse_repo_and_filename(repo_id, model_filename)

    # 1. Check if model exists locally
    local_model_path = resolve_model_path(clean_repo_id, effective_model_filename, base_dir=base_dir)
    if local_model_path is not None:
        print(f"Loaded local checkpoint: {local_model_path}")
        model_path = local_model_path
    else:
        if hf_hub_download is None:
            raise ImportError(
                f"Model {effective_model_filename} not found locally at {os.path.join(base_dir, reponame, effective_model_filename)} "
                "and 'huggingface_hub' is not installed to download it. Please install huggingface-hub or provide the local model."
            )
        target_dir = os.path.join(base_dir, reponame)
        os.makedirs(target_dir, exist_ok=True)
        expected_dest = os.path.join(target_dir, effective_model_filename)
        print(
            f"Checkpoint not found locally at {expected_dest}. "
            f"Downloading from Hugging Face ({clean_repo_id}/{effective_model_filename})..."
        )
        try:
            model_path = hf_hub_download(
                repo_id=clean_repo_id,
                filename=effective_model_filename,
                local_dir=target_dir,
            )
        except (TypeError, Exception) as e:
            print(f"Notice: local_dir download encountered: {e}. Falling back to default cache...")
            model_path = hf_hub_download(
                repo_id=clean_repo_id,
                filename=effective_model_filename,
                cache_dir=os.path.join(base_dir, "hf_cache"),
            )

    if config_filename is None:
        return model_path

    # 2. Check if config exists locally
    local_config_path = resolve_model_path(clean_repo_id, config_filename, base_dir=base_dir)
    if local_config_path is not None:
        print(f"Loaded local config: {local_config_path}")
        config_path = local_config_path
    else:
        if hf_hub_download is None:
            raise ImportError(
                f"Config {config_filename} not found locally at {os.path.join(base_dir, reponame, config_filename)} "
                "and 'huggingface_hub' is not installed to download it."
            )
        target_dir = os.path.join(base_dir, reponame)
        os.makedirs(target_dir, exist_ok=True)
        expected_cfg_dest = os.path.join(target_dir, config_filename)
        print(
            f"Config not found locally at {expected_cfg_dest}. "
            f"Downloading from Hugging Face ({clean_repo_id}/{config_filename})..."
        )
        try:
            config_path = hf_hub_download(
                repo_id=clean_repo_id,
                filename=config_filename,
                local_dir=target_dir,
            )
        except (TypeError, Exception) as e:
            print(f"Notice: local_dir download encountered: {e}. Falling back to default cache...")
            config_path = hf_hub_download(
                repo_id=clean_repo_id,
                filename=config_filename,
                cache_dir=os.path.join(base_dir, "hf_cache"),
            )

    return model_path, config_path
