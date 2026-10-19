import os
import shutil
import tempfile
import unittest
from hf_utils import (
    parse_repo_and_filename,
    extract_repo_name,
    resolve_model_path,
    resolve_pretrained_path,
    load_custom_model_from_hf,
    set_checkpoints_dir,
    get_checkpoints_dir,
)


class TestHfUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_dir = get_checkpoints_dir()
        set_checkpoints_dir(self.temp_dir)

    def tearDown(self):
        set_checkpoints_dir(self.orig_dir)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_repo_and_filename(self):
        # Standard repo ID
        repo, name, fn = parse_repo_and_filename("Plachta/Seed-VC", "v2/cfm_small.pth")
        self.assertEqual(repo, "Plachta/Seed-VC")
        self.assertEqual(name, "Seed-VC")
        self.assertEqual(fn, "v2/cfm_small.pth")

        # Full URL with resolve/main
        repo, name, fn = parse_repo_and_filename(
            "https://huggingface.co/author/reponame/resolve/main/subdir1/mymodel.gguf"
        )
        self.assertEqual(repo, "author/reponame")
        self.assertEqual(name, "reponame")
        self.assertEqual(fn, "subdir1/mymodel.gguf")

        # Full URL with blob/main
        repo, name, fn = parse_repo_and_filename(
            "https://huggingface.co/author/reponame/blob/main/sub/path/model.pt"
        )
        self.assertEqual(repo, "author/reponame")
        self.assertEqual(name, "reponame")
        self.assertEqual(fn, "sub/path/model.pt")

        # User prompt example: huggingface.com/author/reponame/subdir1/mymodel.gguf
        repo, name, fn = parse_repo_and_filename("huggingface.com/author/reponame/subdir1/mymodel.gguf")
        self.assertEqual(repo, "author/reponame")
        self.assertEqual(name, "reponame")
        self.assertEqual(fn, "subdir1/mymodel.gguf")

        # Bare name
        repo, name, fn = parse_repo_and_filename("campplus", "campplus_cn_common.bin")
        self.assertEqual(repo, "campplus")
        self.assertEqual(name, "campplus")
        self.assertEqual(fn, "campplus_cn_common.bin")

    def test_extract_repo_name(self):
        self.assertEqual(extract_repo_name("Plachta/Seed-VC"), "Seed-VC")
        self.assertEqual(extract_repo_name("funasr/campplus"), "campplus")
        self.assertEqual(extract_repo_name("nvidia/bigvgan_v2_22khz_80band_256x"), "bigvgan_v2_22khz_80band_256x")
        self.assertEqual(extract_repo_name("openai/whisper-small"), "whisper-small")
        self.assertEqual(extract_repo_name("huggingface.co/org/custom-model"), "custom-model")

    def test_resolve_model_path_nested(self):
        # Create a mock checkpoint in temp_dir/Seed-VC/v2/cfm_small.pth
        sub_dir = os.path.join(self.temp_dir, "Seed-VC", "v2")
        os.makedirs(sub_dir, exist_ok=True)
        file_path = os.path.join(sub_dir, "cfm_small.pth")
        with open(file_path, "w") as f:
            f.write("mock")

        resolved = resolve_model_path("Plachta/Seed-VC", "v2/cfm_small.pth")
        self.assertIsNotNone(resolved)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(file_path))

    def test_resolve_model_path_flat_in_repo_dir(self):
        # Create a mock checkpoint in temp_dir/campplus/campplus_cn_common.bin
        sub_dir = os.path.join(self.temp_dir, "campplus")
        os.makedirs(sub_dir, exist_ok=True)
        file_path = os.path.join(sub_dir, "campplus_cn_common.bin")
        with open(file_path, "w") as f:
            f.write("mock-campplus")

        resolved = resolve_model_path("funasr/campplus", "campplus_cn_common.bin")
        self.assertIsNotNone(resolved)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(file_path))

    def test_resolve_pretrained_path(self):
        # Create a mock directory in temp_dir/whisper-small
        model_dir = os.path.join(self.temp_dir, "whisper-small")
        os.makedirs(model_dir, exist_ok=True)

        resolved = resolve_pretrained_path("openai/whisper-small")
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(model_dir))

        # BigVGAN directory
        bigvgan_dir = os.path.join(self.temp_dir, "bigvgan_v2_22khz_80band_256x")
        os.makedirs(bigvgan_dir, exist_ok=True)
        resolved_bv = resolve_pretrained_path("nvidia/bigvgan_v2_22khz_80band_256x")
        self.assertEqual(os.path.abspath(resolved_bv), os.path.abspath(bigvgan_dir))

        # Not found should return original identifier
        not_found = resolve_pretrained_path("openai/nonexistent-model")
        self.assertEqual(not_found, "openai/nonexistent-model")

    def test_campplus_resolution(self):
        # When campplus is present in checkpoints/campplus, it resolves
        campplus_dir = os.path.join(self.temp_dir, "campplus")
        os.makedirs(campplus_dir, exist_ok=True)
        cp_file = os.path.join(campplus_dir, "campplus_cn_common.bin")
        with open(cp_file, "w") as f:
            f.write("mock-campplus-weights")

        resolved = resolve_model_path("funasr/campplus", "campplus_cn_common.bin")
        self.assertIsNotNone(resolved)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(cp_file))

        # Also test with full HF URL as repo_id
        resolved_url = resolve_model_path("https://huggingface.co/funasr/campplus/blob/main/campplus_cn_common.bin", "campplus_cn_common.bin")
        self.assertIsNotNone(resolved_url)
        self.assertEqual(os.path.abspath(resolved_url), os.path.abspath(cp_file))

    def test_load_custom_model_local(self):
        # Create a mock model and config under Seed-VC/
        model_dir = os.path.join(self.temp_dir, "Seed-VC")
        os.makedirs(model_dir, exist_ok=True)
        m_path = os.path.join(model_dir, "my_model.pth")
        c_path = os.path.join(model_dir, "config.yml")
        with open(m_path, "w") as f:
            f.write("model-data")
        with open(c_path, "w") as f:
            f.write("config-data")

        res_m, res_c = load_custom_model_from_hf("Plachta/Seed-VC", "my_model.pth", "config.yml")
        self.assertEqual(os.path.abspath(res_m), os.path.abspath(m_path))
        self.assertEqual(os.path.abspath(res_c), os.path.abspath(c_path))

    def test_custom_base_dir_param(self):
        # Verify custom base_dir parameter override
        custom_base = tempfile.mkdtemp()
        try:
            m_dir = os.path.join(custom_base, "VoiceConversionWebUI")
            os.makedirs(m_dir, exist_ok=True)
            m_path = os.path.join(m_dir, "rmvpe.pt")
            with open(m_path, "w") as f:
                f.write("rmvpe-data")

            res = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", base_dir=custom_base)
            self.assertEqual(os.path.abspath(res), os.path.abspath(m_path))
        finally:
            shutil.rmtree(custom_base, ignore_errors=True)

    def test_config_presets_fallback(self):
        # Test that a config file in configs/presets is found if not in checkpoints
        resolved = resolve_model_path("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        self.assertIsNotNone(resolved)
        self.assertTrue(os.path.exists(resolved))

    def test_entry_scripts_have_checkpoint_dir_argument(self):
        import ast
        scripts_to_test = [
            "app.py",
            "app_vc.py",
            "app_svc.py",
            "app_vc_v2.py",
            "inference.py",
            "inference_v2.py",
            "eval.py",
            "real-time-gui.py",
            "train.py",
            "train_v2.py",
        ]
        for script in scripts_to_test:
            with open(script, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=script)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and arg.value == "--checkpoint-dir":
                                found = True
                                break
                if found:
                    break
            self.assertTrue(found, f"Script {script} is missing --checkpoint-dir argument definition")


if __name__ == "__main__":
    unittest.main()
