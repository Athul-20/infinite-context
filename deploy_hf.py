from huggingface_hub import HfApi
api = HfApi()

USERNAME = api.whoami()["name"]
REPO_ID = f"{USERNAME}/infinite-context"
REPO_TYPE = "space"

print(f"Creating Space repository '{REPO_ID}'...")
try:
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, space_sdk="gradio", exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

print("Uploading files to the Space...")
api.upload_folder(
    folder_path=".",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    ignore_patterns=[
        ".git/*",
        ".git",
        "venv311/*",
        "venv310/*",
        "__pycache__/*",
        "unsloth_compiled_cache/*",
        "*.pyc",
        "*.log",
        ".env",
        "deploy_hf.py",
        "infinite_context.egg-info/*",
    ]
)

print(f"\nSuccess! Your Space is being built and will be available at: https://huggingface.co/spaces/{REPO_ID}")
