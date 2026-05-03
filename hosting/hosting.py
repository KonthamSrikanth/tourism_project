from huggingface_hub import HfApi
import os

# Use environment variable for HF token (do not hardcode credentials)
api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",   # local folder containing app.py, Dockerfile, requirements.txt
    repo_id="SrikanthKontham/tourism_project",  # target Hugging Face Space
    repo_type="space",
    path_in_repo="",                            # upload to root of the space
)
print("Deployment files pushed to Hugging Face Space successfully.")
