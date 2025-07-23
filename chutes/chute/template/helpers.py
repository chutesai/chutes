from huggingface_hub import HfApi


def get_current_hf_commit(model_name: str):
    """
    Helper to load the current main commit for a given repo.
    """
    api = HfApi()
    for ref in api.list_repo_refs(model_name).branches:
        if ref.ref == "refs/heads/main":
            return ref.target_commit
    return None
