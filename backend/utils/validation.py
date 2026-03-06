'''
Not needed for this starter, can be used in the future when other models are added.
'''

def require_model_supported(model: str):
    if model != "tfim":
        raise ValueError(f"Unsupported model '{model}'. Only 'tfim' is implemented in this starter.")