import src.config as config

RED = "\x1B[0;31m"
GREEN = "\x1B[0;32m"
YELLOW = "\x1B[0;33m"
BLUE = "\x1B[0;34m"
MAGENTA = "\x1B[0;35m"
CYAN = "\x1B[0;36m"
LIGHT_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
LIGHT_YELLOW = "\033[93m"
LIGHT_BLUE = "\033[94m"
RESET = "\033[0m"

def print_color(text, color):
    allowed_colors = {
        'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN',
        'LIGHT_GREEN', 'LIGHT_RED', 'LIGHT_YELLOW', 'LIGHT_BLUE'
    }
    if color not in allowed_colors:
        raise ValueError("Unsupported color")
    color_code = globals()[color]
    print(f"{color_code}{text}{RESET}")

def select_tokenizer():
    """
    Selects the appropriate tokenizer based on the model name.
    Includes fallback to AutoTokenizer for flexibility.
    """
    try:
        if config.MODEL_NAME == "bert-base-uncased":
            from transformers import BertTokenizer
            return BertTokenizer.from_pretrained(config.MODEL_NAME)
        elif config.MODEL_NAME == "distilbert-base-uncased":
            from transformers import DistilBertTokenizer
            return DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        else:
            # fallback to AutoTokenizer for other models
            from transformers import AutoTokenizer
            print_color(f"Using AutoTokenizer for model: {config.MODEL_NAME}", "YELLOW")
            return AutoTokenizer.from_pretrained(config.MODEL_NAME)
    except Exception as e:
        # handle any errors during tokenizer loading
        print_color(f"Error loading tokenizer: {e}", "RED")
        print_color("Falling back to bert-base-uncased tokenizer", "YELLOW")
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('bert-base-uncased')
    
def select_model():
    """
    Selects the appropriate model based on the model name.
    Includes fallback to AutoModel for flexibility.
    """
    try:
        if config.MODEL_NAME == "bert-base-uncased":
            from transformers import BertModel
            return BertModel.from_pretrained(config.MODEL_NAME)
        elif config.MODEL_NAME == "distilbert-base-uncased":
            from transformers import DistilBertModel
            return DistilBertModel.from_pretrained(config.MODEL_NAME)
        else:
            # fallback to AutoModel for other models
            from transformers import AutoModel
            print_color(f"Using AutoModel for model: {config.MODEL_NAME}", "YELLOW")
            return AutoModel.from_pretrained(config.MODEL_NAME)
    except Exception as e:
        # handle any errors during model loading
        print_color(f"Error loading model: {e}", "RED")
        print_color("Falling back to bert-base-uncased model", "YELLOW")
        from transformers import BertModel
        return BertModel.from_pretrained('bert-base-uncased')