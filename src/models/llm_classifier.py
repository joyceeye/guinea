from src.core.utils import print_color
from src.llm_wrapper import LLM


def classify(llm: LLM, text, possible_labels):
    prompt = f"""
    Classify the following academic text into one or more of these categories: {', '.join(possible_labels)}
    
    Text: {text}
    
    Return only the labels, separated by commas.
    """
    print("====== DEBUG ======")
    print_color(f"LLM prompt: {prompt}", "RED")
    
    response = llm.query(prompt)

    print("====== RESPONSE ======")
    print_color(f"LLM response: {response}", "RED")
    
    predicted_labels = [label.strip() for label in response.split(',')]
    
    return predicted_labels