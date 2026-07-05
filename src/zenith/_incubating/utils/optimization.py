import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

def apply_dynamic_quantization(model):
    """
    Applies dynamic quantization to a trained model.
    This is most effective for speeding up models on CPU.

    Args:
        model (nn.Module): The trained PyTorch model.

    Returns:
        nn.Module: The quantized model.
    """
    print("Applying dynamic quantization...")
    # Make sure the model is on the CPU
    model.to('cpu')
    model.eval()

    # Quantize the model
    # We specify that we want to quantize Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    print("Quantization complete.")
    return quantized_model

def apply_structured_pruning(model, modules_to_prune, amount=0.5):
    """
    Applies structured L1-norm pruning to specified layers of a model.
    This type of pruning removes entire channels/neurons.

    Args:
        model (nn.Module): The model to be pruned.
        modules_to_prune (list of tuples): A list of tuples, where each tuple contains
                                            a module and the name of the parameter to prune 
                                            (e.g., [(model.fc, 'weight')]).
        amount (float): The fraction of channels to prune (0.0 to 1.0).

    Returns:
        nn.Module: The pruned model.
    """
    print(f"Applying structured pruning with amount {amount}...")
    
    for module, name in modules_to_prune:
        prune.ln_structured(module, name=name, amount=amount, n=1, dim=0)
        
        prune.remove(module, name)

    print("Pruning complete.")
    return model

if __name__ == '__main__':
    from ..models.bert_model import BERT

    dummy_bert = BERT(vocab_size=10000, num_layers=2, heads=2, ff_hidden_dim=512)
    dummy_bert.eval()
    
    # Apply quantization
    quantized_bert = apply_dynamic_quantization(dummy_bert)
    
    # It will be smaller and faster on CPU
    print("\nOriginal BERT model:")
    print(dummy_bert)
    print("\nQuantized BERT model:")
    print(quantized_bert)

    # Create another dummy model
    dummy_bert_to_prune = BERT(vocab_size=10000, num_layers=2, heads=2, ff_hidden_dim=512)
    
    # Define which layers to prune
    # For a Transformer, we can prune the linear layers in the feed-forward network
    modules_to_prune = []
    for layer in dummy_bert_to_prune.encoder_layers:
        modules_to_prune.append((layer.feed_forward.linear1, 'weight'))
        modules_to_prune.append((layer.feed_forward.linear2, 'weight'))
        
    # Apply pruning
    pruned_bert = apply_structured_pruning(dummy_bert_to_prune, modules_to_prune, amount=0.3)
    
    print("\n--- Pruning Results ---")
    print("Original model's feed-forward layer weight shape:")
    print(dummy_bert.encoder_layers[0].feed_forward.linear1.weight.shape)
    
    print("\nPruned model's feed-forward layer weight shape:")
    print(pruned_bert.encoder_layers[0].feed_forward.linear1.weight.shape)
    

