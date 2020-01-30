import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel, BertConfig

# Initializing a BERT bert-base-uncased style configuration
config = BertConfig()
config.output_hidden_states = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel(config)
model.eval()
model.to('cuda')


def get_vector(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([0]*len(tokenized_text))


    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        # take the average to get a decent semantic embedding?
        embedding = outputs[0].detach().cpu().numpy().mean(axis=1)[0]
    return embedding