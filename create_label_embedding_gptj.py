import torch
import os
import tqdm
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    GPTNeoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTJModel,
)


def configure_lm(lm):
    """
    Configure the language model, tokenizer, and embedding generator function.

    Args:
        lm: str representing name of LM to use

    Returns:
        None
    """

    if lm == "GPT-J":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        lm_model = GPTJModel.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,  # low_cpu_mem_usage=True
        )
        # revision="float16",
    else:
        print("Model option " + lm + " not implemented yet")
        raise
    device = None
    device = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None else device)

    lm_model = lm_model
    lm_model.eval()
    lm_model = lm_model.to(device)

    embedder = _initialize_embedder(False, lm_model, tokenizer, device)

    return embedder

def _initialize_embedder(is_mlm, lm_model, tokenizer,device, start=None, end=None):
    """
    Returns a function that embeds sentences with the selected
    language model.

    Args:
        is_mlm: bool (optional) indicating if lm_model is an mlm.
            Default
        start: str representing start token for MLMs.
            Must be set if is_mlm == True.
        end: str representing end token for MLMs.
            Must be set if is_mlm == True.

    Returns:
        function that takes in a query string and outputs a
            [batch size=1, hidden state size] summary embedding
            using lm_model
    """
    if not is_mlm:

        def embedder(query_str):
            tokens_tensor = torch.tensor(
                tokenizer.encode(query_str,
                                        add_special_tokens=False,
                                        return_tensors="pt").to(device))

            outputs = lm_model(tokens_tensor)
            # print(outputs)
            # print(outputs.last_hidden_state.shape)
            # Shape (batch size=1, hidden state size)
            return outputs.last_hidden_state[:, -1]

    else:

        def embedder(query_str):
            query_str = start + " " + query_str + " " + end
            tokenized_text = tokenizer.tokenize(query_str)
            tokens_tensor = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokenized_text)])
            tokens_tensor = tokens_tensor.to(device)  # if you have gpu

            with torch.no_grad():
                outputs = lm_model(tokens_tensor)
                # hidden state is a tuple
                hidden_state = outputs.last_hidden_state

            # Shape (batch size=1, num_tokens, hidden state size)
            # Return just the start token's embeddinge
            return hidden_state[:, -1]

    return embedder

def _label_str_constructor(room):
    if room == 'office':
        label_string = "This room is an " + room + "."
    else:
        label_string = "This room is a " + room + "."
    return label_string

def generate_data(lm, data_folder):
    """
    Constructs query string using selected number of objects

    Args:

    Returns:
        Tuple of (list of strs, torch.tensor, torch.tensor, torch.tensor).
            Respectively:
            1) list of query sentences of length
                (# rooms) * (num_obs P k)
            2) tensor of int room labels corresponding to above list
            3) tensor of sentence embeddings corresponding to above list
            4) tensor of sentence embeddings corresponding to room label string
    """
    room_list = [
        'none', 'balcony', 'bar', 'bathroom', 'bedroom', 'classroom', 'closet',
        'conference room', 'dining room', 'family room', 'game room', 'garage',
        'gym', 'hallway', 'kitchen', 'laundry room', 'library', 'living room',
        'lobby', 'lounge', 'office', 'porch', 'spa', 'staircase',
        'television room', 'utility room', 'yard'
    ]
    embedder = configure_lm(lm)
    for room in room_list:

        label_str = _label_str_constructor(room)
        label_embedding = embedder(label_str)                

        # Save query embeddings
        if os.path.isfile(os.path.join(data_folder, "GPT-J.pt")):
            label_embeddings = torch.load(os.path.join(data_folder, "GPT-J.pt"))
            label_embeddings = torch.vstack((label_embeddings,label_embedding))
            torch.save(
                label_embeddings,
                os.path.join(data_folder, "GPT-J.pt"),
            )
        else:
            torch.save(
                label_embedding,
                os.path.join(data_folder, "GPT-J.pt"),
            )
    tmp = 1


if __name__ == "__main__":


    lm = "GPT-J"
    data_folder = "./label_embeddings"
    generate_data(lm, data_folder)
