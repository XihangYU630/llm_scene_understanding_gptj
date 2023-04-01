from transformers import GPTJModel, GPTJForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import os
import sys
import time
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  #'0,1'
device_ids = [0, 1]    #[0,1]

query_embeddings_GPT_J = torch.load(os.path.join("/media/xihang/Elements/487_project/llm_scene_understanding/data/GPT-J_nyuClass_useGT_True_502030/train/query_embeddings_train_k1_total1.pt"))
query_embeddings_RoBERTa = torch.load(os.path.join("/media/xihang/Elements/487_project/data/RoBERTa-large_nyuClass_useGT_True_502030/train/query_embeddings_train_k1_total1.pt"))

model = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16).cuda()
# model = GPTJModel.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).cuda()
# model = nn.DataParallel(model, device_ids=device_ids).cuda()

model.eval()
model = model.cuda()


# no revision
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

out_list = []
for i in range (10000):
    print(str(i) + "th iteration")
    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    tokens_tensor = torch.tensor(tokenizer.encode(prompt,
                                add_special_tokens=False,
                                return_tensors="pt").cuda())
    with torch.no_grad():
        outputs = model(tokens_tensor)
        # output_tmp = copy.deepcopy(outputs)
        out_list.append(outputs)
    # size_obj = sys.getsizeof(outputs)
    # output_tmp = outputs
    # time.sleep(2)
    # out_list.append(outputs)
    # size_obj = sys.getsizeof(out_list)
    # print("size_obj: ", size_obj)