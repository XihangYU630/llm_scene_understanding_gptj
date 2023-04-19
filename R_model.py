from models import *
from transformers import BertModel


class BERT_CLASSIFIER(nn.Module):
    def __init__(self, output_size):
        super(BERT_CLASSIFIER, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.ff_net = FeedforwardNet(768, output_size)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        
        final_output = self.ff_net(pooled_output)
        return final_output
    