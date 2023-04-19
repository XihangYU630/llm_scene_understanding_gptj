# First get query
# Tokenize
# Convert to tensor
from tqdm import tqdm, trange
from R_generate_data import *
from torch.utils.data import Dataset
from R_model import *

def initialize_embedder(is_mlm, start=None, end=None, tokenizer = None):
        """
        Returns a function that embeds sentences with the selected
        language model.

        Args:
            is_mlm: bool (optional) indicating if self.lm_model is an mlm.
                Default
            start: str representing start token for MLMs.
                Must be set if is_mlm == True.
            end: str representing end token for MLMs.
                Must be set if is_mlm == True.

        Returns:
            function that takes in a query string and outputs a
                [batch size=1, hidden state size] summary embedding
                using self.lm_model
        """
        if not is_mlm:

            quit()

        else:

            def embedder(query_str):
                
                # Return token_id, mask   with padding
                
                #query_str = start + " " + query_str + " " + end
                #print(query_str)
                #tokenized_text_orig = tokenizer.tokenize(query_str)
                tokenized_text = tokenizer(query_str, padding='max_length', max_length = 30)
                #print(tokenized_text['input_ids'])
                
                
                
                input_ids = torch.tensor(tokenized_text['input_ids']).to('cuda')
                mask = torch.tensor(tokenized_text['attention_mask']).to('cuda')
                
                #tokens_tensor = torch.tensor(
                #    [tokenizer.convert_tokens_to_ids(tokenized_text_orig)])
                #print(tokens_tensor)
                #print(tokenizer.decode(tokenized_text.input_ids))
                
                """ tokens_tensor = torch.tensor([indexed_tokens.to(self.device)])
                 """
                # if you have gpu

                # Shape (batch size=1, num_tokens, hidden state size)
                # Return just the start token's embeddinge
                return input_ids, mask

        return embedder


class R_RoomDataset(Dataset):

    def __init__(self,
                 path_to_data,
                 device="cuda",
                 return_sentences=False,
                 return_all_objs=True):
        # Extract object, room, and bldg labels
        dataset = Matterport3dDataset(
            "./mp_data/nyuClass_matterport3d_w_edge_new.pkl")
        labels, pl_labels = create_label_lists(dataset)
        self.building_list, self.room_list, self.object_list = labels
        self.building_list_pl, self.room_list_pl, self.object_list_pl = pl_labels

        del dataset

        self.device = device
        
        self.return_all_objs = return_all_objs

        # Initialize data attrs
        
        self.labels = []
        
        self.sentences = []
        if self.return_all_objs:
            self.all_objs = []

        # Extract all suffixes
        suffixes = []
        for file in os.listdir(path_to_data):
            if "labels_" in file:
                suffixes.append(file[len("labels"):-len(".pt")])
        
        for s in suffixes:
            '''
            query_embeddings = torch.load(
                os.path.join(path_to_data, "query_embeddings" + s + ".pt"))
            room_embeddings = torch.load(
                os.path.join(path_to_data, "room_embeddings" + s + ".pt"))
            '''
            labels = torch.load(
                os.path.join(path_to_data, "labels" + s + ".pt"))
            
            with open(
                    os.path.join(path_to_data,
                                    "query_sentences" + s + ".pkl"),
                    "rb") as fp:
                self.sentences += pickle.load(fp)
            if self.return_all_objs:
                with open(os.path.join(path_to_data, "all_objs" + s + ".pkl"),
                          "rb") as fp:
                    self.all_objs += pickle.load(fp)
            #self.query_embeddings.append(query_embeddings)
            #self.room_embeddings.append(room_embeddings)
            self.labels.append(labels)
        
        
        self.sentences_token = []
        self.sentences_mask = []
        # Tokenize the sentence
        # self.sentences: list of queries
        start = "[CLS]"
        end = "[SEP]"
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        embedder = initialize_embedder(True,
                                                      start=start,
                                                      end=end, 
                                                      tokenizer=tokenizer)
        for s in self.sentences:
            
            s_tok, s_mask = embedder(s)
            self.sentences_token.append(s_tok)
            self.sentences_mask.append(s_mask)
            
        self.sentences_token = torch.vstack(self.sentences_token)
        self.sentences_mask = torch.vstack(self.sentences_mask)
        
        
        '''
        self.query_embeddings = torch.cat(self.query_embeddings).to(
            self.device)
        self.room_embeddings = torch.cat(self.room_embeddings).to(self.device)
        '''
        self.labels = torch.cat(self.labels).to(self.device)
        self.sentences = None 
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        res = [
            self.query_embeddings[idx], self.room_embeddings[idx],
            self.labels[idx]
        ]
        if self.return_sentences:
            res += [self.sentences[idx]]
        if self.return_all_objs:
            res += [self.all_objs[idx]]
        '''
        return self.sentences_token[idx], self.sentences_mask[idx], self.labels[idx]


def R_create_room_splits(path_to_data,
                       device="cuda",
                       return_sentences=False,
                       return_all_objs=False):

    train_ds = R_RoomDataset(os.path.join(path_to_data, "train"),
                           device=device,
                           return_sentences=return_sentences,
                           return_all_objs=return_all_objs)
    val_ds = R_RoomDataset(os.path.join(path_to_data, "val"),
                         device=device,
                         return_sentences=return_sentences,
                         return_all_objs=return_all_objs)
    test_ds = R_RoomDataset(os.path.join(path_to_data, "test"),
                          device=device,
                          return_sentences=return_sentences,
                          return_all_objs=return_all_objs)

    return train_ds, val_ds, test_ds














def R_train_job(lm, label_set, use_gt, epochs, batch_size, seed=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    def ff_loss(pred, label):
        return F.cross_entropy(pred, label)

    # Create datasets
    suffix = lm + "_" + label_set + "_useGT_" + str(use_gt) + "_502030"
    path_to_data = os.path.join("./data/", suffix)
    train_ds, val_ds, test_ds = R_create_room_splits(path_to_data, device="cuda")

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    output_size = len(train_ds.room_list)

    R_model = BERT_CLASSIFIER(output_size)
    R_model.to(device)

    # 63.42, lr=0.00001, wd=0.001, ss=50, g=0.1
    # 64.49, lr=0.0001, wd=0.001, ss=10, g=0.5
    optimizer = torch.optim.Adam(R_model.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)

    loss_fxn = ff_loss

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    desc = ""
    with trange(epochs) as pbar:
        for epoch in pbar:
            train_epoch_loss = []
            val_epoch_loss = []
            train_epoch_acc = []
            val_epoch_acc = []
            for batch_idx, (sentence, mask, label) in enumerate(train_dl):
                pred = R_model(sentence, mask)
                loss = loss_fxn(pred, label)

                R_model.zero_grad()

                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

                accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
                train_epoch_acc.append(accuracy)

                if batch_idx % 100 == 0:
                    pbar.set_description((desc).rjust(20))

            scheduler.step()
            train_losses.append(torch.mean(torch.tensor(train_epoch_loss)))
            train_acc.append(torch.mean(torch.tensor(train_epoch_acc)))

            for batch_idx, (sentence, mask, label) in enumerate(val_dl):
                with torch.no_grad():
                    pred = R_model(sentence, mask)
                    loss = loss_fxn(pred, label)
                    val_epoch_loss.append(loss.item() * len(label))

                    accuracy = ((torch.argmax(pred, dim=1) == label) *
                                1.0).mean()
                    val_epoch_acc.append(accuracy * len(label))
                    if batch_idx % 100 == 0:
                        desc = (f"{loss.item():6.4}" + ", " +
                                f"{accuracy.item():6.4}")
                        pbar.set_description((desc).rjust(20))
            val_losses.append(
                torch.sum(torch.tensor(val_epoch_loss)) / len(val_ds))
            val_acc.append(
                torch.sum(torch.tensor(val_epoch_acc)) / len(val_ds))
            if epoch == 0:
                best_val_acc = val_acc[-1]
                torch.save(R_model.state_dict(),
                           "./checkpoints/best_ff_" + suffix + ".pt")
            elif val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                torch.save(R_model.state_dict(),
                           "./checkpoints/best_ff_" + suffix + ".pt")

    R_model.load_state_dict(
        torch.load("./checkpoints/best_ff_" + suffix + ".pt"))
    R_model.eval()
    test_loss, test_acc = [], []
    for batch_idx, (sentence, mask, label) in enumerate(test_dl):
        pred = R_model(sentence, mask)
        loss = loss_fxn(pred, label)
        test_loss.append(loss.item())

        accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
        test_acc.append(accuracy)

    print("test loss:", torch.mean(torch.tensor(test_loss)))
    print("test acc:", torch.mean(torch.tensor(test_acc)))
    return train_losses, val_losses, train_acc, val_acc, test_loss, test_acc


if __name__ == "__main__":
    (
        train_losses_list,
        val_losses_list,
        train_acc_list,
        val_acc_list,
        test_loss_list,
        test_acc_list,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for lm in ["BERT"]:
        for label_set in ["nyuClass", "mpcat40"]:
            for use_gt in [True, False]:
                print("Starting:", lm, label_set, "use_gt =", use_gt)
                (
                    train_losses,
                    val_losses,
                    train_acc,
                    val_acc,
                    test_loss,
                    test_acc,
                ) = R_train_job(lm, label_set, use_gt, 100, 128)
                train_losses_list.append(train_losses)
                val_losses_list.append(val_losses)
                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)

    pickle.dump(train_losses_list, open("./ff_results/train_losses.pkl", "wb"))
    pickle.dump(train_acc_list, open("./ff_results/train_acc.pkl", "wb"))
    pickle.dump(val_losses_list, open("./ff_results/val_losses.pkl", "wb"))
    pickle.dump(val_acc_list, open("./ff_results/val_acc.pkl", "wb"))
    pickle.dump(test_loss_list, open("./ff_results/test_loss.pkl", "wb"))
    pickle.dump(test_acc_list, open("./ff_results/test_acc.pkl", "wb"))
