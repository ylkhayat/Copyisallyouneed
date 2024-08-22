from header import *
from .util_func import *

local_rank = int(os.environ['LOCAL_RANK'])
class CopyisallyouneedDataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_encoder_tokenizer'][args['lang']])
        self.bert_vocab.add_tokens(['<|endoftext|>', '[PREFIX]'])
        self.prefix_token_id = self.bert_vocab.convert_tokens_to_ids('[PREFIX]')
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])
        self.data_root_path = args['data_root_dir']
        print(f'[!] load the data from {self.data_root_path}')
        self.file_lists = [f'{self.data_root_path}/dpr_search_result_128_{i}.txt' for i in range(1)]
        # count the number of the samples
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['global_rank']
            random.seed(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker {local_rank}:')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        # load the base_data
        base_data = {}
        with open(f'{self.data_root_path}/base_data_128.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1].strip()
                if id_label:
                    base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over')

    def __len__(self):
        return self.size

    def load_one_part(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def __getitem__(self, i):
        '''
        gpt2_batch: [B_v, S_v]
        bert_batch: [B_doc, S_doc]
        phrase_to_doc: [B_p]
        start_index: [B_p]
        end_index: [B_p]
        '''

        gpt2_batch, cache_doc, docs, counter = [], set(), [], 0
        while counter < self.args['max_doc_size']:
            if len(self.cache) == 0:
                self.load_one_part()
            item = json.loads(self.cache[0].strip())
            base_index = item['index']

            cache_phrase, delta = [], 0
            for phrase, metadata in item['results'][self.last_delta:]:
                if metadata and delta > 0:
                    phrase_ = ' ' + phrase
                    doc, start_pos = metadata[0]

                    # Ġ for minus 1
                    end_pos = start_pos + len(phrase_) - 1

                    if doc:
                        # truncate length at max_length
                        truncate_length = None
                    else:
                        # in the prefix
                        try:
                            phrase_length = len(phrase_)
                            # minus 1 means due to the empty character before the phrase
                            doc_end_pos = self.base_data[base_index][start_pos-1+phrase_length:].index(phrase_)
                        except:
                            # this phrase will not be used
                            cache_phrase.append((phrase_, 0))
                            delta += 1
                            continue

                        # minus the special token
                        truncate_length = start_pos + phrase_length + doc_end_pos - 1
                    cache_doc.add(doc)
                    if doc:
                        docs.append((doc, phrase_, start_pos, end_pos, truncate_length))
                    else:
                        docs.append((base_index, phrase_, start_pos, end_pos, truncate_length))

                    counter += 1

                    # save the phrase and its map to the cache_doc
                    cache_phrase.append((phrase_, 1))
                    # valid phrase add 1
                else:
                    if delta > 0:
                        phrase = ' ' + phrase
                    cache_phrase.append((phrase, 0))

                if counter >= self.args['max_doc_size']:
                    self.last_delta = delta
                    self.if_last_over = False
                    break
                delta += 1
            else:
                self.if_last_over = True

            # update the cache
            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # print(f"cache_phrase size: {np.asarray(cache_phrase).shape}")
            # collect
            gpt2_batch.append(cache_phrase)

        # process the bert_batch
        bert_batch, phrase_to_doc, phrase_start_index, phrase_end_index = [], [], [], []
        error_label = []
        phrase_doc_dict = {}
        for doc_id, phrase, start_pos, end_pos, truncate_length in docs:
            text = self.base_data[doc_id]
            item = self.bert_vocab(text, add_special_tokens=False, return_offsets_mapping=True)
            doc_ids = item['input_ids']
            start_mapping = [s for s, e in item['offset_mapping']]
            end_mapping = [e for s, e in item['offset_mapping']]
            try:
                start_index = start_mapping.index(start_pos)
                end_index = end_mapping.index(end_pos)
            except:
                # wrong phrase, not consider
                error_label.append(True)
                continue
            # target_phrase = ' ' + self.bert_vocab.decode(doc_ids[start_index:end_index+1])
            # assert phrase == target_phrase, f'{phrase} {target_phrase}'
            if truncate_length:
                try:
                    end_doc_index = end_mapping.index(truncate_length)
                except:
                    error_label.append(True)
                    continue
                # special case with the [CLS] as the end, not the [SEP]
                bert_batch.append(
                    [self.prefix_token_id] + doc_ids[:end_doc_index] + [self.bert_vocab.sep_token_id]
                )
                phrase_to_doc.append(len(bert_batch) - 1)
            else:
                if doc_id in phrase_doc_dict:
                    phrase_to_doc.append(phrase_doc_dict[doc_id])
                else:
                    bert_batch.append(
                        [self.bert_vocab.cls_token_id] + doc_ids + [self.bert_vocab.sep_token_id]
                    )
                    phrase_to_doc.append(len(bert_batch) - 1)
            if doc_id not in phrase_doc_dict:
                phrase_doc_dict[doc_id] = len(bert_batch) - 1
            # +1 cause the [CLS] token is added
            phrase_start_index.append(start_index + 1)
            phrase_end_index.append(end_index + 1)
            error_label.append(False)
            
        # print(f"phrase_start_index size: {np.asarray(phrase_start_index).shape}")
        # print(f"phrase_end_index size: {np.asarray(phrase_end_index).shape}")
        # print(f"error_label size: {np.asarray(error_label).shape}")

        max_bert_length = max([len(i) for i in bert_batch])

        # process the gpt2_batch
        gpt2_ids, counter, valid_counter = [], 0, 0
        start_labels, end_labels = [], []
        for text in gpt2_batch:
            phrases = [phrase for phrase, _ in text]
            is_phrase = [label for _, label in text]
            phrase_ids = self.vocab(phrases, add_special_tokens=False)['input_ids']
            # print(f"phrase_ids size: {len(phrase_ids)}")
            ids, end_labels_, start_labels_ = [], [], []
            for ids_, label in zip(phrase_ids, is_phrase):
                start_ids_ = deepcopy(ids_)
                end_ids_ = deepcopy(ids_)
                if label and error_label[counter] is False:
                    # replace the first token with OOV token to label it
                    bert_doc_id = phrase_to_doc[valid_counter]
                    chunk_length = max_bert_length * bert_doc_id
                    chunk_start_delta = phrase_start_index[valid_counter]
                    chunk_end_delta = phrase_end_index[valid_counter]
                    start_ids_[0] = len(self.vocab) + chunk_length + chunk_start_delta
                    end_ids_[0] = len(self.vocab) + chunk_length + chunk_end_delta
                    valid_counter += 1
                start_labels_.extend(start_ids_)
                end_labels_.extend(end_ids_)
                ids.extend(ids_)
                if label:
                    counter += 1
            gpt2_ids.append(ids)
            start_labels.append(start_labels_)
            end_labels.append(end_labels_)
        # print(f"gpt2_ids size: {len(gpt2_ids)}")
        # print(f"start_labels size: {len(start_labels)}")
        # print(f"end_labels size: {len(end_labels)}")
        assert counter == len(phrase_to_doc) + sum(error_label)
        query_num = counter - sum(error_label)
            
        ######
        # prepare the batch
        gpt2_ids = pad_sequence([torch.LongTensor(i) for i in gpt2_ids], padding_value=self.vocab.eos_token_id, batch_first=True)
        start_labels = pad_sequence([torch.LongTensor(i) for i in start_labels], padding_value=self.vocab.eos_token_id, batch_first=True)
        end_labels = pad_sequence([torch.LongTensor(i) for i in end_labels], padding_value=self.vocab.eos_token_id, batch_first=True)
        bert_ids = pad_sequence([torch.LongTensor(i) for i in bert_batch], padding_value=self.bert_vocab.pad_token_id, batch_first=True)
        gpt2_mask = generate_mask(gpt2_ids, pad_token_idx=self.vocab.eos_token_id)
        bert_mask = generate_mask(bert_ids, pad_token_idx=self.bert_vocab.pad_token_id)
        ###### prepare the document mask (only for the query position)
        ###### [Q, N_p]
        total_phrase_num = bert_ids.size(0) * bert_ids.size(1)
        query_num = counter - sum(error_label)
        pos_mask = torch.zeros(query_num, total_phrase_num).to(torch.long)
        chunk_size = bert_ids.size(1)
        for i in range(query_num):
            start_pos = phrase_to_doc[i] * chunk_size
            end_pos = phrase_to_doc[i] * chunk_size + chunk_size 
            pos_mask[i, start_pos:end_pos] = 1
            # print(f"phrase_to_doc item: {phrase_to_doc[i]}")
            # print(f"start_pos: {start_pos}, end_pos: {end_pos}")
            # print(f"pos_mask size: {pos_mask[i].shape}")


        # print(f"gpt2_ids size: {gpt2_ids.size()}")
        # print(f"start_labels size: {start_labels.size()}")
        # print(f"end_labels size: {end_labels.size()}")
        # print(f"bert_ids size: {bert_ids.size()}")
        # print(f"gpt2_mask size: {gpt2_mask.size()}")
        # print(f"bert_mask size: {bert_mask.size()}")
        # print(f"pos_mask size: {pos_mask.size()}")
        ###### get the query_pos position
        # labels = (start_labels.reshape(-1) > len(self.vocab)).to(torch.long)
        return gpt2_ids, start_labels, end_labels, bert_ids, gpt2_mask, bert_mask, pos_mask

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        gpt2_ids, start_labels, end_labels, bert_ids, gpt2_mask, bert_mask, pos_mask = batch[0]
        return {
            'gpt2_ids': gpt2_ids.cuda(),
            'bert_ids': bert_ids.cuda(),
            'gpt2_mask': gpt2_mask.cuda(),
            'bert_mask': bert_mask.cuda(),
            'start_labels': start_labels.cuda(),
            'end_labels': end_labels.cuda(),
            'pos_mask': pos_mask.cuda()
        }
