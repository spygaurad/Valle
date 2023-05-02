import numpy as np
import torch
import torch.nn as nn
import random
from torch.nn.functional import mse_loss
import pandas as pd
from src.data.handwriting.handwriting_dataset import DatasetHelper


class TextDecoder(nn.Module):

    def __init__(self, vocab_size, char_embedding, img_embedding, text_pos_encoding, audio_pos_encoding, config, device):
        super(TextDecoder, self).__init__()

        self.device = device
        self.char_embedding = char_embedding
        self.text_pos_encoding = text_pos_encoding
        self.audio_pos_encoding = audio_pos_encoding

        self.img_embedding = img_embedding
        self.dropout = nn.Dropout(p=config.transformer_decoder['dropout'])

        decoder_layer = nn.TransformerEncoderLayer(d_model=config.char_embedding_dim,
                                                   nhead=config.transformer_decoder['num_heads'],
                                                   dropout=config.transformer_decoder['dropout'],
                                                   dim_feedforward=config.transformer_decoder['ffn'],
                                                   activation='relu')
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer,
                                                         num_layers=config.transformer_decoder['num_layers'])

        self.linear = nn.Linear(config.char_embedding_dim, vocab_size)
        # embed_shape = self.img_embedding.weight.shape
        # self.linear = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        print("Linear Embedding:", self.linear)
        # print("Vocab:", vocab_size)
        # self.linear.weight = self.img_embedding.weight # Tied weights

        self.aux_linear = nn.Linear(config.char_embedding_dim, vocab_size+1)
        # self.linear_1 = nn.Linear(config.char_embedding_dim, vocab_size)
        # self.linear_2 = nn.Linear(config.char_embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    # """
    # def forward(self, memory, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask):

    #     tgt_mask = tgt_mask[0]

    #     tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))

    #     memory = memory.permute(1, 0, 2)
    #     tgt = tgt.permute(1, 0, 2)

    #     output = self.transformer_decoder(
    #         tgt=tgt,
    #         memory=memory,
    #         tgt_mask=tgt_mask,  # to avoid looking at the future tokens (the ones on the right)
    #         tgt_key_padding_mask=tgt_key_padding_mask,  # to avoid working on padding
    #         memory_key_padding_mask=src_key_padding_mask  # avoid looking on padding of the src
    #     )

    #     output = output.permute(1, 0, 2)
    #     output = self.linear(output)

    #     return output
    #   """

    def forward(self, memory, tgt, src_mask, src_key_padding_mask, **kwargs):
        text_encodec = kwargs.pop("text_encodec")
        style_codec = None
        output_codec = kwargs.pop("output_codec")

        text_encodec = text_encodec + self.text_pos_encoding(text_encodec)
        style_encodec = None
        # print(text_encodec)
        # print(output_codec)

        output_encodec = self.img_embedding(output_codec) + self.audio_pos_encoding(output_codec)
        # print(text_encodec.shape, output_encodec.shape)

        bs = text_encodec.shape[0]
        
        # memory = torch.cat([text_encodec, style_encodec], axis=1)
        memory = text_encodec
        tgt = output_encodec
        # print("Memory:", memory.shape)
        # print("Tgt:", tgt.shape)

        first_seq_len = memory.shape[1]
        second_seq_len = tgt.shape[1]

        # print("First Seq Len:", first_seq_len)
        # print("Second Seq Len:", second_seq_len)

        src = torch.cat([memory, tgt], axis=1)
        src = src.permute(1, 0, 2)

        # print("SRC:", src.shape)


        src_mask = get_mask_seq_cat(first_seq_len=first_seq_len, second_seq_len=second_seq_len).to(self.device)
        src_key_padding_mask = src_key_padding_mask[:,:-1]

        output = self.transformer_decoder(
            src=src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,  # to avoid working on padding
        )


        output = output.permute(1, 0, 2)
        '''
        # torch.set_printoptions(edgeitems=5)

        p_mask = src_key_padding_mask[0].cpu().numpy().tolist()
        target_tgt = src.cpu().numpy().tolist()
        # s_mask = src_mask.cpu().numpy().tolist()

        # print("Input:  ", tgt)
        # print('-'*200)
        # print("Output: ",output)
        # print('-'*200)
        # print("Src mask: ",tgt_mask)
        # print(indices.shape)
        indices_output = indices.cpu().numpy().tolist()
        df = pd.DataFrame({'pad_mask':p_mask, 'target_tgt':target_tgt, "indices_output":indices_output, "src_mask":s_mask})
        df.to_csv('greedy_tracking.csv', index=False)

        quit() 
        # '''
        # print("*************")
        # print(output.shape)
        # print(self.linear)
        # print(self.linear(output).shape)
        # print("*************")

        final_output = self.linear(output[:,first_seq_len:,:])

        return final_output

    # """
    # def generate(self, memory, src_key_padding_mask, sos_idx, eos_idx, max_char_len):

    #     memory = memory.permute(1, 0, 2)
    #     result = [sos_idx]

    #     for i in range(max_char_len):

    #         tgt = torch.LongTensor([result]).to(self.device)
    #         tgt_mask = DatasetHelper.get_mask(i+1).to(self.device)
    #         tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))

    #         tgt = tgt.permute(1, 0, 2)

    #         output = self.transformer_decoder(
    #             tgt=tgt,
    #             memory=memory,
    #             tgt_mask=tgt_mask,  # to avoid looking at the future tokens (the ones on the right)
    #             tgt_key_padding_mask=None,  # to avoid working on padding
    #             memory_key_padding_mask=src_key_padding_mask  # avoid looking on padding of the src
    #         )

    #         output = output.permute(1, 0, 2)
    #         output = self.linear(output)
    #         output = self.softmax(output)
    #         output = output[0][-1]  # the last timestep

    #         values, indices = output.max(dim=0)
    #         pred_token = indices.item()
    #         result.append(pred_token)

    #         if pred_token == eos_idx:
    #             break

    #     return result
    #  """

    def generate(self, memory, src_key_padding_mask, sos_idx, eos_idx, max_char_len, **kwargs):
        text_encodec = kwargs.pop("text_encodec")
        style_codec = kwargs.pop("style_codec")

        text_encodec = text_encodec + self.text_pos_encoding(text_encodec)
        style_encodec = self.img_embedding(style_codec) + self.audio_pos_encoding(style_codec)

        memory = torch.cat([text_encodec, style_encodec], axis=1)


        first_seq_len = memory.shape[1]
        bs = memory.shape[0]
        result = [sos_idx]

        for i in range(max_char_len):

            tgt = torch.LongTensor([result]).to(self.device)
            tgt_mask = get_mask_seq_cat(first_seq_len=first_seq_len, second_seq_len=i+1).to(self.device)

            tgt = (self.img_embedding(tgt) + self.pos_encoding(tgt))
            # tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))
            
            tgt = torch.cat([memory, tgt], axis=1)

            tgt = tgt.permute(1, 0, 2)

            # src_key_padding_mask = src_key_padding_mask[:,:-1]


            src_key_padding_mask = torch.cat([src_key_padding_mask, get_last_seq_pad_mask(258, i+1, bs).to(self.device)], axis=-1)

            output = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,  # to avoid working on padding
            )

            output = output.permute(1, 0, 2)

            output = output[:,first_seq_len:,:]

            output = self.linear(output)
            output = self.softmax(output)
            output = output[0][-1]  # the last timestep

            values, indices = output.max(dim=0)
            pred_token = indices.item()
            result.append(pred_token)


            # if pred_token == eos_idx:
            #     break

        return result

    def generate_batch(self, memory, src_key_padding_mask, sos_idx, eos_idx, max_char_len, **kwargs):
        text_encodec = kwargs.pop("text_encodec")
        style_codec = None

        text_encodec = text_encodec + self.text_pos_encoding(text_encodec)
        style_encodec = None

        # memory = torch.cat([text_encodec, style_encodec], axis=1)
        memory = text_encodec

        first_seq_len = memory.shape[1]
        bs = memory.shape[0]
        result = torch.ones(bs, 1).long().to(self.device) * sos_idx
        # delete_later = torch.randint(1, 1024, (bs, 1)).long().to(self.device)

        # result = torch.cat([result, delete_later],dim=-1)

        for i in range(max_char_len):

            #Delete this later, just for testing

            # i += 1

            tgt = result
            # tgt_mask = get_mask_seq_cat(first_seq_len=first_seq_len, second_seq_len=i+1).to(self.device)
            tgt_mask = get_mask_seq_cat(first_seq_len=first_seq_len, second_seq_len=i+1).to(self.device)

            tgt = (self.img_embedding(tgt) + self.audio_pos_encoding(tgt))

            tgt = torch.cat([memory, tgt], axis=1)

            
            tgt = tgt.permute(1, 0, 2)

            # print('£'*20)
            # print('Text Decoder')
            # print('src_key_mask_pad shape', src_key_padding_mask.shape)
            # print('£'*20)

            
            pad_mask = torch.cat([src_key_padding_mask, get_last_seq_pad_mask(0, i+1, bs).to(self.device)], axis=-1)
            # src_key_padding_mask = src_key_padding_mask[:,:-1]

            output = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
                src_key_padding_mask=pad_mask,  # to avoid working on padding
            )

            output = output.permute(1, 0, 2)
            # print('-'*200)
            # print(output.shape)
            # print('-'*200)

            output = output[:,first_seq_len+i,:]

            output = self.linear(output)
            output = self.softmax(output)

            values, indices = output.max(dim=-1)
            indices = indices.unsqueeze(-1)
            result = torch.cat([result, indices],dim=-1)

            '''
            # f = open("greedy.txt", "a")
            torch.set_printoptions(edgeitems=5)
            # print("Input: "img2txt_enc_char)
            if i == 3:
                p_mask = pad_mask.cpu().numpy().tolist()
                target_tgt = tgt.cpu().numpy().tolist()
                # print("Input:  ", tgt)
                # print('-'*200)
                # print("Output: ",output)
                # print('-'*200)
                # print("Src mask: ",tgt_mask)
                # print(indices.shape)
                indices_output = indices.cpu().numpy().tolist()
                df = pd.DataFrame({'pad_mask':[p_mask[1]], 'target_tgt':[target_tgt[1]], "indices_output":[indices_output[1]]})
                df.to_csv('greedy_tracking.csv', index=False)

                quit()    
                # f.close()
            # '''

        return result

    def beam_search_batch(self, memory, sos_idx, eos_idx, max_char_len, beam_size=4, gt=None, lm_model=None, **kwargs):
        # print("BEAM search")

        first_seq_len = memory.shape[1]
        bs = memory.shape[0]
        result = torch.ones(bs, 1).long().to(self.device) * sos_idx
        repeat_factor = beam_size
        r_memory = memory.repeat_interleave(repeat_factor, dim=0)

        beam_memory_candidates =[]
        beam_memory_probabilites = []
        lm_next_probabilities = 0

        beam_penalty = kwargs.pop("beam_penalty", 0.5)
        lm_weight = kwargs.pop("lm_weight", 0.1)
        char_model = kwargs.pop("char_model", None)
        tgt = kwargs.pop("tgt", None)
        char2lm = None
        if char_model:
            char2lm = torch.tensor(char_model.char2lm).long().to(self.device)
            # print(char2lm)
        # print(f"Beam Penalty: {beam_penalty}")
        # print(f"LM Weight: {lm_weight}")
        # print(probabilites_mask.shape)
        

        def model_step(result, i):
            tgt = result
            t_memory = memory

            tgt_bs = tgt.shape[0]
            t_memory_bs = t_memory.shape[0]

            if tgt_bs != t_memory_bs and tgt_bs > t_memory_bs:
                repeat_factor = tgt_bs #- t_memory_bs
                t_memory = memory.repeat_interleave(repeat_factor, dim=0)

            tgt_mask = DatasetHelper.get_mask_seq_cat(first_seq_len, i+1).to(self.device)
            tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))
            # print("Memory:", t_memory.shape)
            # print("Tgt:",tgt.shape)

            tgt = torch.cat([t_memory, tgt], axis=1)

            tgt = tgt.permute(1, 0, 2)

            output = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
                src_key_padding_mask=None,  # to avoid working on padding
            )

            output = output.permute(1, 0, 2)

            output = output[:,first_seq_len+i,:]

            output = self.linear(output)
            # output = self.softmax(output)
            return output

        next_probabilities = model_step(result, 0)
        vocab_size = next_probabilities.shape[-1]

        next_probabilities = next_probabilities.log_softmax(-1)
        # print("Result:", result.shape)
        # print("Next Probabilites:", next_probabilities.shape)

        # if lm_model:
        #     # lm_next_probabilities = lm_weight * lm_model.step(result,0).log_softmax(-1)
        #     # lm_next_probabilities = lm_weight * lm_model.step(char2lm[result],0).log_softmax(-1)
        #     seq_len = result.shape[-1]
        #     lm_next_probabilities = lm_model.generate(char2lm[result], do_sample=False, max_length=seq_len+1, output_scores=True, return_dict_in_generate=True)
        #     lm_next_probabilities = lm_next_probabilities["scores"][0].log_softmax(-1)
        #     # print("LM Next:", lm_next_probabilities.shape)

        #     # lm_next_probabilities =  0 * lm_model.step(result,0).log_softmax(-1)
        #     # lm_next_probabilities = lm_model.step(result,0)
        #     # print(lm_next_probabilities)
        #     # lm_next_probabilities = lm_next_probabilities.log_softmax(-1)
        #     # print(lm_next_probabilities)
        #     # lm_next_probabilities = 0.2 * lm_next_probabilities
        #     # print(lm_next_probabilities)

        #     # print("LM next probabilites:", lm_next_probabilities.shape)
       
        # print(next_probabilities.shape)
        # next_probabilities = next_probabilities + lm_next_probabilities

        # probabilities, next_chars = next_probabilities.log_softmax(-1).topk(k=beam_size, axis=-1)
        probabilities, next_chars = next_probabilities.topk(k=beam_size, axis=-1)

        def decode_lm(outputs_ids):
            decoded_outputs = []
            for output_ids in outputs_ids.tolist():
                # transform id back to char IDs < 2 are simply transformed to ""
                # print(output_ids)
                decoded_outputs.append("".join([chr(x - 2) if x > 1 else str(x) for x in output_ids]))
            # print(decoded_outputs)
            return decoded_outputs

        def decode_char_model(outputs_ids):
            decoded_outputs = []
            for output_ids in outputs_ids.tolist():
                # transform id back to char IDs < 2 are simply transformed to ""
                # print(output_ids)
                decoded_outputs.append("".join([char_model.index2char[x] if x > 2 else char_model.index2char[x]+" " for x in output_ids]))
            # print(decoded_outputs)
            return decoded_outputs

        # if lm_model:
        #     # lm_next_probabilities = lm_next_probabilities[:, next_chars.squeeze()]
        #     # print("Next chars:", next_chars)
        #     # print("Next chars decoded:", decode_char_model(next_chars))
        #     # print("Char2LM next chars:", char2lm[next_chars])
        #     # print("Char2LM next chars decoded: ", decode_lm(char2lm[next_chars]))
        #     lm_next_probabilities = lm_next_probabilities[:, char2lm[next_chars].squeeze()]
        #     probabilities = probabilities + lm_next_probabilities
        #     probabilities, next_chars_indicies = probabilities.topk(k=beam_size, axis=-1)
        #     next_chars = next_chars[:, next_chars_indicies]

        result = result.repeat((beam_size, 1))
        next_chars = next_chars.reshape(-1, 1)
        result = torch.cat([result, next_chars], axis=-1)
        # print("inside loop:")

        for i in range(1, max_char_len):
            # print(result)
            next_probabilities = model_step(result, i).log_softmax(-1)
            if lm_model:
                # lm_next_probabilities = lm_weight * lm_model.step(result, i).log_softmax(-1)
                # lm_next_probabilities = lm_weight * lm_model.step(char2lm[result],i).log_softmax(-1)
                seq_len = result.shape[-1]
                # print(char2lm[result])
                # print(char2lm[result[:, 1:]])
                with torch.no_grad():
                    lm_next_probabilities = lm_model.generate(char2lm[result[:, 1:]], do_sample=False, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                lm_next_probabilities = lm_weight * lm_next_probabilities["scores"][0].log_softmax(-1)
            # next_probabilities = next_probabilities.reshape(-1, beam_size, next_probabilities.shape[-1])
            # next_probabilities = next_probabilities + lm_next_probabilities
            probabilities = probabilities.permute(1,0) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 0)
            probabilities, idx = probabilities.topk(k = beam_size, axis = -1)
            next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocab_size).long()

            if lm_model:
                # print("Ground truth:", decode_char_model(tgt)) 
                # print("Result till now:", decode_char_model(result))
                # print('\n\n')
                # print("LM next probabilities shape:", lm_next_probabilities.shape)
                lm_next_probabilities = lm_next_probabilities.flatten(start_dim=0)
                # print("IDX", idx)
                # print(idx.shape)
                # print("next chars:", next_chars)
                # print(next_chars.shape)
                # print("best_candidates:", best_candidates)
                # print(best_candidates.shape)
                lm_idx = char2lm[next_chars.squeeze()] + (best_candidates * 258)
                # print("LM idx:", lm_idx)
                # print(lm_idx.shape)
                # lm_next_probabilities = lm_next_probabilities[idx.squeeze()]

                # print("Next chars:", next_chars)
                # print("Next chars decoded:", decode_char_model(next_chars))
                # print("Char2LM next chars:", char2lm[next_chars])
                # print("Char2LM next chars decoded: ", decode_lm(char2lm[next_chars]))
                # print("Idx:", idx)
                # print("Best candidates:", best_candidates)
                # print("LM Idx:", lm_idx)
                # print("\n\n")

                lm_next_probabilities = lm_next_probabilities[lm_idx.squeeze()]
                probabilities = probabilities + lm_next_probabilities
                probabilities, next_chars_indicies = probabilities.topk(k=beam_size, axis=-1)
                # print(next_chars.shape)
                next_chars = next_chars[next_chars_indicies, :]
                best_candidates = best_candidates.unsqueeze(0)[:, next_chars_indicies.squeeze()]



            # best_candidates += torch.arange(result.shape[0] // beam_size, device=self.device).unsqueeze(-1) * beam_size
            result = result[best_candidates].flatten(end_dim=-2)
            # print(result.shape)
            # print(next_chars.shape)
            result = torch.cat([result, next_chars], dim=-1)

            eos_mask = result[:,-1]==eos_idx

            #If current best ends with eos put it in a memory
            if eos_mask[0].item():

                cur_candidate = result[0]
                cur_prob = probabilities[0]/(i**beam_penalty)


                if beam_memory_probabilites:
                    sorted_probs, _ = torch.stack(beam_memory_probabilites).squeeze().topk(k=len(beam_memory_probabilites), largest=False)
                    sorted_probs = sorted_probs.reshape(1, -1)
                    cur_lowest_prob = sorted_probs[0, 0].item()

                    if len(beam_memory_probabilites) < beam_size:
                            beam_memory_candidates.append(cur_candidate)
                            beam_memory_probabilites.append(cur_prob)

                    elif len(beam_memory_probabilites) >= beam_size:
                        if cur_lowest_prob > cur_prob:
                            beam_memory_probabilites = torch.stack(beam_memory_probabilites).squeeze()

                            r, i = beam_memory_probabilites.topk(k=beam_size)
                            return beam_memory_candidates[i[0]], r[0]

                        else:
                            beam_memory_candidates.append(cur_candidate)
                            beam_memory_probabilites.append(cur_prob)

                else:
                    beam_memory_candidates.append(cur_candidate)
                    beam_memory_probabilites.append(cur_prob)

            if i==max_char_len-1:
                probabilities = probabilities / (i**beam_penalty)

                if beam_memory_probabilites:
                    result = list(result) + beam_memory_candidates
                    probabilities = list(probabilities) + beam_memory_probabilites
                else:
                    result = list(result)
                    probabilities = list(probabilities)
                try:
                    probabilities = torch.stack(probabilities).squeeze()
                    r,i = probabilities.topk(k=beam_size)
                    if i.dim() == 0 and len(result)==1:
                        return result[0], r.item()

                    return result[i[0]], r[0]

                except Exception as e:
                    print(e)

                

            result_mask = result[:,-1] != eos_idx

            if result_mask.sum().item() == 0:
                probabilities = probabilities / (i**beam_penalty)

                if beam_memory_probabilites:
                    result = list(result) + beam_memory_candidates
                    probabilities = list(probabilities) + beam_memory_probabilites
                else:
                    result = list(result)
                    probabilities = list(probabilities)
                probabilities = torch.stack(probabilities).squeeze()

                r,i = probabilities.topk(k=beam_size)
                return result[i[0]], r[0]

            result = result[result_mask]
            probabilities = probabilities[result_mask].unsqueeze(0)
                


    def generate_next_beam(self, memory, result, current_score, beam_size):

        first_seq_len = memory.shape[1]
        tgt = torch.LongTensor([result]).to(self.device)
        tgt_mask = DatasetHelper.get_mask_seq_cat(first_seq_len, len(result)).to(self.device)
        tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))
        
        tgt = torch.cat([memory, tgt], axis=1)

        tgt = tgt.permute(1, 0, 2)

        output = self.transformer_decoder(
            src=tgt,
            mask=tgt_mask,
            src_key_padding_mask=None,  # to avoid working on padding
        )

        output = output.permute(1, 0, 2)

        output = output[:,first_seq_len:,:]
        # output = output[:,-1,:]

        output = self.linear(output)
        output = output.log_softmax(-1)
        output = self.softmax(output)
        output = output[0][-1]  # the last timestep

        values, indices = output.topk(k=beam_size)
        scores = current_score + torch.log(values)
        
        pred_token = indices.cpu().numpy().tolist()
        scores = scores.cpu().numpy().tolist()

        candidate_result = [result[:] + [i] for i in pred_token]
        candidate_scores = scores

        return candidate_result, candidate_scores

    def get_gt_score(self, memory, gt, sos_idx, eos_idx, max_char_len, alpha=0.7):

        first_seq_len = memory.shape[1]
        result = [sos_idx]
        scores  = 0

        for i in range(max_char_len):

            tgt = torch.LongTensor([result]).to(self.device)
            tgt_mask = DatasetHelper.get_mask_seq_cat(first_seq_len, len(result)).to(self.device)
            tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))
        
            tgt = torch.cat([memory, tgt], axis=1)

            tgt = tgt.permute(1, 0, 2)

            output = self.transformer_decoder(
                src=tgt,
                mask=tgt_mask,
                src_key_padding_mask=None,  # to avoid working on padding
            )

            output = output.permute(1, 0, 2)

            output = output[:,first_seq_len:,:]

            output = self.linear(output)
            output = self.softmax(output)
            output = output[0][-1]  # the last timestep

            actual_output_index = gt[0][i]
            actual_output_prob = output[actual_output_index]

            scores +=  torch.log(actual_output_prob)

            result.append(actual_output_index.item())

            if result[-1] == eos_idx:
                break

        scores = scores/len(result)**alpha
        scores = scores.item()

        return result, scores

    def beam_search(self, memory, sos_idx, eos_idx, max_char_len, beam_size=3, gt=None):
        alpha = 0.7
        result = [sos_idx]
        current_score = 0

        candidate_result, candidate_scores = self.generate_next_beam(memory, result, current_score, beam_size)

        final_result = []
        final_score = []

        terminated_result = []
        terminated_score = []

        for i in range(max_char_len-1):
            beam_result = []
            beam_scores = []

            for result, current_score in zip(candidate_result, candidate_scores):
                result, scores = self.generate_next_beam(memory, result, current_score, beam_size)
                beam_result.extend(result)
                beam_scores.extend(scores)
            
            scores, indices = torch.tensor(beam_scores).topk(k=beam_size)

            indices = indices.cpu().numpy().tolist()
            scores = scores.cpu().numpy().tolist()

            candidate_result = [beam_result[i] for i in indices]
            candidate_scores = [beam_scores[i] for i in indices]

            assert candidate_scores==scores

            index_to_remove = []
            for index, item  in enumerate(zip(candidate_result, candidate_scores)):
                result, score = item
                last_token = result[-1]
                if last_token == eos_idx and index==0: 
                    final_result.append(result)
                    final_score.append(score/len(result)**alpha)
                    index_to_remove.append(index)
                elif last_token == eos_idx:
                    terminated_result.append(result)
                    terminated_score.append(score/len(result)**alpha)
                    index_to_remove.append(index)

            # Remove hypothesis ending with <eos> token from branch of beam search
            candidate_result = [i for j, i in enumerate(candidate_result) if j not in index_to_remove]
            candidate_scores = [i for j, i in enumerate(candidate_scores) if j not in index_to_remove]

            # Stopping Criteria
            if not candidate_result:
                break

            if len(final_result) >= beam_size:
                memory_min_score_index = np.argmin(final_score)
                memory_min_score = final_score[memory_min_score_index]

                candidate_normalized_score = [s/len(candidate_result[i])**alpha for i,s in enumerate(candidate_scores)]
                candidate_max_score_index = np.argmax(candidate_normalized_score)
                candidate_max_score = candidate_normalized_score[candidate_max_score_index]

                if memory_min_score > candidate_max_score:
                    break
                else:
                    pass
              
            if i==max_char_len-2:
                for result, score in zip(candidate_result, candidate_scores):
                    final_result.append(result)
                    final_score.append(score/len(result)**alpha)
        

        final_result += terminated_result
        final_score += terminated_score

        _, index = torch.tensor(final_score).max(dim=0)

        if gt is not None:
            gt_seq, gt_score = self.get_gt_score(memory, gt, sos_idx, eos_idx, max_char_len)
            # return final_result[index], final_score[index], gt_seq, gt_score
            return final_result, final_score, gt_seq, gt_score

        # print("######################################")
        # for i, j in zip(final_result, final_score):
        #     print(i,j)
        # print("######################################")
        return final_result[index], final_score[index]
        # return final_result, final_score




def get_last_seq_pad_mask(second_seq_len=258, third_seq_len=258, bs=1):
        last_sequence_pad_mask = torch.zeros(second_seq_len+third_seq_len, dtype=torch.bool)
        last_sequence_pad_mask = last_sequence_pad_mask.repeat(bs, 1)
        return last_sequence_pad_mask

def get_mask_seq_cat(first_seq_len=130, second_seq_len=128):

        def add_additional_top_mask(mat_mask, mp):
            for seq in range(mat_mask.shape[0]):
                position = [random.randint(1,first_seq_len-2) for _ in range(mp)]

                for pos in position:
                    mat_mask[seq][pos] = float('-inf')
                       
            
            return mat_mask
        
        def add_additional_bottom_mask(mat_mask, mp):
            for seq in range(mat_mask.shape[0]):
                position = [random.randint(1, first_seq_len-2) for _ in range(mp)]
                position += [random.randint(first_seq_len+1, first_seq_len+second_seq_len-2) for _ in range(mp)]
                #For masking the text part code needs to here
                
                for pos in position:
                    mat_mask[seq][pos] = float('-inf')
                      
            return mat_mask


        second_mask = torch.triu(torch.full((second_seq_len, second_seq_len), float('-inf')), diagonal=1)
        first_mask = torch.zeros(second_seq_len, first_seq_len)

        bottom_mask = torch.cat([first_mask, second_mask], axis=-1)

        # top_mask = torch.clone(bottom_mask[0])
        # top_mask[first_seq_len] = float('-inf')
        # top_mask = top_mask.unsqueeze(0).repeat(first_seq_len,1)

        top_mask = bottom_mask[0].unsqueeze(0).repeat(first_seq_len,1)
        mask_ua = torch.cat([top_mask, bottom_mask], axis=0)
        return mask_ua

        # # Additional masking attention in image & text segment
        # bottom_mask = add_additional_bottom_mask(bottom_mask, int(0.1 * first_seq_len))
        # # top_mask = add_additional_top_mask(top_mask, int(0.1 * first_seq_len))
        # mask_a = torch.cat([top_mask, bottom_mask], axis=0)

        # if random.random() < 0.5:
        #     return mask_a
        # return mask_ua


def get_mask_seq_cat_batch(first_seq_len=130, second_seq_len=128, bs=1, no_of_heads=1):
    batch_mask = []
    for i in range(bs):
        mask = get_mask_seq_cat(first_seq_len, second_seq_len)
        mask = mask.unsqueeze(0).repeat(no_of_heads, 1, 1)
        batch_mask.append(mask)
    mask = torch.cat(batch_mask, axis=0)
    return mask
