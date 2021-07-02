
class Tokenizer():
    def adjust_bert_tokens(self, tokenized_text:list, masked_index=5):
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        tokenized_text[masked_index] = '[MASK]'



class Juman_Tokenizer(Tokenizer):
    def __init__(self):
        from pyknp import Juman
        self.jumanpp = Juman()

    def toknize(self, text):
        result = self.jumanpp.analysis(text)
        print([mrph.midasi for mrph in result.mrph_list()])
        return [mrph.midasi for mrph in result.mrph_list()]
        # ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']


class Transformers_Tokenizer(Tokenizer):
    def __init__(self, model_name:str):
        if model_name == "T5":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            return T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
        else:
            from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
            if model_name == "BERT_kyoto":
                return AutoTokenizer.from_pretrained(
                            "Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt",
                            do_lower_case=False, do_basic_tokenize=False)
            elif model_name == "BERT":
                # return AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
                return AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        raise Exception(f"[Error] モデル：{model_name} は未調整です。")

    def toknize(self, text):
        result = self.jumanpp.analysis(text)
        print([mrph.midasi for mrph in result.mrph_list()])
        return [mrph.midasi for mrph in result.mrph_list()]
        # ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']
