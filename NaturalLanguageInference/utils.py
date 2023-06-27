from dataclasses import dataclass

@dataclass
class WNLI_Item:
    id: int
    sentence: str
    inference: str
    label: int
        

def get_data(fname):
    data:list[WNLI_Item] = []
    with open(fname) as fs:
        lines_after_first = fs.readlines()[1:]
        for line in lines_after_first:
            id, sent1, sent2, label = line.strip().split('\t')
            data.append(WNLI_Item(id, sent1, sent2, label))
    return data


