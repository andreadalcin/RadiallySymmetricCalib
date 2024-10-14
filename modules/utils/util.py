import os

def join_paths(*args):
    assert len(args) > 1

    base = args[0]

    for path in args[1:]:
        base = os.path.join(base, path)

    return base

def remove_indxs_from_list(list: list, indxs:list)->None:
    for idx in sorted(indxs, reverse=True):
        del list[idx]

if __name__=="__main__":
    print(join_paths("asd","2","3","4"))