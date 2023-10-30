import sys


def pprint_summary(title: str, file=sys.stdout, n_spliter=15, **kv_pairs):
    """
    prettry print summary for parameters, the style is like
    =====title=====
    key1: 	 abc
    key2: 	 123
    key3: 	 0.456
    ===============
        Args:
            title (str): title of the summary
            file (_type_, optional): The file argument must be an object with
                a write(string) method; if it is not present or None,
                sys.stdout will be used. Since printed arguments are converted to
                text strings, print() cannot be used with binary mode file objects.
                For these, use file.write(...) instead.
            n_spliter (int, optional): # of spliter in separate line
            kv_pairs (dict): the key-value pair that will be displayed in the summary
    """
    print("="*n_spliter+title+"="*n_spliter, file=file)
    for key in kv_pairs:
        print("%15s:\t%s" % (key, str(kv_pairs[key])), file=file)

    print("="*(n_spliter*2+len(title)), file=file)
