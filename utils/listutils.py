

def list_to_str(input_list, delimiter=','):
    return delimiter.join([x if type(x) == str else repr(x) for x in input_list])  # list comprehension


def list_to_csv_str(input_list):
    return list_to_str(input_list)  # list comprehension


def fill_list_with_new_values(original_list, new_list_values):
    for i in xrange(len(new_list_values)):
        type1 = type(original_list[i])
        if type1 == int:
            original_list[i] = int(new_list_values[i])
        elif type1 == float:
            original_list[i] = float(new_list_values[i])
        else:
            original_list[i] = new_list_values[i]


def list_to_svm_line(original_list):
    list_size = len(original_list)
    svm_line = '%s ' % original_list[0]
    for i in xrange(1, list_size):
        #svm_line += '{}:{} '.format(i, repr(original_list[i]))
        svm_line += '{}:{} '.format(i, original_list[i])
    return svm_line.rstrip()
