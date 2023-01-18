
import sys
import numpy as np

def sample_neg_items_for_u(u, num, n_items, train_user_dict):
    neg_items = []
    while True:
        if len(neg_items) == num: break
        neg_i_id = str(np.random.randint(low=0, high=n_items, size=1)[0])

        if neg_i_id not in train_user_dict[u] and neg_i_id not in neg_items:
            neg_items.append(neg_i_id)
    return neg_items

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    train = int(sys.argv[3])
    n_item = 0
    user_item_dict = {}
    with open(input_file) as f:
        for line in f:
            parts = line.strip('\n').split(' ')
            if len(parts) <= 1:
                continue
            user = parts[0]
            for i in range(1, len(parts)):
                item = parts[i]
                # f.write("%s\n" % " ".join([user, item, "1"]))
                user_item_dict.setdefault(user, set())
                user_item_dict[user].add(item)
                n_item = max(n_item, int(item))
    n_item += 1

    with open(input_file) as f, open(output_file, "w") as f2:
        for line in f:
            parts = line.strip('\n').split(' ')
            if len(parts) <= 1:
                continue
            user = parts[0]
            for i in range(1, len(parts)):
                item = parts[i]
                f2.write("%s\n" % " ".join([user, item, "1"]))
                if train == 1:
                    neg_item = sample_neg_items_for_u(user, 1, n_item, user_item_dict)
                    f2.write("%s\n" % " ".join([user, neg_item[0], "0"]))
