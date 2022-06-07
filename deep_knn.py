import argparse
import random
import time

import numpy as np
import torchvision.transforms as transforms


import clip
from data.datasets import input_dataset_clip
from hoc import *
import global_var

# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pre_type", type=str, default='CLIP')  # image, cifar
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.6)
parser.add_argument('--noise_type', type=str, default='manual')  # manual
parser.add_argument('--dataset', type=str, help='cifar10, cifar100', default='cifar10')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--G', type=int, default=10, help='num of rounds (parameter G in Algorithm 1)')
parser.add_argument('--k', type=int, default=10, help='knn')
parser.add_argument('--cnt', type=int, default=15000, help='num of examples in each round')
parser.add_argument('--max_iter', type=int, default=400, help='num of iterations to get a T')
parser.add_argument("--local", default=False, action='store_true')
parser.add_argument('--loss', type=str, help='ce, fw', default='fw')
parser.add_argument('--label_file_path', type=str, help='the path of noisy labels',
                    default='./data/noise_label_human.pt')
parser.add_argument('--num_epoch', type=int, default=1, help='num of epochs')
parser.add_argument('--min_similarity', type=float, help='min_similarity', default=0.0)
parser.add_argument('--Tii_offset', type=float, help='Tii_offset', default=1.0)
parser.add_argument('--num_classes', type=int, default=10, help='num of classes')
parser.add_argument('--method', type=str, help='mv or rank1', default='rank1')


def set_model_min(config):
    # use resnet18 (pretrained with CIFAR-10). Only for the minimum implementation of HOC
    print(f'Use model {config.pre_type}')
    if config.pre_type == 'CLIP':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device)  # RN50, RN101, RN50x4, ViT-B/32
        return model, preprocess

    else:

        if  config.pre_type == 'image18':
            model = res_image.resnet18(pretrained=True)
        elif config.pre_type == 'image34':
            model = res_image.resnet34(pretrained=True)
        elif config.pre_type == 'image50':
            model = res_image.resnet50(pretrained=True)
        else:
            RuntimeError('Undefined pretrained model.')
        for param in model.parameters():
            param.requires_grad = False
        if 'image' in config.pre_type:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        model.to(config.device)
        return model, None


def data_transform(record, noise_or_not, sel_noisy):
    # assert noise_or_not is not None
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    noise_or_not_reorder = np.empty(total_len, dtype=bool)
    index_rec = np.zeros(total_len, dtype=int)
    cnt, lb = 0, 0
    sel_noisy = np.array(sel_noisy)
    noisy_prior = np.zeros(len(record))

    for item in record:
        for i in item:
            # if i['index'] not in sel_noisy:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            noise_or_not_reorder[cnt] = noise_or_not[i['index']] if noise_or_not is not None else False
            index_rec[cnt] = i['index']
            cnt += 1 - np.sum(sel_noisy == i['index'].item())
            # print(cnt)
        noisy_prior[lb] = cnt - np.sum(noisy_prior)
        lb += 1
    data_set = {'feature': origin_trans[:cnt], 'noisy_label': origin_label[:cnt],
                'noise_or_not': noise_or_not_reorder[:cnt], 'index': index_rec[:cnt]}
    return data_set, noisy_prior / cnt


def get_knn_acc_all_class(args, data_set, k=10, noise_prior=None, sel_noisy=None, thre_noise_rate=0.5, thre_true=None):
    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes

    all_point_cnt = data_set['feature'].shape[0]
    # global
    sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    noisy_label = data_set['noisy_label'][sample]
    noise_or_not_sample = data_set['noise_or_not'][sample]
    sel_idx = data_set['index'][sample]
    knn_labels_cnt = count_knn_distribution(args, final_feat, noisy_label, all_point_cnt, k=k, norm='l2')


    method = 'ce'
    # time_score = time.time()
    score = get_score(knn_labels_cnt, noisy_label, k=k, method=method, prior=noise_prior)  # method = ['cores', 'peer']
    # print(f'time for get_score is {time.time()-time_score}')
    score_np = score.cpu().numpy()

    if args.method == 'mv':
        # test majority voting
        print(f'Use MV')
        label_pred = np.argmax(knn_labels_cnt, axis=1).reshape(-1)
        sel_noisy += (sel_idx[label_pred != noisy_label]).tolist()
    elif args.method == 'rank1':
        print(f'Use rank1')
        print(f'Tii offset is {args.Tii_offset}')
        # fig=plt.figure(figsize=(15,4))
        for sel_class in range(KINDS):
            thre_noise_rate_per_class = 1 - min(args.Tii_offset * thre_noise_rate[sel_class][sel_class], 1.0)
            if thre_noise_rate_per_class >= 1.0:
                thre_noise_rate_per_class = 0.95
            elif thre_noise_rate_per_class <= 0.0:
                thre_noise_rate_per_class = 0.05
            sel_labels = (noisy_label.cpu().numpy() == sel_class)
            thre = np.percentile(score_np[sel_labels], 100 * (1 - thre_noise_rate_per_class))

            indicator_all_tail = (score_np >= thre) * (sel_labels)
            sel_noisy += sel_idx[indicator_all_tail].tolist()
    else:
        raise NameError('Undefined method')

    return sel_noisy

    # plot_score(score, name = f'{args.noise_type}_{args.noise_rate}_{k}_{method}', noise_or_not_sample = noise_or_not_sample, thre_noise_rate = thre_noise_rate, sel_class = sel_class)

    # method = 'avg'
    # score = get_score(knn_labels_cnt, noisy_label, k = k, method = method, prior = noise_prior) # method = ['cores', 'peer']
    # plot_score(score, name = f'{args.noise_type}_{args.noise_rate}_{k}_{method}', noise_or_not_sample = noise_or_not_sample)
    # method = 'new'
    # score = get_score(knn_labels_cnt, noisy_label, k = k, method = method, prior = noise_prior) # method = ['cores', 'peer']
    # plot_score(score, name = f'{args.noise_type}_{args.noise_rate}_{k}_{method}', noise_or_not_sample = noise_or_not_sample)
    # exit()


def get_T_global_min_new(args, data_set, max_step=501, T0=None, p0=None, lr=0.1, NumTest=50, all_point_cnt=15000):


    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes
    # NumTest = 50
    all_point_cnt = args.cnt
    print(f'Use {all_point_cnt} in each round. Total rounds {NumTest}.')

    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    # p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        # print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    args.device = set_device()
    loss_min, E_calc, P_calc, _ = calc_func(KINDS, p_estimate, False, args.device, max_step, T0, p0, lr=lr)
    E_calc = E_calc.cpu().numpy()
    P_calc = P_calc.cpu().numpy()
    return E_calc, P_calc


# def error(T, T_true):
#     error = np.sum(np.abs(T - T_true)) / np.sum(np.abs(T_true))
#     return error


def noniterate_detection(config, record, train_dataset, sel_noisy=[]):

    T_given_noisy_true = None
    T_given_noisy = None


    # non-iterate
    # sel_noisy = []
    data_set, noisy_prior = data_transform(record, train_dataset.noise_or_not, sel_noisy)
    # print(data_set['noisy_label'])
    if config.method == 'rank1':
        T_init = global_var.get_value('T_init')
        p_init = global_var.get_value('p_init')

        # print(f'T_init is {T_init}')
        T, p = get_T_global_min_new(config, data_set=data_set, max_step=config.max_iter if T_init is None else 20,
                                    lr=0.1 if T_init is None else 0.01, NumTest=config.G, T0=T_init, p0=p_init)


        T_given_noisy = T * p / noisy_prior
        print("T given noisy:")
        print(np.round(T_given_noisy, 2))
        # add randomness
        for i in range(T.shape[0]):
            T_given_noisy[i][i] += np.random.uniform(low=-0.05, high=0.05)


    sel_noisy = get_knn_acc_all_class(config, data_set, k=config.k, noise_prior=noisy_prior, sel_noisy=sel_noisy,
                                      thre_noise_rate=T_given_noisy, thre_true=T_given_noisy_true)

    sel_noisy = np.array(sel_noisy)
    sel_clean = np.array(list(set(data_set['index'].tolist()) ^ set(sel_noisy)))

    noisy_in_sel_noisy = np.sum(train_dataset.noise_or_not[sel_noisy]) / sel_noisy.shape[0]
    precision_noisy = noisy_in_sel_noisy
    recall_noisy = np.sum(train_dataset.noise_or_not[sel_noisy]) / np.sum(train_dataset.noise_or_not)


    print(f'[noisy] precision: {precision_noisy}')
    print(f'[noisy] recall: {recall_noisy}')
    print(f'[noisy] F1-score: {2.0 * precision_noisy * recall_noisy / (precision_noisy + recall_noisy)}')

    return sel_noisy, sel_clean, data_set['index']


if __name__ == "__main__":

    # Setup ------------------------------------------------------------------------
    torch.multiprocessing.set_sharing_strategy('file_system')
    config = parser.parse_args()
    config.device = set_device()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model_pre, preprocess = set_model_min(config)
    if config.noise_type in ['clean', 'worst', 'aggre', 'rand1', 'rand2', 'rand3', 'clean100', 'noisy100']:
        noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label',
                          'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3',
                          'clean100': 'clean_label', 'noisy100': 'noisy_label'}
        config.noise_type = noise_type_map[config.noise_type]

    # set transforms
    if config.dataset in ['cifar10', 'cifar100']:
        crop = transforms.RandomCrop(32, padding=4)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif config.dataset in ['stl10']:
        crop = transforms.RandomCrop(96, padding=12)
        normalize = transforms.Normalize((0.44671097, 0.4398105, 0.4066468), (0.2603405, 0.25657743, 0.27126738))
    else:
        raise NameError('Undefined dataset')

    if config.pre_type == "CLIP":
        preprocess_rand = transforms.Compose([crop,
                                              transforms.RandomHorizontalFlip(),
                                              preprocess])
    elif 'image' in config.pre_type:
        preprocess_rand = transforms.Compose([crop,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Resize(224),
                                              transforms.ToTensor(),
                                              normalize])
    else:  # cifar pretrain
        preprocess_rand = transforms.Compose([crop,
                                              transforms.Resize(32),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

    # load dataset
    train_dataset, _, num_classes, num_training_samples, _ = input_dataset_clip(config.dataset, config.noise_type,
                                                                                config.noise_rate,
                                                                                transform=preprocess_rand,
                                                                                noise_file=config.label_file_path)
    config.num_classes = num_classes
    config.num_training_samples = num_training_samples
    print(f'num_training_samples is {num_training_samples}')

    sel_noisy_rec = []
    # for config.cnt in [5000, 15000, 50000]:
    for loop_i in range(1):
        train_dataloader_EF = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=256,
                                                          shuffle=True,
                                                          num_workers=4,
                                                          drop_last=False)
        model_pre.eval()

        sel_clean_rec = np.zeros((config.num_epoch, num_training_samples))
        sel_times_rec = np.zeros(num_training_samples)
        global_var._init()
        global_var.set_value('T_init', None)
        global_var.set_value('p_init', None)
        for epoch in range(config.num_epoch):
            print(f'Epoch {epoch}')

            record = [[] for _ in range(config.num_classes)]

            for i_batch, (feature, label, index) in enumerate(train_dataloader_EF):
                feature = feature.to(config.device)
                label = label.to(config.device)
                with torch.no_grad():
                    if config.pre_type == "CLIP":
                        extracted_feature = model_pre.encode_image(feature)
                    elif 'ssl' in config.pre_type:
                        extracted_feature, _ = model_pre(feature)
                    else:
                        extracted_feature, _ = model_pre(feature)
                for i in range(extracted_feature.shape[0]):
                    record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})
                if i_batch > 200:
                    break

            time1 = time.time()

            if config.method == 'both':
                # rank1 + mv
                config.method = 'rank1'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(config, record, train_dataset,
                                                                     sel_noisy=sel_noisy_rec.copy())
                sel_clean_rec[epoch][np.array(sel_clean)] += 0.5
                sel_times_rec[np.array(sel_idx)] += 0.5
                config.method = 'mv'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(config, record, train_dataset,
                                                                     sel_noisy=sel_noisy_rec.copy())
                sel_clean_rec[epoch][np.array(sel_clean)] += 0.5
                config.method = 'both'
                sel_times_rec[np.array(sel_idx)] += 0.5
            else:
                # use one method
                sel_noisy, sel_clean, sel_idx = noniterate_detection(config, record, train_dataset,
                                                                     sel_noisy=sel_noisy_rec.copy())
                if config.num_epoch > 1:
                    sel_clean_rec[epoch][np.array(sel_clean)] = 1
                    sel_times_rec[np.array(sel_idx)] += 1


            print(f'Time for one detection is {time.time() - time1}')

            # config.method = 'rank1'
            if epoch % 1 == 0:
                # config.method = 'mv'
                aa = np.sum(sel_clean_rec[:epoch + 1], 0) / sel_times_rec
                nan_flag = np.isnan(aa)
                aa[nan_flag] = 0
                # aa += 0.1

                sel_clean_summary = np.round(aa).astype(bool)
                sel_noisy_summary = np.round(1.0 - aa).astype(bool)
                sel_noisy_summary[nan_flag] = False
                print(
                    f'We find {sel_clean_summary.shape[0] - np.sum(sel_clean_summary) - np.sum(nan_flag * 1)} corrupted instances from {sel_clean_summary.shape[0] - np.sum(nan_flag * 1)} instances')

                # noisy
                noisy_in_sel_noisy = np.sum(train_dataset.noise_or_not[sel_noisy_summary]) / np.sum(sel_noisy_summary)
                precision_noisy = noisy_in_sel_noisy
                recall_noisy = np.sum(train_dataset.noise_or_not[sel_noisy_summary]) / np.sum(
                    train_dataset.noise_or_not[(1 - nan_flag).astype(bool)])

                print(f'[Epoch {epoch + 1}] precision noisy: {precision_noisy}')
                print(f'[Epoch {epoch + 1}] recall noisy: {recall_noisy}')
                print(
                    f'[Epoch {epoch + 1}] F1-score noisy: {2.0 * precision_noisy * recall_noisy / (precision_noisy + recall_noisy)}')
                torch.save(sel_clean_rec,
                           f'result_{config.pre_type}_{config.method}_{config.dataset}_{config.noise_type}_e{config.num_epoch}_k{config.k}.pt')
