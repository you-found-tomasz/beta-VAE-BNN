import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import torch
import nltk

def frequencies(x_short_hist):
    result1_zeros, result2_zeros, result3_zeros, result4_zeros, sequences_zeros = np.zeros(20), np.zeros(20), np.zeros(
        20), np.zeros(20), np.zeros(200)
    result1 = np.where(x_short_hist == 1)
    result1 = (np.diff(result1[0], prepend=0))[:20]
    result1_zeros[:result1.shape[0]] = result1
    result2 = np.where(x_short_hist == 2)
    result2 = (np.diff(result2[0], prepend=0))[:20]
    result2_zeros[:result2.shape[0]] = result2
    result3 = np.where(x_short_hist == 3)
    result3 = (np.diff(result3[0], prepend=0))[:20]
    result3_zeros[:result3.shape[0]] = result3
    result4 = np.where(x_short_hist == 4)
    result4 = (np.diff(result4[0], prepend=0))[:20]
    result4_zeros[:result4.shape[0]] = result4
    frequencies = np.concatenate([result1_zeros, result2_zeros, result3_zeros, result4_zeros])

    return frequencies

def bleu(sequences):
    ref1_clockwise = [
        '1 2 3'.split(),
        '3 1 2'.split(),
        '2 3 1'.split(),
    ]
    ref1_cclockwise = [
        '1 3 2'.split(),
        '2 1 3'.split(),
        '3 2 1'.split(),
    ]
    ref2_clockwise = [
        '2 4 3'.split(),
        '3 2 4'.split(),
        '4 3 2'.split(),
    ]
    ref2_cclockwise = [
        '2 3 4'.split(),
        '3 4 2'.split(),
        '4 2 3'.split(),
    ]
    ref3_clockwise = [
        '1 2 4'.split(),
        '2 4 1'.split(),
        '4 1 2'.split(),
    ]
    ref3_cclockwise = [
        '1 4 2'.split(),
        '4 2 1'.split(),
        '2 1 4'.split(),
    ]
    ref4_clockwise = [
        '4 3 1'.split(),
        '3 1 4'.split(),
        '1 4 3'.split(),
    ]
    ref4_cclockwise = [
        '1 3 4'.split(),
        '3 4 1'.split(),
        '4 1 3'.split(),
    ]

    #weights = (0, 0.25, 0.5, 0)
    weights = (0, 0.25, 0.25, 0)
    chencherry = nltk.translate.bleu_score.SmoothingFunction(epsilon=1e-3)
    test04 = (str(sequences)[1:-1]).split()
    bleu1c = sentence_bleu(ref1_clockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu1cc = sentence_bleu(ref1_cclockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu2c = sentence_bleu(ref2_clockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu2cc = sentence_bleu(ref2_cclockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu3c = sentence_bleu(ref3_clockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu3cc = sentence_bleu(ref3_cclockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu4c = sentence_bleu(ref4_clockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu4cc = sentence_bleu(ref4_cclockwise, test04, weights=weights)#,smoothing_function=chencherry.method1)
    bleu_array = np.array([bleu1c, bleu1cc, bleu2c, bleu2cc, bleu3c, bleu3cc, bleu4c, bleu4cc])

    return bleu_array

def spike_counter(x):
    spike_mask = x > 0
    spike_count_class_0 = torch.count_nonzero(spike_mask[:,0,:], dim=1)
    spike_count_class_1 = torch.count_nonzero(spike_mask[:,1,:], dim=1)
    spike_count_class_2 = torch.count_nonzero(spike_mask[:,2,:], dim=1)
    spike_count_class_3 = torch.count_nonzero(spike_mask[:,3,:], dim=1)
    spike_count_class = spike_count_class_0+spike_count_class_1+spike_count_class_2+spike_count_class_3
    return spike_count_class, spike_count_class_0, spike_count_class_1, spike_count_class_2, spike_count_class_3

def continous_spike_counter(x):

    time_tensor = torch.arange(0,x.shape[1],1)
    x_normalized = torch.nan_to_num((x-torch.min(x, dim=1)[0][:, None])/(torch.max(x, dim=1)[0]-torch.min(x, dim=1)[0])[:, None])
    x_time_weighted = (time_tensor[None,:]*x_normalized)

    correlation_01 = np.corrcoef(x_time_weighted[0,:], x_time_weighted[1,:], rowvar=False)[0, 1]
    correlation_02 = np.corrcoef(x_time_weighted[0,:], x_time_weighted[2,:], rowvar=False)[0, 1]
    correlation_03 = np.corrcoef(x_time_weighted[0,:], x_time_weighted[3,:], rowvar=False)[0, 1]
    correlation_12 = np.corrcoef(x_time_weighted[1,:], x_time_weighted[2,:], rowvar=False)[0, 1]
    correlation_13 = np.corrcoef(x_time_weighted[1,:], x_time_weighted[3,:], rowvar=False)[0, 1]
    correlation_23 = np.corrcoef(x_time_weighted[2,:], x_time_weighted[3,:], rowvar=False)[0, 1]
    correlation_array = np.array([correlation_01, correlation_02,correlation_03, correlation_12, correlation_13, correlation_23])

    x_time_weighted_sum = torch.nan_to_num(torch.sum((time_tensor[None,:]*x_normalized),dim=1)/ torch.sum(x_normalized,dim=1)).numpy()

    x_cum_sum_spike = np.cumsum(x_normalized, axis=1)//1
    x_cum_sum_spike = np.diff(x_cum_sum_spike, axis=1)
    x_cum_sum_spike[x_cum_sum_spike>0] = 1

    crossentropy_01 = torch.nn.functional.binary_cross_entropy(x_normalized[0,:], x_normalized[1,:]).numpy()
    crossentropy_02 = torch.nn.functional.binary_cross_entropy(x_normalized[0,:], x_normalized[2,:]).numpy()
    crossentropy_03 = torch.nn.functional.binary_cross_entropy(x_normalized[0,:], x_normalized[3,:]).numpy()
    crossentropy_12 = torch.nn.functional.binary_cross_entropy(x_normalized[1,:], x_normalized[2,:]).numpy()
    crossentropy_13 = torch.nn.functional.binary_cross_entropy(x_normalized[1,:], x_normalized[3,:]).numpy()
    crossentropy_23 = torch.nn.functional.binary_cross_entropy(x_normalized[2,:], x_normalized[3,:]).numpy()

    crossentropy_array = np.array([crossentropy_01, crossentropy_02, crossentropy_03, crossentropy_12, crossentropy_13, crossentropy_23])

    spike_count_class, spike_count_class_0, spike_count_class_1, spike_count_class_2, spike_count_class_3 = spike_counter(torch.from_numpy(x_cum_sum_spike[None, :]))
    spike_count_array = np.array([spike_count_class_0.numpy(), spike_count_class_1.numpy(), spike_count_class_2.numpy(), spike_count_class_3.numpy()])

    return correlation_array, x_time_weighted_sum, crossentropy_array, spike_count_array

def spike_counter_one_line(x):
    spike_mask = x > 0
    spike_count_class_0 = torch.count_nonzero(spike_mask[0:1, :], dim=0)[:2000]
    spike_count_class_1 = torch.count_nonzero(spike_mask[1:2, :], dim=0)[:2000]
    spike_count_class_2 = torch.count_nonzero(spike_mask[2:3, :], dim=0)[:2000]
    spike_count_class_3 = torch.count_nonzero(spike_mask[3:4, :], dim=0)[:2000]
    x_short_hist = (spike_count_class_0 + spike_count_class_1 * 2 + spike_count_class_2 * 3 + spike_count_class_3 * 4).unsqueeze(dim=1)
    x_short_hist = x_short_hist.cpu().numpy()

    return x_short_hist

def x_hat_plotter(x, x_hat, device):

    x = torch.kron(x, torch.ones((5, 5), device=device))
    x_hat_mean = x_hat.reshape(x.shape[0], 4, -1)
    x_hat = torch.kron(x_hat, torch.ones((5, 5), device=device))
    image_concate = torch.cat((x, x_hat), axis=0)
    return image_concate

def y_decoder(y):
    y_decoded1 = (y // 25)
    y_decoded2 = ((y // 5) % 5)
    y_decoded3 = (y % 5)
    y32 = torch.stack([y_decoded1, y_decoded2, y_decoded3], dim=0).permute(1, 0)
    y_decoded = torch.nn.functional.one_hot(y32, num_classes = 5).reshape(y.shape[0], -1).float()
    return y_decoded

def parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--logging_string', type=str, default='default')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lbb', type=bool, default=False)
    parser.add_argument('--cluster', type=bool, default=False)
    parser.add_argument('--network_enc', type=str, default="deep")
    parser.add_argument('--network_dec', type=str, default="deep")
    parser.add_argument('--data', type=str, default="raw", help="raw or other(for different location)")
    parser.add_argument('--raytune', type=str, default="False", help="ray tune enabled?")
    parser.add_argument('--number_classes', type=int, default=125, help="3 or 125")
    parser.add_argument('--images', type=str, default="True", help="tensorboard images")
    parser.add_argument('--seq_length', type=int, default=4000, help="3 or 125")
    parser.add_argument('--network_config', type=str, default="config_files/dense_dense/6ba7d_00151_best_4000.csv", help="beta for VAE")
    parser.add_argument('--max_epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--gpu_number', type=str, default="7", help="cluster number")
    parser.add_argument('--beta', type=float, default=1, help="beta for VAE")
    parser.add_argument('--checkpoint_directory', type=str, default='/trained_model', help="beta for VAE")
    parser.add_argument('--data_location', type=str, default="example/", help="should be data_frames/")
    parser.add_argument('--alpha', type=float, default=0.00, help="spike loss")
    parser.add_argument('--tanh', type=str, default="False", help="tanh after decodeer")
    parser.add_argument('--y_decoder_only', type=bool, default=False, help="only decoder?")
    parser.add_argument('--y_encoder_only', type=bool, default=False, help="only encoder?")
    parser.add_argument('--model', type=str, default="baseline125_beta1_z3_x_single", help="model name")
    #parser.add_argument('--model', type=str, default="baseline125_beta1_z3_x_single_x_hat_single", help="model name")
    parser.add_argument('--data_size', type=float, default="0.01", help="how much data to be analysed")

    args = parser.parse_args()
    print(args)

    return args