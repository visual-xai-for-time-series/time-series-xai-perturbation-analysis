from tqdm import tqdm

import numpy as np
import scipy as sp

import torch



def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points in N-dimensional space.
    
    Args:
    p1 (numpy.ndarray): An array representing the first point.
    p2 (numpy.ndarray): An array representing the second point.
    
    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum(np.square(p1 - p2)))


def cosine_distance(v1, v2):
    """
    Calculate the cosine distance between two vectors.
    
    Args:
    v1 (numpy.ndarray): An array representing the first vector.
    v2 (numpy.ndarray): An array representing the second vector.
    
    Returns:
    float: The cosine distance between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return 1 - (dot_product / (norm_v1 * norm_v2))


def copy_data(d):
    if isinstance(d, np.ndarray):
        return d.copy()
    elif isinstance(d, torch.Tensor):
        return d.clone()
    return None


def deletion(time_series_sample, attribution_values, threshold, value=0):
    tmp_ts = copy_data(time_series_sample)
    mask = 1 - (attribution_values > threshold).astype(int)
    return tmp_ts * mask + (1 - mask) * value, 1 - mask


def deletion_sequence(time_series_sample, attribution_values, threshold, value=0, length=1):
    tmp_ts = copy_data(time_series_sample)

    mask = np.ones(tmp_ts.shape).astype(int)
    idc_masks = np.argwhere(attribution_values > threshold)
    idc_masks_with_around = np.clip(np.unique(np.concatenate([idc_masks - (l - int(length / 2)) for l in range(length)])), 0, len(tmp_ts) - 1)
    mask[idc_masks_with_around] = 0

    return tmp_ts * mask + (1 - mask) * value, 1 - mask


def deletion_to_inverse(**kwargs):
    if 'ts' in kwargs:
        return 0 - kwargs['ts'] + np.mean(kwargs['ts']) # inverse for zero centered + fix for non-zero centered
    else:
        return 0


def delete_to_zero(**kwargs):
    return 0


def delete_to_mean(**kwargs):
    if 'ts' in kwargs:
        return np.mean(kwargs['ts'])
    else:
        return 0


def delete_to_global_mean(**kwargs):
    if 'gts' in kwargs:
        return np.mean(kwargs['gts'], axis=0)
    else:
        return 0


def delete_to_global_max(**kwargs):
    if 'gts' in kwargs:
        return np.max(kwargs['gts'])
    else:
        return 0


def delete_to_global_min(**kwargs):
    if 'gts' in kwargs:
        return np.min(kwargs['gts'])
    else:
        return 0
    
    
def random_min_max(**kwargs):
    if 'gts' in kwargs:
        min_ = np.min(kwargs['gts'])
        max_ = np.max(kwargs['gts'])
        return np.random.uniform(low=min_, high=max_)
    else:
        return 0
    
    
def ood_h(**kwargs):
    if 'gts' in kwargs:
        min_ = np.min(kwargs['gts'])
        max_ = np.max(kwargs['gts'])
        return kwargs['ts'] * max(max_, min_)
    else:
        return 0
    
    
def ood_l(**kwargs):
    if 'gts' in kwargs:
        min_ = np.min(kwargs['gts'])
        max_ = np.max(kwargs['gts'])
        return -1 * kwargs['ts'] * max(max_, min_)
    else:
        return 0


values = [
    # single time point
    [delete_to_zero, 1],
    [delete_to_mean, 1],
    [delete_to_global_mean, 1],
    [deletion_to_inverse, 1],
    [delete_to_global_max, 1],
    [delete_to_global_min, 1],
    [random_min_max, 1],
    [ood_h, 1],
    [ood_l, 1],
    # subsequence
    [delete_to_zero, 5],
    [delete_to_mean, 5],
    [delete_to_global_mean, 5],
    [deletion_to_inverse, 5],
    [random_min_max, 5],
    [ood_h, 5],
    [ood_l, 5],
]


def perturbation_analysis(attributions, data, labels, model, device, base_dir):

    overall_results = {}
    
    att_values = tqdm(attributions, leave=False, desc=f'Start with {" ":<25s}')
    for cur_attr_name in att_values:
        att_values.set_description(f'Start with {cur_attr_name:<25s}')

        raw_data = data.copy()
        raw_label = labels.copy()

        cur_attr = attributions[cur_attr_name].copy().reshape(raw_data.shape)

        cur_attr_name = cur_attr_name.lower().replace(' ', '_')
        cur_dir = f'{base_dir}/method-{cur_attr_name}'

        overall_results[cur_attr_name] = {}

        deletion_value = values[0]
        deletion_value_fnc, deletion_length = deletion_value
        name = deletion_value_fnc.__name__ + ' ' + str(deletion_length)
        deletion_strategy = tqdm(values, leave=False, desc=f'Start with {name:<25s}')
        for deletion_value in deletion_strategy:

            deletion_value_fnc, deletion_length = deletion_value
            name = deletion_value_fnc.__name__ + ' ' + str(deletion_length)
            attribution = cur_attr_name

            deletion_strategy.set_description(f'Start with {name:<25s}')

            results = {'changed': [], 'unchanged': []}
            for x in range(len(raw_data)):

                ts = raw_data[x]
                l = raw_label[x]
                a = cur_attr[x]

                ts_new = torch.from_numpy(ts).reshape(1, 1, -1).float().to(device)
                org_pred = model(ts_new).cpu().detach().numpy().flatten()
                org_pl = np.argmax(org_pred)

                value = deletion_value_fnc(ts=ts, gts=raw_data)

                for i in range(75):
                    i += 1
                    ts_new, deleted = deletion_sequence(ts, a, np.percentile(a, 100 - i), value, deletion_length)
                    ts_new_t = torch.from_numpy(ts_new).reshape(1, 1, -1).float().to(device)
                    pred = model(ts_new_t).cpu().detach().numpy().flatten()
                    pl = np.argmax(pred)
                    if org_pl != pl:
                        ts_euc_dist = euclidean_distance(ts, ts_new)
                        ts_cos_dist = cosine_distance(ts, ts_new)
                        single_result = {
                            'index': x,
                            'stats': {
                                'skew': sp.stats.skew(a),
                                'mean': np.mean(a),
                                'variance': np.var(a),
                                'standard_deviation': np.std(a),
                            },
                            'dist': {
                                'euclidean': ts_euc_dist,
                                'cosine': ts_cos_dist,
                            },
                            'perturbed_percentile': i,
                            'perturbed_values': np.sum(deleted),
                            'old_label': org_pl,
                            'new_label': pl,
                            'label': l,
                            'old_prediction': org_pred,
                            'new_prediction': pred,
                        }
                        results['changed'].append(single_result)
                        break

                if org_pl == pl:
                    ts_euc_dist = euclidean_distance(ts, ts_new)
                    ts_cos_dist = cosine_distance(ts, ts_new)
                    single_result = {
                        'index': x,
                        'stats': {
                            'skew': sp.stats.skew(a),
                            'mean': np.mean(a),
                            'variance': np.var(a),
                            'standard_deviation': np.std(a),
                        },
                        'dist': {
                            'euclidean': ts_euc_dist,
                            'cosine': ts_cos_dist,
                        },
                        'perturbed_percentile': 0,
                        'perturbed_values': 0,
                        'old_label': org_pl,
                        'new_label': pl,
                        'label': l,
                        'old_prediction': org_pred,
                        'new_prediction': pred,
                    }
                    results['unchanged'].append(single_result)

            overall_results[cur_attr_name][name] = results

    return overall_results
