from copy import deepcopy
from Util.genericUtil import interval_overlap, sortOutTypePair
from functools import reduce


base_kern_interp_order = dict(zip(['PER', 'SE', 'C', 'WN', 'LIN', 'Sr', 'S', 'SIr', 'SI'], range(10)))
def order_base_kerns(bs): return sorted(bs, key = lambda b: base_kern_interp_order[b])


def base_factors_interpretation(bs_params):
    bps_copy = deepcopy(bs_params)
    var = bps_copy['ProductKE'][0]['variance']
    description_end = '. The overall variance of this component is ' + var
    del bps_copy['ProductKE']

    ordered_ps = sorted(bps_copy.items(), key = lambda bps: base_kern_interp_order[bps[0]])
    description = first_term_interpretation(ordered_ps[0])
    for bps in ordered_ps[1:]: description += ', ' + postmodifier_interpretation(bps)

    return description + description_end


def first_term_interpretation(bps):
    b, ps = bps
    res = ''
    if b == 'WN': res = 'Uncorrelated noise'
    elif b == 'C': res = 'A constant'
    elif b == 'SE': res = 'A smooth function with lengthscale {:.2f}'.format(ps[0]['lengthscale'])
    elif b == 'PER':
        if len(ps) > 1:
            ps_by_desc_period = sorted(ps, key = lambda p: p['period'], reverse = True)
            res = 'A periodic function with main period {:.2f} and lengthscale {:.2f}, then '.format(ps_by_desc_period[0]['period'], ps_by_desc_period[0]['lengthscale'])
            res += 'modulated by another periodic function of ' if len(ps) == 2 else f'repeatedly modulated by {len(ps)-1} more periodic functions (by decreasing period) of '
            further_periodics = ['period {:.2f} and lengthscale {:.2f}'.format(p['period'], p['lengthscale']) for p in ps_by_desc_period[1:]]
            if len(ps) > 2: further_periodics[-1] = 'and ' + further_periodics[-1]
            res += ', '.join(further_periodics)
        else: res = 'A periodic function with period {:.2f} and lengthscale {:.2f}'.format(ps[0]['period'], ps[0]['lengthscale'])
    elif b == 'LIN':
        if len(ps) > 1:
            res = 'A polynomial function of order {:d} with offsets '.format(len(ps))
            offsets = ['{:.2f}'.format(p['offset']) for p in ps]
            if len(ps) > 1: offsets[-1] = 'and ' + offsets[-1]
            res += ', '.join(offsets)
        else: res = 'A linear function with offset {:.2f}'.format(ps[0]['offset'])
    else: raise ValueError(f'An unexpected type of first term in a pure product has arisen: {b}')
    return res


# Functions to apply to each single sigmoid type ps if multiple ones present
def Sr_overlap(Sr_params):
    end = sorted(Sr_params, key = lambda p: p['location'])[0]
    return {'end': end['location'], 'end_slope': end['slope']}

def S_overlap(S_params):
    start = sorted(S_params, key = lambda p: p['location'], reverse = True)[0]
    return {'start': start['location'], 'start_slope': start['slope']}

def SI_ps_left(SI_p): return SI_p['location'] - SI_p['width'] / 2
def SI_ps_right(SI_p): return SI_p['location'] + SI_p['width'] / 2
def SI_overlap(SI_params):
    start = max(SI_params, key = SI_ps_left)
    end = min(SI_params, key = SI_ps_right)
    interval = {'start': SI_ps_left(start), 'end': SI_ps_right(end), 'start_slope': start['slope'], 'end_slope': end['slope']}
    return interval if interval['end'] > interval['start'] else None

def get_SIr_intervals(STr_p): return [{'end': SI_ps_left(STr_p), 'end_slope': STr_p['slope']}, {'start': SI_ps_right(STr_p), 'start_slope': STr_p['slope']}]
def SIr_overlap_step(acc, p): # Acc is a list of dictionaries representing intervals
    p_end = SI_ps_left(p)
    p_start = SI_ps_right(p)
    new_acc = []
    for a in acc:
        if 'start' not in a: # Intersect a left-open semi-line with a line without a segment
            if p_end < a['end']:
                new_acc.append({'end': p_end, 'end_slope': p['slope']})
                if p_start < a['end']: new_acc.append({'start': p_start, 'end': a['end'], 'start_slope': p['slope'], 'end_slope': a['end_slope']})
            else: new_acc.append(a)
        elif 'end' not in a: # Intersect a right-open semi-line with a line without a segment
            if p_start > a['start']:
                if p_end > a['start']: new_acc.append({'start': a['start'], 'end': p_end, 'start_slope': a['start_slope'], 'end_slope': p['slope']})
                new_acc.append({'start': p_start, 'start_slope': p['slope']})
            else: new_acc.append(a)
        else: # Intersect a segment with a line without a segment
            if a['start'] < p_end:
                if a['end'] < p_end: new_acc.append(a)
                else: new_acc.append({'start': a['start'], 'end': p_end, 'start_slope': a['start_slope'], 'end_slope': p['slope']})
            if a['end'] > p_start:
                if a['start'] > p_start: new_acc.append(a)
                else: new_acc.append({'start': p_start, 'end': a['end'], 'start_slope': p['slope'], 'end_slope': a['end_slope']})
    return new_acc
def SIr_overlap(SIr_params): return reduce(SIr_overlap_step, SIr_params[1:], get_SIr_intervals(SIr_params[0]))



# # Intersect two DIFFERENT TYPES of sigmoid parameters (apply after reducing same type ones)
# def simplify_sigmoidals(sig_bps1, sig_bps2):
#     b1, p1 = sig_bps1
#     b2, p2 = sig_bps2
#     pair = {b1: p1, b2: p2}
#     if 'S' in pair:
#         if 'Sr' in pair:
#
#         elif 'SI' in pair:
#
#         else: # elif 'SIr' in pair:
#
#     elif 'Sr' in pair:
#         if 'SI' in pair:
#
#         else: # elif 'SIr' in pair:
#
#     else: # elif 'SI' in pair and 'SIr' in pair:
#
#     return res





def postmodifier_interpretation(bps):
    b, ps = bps
    res = ''
    if b == 'SE': res = 'whose shape changes smoothly with lengthscale {:.2f}'.format(ps[0]['lengthscale'])
    elif b == 'LIN':
        if len(ps) > 1:
            res = 'with polynomially varying amplitude of order {:d} with offsets '.format(len(ps))
            offsets = ['{:.2f}'.format(p['offset']) for p in ps]
            if len(ps) > 1: offsets[-1] = 'and ' + offsets[-1]
            res += ', '.join(offsets)
        else: res = 'with linearly varying amplitude with offset {:.2f}'.format(ps[0]['offset'])
    elif b == 'Sr':
        first_location = sorted(ps, key = lambda p: p['location'])[0]
        res = 'which applies from {:.2f} (with change slope {:.f})'.format(first_location['location'], first_location['slope'])
    elif b == 'S':
        last_location = sorted(ps, key = lambda p: p['location'], reverse = True)[0]
        res = 'which applies until {:.2f} (with change slope {:.f})'.format(last_location['location'], last_location['slope'])
    elif b == 'SIr':
        if len(ps) > 1:
            interval = reduce(interval_overlap_step, ps[1:], (ps[0], ps[0]))
            'which applies before {:.2f} and after {:.2f}'.format(last_location['location'], last_location['slope'])
        else: res = 'which applies before {:.2f} and after {:.2f}'.format(ps[0]['location'] - ps[0]['width'] / 2, ps[0]['location'] + ps[0]['width'] / 2)
    elif b == 'SI':
        if len(ps) > 1:
            reduce(interval_overlap, [])
            'which applies before {:.2f} and after {:.2f}'.format(last_location['location'], last_location['slope'])
        else: res = 'which applies between {:.2f} and {:.2f}'.format(ps[0]['location'] - ps[0]['width'] / 2, ps[0]['location'] + ps[0]['width'] / 2)
    else: raise ValueError(f'An unexpected type of postmodifier term in a pure product has arisen: {b}')
    return res
