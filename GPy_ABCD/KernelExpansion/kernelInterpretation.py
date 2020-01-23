from copy import deepcopy
from functools import reduce
from operator import add

from GPy_ABCD.KernelExpansion.kernelOperations import base_sigmoids


base_kern_interp_order = dict(zip(['PER', 'SE', 'C', 'WN', 'LIN', 'Sr', 'S', 'SIr', 'SI', 'sigmoidal_intervals'], range(10)))
def order_base_kerns(bs): return sorted(bs, key = lambda b: base_kern_interp_order[b])


def base_factors_interpretation(bs_params):
    bps_copy = sigmoids_to_intervals(deepcopy(bs_params))
    var = bps_copy['ProductKE'][0]['variance']
    description_end = (', and it' if 'sigmoidal_intervals' in bps_copy else '; this component') + ' has overall variance {:.2f}'.format(var)
    del bps_copy['ProductKE']

    ordered_ps = sorted(bps_copy.items(), key = lambda bps: base_kern_interp_order[bps[0]])
    postmodifiers = []
    for bps in ordered_ps[1:]:
        interpretation = postmodifier_interpretation(bps)
        if interpretation != '': postmodifiers.append(('; this component applies ' if bps[0] == 'sigmoidal_intervals' else ', ') + interpretation)
    return first_term_interpretation(ordered_ps[0]) + reduce(add, postmodifiers, '') + description_end


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
    elif b == 'sigmoidal_intervals':
        if not isinstance(ps, list): ps = [ps]
        interval_ress = []
        for interval in ps:
            if 'start' not in interval: interval_ress.append('until {:.2f} (with change slope {:.2f})'.format(interval['end'], interval['end_slope']))
            elif 'end' not in interval: interval_ress.append('from {:.2f} (with change slope {:.2f})'.format(interval['start'], interval['start_slope']))
            else:
                interval_ress.append('between {:.2f} and {:.2f}'.format(interval['start'], interval['end']))
                if interval['start_slope'] == interval['end_slope']: interval_ress[-1] += ' (with same change slopes {:.2f})'.format(interval['start_slope'])
                else: interval_ress[-1] += ' (with change slopes {:.2f} and {:.2f})'.format(interval['start_slope'], interval['end_slope'])
        if len(interval_ress) > 1: interval_ress[-1] = 'and ' + interval_ress[-1]
        res = ', '.join(interval_ress)
    elif b not in base_sigmoids: raise ValueError(f'An unexpected type of postmodifier term in a pure product has arisen: {b}')
    return res



# Sigmoid-to-interval functions

def sigmoids_to_intervals(bpss):
    sigmoidals = {}
    for b in list(bpss.keys()):
        if b == 'S': sigmoidals[b] = S_overlap([S_interval(p) for p in bpss[b]])
        elif b == 'Sr': sigmoidals[b] = Sr_overlap([Sr_interval(p) for p in bpss[b]])
        elif b == 'SI': sigmoidals[b] = SI_overlap([SI_interval(p) for p in bpss[b]])
        elif b == 'SIr': sigmoidals[b] = SIr_overlap([SIr_hole_interval(p) for p in bpss[b]])
        # if b in sigmoidals: del bpss[b]
    if len(sigmoidals) > 0:
        res = simplify_sigmoidal_intervals(sigmoidals)
        bpss['sigmoidal_intervals'] = res if isinstance(res, list) else [res]
    return bpss

def S_interval(S_p): return {'start': S_p['location'], 'start_slope': S_p['slope']}
def Sr_interval(Sr_p): return {'end': Sr_p['location'], 'end_slope': Sr_p['slope']}
def SI_interval(ST_p): return {'start': SI_ps_left(ST_p), 'end': SI_ps_right(ST_p), 'start_slope': ST_p['slope'], 'end_slope': ST_p['slope']}
def SIr_hole_interval(STr_p): return {'end': SI_ps_left(STr_p), 'start': SI_ps_right(STr_p), 'end_slope': STr_p['slope'], 'start_slope': STr_p['slope']}

def SIr_split_intervals(hole_interval): return [{'end': hole_interval['end'], 'end_slope': hole_interval['end_slope']}, {'start': hole_interval['start'], 'start_slope': hole_interval['start_slope']}]

def SI_ps_left(SI_p): return SI_p['location'] - SI_p['width'] / 2
def SI_ps_right(SI_p): return SI_p['location'] + SI_p['width'] / 2



# Functions to apply to each single sigmoid type ps if multiple ones present

def Sr_overlap(Sr_params): return min(Sr_params, key = lambda p: p['end'])

def S_overlap(S_params): return max(S_params, key = lambda p: p['start'])


def SI_overlap(SI_params):
    start = S_overlap(SI_params)
    end = Sr_overlap(SI_params)
    interval = {'start': start['start'], 'end': end['end'], 'start_slope': start['start_slope'], 'end_slope': end['end_slope']}
    return interval if interval['end'] > interval['start'] else None

def SIr_overlap_step(acc, p): # Acc is a list of dictionaries representing intervals
    new_acc = []
    for a in acc:
        if 'start' not in a or 'end' not in a: intersect_semiline_hole(a, p, new_acc)
        else: intersect_segment_hole(a, p, new_acc)
    return new_acc
def SIr_overlap(SIr_params): return reduce(SIr_overlap_step, SIr_params[1:], SIr_split_intervals(SIr_params[0]))


# Basic interval intersection functions

def intersect_segment_semiline(segment, semiline, res = None):
    if 'start' not in semiline and segment['start'] < semiline['end']:
        if segment['end'] < semiline['end']: res = segment
        else: res = {'start': segment['start'], 'end': semiline['end'], 'start_slope': segment['start_slope'], 'end_slope': semiline['end_slope']}
    elif 'end' not in semiline and segment['end'] > semiline['start']:
        if segment['start'] > semiline['start']: res = segment
        else: res = {'start': semiline['start'], 'end': segment['end'], 'start_slope': semiline['start_slope'], 'end_slope': segment['end_slope']}
    else: res = None
    return res

def intersect_segment_hole(segment, hole, res = []):
    if segment['start'] < hole['end']:
        if segment['end'] < hole['end']: res.append(segment)
        else:res.append({'start': segment['start'], 'end': hole['end'], 'start_slope': segment['start_slope'], 'end_slope': hole['slope']})
    if segment['end'] > hole['start']:
        if segment['start'] > hole['start']: res.append(segment)
        else: res.append({'start': hole['start'], 'end': segment['end'], 'start_slope': hole['slope'], 'end_slope': segment['end_slope']})
    return res

def intersect_semiline_hole(semiline, hole, res = []):
    if 'start' not in semiline:
        if hole['end'] < semiline['end']:
            res.append({'end': hole['end'], 'end_slope': hole['slope']})
            if hole['start'] < semiline['end']: res.append({'start': hole['start'], 'end': semiline['end'], 'start_slope': hole['start_slope'], 'end_slope': semiline['end_slope']})
        else: res.append(semiline)
    else: # elif 'end' not in semiline:
        if hole['start'] > semiline['start']:
            if hole['end'] > semiline['start']: res.append({'start': semiline['start'], 'end': hole['end'], 'start_slope': semiline['start_slope'], 'end_slope': hole['end_slope']})
            res.append({'start': hole['start'], 'start_slope': hole['start_slope']})
        else: res.append(semiline)
    return res



# Full reduction of sigmoidals to intervals functions

def simplify_sigmoidal_intervals(sigmoidals): return reduce(simplify_sigmoidal_intervals_step, sigmoidals.items())[1]

def simplify_sigmoidal_intervals_step(acc, new):
    if acc is None: return acc
    step_res = simplify_sigmoidal_interval_pair(acc, new)
    return None if step_res is None else identify_interval_type(step_res)

# Intersect two DIFFERENT TYPES of sigmoid parameters (apply after reducing same type ones)
def simplify_sigmoidal_interval_pair(b_ips1, b_ips2): # Arguments are tuples of (sigmoid_str, parameters)
    pair = {b_ips1[0]: [b_ips1[1], b_ips2[1]]} if b_ips1[0] == b_ips2[0] else dict([b_ips1, b_ips2])
    res = []
    if len(pair) == 2: # Always when called directly
        if 'SIr' in pair:
            for interval in pair['SIr']:
                new_interval = simplify_sigmoidal_interval_pair([(x, pair[x]) for x in pair if x != 'SIr'][0], identify_interval_type(interval))
                if new_interval != []: res.append(new_interval)
        elif 'SI' in pair:
            new_interval = intersect_segment_semiline(pair['SI'], pair['S' if 'S' in pair else 'Sr'])
            if new_interval is not None: res.append(new_interval)
        elif pair['S']['start'] < pair['Sr']['end']: res.append({**pair['S'], **pair['Sr']})
    else: # Only happens when called by itself for SIr case
        if 'SI' in pair: res = SI_overlap(pair['SI'])
        elif 'S' in pair: res = S_overlap(pair['S'])
        else: res = (Sr_overlap(pair['Sr']))
    return res[0] if isinstance(res, list) and len(res) == 1 else res


def identify_interval_type(interval):
    if isinstance(interval, list): interval_type = 'SIr'
    elif 'start' not in interval: interval_type = 'Sr'
    elif 'end' not in interval: interval_type = 'S'
    else: interval_type = 'SI'
    return (interval_type, interval)
