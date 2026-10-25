"""Small, reviewed Mendelian genotype-transition helpers for the public app."""
import numpy as np


def mendelian_tensor(birds, child_indices, mode):
    """P(child state | father state, mother state): shape (n,3,3,3).

    States: 0 wild; 1 heterozygous (autosome or ZZ); 2 mutant homozygous
    (autosome/ZZ), or mutant Z/W in females. In z_linked mode female state 1
    is biologically invalid and always receives prior probability zero.
    """
    n = len(child_indices)
    transitions = np.zeros((n, 3, 3, 3))
    for k, idx in enumerate(child_indices):
        sex = birds[idx]['sex']
        for a in range(3):
            for b in range(3):
                father_gamete = a / 2.0
                if mode == 'autosomal':
                    mother_gamete = b / 2.0
                    transitions[k,a,b] = [
                        (1-father_gamete)*(1-mother_gamete),
                        father_gamete*(1-mother_gamete)+(1-father_gamete)*mother_gamete,
                        father_gamete*mother_gamete]
                elif mode == 'z_linked':
                    # A ZW mother is state 0 (+/W) or 2 (mut/W).
                    # state 1 for a female is invalid, never sampled.
                    mother_z_mut = b / 2.0
                    son = [(1-father_gamete)*(1-mother_z_mut),
                           father_gamete*(1-mother_z_mut)+(1-father_gamete)*mother_z_mut,
                           father_gamete*mother_z_mut]
                    daughter = [1-father_gamete, 0.0, father_gamete]
                    if sex == 'M':
                        transitions[k,a,b] = son
                    elif sex == 'F':
                        transitions[k,a,b] = daughter
                    else:
                        transitions[k,a,b] = (np.array(son)+daughter)*0.5
                else:
                    raise ValueError(f'Unsupported inheritance mode: {mode}')
    return transitions


def sex_transition(probs_f, probs_m, mode, sex):
    if sex not in ('M','F','?'):
        raise ValueError('Child sex must be M, F or ?')
    t = mendelian_tensor([{'sex': sex}], [0], mode)[0]
    return np.einsum('a,b,abc->c', np.asarray(probs_f), np.asarray(probs_m), t)
