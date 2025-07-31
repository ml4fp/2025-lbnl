from .constants import c6_to_cH, ct_to_ctH, cg_to_cHG

cH_val = 20.0
ctH_val = 10.0
cHG_val = 0.1

c6_val = cH_val / c6_to_cH
ct_val = ctH_val / ct_to_ctH
cg_val = cHG_val / cg_to_cHG

eft_terms = [
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [4, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [0, 0, 1],
    [0, 0, 2],
    [1, 1, 0],
    [1, 0, 1],
    [2, 1, 0],
    [2, 0, 1],
    [0, 1, 1],
]

c6_degree = max([ctup[0] for ctup in eft_terms])
ct_degree = max([ctup[1] for ctup in eft_terms])
cg_degree = max([ctup[2] for ctup in eft_terms])