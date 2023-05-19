def test_load():
  return 'loaded'


def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]


def cond_prob (table, a_evidence, corresponding_evidence_value, a_target, corresponding_target_value):
  t_subset = up_table_subset(table, a_target, 'equals', corresponding_target_value)
  e_list = up_get_column(t_subset, a_evidence)
  p_b_a = sum([1 if v==corresponding_evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a


def cond_probs_product (table, evidence_row, a_target, corresponding_target_value):
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)

  cond_prob_list= []
  for pair in evidence_complete:
    evidence_n = pair[0]
    evidence_value_n = pair[1]
    cond_prob_list += [cond_prob(table, evidence_n, evidence_value_n, a_target, corresponding_target_value)]

  return up_product(cond_prob_list)


def prior_prob (table, a_target, corresponding_target_value):
  t_list = up_get_column(table, a_target)
  p_a = sum([1 if v==corresponding_target_value else 0 for v in t_list])/len(t_list)
  return p_a


def naive_bayes(table, evidence_row, a_target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the product of the list, finally multiply by P(Flu=0)
  p_partial_numerator_0 = cond_probs_product (table, evidence_row, a_target, 0)
  p_a_0 = prior_prob (table, a_target, 0)
  p_a_0_b_n = p_partial_numerator_0*p_a_0

  #do same for P(Flu=1|...)
  p_partial_numerator_1 = cond_probs_product (table, evidence_row, a_target, 1)
  p_a_1 = prior_prob (table, a_target, 1)
  p_a_1_b_n = p_partial_numerator_1*p_a_1

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(p_a_0_b_n, p_a_1_b_n)
  
  #return your 2 results in a list
  return [neg, pos]
