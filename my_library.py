def test_load():
  return 'loaded'


def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]


def cond_prob(table, a_evidence, corresponding_evidence_value, a_target, corresponding_target_value):
  t_subset = up_table_subset(table, target, 'equals', corresponding_target_value)
  e_list = up_get_column(t_subset, a_evidence)
  p_b_a = sum([1 if v==corresponding_evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor


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


def metrics(a_list):
  assert isinstance(a_list,list), f'Expecting a_list to be a list but instead is {type(a_list)}'
  for item in a_list:
    assert isinstance(item, list), f'Expecting a_list to be a list of lists'
    assert len(item) == 2, f'Expecting a_list to be a zipped list with each value being a pair of items'
    for value in item:
      assert isinstance(value, (int, float)), f'Expecting each value in the pair to be an int'
      assert value >= 0, f'Expecting each value in the pair to be >= 0'

  TN = sum([1 if pair==[0,0] else 0 for pair in a_list])
  TP = sum([1 if pair==[1,1] else 0 for pair in a_list])
  FP = sum([1 if pair==[1,0] else 0 for pair in a_list])
  FN = sum([1 if pair==[0,1] else 0 for pair in a_list])

  precision = TP/(TP+FP) if TP+FP>0 else 0
  recall = TP/(TP+FN) if TP+FN>0 else 0
  f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
  accuracy = sum(p==a for p,a in a_list)/len(a_list)

  return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}


def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)

  #copy paste code here
  
  #loop through your architecutes and get results
  for structure in architectures:
    all_results = up_neural_net(train_table, test_table, structure, target)
  
    all_mets = []
    #loop through thresholds
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]
      
    print(f'Architecture: {structure}')
    print(up_metrics_table(all_mets))

  return None  #main use is to print out threshold tables, not return anything useful.


from sklearn.ensemble import RandomForestClassifier  #make sure this makes it into your library

def run_random_forest(train, test, target, n):

  #your code below
  X = up_drop_column(train, target)
  y = up_get_column(train,target)
  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)
  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
  clf.fit(X, y)  #builds the trees as specified above
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]  #probs is list of [neg,pos] like we are used to seeing.
  pos_probs[:5]

  all_mets = []
  for t in thresholds:
   all_predictions = [1 if pos>t else 0 for pos in pos_probs]
   pred_act_list = up_zip_lists(all_predictions, k_actuals)
   mets = metrics(pred_act_list)
   mets['Threshold'] = t
   all_mets = all_mets + [mets]

  all_mets[:2]
  metrics_table = up_metrics_table(all_mets)
  metrics_table

  print(metrics_table)  #output we really want - to see the table
  return None
