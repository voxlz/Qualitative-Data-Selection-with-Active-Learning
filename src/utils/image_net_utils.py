from itertools import accumulate
import random

def prep_budget(budget, n_unseen, rnd_budget, n_batch, n_epoch):
    '''Prepares the budget schedule for the experiment. If rnd_budget is True, the budget is randomly sampled from the budget list. 
    Otherwise, the budget is sampled in a round-robin fashion.'''
    
    budget_schedule = []
    
    # Accept both list and single int as budget
    if type(budget) is int: budget = [budget]

    # Schedule budget
    index = 0
    while sum(budget_schedule) < n_unseen:
        sampel_index = random.randint(0, len(budget)-1) if rnd_budget else index % len(budget)
        budget_schedule.append(budget[sampel_index])
        index += 1
  
    # Subtract from the last element if budget_schedule sum is larger than unseen images 
    budget_schedule[-1] -= sum(budget_schedule) - n_unseen
    
    assert sum(budget_schedule) == n_unseen, "budget values must sum to unseen_count."
    
    # Check budget assertions
    
    for b in budget:
        assert b % n_batch  == 0, "budget values must be divisible by n_batch."
        if n_epoch not in [0, 1000]: 
            # If not using early-stop or 0 epochs
            assert b % n_epoch == 0 or n_epoch % b == 0 , "budget values must be divisible by n_epoch."
  
    # test = list(accumulate(budget_schedule, lambda x, y: x - y, initial=n_unseen))
    # map(lambda budget: budget  n_batch, budget_schedule)
  
    return budget_schedule