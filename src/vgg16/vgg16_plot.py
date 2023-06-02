from active_learning.active_learning_plot import plot_history_files
from utils.plot_utils import  get_history_files_from_tests

''' File used to keep track of previous vgg16 test results and plotting history files '''

# Pivot Selection
vgg16_variable_pivot                    = "saved/2023-04-25_16-27-16__variable_pivot"
vgg16_variable_pivot_v2                 = "saved/2023-04-26_14-27-55__variable_pivot_500_2000_eval"
vgg16_top_100_selection_entropy         = "saved/2023-04-26_15-20-43__top_100_img"
vgg16_pivot_early_stop                  = "saved/2023-05-15_15-00-41__pivot_early_stop"

# Early Stopping and Fixed steps per epoch
vgg16_early_stop_fixed_step_v1          = "saved/2023-04-27_14-06-32__early_stop_VS_fixed_epoch_VS_fixed_step_Inconsequent_test_data"
vgg16_early_stop_fixed_step_v2          = "saved/2023-04-28_13-31-26__early_stop_VS_fixed_epoch_VS_fixed_step_v2"
vgg16_es_fs_v2_loss_VS_es_acc           = "saved/2023-04-28_13-31-26__early_stop_VS_fixed_epoch_VS_fixed_step_v2_NO_FIXED"
vgg16_es_fs_v2_loss_VS_es_loss_fixed    = "saved/2023-04-28_13-31-26__early_stop_VS_fixed_epoch_VS_fixed_step_v2_ES_LOSS_ONLY"

# Budget / Budget Schedule
vgg16_budget_size_test                  = "saved/2023-05-02_13-19-54__budget_size_proper"

# Diversity evaluation
vgg16_div_eval_v2_entropy               = "saved/2023-05-08_15-10-52__diversity_reevaluation_normal/entropy"
vgg16_div_eval_v2_random                = "saved/2023-05-08_15-10-52__diversity_reevaluation_normal/random"
vgg16_div_eval_mixed_entropy            = "saved/2023-05-08_13-35-24__diversity_reevaluation/entropy"
vgg16_div_eval_mixed_random             = "saved/2023-05-08_13-35-24__diversity_reevaluation/random"
vgg16_div_eval_duplicates               = "saved/2023-05-12_16-21-22__diversity_reevaluation_duplicates"
vgg16_div_eval_duplicates_cont          = "saved/2023-05-12_22-35-28__diversity_reevaluation_duplicates_cont"

# Method comparison
vgg16_v2_method_comparison              = "saved/2023-05-09_15-33-33__compare_methods_v2"
vgg16_v2_method_comparison_cont         = "saved/2023-05-09_15-33-40__compare_methods_v2_cont"
vgg16_v2_method_comparison_cont2        = "saved/2023-05-10_09-34-38__compare_methods_v2_cont2"
vgg16_mixed_method_comparison           = "saved/2023-05-10_16-23-54__compare_methods_mixed"
vgg16_mixed_method_comparison_cont      = "saved/2023-05-10_16-24-45__compare_methods_mixed_cont"
vgg16_corrupt_method_comparison         = "saved/2023-05-12_09-34-55__compare_methods_corrupt"
vgg16_corrupt_method_comparison_cont    = "saved/2023-05-12_22-36-24__compare_methods_corrupt_cont"
vgg16_hard_method_comparison            = "saved/2023-05-11_15-01-37__compare_methods_hard"
vgg16_hard_method_comparison_cont       = "saved/2023-05-11_15-02-09__compare_methods_hard_cont"

# Add base set in the beginning. Rerun to ensure same starting point (fixed dataset splits, with base added to train or not)
vgg16_base_set_v2                       = "saved/2023-05-29_13-10-56__v2_test_(without_base)"
vgg16_base_set_v2_base                  = "saved/2023-05-29_16-22-18__v2_(with_base)"
vgg16_base_set_mixed_base               = "saved/2023-05-29_16-22-27__mixed_(with_base)"
vgg16_mixed_1000_plus_5000              = "saved/2023-05-30_16-50-39__mixed_1000+5000"
vgg16_mixed_1000_plus_5000_cont         = "saved/2023-05-30_16-51-02__mixed_1000+5000_cont"

# Accumulation
vgg16_accumulation_v2                   = "saved/2023-05-15_14-32-04__accumulation_vs_prune" # Indecisive results. Needs more testing
vgg16_accumulation_on_off_pivot         = "saved/2023-05-16_11-56-53__accumulation_on_off_pivot" # Indecisive results. Needs more testing

# Plotting
folders = [vgg16_mixed_1000_plus_5000, vgg16_mixed_1000_plus_5000_cont]
files, eval_base = get_history_files_from_tests(folders)

plot_history_files(files,
                   suptitle="ds", # ds is special value, add dataset name to suptitle
                   skip_first_eval=False, 
                   eval_baseline=eval_base, 
                   use_standard_deviation=True, 
                   export_latex=True,
                   separate_budgets=False, 
                   separate_pivots=False, 
                   separate_accumulate=False)
print("finished")