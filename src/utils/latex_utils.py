import numpy as np
import pandas as pd
from scipy import stats


def print_export_auc_table(processed_data, metric_keys):    #area_under_curve):
    '''Print and export the area under the curve table for all evaluation metrics'''

    print("\nArea under curve (higher is better)")
    values = []
    methods_keys = processed_data.keys()
    for method in methods_keys:
        values.extend(
            processed_data[method]['eval'][metric]['avg_auc']
            for metric in metric_keys
        )
    values = np.array(values)
    values = values.reshape(len(methods_keys), len(metric_keys)) # transpose to get reshape to do what we want

    columns = map(lambda x : x.replace("_", " "), metric_keys)
    rows    = map(lambda x : x.replace("_", " ").split("-")[0], methods_keys)

    df = pd.DataFrame(values/100, columns=columns, index=rows).rename_axis('method', axis=1)
    style = df.style.highlight_max(props='textbf:--rwrap;').format("{:.2f}").hide(["loss"], axis=1)
    style.to_latex("latex/" + "AUC" + ".tex", caption="AUC values for evaluation (test) dataset. AUC is scaled by $10^{-2}$.", label="tab:auc", column_format="l|ccc", hrules=True, position_float='centering')
    print(style.to_string())
        
        
def print_export_ttest_file(processed_data, metric_keys):
    '''T-Test against all areas under curve '''
    
    print("\nEval t-test p-value (lower is better, <0.05 = significant)")
    for metric in metric_keys:
        
        methods = list(processed_data.keys())

        # Calculate t-test against each other method, and save it into a value array
        values = np.zeros((len(methods), len(methods)))

        for i, method_1 in enumerate(methods):
            for j, method_2 in enumerate(methods):
                auc_samples_1 = processed_data[method_1]['eval'][metric]
                auc_samples_2 = processed_data[method_2]['eval'][metric]
                t_test = stats.ttest_ind(auc_samples_1['auc'], auc_samples_2['auc'])[1] # p-value
                values[i][j] = float(t_test)

        columns    = list(map(lambda x : x.replace("_", " ").split("-")[0], methods))
        methods    = list(map(lambda x : x.replace("_", " ").split("-")[0], methods))

        df = pd.DataFrame(values, columns=columns, index=methods)
        style = df.style.format_index('\\rotatebox{{90}}{{{}}}', axis=1).format("{:.2f}").highlight_between(left=0.05, right=1, inclusive='neither', props='textbf:--rwrap;')
        style.to_latex(
            f"latex/t-test_{metric}.tex",
            caption=f't-test p-value for {metric.replace("_", " ")}',
            label=f"tab:t-test_{metric}",
            column_format="l|cccccc",
            hrules=True,
            position_float='centering',
        )

        # PRINT
        pd.options.display.float_format = '{:,.2f}'.format
        print("\n" + str(metric))
        print(df)