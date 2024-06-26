import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def load_data_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_classifier_data(data, fig_path=None, weight_type=None, mask_percent=None, sort_bars=False, annotate=False):    
    for dataset_name, experiments  in data["details"].items():        
        for experiment in experiments:
            classifiers = experiment["classifiers"]            
            classifier_names = []
            means = []
            std_devs = []
            
            title_name_mapping = {'NC_graph_rel': 'NC',
                                 'Scandinavia_weights_partial_labels': 'SC',
                                 'Volga_weights_partial_labels': 'VU',
                                 'Western-Europe_weights_partial_labels': 'WE'}
            
            classifier_names_mapping = {'AttnGCN': 'AttnGCN',
                                       'AttnGCN_gb': 'AttnGCN_gb',
                                       'AttnGCN_narrow_long': 'AttnGCN_narrow_long',
                                       'AttnGCN_narrow_long_gb': 'AttnGCN_narrow_long_gb',
                                       'AttnGCN_narrow_short': 'AttnGCN_narrow_short',
                                       'AttnGCN_narrow_short_gb': 'AttnGCN_narrow_short_gb',
                                       'AttnGCN_wide_long': 'AttnGCN_wide_long',
                                       'AttnGCN_wide_long_gb': 'AttnGCN_wide_long_gb',
                                       'AttnGCN_wide_short': 'AttnGCN_wide_short',
                                       'AttnGCN_wide_short_gb': 'AttnGCN_wide_short_gb',
                                       'EdgeCount': 'EdgeCount',
                                       'EdgeCountPerClassize': 'EdgeCountPerClassize',
                                       'GCNConv_3l_128h_w': 'GCNConv_narrow_short',
                                       'GCNConv_3l_128h_w_gb': 'GCNConv_narrow_short_gb',
                                       'GINNet': 'GINNet',
                                       'GINNet_gb': 'GINNet_gb',
                                       'GINNet_narrow_long': 'GINNet_narrow_long',
                                       'GINNet_narrow_long_gb': 'GINNet_narrow_long_gb',
                                       'GINNet_narrow_short': 'GINNet_narrow_short',
                                       'GINNet_narrow_short_gb': 'GINNet_narrow_short_gb',
                                       'GINNet_wide_long': 'GINNet_wide_long',
                                       'GINNet_wide_long_gb': 'GINNet_wide_long_gb',
                                       'GINNet_wide_short': 'GINNet_wide_short',
                                       'GINNet_wide_short_gb': 'GINNet_wide_short_gb',
                                       'IbdSum': 'IbdSum',
                                       'IbdSumPerEdge': 'IbdSumPerEdge',
                                       'LongestIbd': 'LongestIbd',
                                       'MLP_3l_128h': 'MLP_narrow_short',
                                       'MLP_3l_512h': 'MLP_wide_short',
                                       'MLP_9l_128h': 'MLP_narrow_long',
                                       'MLP_9l_512h': 'MLP_wide_long',
                                       'SegmentCount': 'SegmentCount',
                                       'TAGConv_3l_128h_w_k3': 'TAGConv_narrow_short',
                                       'TAGConv_3l_128h_w_k3_gb': 'TAGConv_narrow_short_gb',
                                       'TAGConv_3l_512h_w_k3': 'TAGConv_wide_short',
                                       'TAGConv_3l_512h_w_k3_gb': 'TAGConv_wide_short_gb',
                                       'TAGConv_9l_128h_k3': 'TAGConv_narrow_long',
                                       'TAGConv_9l_128h_k3_gb': 'TAGConv_narrow_long_gb',
                                       'TAGConv_9l_512h_nw_k3': 'TAGConv_wide_long_nw',
                                       'TAGConv_9l_512h_nw_k3_gb': 'TAGConv_wide_long_nw_gb',
                                       'SAGEConv_3l_128h': 'SAGEConv_narrow_short',
                                       'SAGEConv_3l_128h_gb': 'SAGEConv_narrow_short_gb',
                                       'SAGEConv_3l_512h': 'SAGEConv_wide_short',
                                       'SAGEConv_3l_512h_gb': 'SAGEConv_wide_short_gb',
                                       'SAGEConv_9l_128h': 'SAGEConv_narrow_long',
                                       'SAGEConv_9l_128h_gb': 'SAGEConv_narrow_long_gb',
                                       'SAGEConv_9l_512h': 'SAGEConv_wide_long',
                                       'SAGEConv_9l_512h_gb': 'SAGEConv_wide_long_gb',
                                       'TAGConv_3l_512h_w_k3_g_norm_mem_pool': 'TAGConv_wide_short_g_norm',
                                       'TAGConv_3l_512h_w_k3_g_norm_mem_pool_gb': 'TAGConv_wide_short_g_norm_gb',
                                       'Agglomerative': 'Agglomerative clustering',
                                       'GirvanNewmann': 'Girvan-Newman',
                                       'LabelPropagation': 'Label propagation',
                                       'MultiRankWalk': 'Multi-Rank-Walk',
                                       'RelationalNeighbor': 'Relational neighbor classifier',
                                       'RidgeRegression': 'Ridge regression',
                                       'Spectral': 'Spectral clustering'}

            for name, metrics in classifiers.items():
                if name == "exp_idx":
                    continue
                classifier_names.append(name)
                means.append(metrics['f1_macro']['mean'])
                std_devs.append(metrics['f1_macro']['std'])

            classifier_names_mapped = []
            for name in classifier_names:
                classifier_names_mapped.append(classifier_names_mapping[name])
            
            # Create a DataFrame for easier plotting with seaborn
            df = pd.DataFrame({
                'Classifier': classifier_names_mapped,
                'Mean': means,
                'StdDev': std_devs
            })

            # Optionally sort the bars by their mean values
            if sort_bars:
                df = df.sort_values('Mean', ascending=False)

            # Plotting
            plt.figure(figsize=(14, 6))
            # sns.set(style="whitegrid")
            sns.set_theme()
            
            cols = []
            color_model_scheme = {'GNN graph based':'#05F140', 'GNN one hot':'#253957', 'MLP':'#FFB400', 'Heuristics':'#00B4D8', 'Community detection': '#EF233C'}#9B7EDE #FFDF64 #5FBFF9 #FF595E
            for model_name in df.Classifier:
                if 'gb' in model_name:
                    cols.append(color_model_scheme['GNN graph based'])
                elif 'MLP' in model_name:
                    cols.append(color_model_scheme['MLP'])
                elif model_name in ["EdgeCount", "EdgeCountPerClassize", "SegmentCount", "LongestIbd", "IbdSum", "IbdSumPerEdge"]:
                    cols.append(color_model_scheme['Heuristics'])
                elif model_name in ["Spectral clustering", "Agglomerative clustering", "Girvan-Newman", "Label propagation", "Relational neighbor classifier", "Multi-Rank-Walk", "Ridge regression"]:
                    cols.append(color_model_scheme['Community detection'])
                else:
                    cols.append(color_model_scheme['GNN one hot'])

            bar_plot = sns.barplot(x='Classifier', y='Mean', data=df, ci=None, palette=cols)  # , palette="viridis")

            # Adding error bars
            for i, (mean, std) in enumerate(zip(df['Mean'], df['StdDev'])):
                bar_plot.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

                # Optionally annotate the bars with the exact mean values
                if annotate:
                    bar_plot.text(i, mean + std + 0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=6)
                    
            
            for k, v in color_model_scheme.items():
                plt.scatter([],[], c=v, label=k)

            plt.title(f'Model performance ({title_name_mapping[dataset_name]})')
            plt.xlabel('Model')
            plt.ylabel('Mean f1-macro score')
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor', verticalalignment='center')  # Rotate x-axis labels for better readability
            plt.legend()
            plt.tight_layout()  # Adjust the layout to make room for the rotated labels
            if fig_path is not None and weight_type is not None and mask_percent is not None:
                plt.savefig(f'{fig_path}{dataset_name}_{weight_type}_mask_{mask_percent}.pdf', bbox_inches="tight")
            plt.show()
