import matplotlib.pyplot as plt


def get_metric(metric_path, run_id, experiment_id):
    path = f'./mlruns/{experiment_id}/{run_id}/metrics/{metric_path}'
    with open(path) as f:
        content = f.readlines()
    metrics_for_step = [float(x.split(' ')[1]) for x in content]
    return metrics_for_step


def load_metric_for_runs(metric_path, run_ids, experiment_id):
    final_metric = list()
    for run_id in run_ids:
        metric = get_metric(metric_path, run_id, experiment_id)
        final_metric.extend(metric)
    return final_metric


def plot_metrics(metrics_info, run_ids, experiment_id, FINAL_NAME):
    for name, path in metrics_info.items():

        if isinstance(path, tuple):
            fig, ax = plt.subplots(1, 1, figsize=(9, 4))
            for sub_path in path:
                
                stage = 'Train' if 'train' in sub_path else 'Val'

                metric = load_metric_for_runs(sub_path, run_ids, experiment_id)
                ax.plot(metric, label=stage)
                ax.set_title(name)
                ax.set_ylabel(name)
                ax.set_xlabel('Epoch')
                plt.grid()
            ax.legend()
            plt.grid()
            plt.savefig(f"{FINAL_NAME}_{name}.png", bbox_inches='tight')
            plt.show()

            #fig = plt.gcf()
        else: 

            metric = load_metric_for_runs(path, run_ids, experiment_id)

            plt.figure(figsize=(9, 4))
            plt.plot(metric)
            plt.title(name)
            plt.ylabel(name)
            plt.xlabel('Epoch')
            plt.grid()
            plt.savefig(f"{FINAL_NAME}_{name}.png", bbox_inches='tight')
            plt.show()
