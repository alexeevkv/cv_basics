EXPERIMENT_TO_MLFLOW_INFO = {
    'ResNet_no_augm_adam': {
        'run_ids': [
            '048e912c53544943971e75e535d9d400',
            'cb9fff5163ad4e68adec856e96727d09',
            'f485a0417c604a449c81f732014f1b3f',
            '03344e427d264200a42bcf944a00745c'
        ],
        'metrics_info': {
            'Accuracy score': ('accuracy_score/train', 'accuracy_score/val'),
            'Cross Entropy Loss': ('loss/train', 'loss/val'),
            'Absolute average weight': 'abs_avg_weight',
            'Absolute average gradient': 'abs_avg_grad',
        },
        'experiment_id': '693429215089447298',
        'FINAL_NAME': 'images/ResNet_no_augm'
    },
    'ResNet_no_augm_adamw': {
        'run_ids': [
            'c700e81e6dae476caec5c0c826a049c1',
        ],
        'metrics_info': {
            'Accuracy score': ('accuracy_score/train', 'accuracy_score/val'),
            'Cross Entropy Loss': ('loss/train', 'loss/val'),
            'Absolute average weight': 'abs_avg_weight',
            'Absolute average gradient': 'abs_avg_grad',
        },
        'experiment_id': '385879499541351603',
        'FINAL_NAME': 'images/ResNet_no_augm_adamw'        
    },
    'Swin_no_augm_adamw': {
        'run_ids': [
            'd84f2f8631b94bc79b763585b40f6a73',
            '6cf629638974424991e3b83fda5014fd'
        ],
        'metrics_info': {
            'Accuracy score': ('accuracy_score/train', 'accuracy_score/val'),
            'Cross Entropy Loss': ('loss/train', 'loss/val'),
            'Absolute average weight': 'abs_avg_weight',
            'Absolute average gradient': 'abs_avg_grad',
        },
        'experiment_id': '781614969400698020',
        'FINAL_NAME': 'images/swin_no_augm_adamw'
    }
}
