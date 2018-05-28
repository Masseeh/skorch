{
    'predict_service': {
        '__factory__': 'palladium.server.PredictService',
        'mapping': [
            ('text', 'str'),
        ],
        'predict_proba': True,
        'unwrap_sample': True,
    },

    'model_persister': {
        '__factory__': 'palladium.persistence.File',
        'path': 'rnn-model-{version}',
    },

}
