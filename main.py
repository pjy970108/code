import os
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np

# mode : 실행모드
if __name__ == '__main__':
    path = '/Users/pjy97/Desktop/한양대/1-1/강화학습/project'
    data_path = path + '/data/'
    data = pd.read_csv(data_path + '수정주가(원).csv')
    data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    data.set_index('date', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.index.get_loc('2012-01-02')
    data_inf = data.iloc[data.index.get_loc('2012-01-02'):, :]
    # 삼성전자와 sk하이닉스만 사용
    data_inf = data_inf.loc[:, ['A005930', 'A000660']]
    econmic_index = pd.read_csv(data_path + '경기종합지수_2020100__10차__20231208161859.csv')
    econmic_index = pd.DataFrame(econmic_index.iloc[1,]).T
    econmic_index.index = econmic_index.iloc[:, 0]
    econmic_index = econmic_index.iloc[:, 1:]
    econmic_index = econmic_index.astype(float)
    # 전달 대비 상위 75%는 상승
    # 전달 대비 하위 25%는 하락
    # 25% ~ 75%는 변동 없음
    econmic_index.T.describe()
    econmic_index = econmic_index.iloc[:, 189:]
    col_list = econmic_index.columns
    threshold_75 = np.percentile(econmic_index, 75)
    threshold_25 = np.percentile(econmic_index, 25)
    econmic_index = np.where(econmic_index >= threshold_75, 1, np.where(econmic_index >= threshold_25, 0, -1))
    econmic_index = pd.DataFrame(econmic_index)
    econmic_index.columns = col_list
    econmic_index = econmic_index.T
    econmic_index.rename(columns={0: 'econmic_index'}, inplace=True)
    data_inf['str_index'] = data_inf.index.strftime('%Y.%m')
    data_inf = data_inf.merge(econmic_index, left_on='str_index', right_index=True)
    data_inf.drop('str_index', axis=1, inplace=True)
    data_inf['A005930'].replace(',', '', regex=True, inplace=True)
    data_inf['A000660'].replace(',', '', regex=True, inplace=True)
    data_inf['A005930'] = data_inf['A005930'].astype(float)
    data_inf['A000660'] = data_inf['A000660'].astype(float)
    samsung = data_inf[['A005930', 'econmic_index']]
    hinix = data_inf[['A000660', 'econmic_index']]
    samsung = data_inf[['A005930']]
    hinix = data_inf[['A000660']]
    samsung.rename(columns={'A005930': 'Close'}, inplace=True)
    hinix.rename(columns={'A000660': 'Close'}, inplace=True)
    # EMA (Exponential Moving Average) 계산
    def calculate_ema(data, window):
        return data['Close'].ewm(span=window, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence) 계산
    def calculate_macd(data, short_window, long_window):
        short_ema = calculate_ema(data, short_window)
        long_ema = calculate_ema(data, long_window)

        macd = short_ema - long_ema

        return macd

    # Log Return 계산
    def calculate_log_return(data):
        return np.log(data['Close'] / data['Close'].shift(1))
    # EMA, MACD, Log Return 계산
    samsung['EMA_5'] = calculate_ema(samsung, 5)
    samsung['EMA_20'] = calculate_ema(samsung, 20)
    samsung['MACD'] = calculate_macd(samsung, 5, 20)
    samsung['Log_Return'] = calculate_log_return(samsung)
    # EMA, MACD, Log Return 계산
    hinix['EMA_5'] = calculate_ema(hinix, 5)
    hinix['EMA_20'] = calculate_ema(hinix, 20)
    hinix['MACD'] = calculate_macd(hinix, 5, 20)
    hinix['Log_Return'] = calculate_log_return(hinix)
    hinix = hinix.iloc[1:,]
    samsung = samsung.iloc[1:,]
    samsung_train = samsung.iloc[:samsung.index.get_loc('2017-12-28')+1, :]
    samsung_test = samsung.iloc[samsung.index.get_loc('2017-12-28')+1:, :]
    hinix_train = hinix.iloc[:hinix.index.get_loc('2017-12-28')+1, :]
    hinix_test = hinix.iloc[hinix.index.get_loc('2017-12-28')+1:, :]
    # mode : 실행모드
    # ['train', 'test', 'update', 'predict']
    mode = 'predict'
    name = "non_samsung_model"
    rl_method = 'dqn'
    net = 'dnn'
    backend = 'pytorch'
    lr = 0.001
    discount_factor = 0.7
    balance = 50000000
    output_name = f'{mode}_{name}_{rl_method}_{net}'
    learning = mode in ['train', 'update']
    reuse_models = mode in ['test', 'update', 'predict']
    value_network_name = f'{name}_{rl_method}_{net}_value.mdl'
    policy_network_name = f'{name}_{rl_method}_{net}_policy.mdl'
    start_epsilon = 1 if mode in ['train', 'update'] else 0
    num_epoches = 100 if mode in ['train', 'update'] else 1
    num_steps = 5 if net in ['lstm', 'cnn'] else 1
    os.environ['RLTRADER_BACKEND'] = backend
    base_path = '/Users/pjy97/Desktop/한양대/1-1/강화학습/project/code/models/'
    output_path = os.path.join('/Users/pjy97/Desktop/한양대/1-1/강화학습/project/code/models/', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)


    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(base_path, 'models', value_network_name)
    policy_network_path = os.path.join(base_path, 'models', policy_network_name)
    LOGGER_NAME = 'rltrader'
    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from learner import DQNLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []
    assert len(samsung_test) >= num_steps
    # 최소/최대 단일 매매 금액 설정
    min_trading_price = 100000
    max_trading_price = 10000000

    # 공통 파라미터 설정
    common_params = {'rl_method': rl_method,
        'net': net, 'num_steps': num_steps, 'lr': lr,
        'balance': balance, 'num_epoches': num_epoches,
        'discount_factor': discount_factor, 'start_epsilon': start_epsilon,
        'output_path': output_path, 'reuse_models': reuse_models}

    samsung_train['date'] = samsung_train.index
    samsung_train.reset_index(drop=True, inplace=True)
    samsung_test['date'] = samsung_test.index
    samsung_test.reset_index(drop=True, inplace=True)
    hinix_train['date'] = hinix_train.index
    hinix_train.reset_index(drop=True, inplace=True)
    hinix_test['date'] = hinix_test.index
    hinix_test.reset_index(drop=True, inplace=True)

    # print(samsung_test.shape, samsung_train.shape)
    # 강화학습 시작
    learner = None
    common_params.update({
        'chart_data': samsung_test,
        'training_data': samsung_test,
        'min_trading_price': min_trading_price,
        'max_trading_price': max_trading_price})
    learner = DQNLearner(**{**common_params, 'value_network_path': value_network_path})
    if mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if mode in ['train', 'update']:
            learner.save_models()
    elif mode == 'predict':
        learner.predict()

