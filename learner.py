# ref: https://github.com/quantylab/rltrader/blob/master/src/quantylab/rltrader/learners.py
import os
import logging
import abc
import collections
import threading
import time
import json
import numpy as np
from tqdm import tqdm
from enviroment import Environment
from agent import Agent
from network import Network, DNN

LOGGER_NAME = 'rltrader'
logger = logging.getLogger(LOGGER_NAME)


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()
    """
    rl_method: 강화학습 방법
    stock_code: 주식 종목 코드
    chart_data: 차트 데이터
    training_data: 학습 데이터
    min_trading_price: 최소 투자 금액
    max_trading_price: 최대 투자 금액
    net: 신경망 모델
    num_steps : 학습 데이터 생성 시 사용할 시간 간격 크기
    discount_factor: 할인율
    num_epoches: 학습 에포크 수
    balance: 초기 자본금
    start_epsilon: 초기 탐험 비율
    value_network: 가치 신경망
    policy_network: 정책 신경망
    output_path : 가시화 결과 저장할 경로
    reuse_models: 기존에 학습한 모델 재사용 여부
    gen_output: 가시화 결과 생성 여부
    """
    def __init__(self, rl_method='dqn', training_data=None, chart_data=None,
                min_trading_price=100000, max_trading_price=10000000,
                net='dnn', num_steps=1, lr=0.0005,
                discount_factor=0.9, num_epoches=1000,
                balance=50000000, start_epsilon=1,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True, gen_output=True):
        # 인자 확인
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]-1
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        # 로그 등 출력 경로
        self.output_path = output_path
        self.gen_output = gen_output

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid',
                            loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        # 에포크마다 새로 쌓이는 변수를 초기화하는 함수
        self.sample = None
        # 데이터를 처음부터 읽기 위함
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        # 무작위 탐험 횟수 초기화
        self.exploration_cnt = 0
        self.batch_size = 0

    # 학습데이터를 구성하는 함수
    def build_sample(self):
        # 현재 인덱스에서 다음 인덱스를 읽어옴
        self.environment.observe()
        # 학습 데이터의 다음 인덱스 존재 여부 확인
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()[:-1]
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self):
        pass

    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)

            self.loss = loss

    def run(self, learning=True):
        # learning : 학습 여부
        # learning이 True이면 학습을 진행하고 False이면 학습된 모델을 통해 투자 전략을 수행
        info = (
            f'[RL:{self.rl_method} NET:{self.net} '
            f'LR:{self.lr} DF:{self.discount_factor} '
        )
        with self.lock:
            logger.debug(info)

        # 시작 시간
        time_start = time.time()

        # 학습에 대한 정보 초기화
        # 수행한 epoch중 가장 높은 가치가 저장됨
        max_portfolio_value = 0
        # 수익이 발생한 epoch의 수
        epoch_win_cnt = 0

        # 에포크 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon

            for i in tqdm(range(len(self.training_data)), leave=False):
                # 샘플 생성

                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                # 정책 신경망의 출력이 없으면 가치 신경망의 출력(손익률)
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 보상 획득
                reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                # 매수 매도인지 기억
                self.memory_action.append(action)
                # 포트폴리오 가치 기억
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            # 에포크 종료 후 학습
            if learning:
                self.fit()

            # 에포크 관련 정보 로그 기록(주식 종목코드, 에포크, 탐험률, 매수행동 수행횟스, 매도 수행 횟수, 관망 횟수, 보유 주식수, 달성한 포트폴리오 가치, 학습 손실, 에포크 소요시간
            num_epoches_digit = len(str(self.num_epoches))
            # 문자열을 정리해주는 함수
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(f'[Epoch {epoch_str}/{self.num_epoches}] '
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')


            # 학습 관련 정보 갱신 - 포트폴리오 가치를 갱신하고 가치가 자본금보다 높으면 epoch_win_cnt를 증가시킴
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self):
        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)

        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample)).tolist()
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample)).tolist()

            # 신경망에 의한 행동 결정
            result.append((self.environment.observation[0], pred_value, pred_policy))

        if self.gen_output:
            with open(os.path.join(self.output_path, f'pred_non_samsung.json'), 'w') as f:
                print(json.dumps(result), file=f)

        return result

# 가치 신경망으로만 학습
class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # 가치 신경망의 경로
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self):
        # 메모리얼 배열 묶어줌, 역으로 묶음
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        # 샘플 배열과 레이블 배열에 값을 채워준다.
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            # 보상을 구해 저장 마지막 수익률 - 현재 수익률
            r = self.memory_reward[-1] - reward
            y_value[i] = value
            # 다음 행동 수행 시점과 현재 행동 수행 시점의 손익률을 빼서 더한다.
            y_value[i, action] = r + self.discount_factor * value_max_next
            value_max_next = value.max()
            # 정책 신경망 사용 X
        return x, y_value, None
